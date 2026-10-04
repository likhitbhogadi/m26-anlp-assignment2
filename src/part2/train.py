"""Pretrains the Q1 dense transformer on HAP-E with one of the from-scratch optimizers.

Usage:
    uv run python -m src.part2.train --optimizer adamw                                   # full 1x run
    uv run python -m src.part2.train --optimizer lion --lr 1.2e-4 --max-steps 500 --no-bleu --no-save  # LR probe
    uv run python -m src.part2.train --optimizer muon --max-steps 50 --no-wandb --no-save                # smoke test

Every optimizer sees the same packed batches in the same order from the same init. Evals at
step 0 and every 1/n_evals of the run (every 0.1x of the dataset in a full run):
  - val loss over the whole packed val stream (+ human-written vs LLM-written tokens, per genre);
  - test BLEU: greedy 64-token continuation of the last 128 tokens of each test doc's chunk_1,
    scored (sacreBLEU, 13a) against 7 references (human chunk_2 + 6 LLM continuations), and
    against the human continuation alone.
Writes `<output_dir>/{train_log,eval_log,samples}_<run>.csv`, upserts `results.csv`, and saves
the final weights to `<checkpoint_dir>/<run>/`.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import shutil
import time

import numpy as np
import sacrebleu
import torch

from src.part1.common import get_device, load_config
from src.part1.model import Transformer, TransformerConfig, count_params, prefill_mask
from src.part1.train import CSVLog, lr_at, target_ce
from src.part2 import data
from src.part2.data import EOS, PAD
from src.part2.optimizers import NAMES, build_optimizer, set_lr, state_bytes
from src.part3.logging_utils import setup_run_log

SOURCES = ["human", "llm"]


@torch.no_grad()
def eval_loss(model, stream, src, genre, genres: list[str], seq_len: int, batch_size: int, device) -> dict:
    """Mean next-token CE over all val windows, overall / by source / by genre."""
    model.eval()
    sums = torch.zeros(len(SOURCES), len(genres), device=device, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    n = data.n_windows(stream, seq_len)
    for i in range(0, n, batch_size):
        x, y, s, g = (t.to(device) for t in data.window_batch(stream, np.arange(i, min(i + batch_size, n)),
                                                             seq_len, src, genre))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        ce = target_ce(logits, y, reduction="none").double()
        cell = (s * len(genres) + g).flatten()
        sums += torch.bincount(cell, weights=ce, minlength=sums.numel()).view_as(sums)
        counts += torch.bincount(cell, minlength=sums.numel()).view_as(counts).double()
    model.train()
    out = {"loss": (sums.sum() / counts.sum()).item()}
    for si, s in enumerate(SOURCES):
        out[f"loss_{s}"] = (sums[si].sum() / counts[si].sum()).item()
    for gi, g in enumerate(genres):
        out[f"loss_{g}"] = (sums[:, gi].sum() / counts[:, gi].sum()).item()
    return out


@torch.no_grad()
def greedy_continue(model, prompts: list[list[int]], n_new: int, device) -> list[list[int]]:
    """Batched greedy continuation with a KV cache: exactly n_new tokens, <eos>/<pad> banned
    (every prompt is cut mid-document, so the text should go on). Left-padded prompts with
    RoPE positions starting at each row's first real token (same scheme as Q1's decoder)."""
    model.eval()
    B, P = len(prompts), max(len(p) for p in prompts)
    ids = torch.full((B, P), PAD, dtype=torch.long)
    valid = torch.zeros(B, P, dtype=torch.bool)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p)
        valid[i, P - len(p):] = True
    ids, valid = ids.to(device), valid.to(device)
    positions = (valid.long().cumsum(1) - 1).clamp_min(0)
    caches = [{} for _ in model.blocks]
    ac = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def pick(logits):
        logits = logits[:, -1].float()
        logits[:, [PAD, EOS]] = -float("inf")
        return logits.argmax(-1)

    with torch.autocast(**ac):
        logits, _ = model(ids, valid=valid, positions=positions, attn_mask=prefill_mask(valid), caches=caches)
    nxt = pick(logits)
    key_valid, pos = valid, positions[:, -1]
    ones = torch.ones(B, 1, dtype=torch.bool, device=device)
    out = [nxt]
    for _ in range(n_new - 1):
        key_valid, pos = torch.cat([key_valid, ones], 1), pos + 1
        with torch.autocast(**ac):
            logits, _ = model(nxt[:, None], valid=ones, positions=pos[:, None],
                              attn_mask=key_valid[:, None, None, :], caches=caches)
        nxt = pick(logits)
        out.append(nxt)
    model.train()
    return torch.stack(out, 1).tolist()


def eval_bleu(model, tok, bleu: dict, ecfg: dict, device) -> tuple[dict, list[str]]:
    items = bleu["items"]
    prompts = [it["prompt_ids"] for it in items]
    hyps = [None] * len(items)
    order = np.argsort([len(p) for p in prompts])  # similar lengths per batch -> little padding
    bs = ecfg["gen_batch_size"]
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        for j, ids in zip(idx, greedy_continue(model, [prompts[j] for j in idx], ecfg["gen_tokens"], device)):
            hyps[j] = data.decode(tok, ids)
    refs = [it["refs"] for it in items]
    multi = sacrebleu.corpus_bleu(hyps, [list(r) for r in zip(*refs)], tokenize="13a")
    human = sacrebleu.corpus_bleu(hyps, [[r[0] for r in refs]], tokenize="13a")
    out = {"bleu": multi.score, "bleu_human": human.score, "bleu_bp": multi.bp,
           **{f"bleu_p{n + 1}": p for n, p in enumerate(multi.precisions)}}
    return out, hyps


def upsert_csv(row: dict, path: str, key: str = "run"):
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r[key] != row[key]]
    rows = old + [row]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part2/config.yaml")
    parser.add_argument("--optimizer", required=True, choices=NAMES)
    parser.add_argument("--lr", type=float, default=None, help="override the optimizer's peak lr (LR probe)")
    parser.add_argument("--max-steps", type=int, default=None, help="shortened run; schedule compressed to it")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tags", default="", help="extra comma-separated WandB tags")
    parser.add_argument("--no-bleu", action="store_true", help="skip the BLEU evals (LR probes)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg, ecfg = cfg["train"], cfg["eval"]
    hp = dict(cfg["optimizers"][args.optimizer])
    if args.lr is not None:
        hp["lr"] = args.lr
    run_name = args.run_name or f"{args.optimizer}_lr{hp['lr']:g}"
    setup_run_log(cfg["log_dir"], f"train_{run_name}", {"optimizer": args.optimizer, "hp": hp, "train": tcfg,
                                                       "eval": ecfg, "model": cfg["model"]})
    device = get_device(cfg, args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = data.load_tokenizer(cfg)
    meta = data.load_meta(cfg)
    L, B = tcfg["seq_len"], tcfg["batch_size"]
    train_stream = data.load_stream(cfg, "train")
    val = [data.load_stream(cfg, "val", s) for s in ("", "_src", "_genre")]
    test = [data.load_stream(cfg, "test", s) for s in ("", "_src", "_genre")]
    bleu_items = data.load_bleu_items(cfg)
    total_steps = data.n_windows(train_stream, L) // B          # 1x dataset (partial last batch dropped)
    run_steps = min(total_steps, args.max_steps or total_steps)
    sched = {"lr": 1.0, "warmup_steps": max(1, round(tcfg["warmup_frac"] * run_steps)),
             "min_lr_ratio": tcfg["min_lr_ratio"]}
    eval_steps = sorted({round(k * run_steps / tcfg["n_evals"]) for k in range(tcfg["n_evals"] + 1)})
    dataset_tokens = total_steps * B * L
    print(f"train stream {len(train_stream):,} tokens -> {total_steps} steps x {B * L} tokens (1x = {dataset_tokens:,}); "
          f"running {run_steps} steps, evals at {eval_steps}")

    torch.manual_seed(tcfg["seed"])
    mcfg = TransformerConfig(vocab_size=tok.get_vocab_size(), n_ctx=L, **cfg["model"])
    model = Transformer(mcfg).to(device)
    params = count_params(model)
    chinchilla_tokens = 20 * params["total_params"]
    print(f"model: " + ", ".join(f"{k}={v:,}" for k, v in params.items()) +
          f"; 1x dataset = {dataset_tokens / chinchilla_tokens:.4f}x Chinchilla")
    opt = build_optimizer(args.optimizer, model, hp)
    print(f"optimizer {args.optimizer}: {hp}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    train_log = CSVLog(os.path.join(cfg["output_dir"], f"train_log_{run_name}.csv"))
    eval_log = CSVLog(os.path.join(cfg["output_dir"], f"eval_log_{run_name}.csv"))

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], group=cfg["wandb"]["group"],
            name=run_name, tags=["part2", args.optimizer, *[t for t in args.tags.split(",") if t]],
            config={"optimizer": args.optimizer, "hp": hp, "train": tcfg, "eval": ecfg, "model": dataclasses.asdict(mcfg),
                    "data": meta, **params, "run_steps": run_steps, "total_steps": total_steps,
                    "dataset_tokens": dataset_tokens},
        )
        wandb_run.define_metric("tokens_seen")
        for prefix in ("train/*", "eval/*"):
            wandb_run.define_metric(prefix, step_metric="tokens_seen")

    tokens_seen = 0
    step_time = 0.0  # optimizer-step + fwd/bwd wall time, excluding evals

    def run_eval(step, final=False):
        t0 = time.perf_counter()
        vals = {f"val_{k}": v for k, v in eval_loss(model, *val, meta["genres"], L, tcfg["eval_batch_size"],
                                                     device).items()}
        hyps = None
        if not args.no_bleu:
            b, hyps = eval_bleu(model, tok, bleu_items, ecfg, device)
            vals.update(b)
        row = {"step": step, "tokens_seen": tokens_seen, "dataset_fraction": round(tokens_seen / dataset_tokens, 4),
               "chinchilla_ratio": round(tokens_seen / chinchilla_tokens, 5), **{k: round(v, 5) for k, v in vals.items()}}
        eval_log.write(row)
        print(f"  eval @ {step} ({row['dataset_fraction']:.2f}x, {time.perf_counter() - t0:.0f}s): val_loss={vals['val_loss']:.4f}"
              + (f" bleu={vals['bleu']:.2f} bleu_human={vals['bleu_human']:.2f}" if hyps else ""), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": v for k, v in row.items() if k not in ("step", "tokens_seen")}
                          | {"tokens_seen": tokens_seen}, step=step)
        if final and hyps is not None:
            with open(os.path.join(cfg["output_dir"], f"samples_{run_name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["doc", "prompt_end", "hypothesis", "ref_human", "ref_llm0"])
                w.writeheader()
                for it, h in list(zip(bleu_items["items"], hyps))[: ecfg["n_samples"]]:
                    w.writerow({"doc": it["doc"], "prompt_end": it["prompt_text"][-200:], "hypothesis": h,
                                "ref_human": it["refs"][0], "ref_llm0": it["refs"][1]})
        return vals

    run_eval(0)
    acc = {"loss": 0.0, "gn": 0.0, "n": 0, "t": 0.0}
    model.train()
    for step, (x, y) in enumerate(data.train_batches(train_stream, L, B, run_steps)):
        t_step = time.perf_counter()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        mult = lr_at(step, run_steps, sched)
        set_lr(opt, mult)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        loss = target_ce(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t_step
        step_time += dt
        acc["t"] += dt
        tokens_seen += x.numel()
        acc["loss"] += loss.detach()
        acc["gn"] += grad_norm.detach()
        acc["n"] += 1

        done = step + 1
        if done % tcfg["log_every"] == 0 or done == run_steps:
            n = acc["n"]
            row = {"step": done, "tokens_seen": tokens_seen, "dataset_fraction": round(tokens_seen / dataset_tokens, 4),
                   "lr_mult": round(mult, 5), "train_loss": round(float(acc["loss"]) / n, 5),
                   "grad_norm": round(float(acc["gn"]) / n, 5), "tokens_per_s": round(n * B * L / max(acc["t"], 1e-9), 1)}
            train_log.write(row)
            if wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in row.items() if k not in ("step", "tokens_seen")}
                              | {"tokens_seen": tokens_seen}, step=done)
            print(f"step {done}/{run_steps} loss={row['train_loss']:.4f} gn={row['grad_norm']:.3f} lr_mult={mult:.3f} "
                  f"{row['tokens_per_s']:,.0f} tok/s tokens={tokens_seen:,}", flush=True)
            acc = {"loss": 0.0, "gn": 0.0, "n": 0, "t": 0.0}
        if done in eval_steps:
            vals = run_eval(done, final=done == run_steps)

    test_vals = eval_loss(model, *test, meta["genres"], L, tcfg["eval_batch_size"], device)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    opt_mb = state_bytes(opt) / 2**20
    print(f"done: tokens_seen={tokens_seen:,} test_loss={test_vals['loss']:.4f} "
          f"{1000 * step_time / run_steps:.1f} ms/step, optimizer state {opt_mb:.1f} MiB, peak mem {peak:.1f} GiB")

    result = {"run": run_name, "optimizer": args.optimizer, "lr": hp["lr"], "run_steps": run_steps,
              "tokens_seen": tokens_seen, "dataset_fraction": round(tokens_seen / dataset_tokens, 4),
              **{k: round(v, 5) for k, v in vals.items()}, "test_loss": round(test_vals["loss"], 5),
              "test_ppl": round(math.exp(test_vals["loss"]), 3), "ms_per_step": round(1000 * step_time / run_steps, 1),
              "opt_state_mib": round(opt_mb, 1), "peak_mem_gib": round(peak, 2), "total_params": params["total_params"],
              "hp": json.dumps(hp)}
    upsert_csv(result, os.path.join(cfg["output_dir"], "results.csv"))

    if not args.no_save:
        out_dir = os.path.join(cfg["checkpoint_dir"], run_name)
        os.makedirs(out_dir, exist_ok=True)
        from safetensors.torch import save_model

        save_model(model, os.path.join(out_dir, "model.safetensors"))
        shutil.copy(cfg["tokenizer"]["path"], os.path.join(out_dir, "tokenizer.json"))
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"model": dataclasses.asdict(mcfg), "optimizer": args.optimizer, "hp": hp, "train": tcfg,
                       **{k: v for k, v in result.items() if k != "hp"}}, f, indent=2)
        print(f"saved {out_dir}")

    if wandb_run is not None:
        wandb_run.summary.update({f"final/{k}": v for k, v in result.items() if k not in ("run", "hp")})
        wandb_run.finish()
    train_log.close()
    eval_log.close()


if __name__ == "__main__":
    main()
