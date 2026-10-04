"""Trains one FFN variant on the fixed 30M-token vi/ja->en training prefix.

Usage:
    uv run python -m src.part1.train --variant dense
    uv run python -m src.part1.train --variant moe_4e_top2 --max-steps 200 --no-wandb   # smoke test

Every variant sees the same pairs in the same order (fixed in data.py --prepare) and starts
from the same seed, so `tokens_seen` at the end is identical across runs. Writes
`<output_dir>/{train_log,val_log,train_routing}_<variant>.csv` and the final weights to
`<checkpoint_dir>/<variant>/` (model.safetensors + config.json).
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_model

from src.part1 import data
from src.part1.common import LANGS, get_device, get_variant, load_config, load_tokenizer, model_config
from src.part1.model import Transformer, count_params
from src.part3.logging_utils import setup_run_log


class CSVLog:
    """Appends dict rows to a CSV (header from the first row); truncated at start."""

    def __init__(self, path: str):
        self.path, self.writer, self.file = path, None, None

    def write(self, rows: list[dict] | dict):
        rows = [rows] if isinstance(rows, dict) else rows
        if not rows:
            return
        if self.writer is None:
            self.file = open(self.path, "w", newline="", buffering=1)
            self.writer = csv.DictWriter(self.file, fieldnames=list(rows[0]))
            self.writer.writeheader()
        self.writer.writerows(rows)

    def close(self):
        if self.file is not None:
            self.file.close()


def lr_at(step: int, total: int, tcfg: dict) -> float:
    """Linear warmup, then cosine from lr down to lr * min_lr_ratio."""
    lr, warm = tcfg["lr"], tcfg["warmup_steps"]
    if step < warm:
        return lr * (step + 1) / warm
    t = min(1.0, (step - warm) / max(1, total - warm))
    return lr * (tcfg["min_lr_ratio"] + (1 - tcfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * t)))


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def target_ce(logits, labels, reduction="mean"):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction=reduction)


@torch.no_grad()
def evaluate_ce(model, split: data.Split, pairs, batch_size: int, device) -> dict[str, float]:
    """Mean target-token CE / PPL per source language and overall."""
    model.eval()
    sums, counts = torch.zeros(len(LANGS)), torch.zeros(len(LANGS))
    for chunk in data.batches(pairs, batch_size):
        b = to_device(data.collate(split, chunk), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(b["input_ids"], valid=b["valid"])
        ce = target_ce(logits, b["labels"], reduction="none").view_as(b["labels"])
        n = (b["labels"] != -100).float()
        for l in range(len(LANGS)):
            m = b["lang"] == l
            sums[l] += ce[m].sum().cpu()
            counts[l] += n[m].sum().cpu()
    model.train()
    out = {"val_ce": (sums.sum() / counts.sum()).item()}
    for l, lang in enumerate(LANGS):
        out[f"val_ce_{lang}"] = (sums[l] / counts[l]).item()
    for k in list(out):
        out[k.replace("ce", "ppl")] = math.exp(out[k])
    return out


def routing_rows(step: int, layers: list[dict], counts: list[torch.Tensor], probs: list[torch.Tensor],
                 entropy: list[float], n_batches: int) -> list[dict]:
    rows = []
    for li in range(len(layers)):
        frac = counts[li] / counts[li].sum().clamp_min(1)
        for e in range(len(frac)):
            rows.append({"step": step, "layer": li, "expert": e, "load_frac": round(frac[e].item(), 6),
                         "mean_prob": round((probs[li][e] / n_batches).item(), 6),
                         "entropy": round(float(entropy[li]) / n_batches, 6)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests / sweeps)")
    parser.add_argument("--lr", type=float, default=None, help="override train.lr (LR sweep)")
    parser.add_argument("--run-name", default=None, help="defaults to the variant name")
    parser.add_argument("--tags", default="", help="extra comma-separated WandB tags")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = dict(cfg["train"])
    if args.lr is not None:
        tcfg["lr"] = args.lr
    variant = get_variant(cfg, args.variant)
    run_name = args.run_name or variant["name"]
    setup_run_log(cfg["log_dir"], f"train_{run_name}", {**cfg, "train": tcfg, "variant": variant})
    device = get_device(cfg, args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = load_tokenizer(cfg)
    meta = data.load_meta(cfg)
    train_split, val_split = data.Split(cfg, "train"), data.Split(cfg, "validation")
    train_pairs, val_pairs = data.load_pairs(cfg, "train"), data.load_pairs(cfg, "val")
    total_steps = math.ceil(len(train_pairs) / tcfg["batch_size"])
    run_steps = min(total_steps, args.max_steps or total_steps)
    sched_steps = run_steps  # a shortened run still gets a full warmup + cosine
    print(f"train pairs: {len(train_pairs)} ({meta['train_tokens']:,} tokens), "
          f"steps: {run_steps}/{total_steps}, batch {tcfg['batch_size']}")

    torch.manual_seed(tcfg["seed"])
    mcfg = model_config(cfg, variant, tok.get_vocab_size())
    model = Transformer(mcfg).to(device)
    params = count_params(model)
    print(f"model: {variant['name']} " + ", ".join(f"{k}={v:,}" for k, v in params.items()))

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["lr"], betas=tuple(tcfg["betas"]), fused=device.type == "cuda",
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)
    train_log = CSVLog(os.path.join(cfg["output_dir"], f"train_log_{run_name}.csv"))
    val_log = CSVLog(os.path.join(cfg["output_dir"], f"val_log_{run_name}.csv"))
    routing_log = CSVLog(os.path.join(cfg["output_dir"], f"train_routing_{run_name}.csv")) if mcfg.use_moe else None

    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], group=cfg["wandb"]["group"],
            name=run_name, tags=["part1", variant["name"], "moe" if mcfg.use_moe else "dense",
                                 *[t for t in args.tags.split(",") if t]],
            config={"variant": variant, "model": dataclasses.asdict(mcfg), "train": tcfg, "data": meta,
                    **params, "run_steps": run_steps},
        )

    tokens_seen = target_seen = 0
    acc = None  # running sums over the current log window
    t_window = time.perf_counter()
    model.train()

    def reset_acc():
        n_layers = mcfg.n_layers if mcfg.use_moe else 0
        return {"ce": 0.0, "lb": 0.0, "z": 0.0, "gn": 0.0, "n": 0, "tok": 0,
                "counts": [torch.zeros(mcfg.n_experts, device=device) for _ in range(n_layers)],
                "probs": [torch.zeros(mcfg.n_experts, device=device) for _ in range(n_layers)],
                "entropy": [0.0] * n_layers, "layers": list(range(n_layers))}

    def run_val(step):
        vals = evaluate_ce(model, val_split, val_pairs, tcfg["eval_batch_size"], device)
        row = {"step": step, "tokens_seen": tokens_seen, **{k: round(v, 5) for k, v in vals.items()}}
        val_log.write(row)
        print(f"  val @ {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in vals.items()} | {"tokens_seen": tokens_seen}, step=step)
        return vals

    acc = reset_acc()
    for step, chunk in enumerate(data.batches(train_pairs[: run_steps * tcfg["batch_size"]], tcfg["batch_size"])):
        b = to_device(data.collate(train_split, chunk), device)
        lr = lr_at(step, sched_steps, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, aux = model(b["input_ids"], valid=b["valid"])
        ce = target_ce(logits, b["labels"])
        loss = ce + aux["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()

        tokens_seen += b["n_tokens"]
        target_seen += b["n_target"]
        acc["ce"] += ce.detach()
        acc["gn"] += grad_norm.detach()
        acc["n"] += 1
        acc["tok"] += b["n_tokens"]
        if mcfg.use_moe:
            acc["lb"] += aux["lb_loss"]
            acc["z"] += aux["z_loss"]
            for li, info in enumerate(aux["layers"]):
                acc["counts"][li] += info["counts"]
                acc["probs"][li] += info["mean_prob"]
                acc["entropy"][li] += info["entropy"]

        done = step + 1
        if done % tcfg["log_every"] == 0 or done == run_steps:
            n = acc["n"]
            elapsed = time.perf_counter() - t_window
            row = {"step": done, "tokens_seen": tokens_seen, "target_tokens_seen": target_seen, "lr": lr,
                   "train_ce": round(float(acc["ce"]) / n, 5),
                   "aux_lb": round(float(acc["lb"]) / n, 5), "aux_z": round(float(acc["z"]) / n, 5),
                   "grad_norm": round(float(acc["gn"]) / n, 5), "tokens_per_s": round(acc["tok"] / elapsed, 1)}
            train_log.write(row)
            log = {f"train/{k}": v for k, v in row.items() if k != "step"}
            if mcfg.use_moe:
                rrows = routing_rows(done, acc["layers"], acc["counts"], acc["probs"], acc["entropy"], n)
                routing_log.write(rrows)
                for r in rrows:
                    log[f"routing/layer{r['layer']}_expert{r['expert']}_load"] = r["load_frac"]
                for li in acc["layers"]:
                    log[f"routing/layer{li}_entropy"] = float(acc["entropy"][li]) / n
            if wandb_run is not None:
                wandb_run.log(log, step=done)
            print(f"step {done}/{run_steps} ce={row['train_ce']:.4f} lb={row['aux_lb']:.4f} z={row['aux_z']:.3f} "
                  f"gn={row['grad_norm']:.3f} lr={lr:.2e} {row['tokens_per_s']:,.0f} tok/s "
                  f"tokens={tokens_seen:,}", flush=True)
            acc = reset_acc()
            t_window = time.perf_counter()

        if done % tcfg["val_every"] == 0 and done != run_steps:
            run_val(done)
            t_window = time.perf_counter()

    vals = run_val(run_steps)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    print(f"done: tokens_seen={tokens_seen:,} target_tokens_seen={target_seen:,} peak_mem={peak:.1f} GiB")

    if not args.no_save:
        out_dir = os.path.join(cfg["checkpoint_dir"], run_name)
        os.makedirs(out_dir, exist_ok=True)
        save_model(model, os.path.join(out_dir, "model.safetensors"))
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"variant": variant, "model": dataclasses.asdict(mcfg), "train": tcfg,
                       "tokens_seen": tokens_seen, "target_tokens_seen": target_seen, "steps": run_steps,
                       **params, **vals}, f, indent=2)
        print(f"saved {out_dir}")

    if wandb_run is not None:
        wandb_run.summary.update({"tokens_seen": tokens_seen, "target_tokens_seen": target_seen,
                                  "peak_mem_gib": peak, **params, **{f"final/{k}": v for k, v in vals.items()}})
        wandb_run.finish()
    for log in (train_log, val_log, routing_log):
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()
