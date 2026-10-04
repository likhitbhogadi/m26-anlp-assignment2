"""Evaluates one trained variant on the test set.

Usage:
    uv run python -m src.part1.evaluate --variant dense
    uv run python -m src.part1.evaluate --variant moe_4e_top2 --bleu-n 200   # quick check

  - perplexity of the target tokens (+ <eos>) over every usable test pair, per source language;
  - BLEU (sacrebleu, 13a) of greedy translations of the fixed BLEU subset; greedy decoding
    is our own batched loop with a KV cache (no HF generate);
  - MoE routing over the full test set: per layer x source language x token role (src = the
    <bos> <lang> source part, tgt = <2en> + English tokens) x routed expert.
Outputs (CSV, in output_dir): results.csv (upsert per variant), routing.csv (upsert per
variant), translations_<variant>.csv.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_model

from src.part1 import data
from src.part1.common import EOS, LANGS, PAD, SPECIAL_TOKENS, get_device, get_variant, load_config, \
    load_tokenizer, model_config
from src.part1.model import Transformer, count_params, prefill_mask
from src.part1.train import to_device
from src.part3.logging_utils import setup_run_log

ROLES = ["src", "tgt"]


def load_trained(cfg: dict, variant: dict, vocab_size: int, device, run_name: str | None = None):
    model = Transformer(model_config(cfg, variant, vocab_size))
    load_model(model, os.path.join(cfg["checkpoint_dir"], run_name or variant["name"], "model.safetensors"))
    return model.to(device).eval()


def upsert_csv(rows: list[dict], path: str, variant: str):
    """Replaces every row of `variant`, keeping the other variants' rows."""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["variant"] != variant]
    rows = old + rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def perplexity_and_routing(model, split: data.Split, pairs: np.ndarray, batch_size: int, device):
    """Target CE per language; for MoE models also routed-expert counts / gate sums of shape
    (layers, langs, roles, experts) and per-token-id counts (layers, experts, vocab)."""
    c = model.config
    ce_sum, n_tok = torch.zeros(len(LANGS)), torch.zeros(len(LANGS))
    counts = gate_sum = tok_counts = None
    if c.use_moe:
        tok_counts = torch.zeros(c.n_layers, c.n_experts * c.vocab_size, dtype=torch.long, device=device)
        counts = torch.zeros(c.n_layers, len(LANGS), len(ROLES), c.n_experts, dtype=torch.long, device=device)
        gate_sum = torch.zeros(c.n_layers, len(LANGS), len(ROLES), c.n_experts, device=device)
    # length-sorted batches: far less padding, same result
    order = np.argsort([len(split.pair(int(t), int(l))[0]) + len(split.pair(int(t), int(l))[1]) for t, l in pairs])
    for chunk in data.batches(pairs[order], batch_size):
        b = to_device(data.collate(split, chunk), device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, aux = model(b["input_ids"], valid=b["valid"])
        # CE on the labelled positions only: the fp32 copy of the full (B, T, V) logits is what
        # fragments the CUDA cache when every length-sorted batch has a new shape
        has_label = b["labels"] != -100
        ce = F.cross_entropy(logits[has_label].float(), b["labels"][has_label], reduction="none")
        tok_lang = b["lang"][:, None].expand_as(has_label)[has_label]
        del logits
        for l in range(len(LANGS)):
            m = tok_lang == l
            ce_sum[l] += ce[m].sum().cpu()
            n_tok[l] += m.sum().cpu()
        if counts is not None:
            valid = b["valid"]
            lang = b["lang"][:, None].expand_as(valid)[valid]       # (N,) same order as the MoE's tokens
            role = has_label[valid].long()                          # 0 = src side, 1 = tgt side
            ids = b["input_ids"][valid]
            for li, info in enumerate(aux["layers"]):
                k = info["top_e"].shape[1]
                cell = ((lang * len(ROLES) + role) * c.n_experts)[:, None] + info["top_e"]   # (N, k)
                size = len(LANGS) * len(ROLES) * c.n_experts
                counts[li] += torch.bincount(cell.flatten(), minlength=size).view_as(counts[li])
                gate_sum[li] += torch.bincount(cell.flatten(), weights=info["gates"].flatten().float(),
                                               minlength=size).view_as(gate_sum[li])
                assert k == c.top_k
                tok_counts[li] += torch.bincount((info["top_e"] * c.vocab_size + ids[:, None]).flatten(),
                                                 minlength=c.n_experts * c.vocab_size)
    ce = {lang: (ce_sum[l] / n_tok[l]).item() for l, lang in enumerate(LANGS)}
    ce["all"] = (ce_sum.sum() / n_tok.sum()).item()
    n = {lang: int(n_tok[l]) for l, lang in enumerate(LANGS)} | {"all": int(n_tok.sum())}
    if tok_counts is not None:
        tok_counts = tok_counts.view(c.n_layers, c.n_experts, c.vocab_size)
    return ce, n, counts, gate_sum, tok_counts


def token_rows(tok, name: str, tok_counts: torch.Tensor, top_k: int, min_freq: int, top: int) -> list[dict]:
    """For each layer x expert: the frequent tokens (>= min_freq test occurrences) most often
    routed to it. purity = share of the token's occurrences that use this expert
    (1 = always; uniform routing gives top_k / n_experts)."""
    tok_counts = tok_counts.cpu().double()
    L, E, _ = tok_counts.shape
    rows = []
    for li in range(L):
        occ = tok_counts[li].sum(0) / top_k                                # occurrences per token id
        frequent = occ >= min_freq
        for e in range(E):
            purity = torch.where(frequent, tok_counts[li, e] / occ.clamp_min(1), torch.zeros_like(occ))
            for rank, t in enumerate(purity.argsort(descending=True)[:top].tolist()):
                rows.append({"variant": name, "layer": li, "expert": e, "rank": rank, "token_id": t,
                             "token": tok.id_to_token(t), "freq": int(occ[t]), "purity": round(purity[t].item(), 4)})
    return rows


@torch.no_grad()
def greedy_translate(model, prompts: list[list[int]], max_new_tokens: int, device) -> list[list[int]]:
    """Batched greedy decoding with a KV cache. Prompts are left-padded; RoPE positions
    start at 0 at each row's first real token, so results match unbatched decoding."""
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
    with torch.autocast(**ac):
        logits, _ = model(ids, valid=valid, positions=positions, attn_mask=prefill_mask(valid), caches=caches)
    nxt = logits[:, -1].float().argmax(-1)
    key_valid, pos = valid, positions[:, -1]
    ones = torch.ones(B, 1, dtype=torch.bool, device=device)
    out, done = [], torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        out.append(nxt)
        done |= nxt == EOS
        if bool(done.all()):
            break
        key_valid, pos = torch.cat([key_valid, ones], 1), pos + 1
        with torch.autocast(**ac):
            logits, _ = model(nxt[:, None], valid=ones, positions=pos[:, None],
                              attn_mask=key_valid[:, None, None, :], caches=caches)
        nxt = logits[:, -1].float().argmax(-1)
    gen = torch.stack(out, 1).tolist()
    return [row[: row.index(EOS)] if EOS in row else row for row in gen]


def decode(tok, ids: list[int]) -> str:
    # drop our control tokens but keep byte-fallback tokens (also "special" in the vocab)
    return tok.decode([i for i in ids if i >= len(SPECIAL_TOKENS)], skip_special_tokens=False).strip()


def write_routing(cfg, tok, name, model, counts, gate_sum, tok_counts):
    rrows = []
    counts, gate_sum = counts.cpu(), gate_sum.cpu()
    for li in range(counts.shape[0]):
        for l, lang in enumerate(LANGS):
            for r, role in enumerate(ROLES):
                total = counts[li, l, r].sum().item()
                for e in range(counts.shape[-1]):
                    cnt = counts[li, l, r, e].item()
                    rrows.append({"variant": name, "layer": li, "src_lang": lang, "token_role": role, "expert": e,
                                  "count": cnt, "frac": round(cnt / max(total, 1), 6),
                                  "mean_gate": round(gate_sum[li, l, r, e].item() / max(cnt, 1), 6)})
    upsert_csv(rrows, os.path.join(cfg["output_dir"], "routing.csv"), name)
    upsert_csv(token_rows(tok, name, tok_counts, model.config.top_k, min_freq=200, top=15),
               os.path.join(cfg["output_dir"], "routing_tokens.csv"), name)
    print("routing (all tokens, per layer):")
    for li in range(counts.shape[0]):
        frac = counts[li].sum((0, 1)).float()
        print(f"  layer {li}: " + " ".join(f"{x:.3f}" for x in (frac / frac.sum()).tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-name", default=None, help="checkpoint folder; defaults to the variant name")
    parser.add_argument("--bleu-n", type=int, default=None, help="limit BLEU pairs per language (quick checks)")
    parser.add_argument("--routing-only", action="store_true",
                        help="only the teacher-forced pass: rewrite routing CSVs, keep results/translations")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = get_variant(cfg, args.variant)
    name = args.run_name or variant["name"]
    setup_run_log(cfg["log_dir"], f"evaluate_{name}", {"variant": variant, "eval": cfg["eval"]})
    device = get_device(cfg, args.device)
    tok = load_tokenizer(cfg)
    model = load_trained(cfg, variant, tok.get_vocab_size(), device, name)
    params = count_params(model)
    test = data.Split(cfg, "test")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # 1. perplexity + routing over every usable test pair
    t0 = time.perf_counter()
    ce, n_tok, counts, gate_sum, tok_counts = perplexity_and_routing(
        model, test, data.load_pairs(cfg, "test"), cfg["train"]["eval_batch_size"], device)
    if device.type == "cuda":
        torch.cuda.empty_cache()  # hand the PPL pass's cached blocks back before decoding
    print(f"test CE/PPL ({time.perf_counter() - t0:.0f}s): " +
          ", ".join(f"{k}: {v:.4f}/{math.exp(v):.2f}" for k, v in ce.items()))

    if args.routing_only:
        if counts is not None:
            write_routing(cfg, tok, name, model, counts, gate_sum, tok_counts)
        return

    # 2. greedy translations + BLEU on the fixed subset
    raw = load_dataset(cfg["dataset"], split="test")
    bleu_pairs = data.load_pairs(cfg, "bleu")
    if args.bleu_n:
        bleu_pairs = np.concatenate([bleu_pairs[bleu_pairs[:, 1] == l][: args.bleu_n] for l in range(len(LANGS))])
    prompts = [test.prompt(int(t), int(l)) for t, l in bleu_pairs]
    order = np.argsort([len(p) for p in prompts])
    hyps = [None] * len(prompts)
    t0 = time.perf_counter()
    bs = cfg["eval"]["gen_batch_size"]
    for i in range(0, len(order), bs):
        idx = order[i : i + bs]
        for j, ids in zip(idx, greedy_translate(model, [prompts[j] for j in idx], cfg["eval"]["max_new_tokens"], device)):
            hyps[j] = decode(tok, ids)
    decode_time = time.perf_counter() - t0
    print(f"decoded {len(hyps)} translations in {decode_time:.0f}s")

    rows, results = [], []
    for l, lang in [*enumerate(LANGS), (None, "all")]:
        sel = [i for i, (_, pl) in enumerate(bleu_pairs) if l is None or pl == l]
        refs = [raw[int(bleu_pairs[i][0])]["en"] for i in sel]
        bleu = sacrebleu.corpus_bleu([hyps[i] for i in sel], [refs], tokenize="13a")
        print(f"BLEU {lang}: {bleu.score:.2f}  ({bleu})")
        results.append({
            "variant": name, "lang": lang, **params,
            "test_ce": round(ce[lang], 5), "test_ppl": round(math.exp(ce[lang]), 4), "test_target_tokens": n_tok[lang],
            "bleu": round(bleu.score, 3), "bleu_bp": round(bleu.bp, 4),
            **{f"bleu_p{n + 1}": round(p, 3) for n, p in enumerate(bleu.precisions)},
            "bleu_n_samples": len(sel), "decode_time_s": round(decode_time, 1),
        })
        if l is None:
            continue
        for i in sel:
            t = int(bleu_pairs[i][0])
            rows.append({"variant": name, "idx": t, "lang": lang, "source": raw[t][lang], "reference": raw[t]["en"],
                         "hypothesis": hyps[i],
                         "sentence_bleu": round(sacrebleu.sentence_bleu(hyps[i], [raw[t]["en"]]).score, 2)})

    upsert_csv(results, os.path.join(cfg["output_dir"], "results.csv"), name)
    with open(os.path.join(cfg["output_dir"], f"translations_{name}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows[:3] + rows[-3:]:
        print(f"  [{r['lang']}] {r['source']}\n     ref: {r['reference']}\n     hyp: {r['hypothesis']}")

    if counts is not None:
        write_routing(cfg, tok, name, model, counts, gate_sum, tok_counts)
    print(f"wrote results / translations{' / routing' if counts is not None else ''} to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
