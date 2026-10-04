"""Tokenized vi->en / ja->en pairs for the decoder-only translation model.

Usage (once, CPU):
    uv run python -m src.part1.data --config src/part1/config.yaml --prepare

Each dataset triplet (en, vi, ja) gives two pairs. A pair is the sequence
    <bos> <vi|ja> src... <2en> tgt... <eos>
with the loss on the target tokens (and <eos>) only. `--prepare` tokenizes every split once
into `<cache_dir>/<split>.npz` and fixes, with `data_seed`:
  - train_pairs.npy: the first triplets of a fixed shuffle whose two pairs together reach
    `token_budget` tokens, as (triplet, lang) rows in a fixed shuffled order (one epoch);
  - val_pairs.npy:   `n_val_pairs` validation pairs for in-training evaluation;
  - test_pairs.npy:  every usable test pair (perplexity + routing stats);
  - bleu_pairs.npy:  `bleu_n_per_lang` test triplets, both languages (BLEU).
Triplets with an empty side or a pair longer than `max_len` are dropped (counts in meta.json).
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_dataset

from src.part1.common import BOS, EOS, LANG_ID, LANG_TOKEN, LANGS, PAD, TO_EN, load_config, load_tokenizer
from src.part3.logging_utils import setup_run_log

SPLITS = ["train", "validation", "test"]
N_SPECIAL_PER_PAIR = 4  # <bos> <lang> <2en> <eos>


# ---------------------------------------------------------------- preparation

def _encode_column(tok, texts: list) -> tuple[np.ndarray, np.ndarray]:
    """Flat int32 ids + int64 offsets (len n+1)."""
    ids, lens = [], []
    for i in range(0, len(texts), 50_000):
        for e in tok.encode_batch([t or "" for t in texts[i : i + 50_000]], add_special_tokens=False):
            ids.append(np.asarray(e.ids, dtype=np.int32))
            lens.append(len(e.ids))
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return np.concatenate(ids), offsets


def _usable(split: dict, max_len: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mask of triplets whose sides are non-empty and whose pairs fit in max_len; pair lengths."""
    en_len = np.diff(split["en_off"])
    pair_len = {l: np.diff(split[f"{l}_off"]) + en_len + N_SPECIAL_PER_PAIR for l in LANGS}
    ok = en_len > 0
    for l in LANGS:
        ok &= (np.diff(split[f"{l}_off"]) > 0) & (pair_len[l] <= max_len)
    return ok, pair_len


def _pairs_of(triplets: np.ndarray) -> np.ndarray:
    return np.array([(t, LANG_ID[l]) for t in triplets for l in LANGS], dtype=np.int64).reshape(-1, 2)


def prepare(cfg: dict):
    dcfg = cfg["data"]
    os.makedirs(dcfg["cache_dir"], exist_ok=True)
    tok = load_tokenizer(cfg)
    meta = {"max_len": dcfg["max_len"], "token_budget": dcfg["token_budget"], "data_seed": dcfg["data_seed"]}

    splits = {}
    for split in SPLITS:
        ds = load_dataset(cfg["dataset"], split=split)
        arrays = {}
        for col in ["en", *LANGS]:
            arrays[f"{col}_ids"], arrays[f"{col}_off"] = _encode_column(tok, list(ds[col]))
        np.savez(os.path.join(dcfg["cache_dir"], f"{split}.npz"), **arrays)
        splits[split] = arrays
        ok, _ = _usable(arrays, dcfg["max_len"])
        meta[f"{split}_triplets"] = len(ds)
        meta[f"{split}_usable_triplets"] = int(ok.sum())
        print(f"{split}: {len(ds)} triplets, {int(ok.sum())} usable ({(~ok).mean():.2%} dropped)")

    rng = np.random.default_rng(dcfg["data_seed"])

    # Train: shortest prefix of a fixed triplet shuffle reaching the token budget.
    ok, pair_len = _usable(splits["train"], dcfg["max_len"])
    order = rng.permutation(len(ok))
    order = order[ok[order]]
    cum = np.cumsum(pair_len["vi"][order] + pair_len["ja"][order])
    if cum[-1] < dcfg["token_budget"]:
        raise SystemExit(f"budget {dcfg['token_budget']} > all usable train tokens {cum[-1]}")
    n_trip = int(np.searchsorted(cum, dcfg["token_budget"]) + 1)
    train_pairs = _pairs_of(order[:n_trip])
    train_pairs = train_pairs[rng.permutation(len(train_pairs))]
    np.save(os.path.join(dcfg["cache_dir"], "train_pairs.npy"), train_pairs)
    en_len = np.diff(splits["train"]["en_off"])
    tgt_tokens = int(2 * (en_len[order[:n_trip]] + 1).sum())  # target + <eos>, both pairs
    batch_size = cfg["train"]["batch_size"]
    meta.update({
        "train_selected_triplets": n_trip,
        "train_pairs": len(train_pairs),
        "train_fraction_of_triplets": n_trip / meta["train_triplets"],
        "train_tokens": int(cum[n_trip - 1]),
        "train_target_tokens": tgt_tokens,
        "mean_pair_len": float(cum[n_trip - 1] / len(train_pairs)),
        "batch_size": batch_size,
        "steps": math.ceil(len(train_pairs) / batch_size),
    })

    # Validation subset (in-training eval) and test sets.
    ok, _ = _usable(splits["validation"], dcfg["max_len"])
    val_trip = rng.permutation(np.flatnonzero(ok))[: dcfg["n_val_pairs"] // 2]
    np.save(os.path.join(dcfg["cache_dir"], "val_pairs.npy"), _pairs_of(np.sort(val_trip)))
    ok, _ = _usable(splits["test"], dcfg["max_len"])
    test_trip = np.flatnonzero(ok)
    np.save(os.path.join(dcfg["cache_dir"], "test_pairs.npy"), _pairs_of(test_trip))
    n_bleu = dcfg["bleu_n_per_lang"] or len(test_trip)
    bleu_trip = np.sort(rng.permutation(test_trip)[:n_bleu])
    np.save(os.path.join(dcfg["cache_dir"], "bleu_pairs.npy"), _pairs_of(bleu_trip))
    meta.update({"val_pairs": 2 * len(val_trip), "test_pairs": 2 * len(test_trip), "bleu_pairs": 2 * len(bleu_trip)})

    with open(os.path.join(dcfg["cache_dir"], "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- loading

def load_meta(cfg: dict) -> dict:
    with open(os.path.join(cfg["data"]["cache_dir"], "meta.json")) as f:
        return json.load(f)


def load_pairs(cfg: dict, name: str) -> np.ndarray:
    """name: train / val / test / bleu -> (n, 2) array of (triplet index, lang id)."""
    return np.load(os.path.join(cfg["data"]["cache_dir"], f"{name}_pairs.npy"))


class Split:
    """Token ids of one split; `pair(t, lang_id)` gives (src_ids, tgt_ids)."""

    def __init__(self, cfg: dict, split: str):
        with np.load(os.path.join(cfg["data"]["cache_dir"], f"{split}.npz")) as z:
            self.arrays = {k: z[k] for k in z.files}

    def _get(self, col: str, t: int) -> np.ndarray:
        off = self.arrays[f"{col}_off"]
        return self.arrays[f"{col}_ids"][off[t] : off[t + 1]]

    def pair(self, t: int, lang_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self._get(LANGS[lang_id], t), self._get("en", t)

    def prompt(self, t: int, lang_id: int) -> list[int]:
        """<bos> <lang> src... <2en>: the generation prompt."""
        src, _ = self.pair(t, lang_id)
        return [BOS, LANG_TOKEN[LANGS[lang_id]], *src.tolist(), TO_EN]


def collate(split: Split, pairs: np.ndarray) -> dict[str, torch.Tensor]:
    """Right-padded teacher-forcing batch. inputs = seq[:-1], labels = seq[1:] with the
    source part (and padding) set to -100; `valid` marks non-pad inputs. Positions with a
    label are exactly the inputs from <2en> on, i.e. the target-side tokens."""
    seqs, src_parts = [], []
    for t, l in pairs:
        src, tgt = split.pair(int(t), int(l))
        seqs.append([BOS, LANG_TOKEN[LANGS[l]], *src.tolist(), TO_EN, *tgt.tolist(), EOS])
        src_parts.append(len(src) + 2)  # index of <2en> in the sequence
    width = max(len(s) for s in seqs) - 1
    input_ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (s, p) in enumerate(zip(seqs, src_parts)):
        s = torch.tensor(s)
        input_ids[i, : len(s) - 1] = s[:-1]
        labels[i, p : len(s) - 1] = s[p + 1 :]
    valid = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, s in enumerate(seqs):
        valid[i, : len(s) - 1] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "valid": valid,
        "lang": torch.tensor(pairs[:, 1], dtype=torch.long),
        "n_tokens": sum(len(s) for s in seqs),
        "n_target": int((labels != -100).sum()),
    }


def batches(pairs: np.ndarray, batch_size: int):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.prepare:
        setup_run_log(cfg["log_dir"], "data_prepare", cfg["data"])
        prepare(cfg)


if __name__ == "__main__":
    main()
