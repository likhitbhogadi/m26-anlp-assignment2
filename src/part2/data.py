"""HAP-E corpus: doc-level split, tokenized training / eval streams and BLEU items.

Usage (once, CPU, after `src.part2.tokenizer`):
    uv run python -m src.part2.data --prepare

Every row is `<type>_<num>@<variant>`; a source doc has 8 variants (chunk_1 = human first half,
chunk_2 = human continuation, 6 LLM continuations of chunk_1). All variants of a doc go to the
same split, so no prompt or continuation leaks across splits. `--prepare` writes to cache_dir:
  - {train,val,test}.npy:      uint16 token streams, every row followed by <eos>, rows in a fixed
                               shuffled doc order (train) -> packed into seq_len windows at load time;
  - {val,test}_src.npy:        uint8 per token: 0 = human-written, 1 = LLM-written;
  - {val,test}_genre.npy:      uint8 per token: index into GENRES;
  - bleu_items.json:           per test doc: prompt ids (last prompt_tokens of chunk_1) and 7
                               references (chunk_2 first, then the LLM continuations), each cut to
                               gen_tokens tokens;
  - meta.json:                 split sizes and token counts.
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset

from src.part1.common import load_config
from src.part3.logging_utils import setup_run_log

SPECIAL_TOKENS = ["<pad>", "<eos>"]
PAD, EOS = 0, 1
SPLITS = ["train", "val", "test"]
N_VARIANTS = 8


def load_tokenizer(cfg: dict):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(cfg["tokenizer"]["path"])
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special token {t} should have id {i}"
    return tok


def decode(tok, ids: list[int]) -> str:
    return tok.decode([i for i in ids if i >= len(SPECIAL_TOKENS)]).strip()


# ---------------------------------------------------------------- split

def load_rows(cfg: dict):
    """The raw dataset plus (doc, genre, variant) per row."""
    ds = load_dataset(cfg["dataset"], split="train")
    docs, variants = zip(*(d.split("@", 1) for d in ds["doc_id"]))
    genres = [d.rsplit("_", 1)[0] for d in docs]
    return ds, list(docs), list(genres), list(variants)


def split_docs(cfg: dict, docs: list[str], genres: list[str]) -> dict[str, str]:
    """doc -> train / val / test; stratified by text type, seeded."""
    dcfg = cfg["data"]
    rng = np.random.default_rng(dcfg["data_seed"])
    doc_genre = dict(zip(docs, genres))
    out = {}
    for g in sorted(set(genres)):
        gd = sorted(d for d in doc_genre if doc_genre[d] == g)
        gd = [gd[i] for i in rng.permutation(len(gd))]
        n_test, n_val = round(len(gd) * dcfg["test_frac"]), round(len(gd) * dcfg["val_frac"])
        for i, d in enumerate(gd):
            out[d] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


# ---------------------------------------------------------------- preparation

def prepare(cfg: dict):
    dcfg, ecfg = cfg["data"], cfg["eval"]
    os.makedirs(dcfg["cache_dir"], exist_ok=True)
    tok = load_tokenizer(cfg)
    ds, docs, genres, variants = load_rows(cfg)
    genre_list = sorted(set(genres))

    # sanity: every doc has exactly the 8 variants incl. the prompt and the human continuation
    per_doc: dict[str, dict[str, int]] = {}
    for i, (d, v) in enumerate(zip(docs, variants)):
        per_doc.setdefault(d, {})[v] = i
    variant_set = sorted(set(variants))
    assert dcfg["prompt_variant"] in variant_set and dcfg["human_variant"] in variant_set, variant_set
    bad = [d for d, vs in per_doc.items() if len(vs) != N_VARIANTS]
    assert not bad, f"{len(bad)} docs without {N_VARIANTS} variants, e.g. {bad[:3]}"
    llm_variants = [v for v in variant_set if v not in (dcfg["prompt_variant"], dcfg["human_variant"])]
    print(f"{len(ds)} rows, {len(per_doc)} docs, genres {genre_list}\nvariants: {variant_set}")

    split_of = split_docs(cfg, docs, genres)
    for s in SPLITS:
        assert len({d for d in per_doc if split_of[d] == s}) > 0
    ids = [e.ids for e in tok.encode_batch(list(ds["text"]))]

    rng = np.random.default_rng(dcfg["data_seed"] + 1)
    meta = {"rows": len(ds), "docs": len(per_doc), "genres": genre_list, "variants": variant_set,
            "vocab_size": tok.get_vocab_size()}
    human = {dcfg["prompt_variant"], dcfg["human_variant"]}
    for s in SPLITS:
        rows = [i for i in range(len(ds)) if split_of[docs[i]] == s]
        rows = [rows[j] for j in rng.permutation(len(rows))]  # fixed shuffled order (all variants mixed)
        stream, src, gen = [], [], []
        for i in rows:
            seq = ids[i] + [EOS]
            stream.extend(seq)
            src.extend([0 if variants[i] in human else 1] * len(seq))
            gen.extend([genre_list.index(genres[i])] * len(seq))
        np.save(os.path.join(dcfg["cache_dir"], f"{s}.npy"), np.asarray(stream, dtype=np.uint16))
        if s != "train":
            np.save(os.path.join(dcfg["cache_dir"], f"{s}_src.npy"), np.asarray(src, dtype=np.uint8))
            np.save(os.path.join(dcfg["cache_dir"], f"{s}_genre.npy"), np.asarray(gen, dtype=np.uint8))
        n_docs = len({docs[i] for i in rows})
        meta[f"{s}_docs"], meta[f"{s}_rows"], meta[f"{s}_tokens"] = n_docs, len(rows), len(stream)
        meta[f"{s}_human_token_frac"] = float(1 - np.mean(src))
        print(f"{s}: {n_docs} docs, {len(rows)} rows, {len(stream):,} tokens")

    # BLEU items: prompt = end of chunk_1, references = first gen_tokens of the 7 continuations
    items = []
    for d in sorted(d for d in per_doc if split_of[d] == "test"):
        vs = per_doc[d]
        refs = [decode(tok, ids[vs[v]][: ecfg["gen_tokens"]]) for v in [dcfg["human_variant"], *llm_variants]]
        items.append({"doc": d, "genre": d.rsplit("_", 1)[0],
                      "prompt_ids": ids[vs[dcfg["prompt_variant"]]][-ecfg["prompt_tokens"]:],
                      "prompt_text": ds[vs[dcfg["prompt_variant"]]]["text"][-400:], "refs": refs})
    with open(os.path.join(dcfg["cache_dir"], "bleu_items.json"), "w") as f:
        json.dump({"ref_names": [dcfg["human_variant"], *llm_variants], "items": items}, f)
    meta["bleu_items"] = len(items)
    with open(os.path.join(dcfg["cache_dir"], "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    ex = items[0]
    print(f"\nBLEU example [{ex['doc']}]\n  prompt: ...{decode(tok, ex['prompt_ids'])[-300:]!r}")
    for name, r in zip([dcfg["human_variant"], *llm_variants], ex["refs"]):
        print(f"  ref {name}: {r!r}")


# ---------------------------------------------------------------- loading

def load_meta(cfg: dict) -> dict:
    with open(os.path.join(cfg["data"]["cache_dir"], "meta.json")) as f:
        return json.load(f)


def load_stream(cfg: dict, split: str, suffix: str = "") -> np.ndarray:
    return np.load(os.path.join(cfg["data"]["cache_dir"], f"{split}{suffix}.npy"))


def load_bleu_items(cfg: dict) -> dict:
    with open(os.path.join(cfg["data"]["cache_dir"], "bleu_items.json")) as f:
        return json.load(f)


def n_windows(stream: np.ndarray, seq_len: int) -> int:
    """Non-overlapping (input, label) windows: inputs stream[i*L:(i+1)*L], labels shifted by one."""
    return (len(stream) - 1) // seq_len


def window_batch(stream: np.ndarray, idx: np.ndarray, seq_len: int, *extra: np.ndarray):
    """(inputs, labels, *extra arrays aligned with the labels) for windows `idx`, as int64 tensors."""
    starts = idx[:, None] * seq_len + np.arange(seq_len + 1)[None]
    seq = torch.from_numpy(stream[starts].astype(np.int64))
    out = [seq[:, :-1], seq[:, 1:]]
    for a in extra:
        out.append(torch.from_numpy(a[starts[:, 1:]].astype(np.int64)))
    return out


def train_batches(stream: np.ndarray, seq_len: int, batch_size: int, n_steps: int):
    """Windows in stream order (the doc order is already shuffled), batch_size per step."""
    for step in range(n_steps):
        yield window_batch(stream, np.arange(step * batch_size, (step + 1) * batch_size), seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part2/config.yaml")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.prepare:
        setup_run_log(cfg["log_dir"], "data_prepare", {"data": cfg["data"], "eval": cfg["eval"]})
        prepare(cfg)


if __name__ == "__main__":
    main()
