"""Trains one shared BPE tokenizer for en + vi + ja on the train split and prints length stats.

Usage:
    uv run python -m src.part1.tokenizer --config src/part1/config.yaml

No romanisation: NFKC-normalised text is split on spaces only (Metaspace), so unspaced
Japanese can still form multi-character merges; characters outside the `limit_alphabet`
most frequent ones fall back to UTF-8 byte tokens, so there is no UNK.
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from src.part1.common import SPECIAL_TOKENS, load_config
from src.part3.logging_utils import setup_run_log

BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]  # vocab entries required by byte_fallback


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(byte_fallback=True))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tok.decoder = build_decoder()
    return tok


def build_decoder():
    """Llama-style: '▁' -> ' ' per token, merge byte tokens, drop the one prepended space."""
    return decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0),
    ])


def train(cfg: dict) -> Tokenizer:
    tcfg = cfg["tokenizer"]
    ds = load_dataset(cfg["dataset"], split="train")
    langs = [cfg["tgt_lang"], *cfg["src_langs"]]

    def corpus():
        for lang in langs:
            for i in range(0, len(ds), 10_000):
                yield [t for t in ds[i : i + 10_000][lang] if t]

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=tcfg["vocab_size"],
        special_tokens=SPECIAL_TOKENS + BYTE_TOKENS,
        limit_alphabet=tcfg["limit_alphabet"],
        max_token_length=tcfg["max_token_length"],
        show_progress=True,
    )
    tok.train_from_iterator(corpus(), trainer=trainer, length=len(langs) * len(ds))
    os.makedirs(os.path.dirname(tcfg["path"]), exist_ok=True)
    tok.save(tcfg["path"])
    print(f"saved {tcfg['path']} (vocab {tok.get_vocab_size()})")
    return tok


def _pct(x: np.ndarray) -> str:
    return " ".join(f"p{q}={np.percentile(x, q):.0f}" for q in (50, 90, 95, 99)) + f" max={x.max()}"


def length_stats(cfg: dict, tok: Tokenizer):
    """Pair length = len(src) + len(tgt) + 4 specials (<bos> <lang> ... <2en> ... <eos>)."""
    ds = load_dataset(cfg["dataset"], split="train")
    n = min(cfg["tokenizer"]["n_stats_triplets"], len(ds))
    rows = ds.shuffle(seed=cfg["data"]["data_seed"]).select(range(n))
    max_len = cfg["data"]["max_len"]
    en = [len(e.ids) for e in tok.encode_batch(rows["en"], add_special_tokens=False)]
    print(f"\nlength stats on {n} random train triplets (tokens):")
    print(f"  en:      {_pct(np.array(en))}  ({np.mean(en) / np.mean([len(t.split()) for t in rows['en']]):.2f} tok/word)")
    for lang in cfg["src_langs"]:
        src = np.array([len(e.ids) for e in tok.encode_batch(rows[lang], add_special_tokens=False)])
        pair = src + np.array(en) + 4
        unit = f"{src.mean() / np.mean([len(t) for t in rows[lang]]):.2f} tok/char" if lang == "ja" else \
               f"{src.mean() / np.mean([len(t.split()) for t in rows[lang]]):.2f} tok/word"
        print(f"  {lang}:      {_pct(src)}  ({unit})")
        print(f"  {lang}->en:  {_pct(pair)}  mean={pair.mean():.1f}  >{max_len}: {(pair > max_len).mean():.2%}")

    byte_ids = {tok.token_to_id(b) for b in BYTE_TOKENS}
    ja_ids = [i for e in tok.encode_batch(rows["ja"], add_special_tokens=False) for i in e.ids]
    print(f"  byte-fallback tokens in ja: {np.mean([i in byte_ids for i in ja_ids]):.3%}")

    print("\nround trip (normalised text -> ids -> text):")
    norm = tok.normalizer
    for lang in ["en", *cfg["src_langs"]]:
        text = rows[0][lang]
        enc = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        print(f"  [{lang}] ok={dec == norm.normalize_str(text)} {text!r}\n        {enc.tokens}\n        -> {dec!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--stats-only", action="store_true", help="reuse the saved tokenizer")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_run_log(cfg["log_dir"], "tokenizer", cfg["tokenizer"])
    if args.stats_only:  # also refreshes the saved decoder (ids / vocab unchanged)
        tok = Tokenizer.from_file(cfg["tokenizer"]["path"])
        tok.decoder = build_decoder()
        tok.save(cfg["tokenizer"]["path"])
    else:
        tok = train(cfg)
    length_stats(cfg, tok)


if __name__ == "__main__":
    main()
