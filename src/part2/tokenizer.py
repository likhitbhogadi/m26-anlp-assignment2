"""Trains an English byte-level BPE on the training-split docs of HAP-E.

Usage:
    uv run python -m src.part2.tokenizer

Only rows of training docs are used (same doc-level split as data.py), so the val / test text
never influences the vocabulary. Byte-level -> no UNK.
"""

import argparse
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.part1.common import load_config
from src.part2.data import SPECIAL_TOKENS, load_rows, split_docs
from src.part3.logging_utils import setup_run_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part2/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["tokenizer"]
    setup_run_log(cfg["log_dir"], "tokenizer", tcfg)

    ds, docs, genres, _ = load_rows(cfg)
    split_of = split_docs(cfg, docs, genres)
    texts = [t for t, d in zip(ds["text"], docs) if split_of[d] == "train"]
    print(f"training BPE on {len(texts)} train rows")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=tcfg["vocab_size"], special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    tok.train_from_iterator((texts[i : i + 1000] for i in range(0, len(texts), 1000)), trainer=trainer,
                            length=len(texts))
    os.makedirs(os.path.dirname(tcfg["path"]), exist_ok=True)
    tok.save(tcfg["path"])
    print(f"saved {tcfg['path']} (vocab {tok.get_vocab_size()})")

    n_tok = sum(len(e.ids) for e in tok.encode_batch(texts[:2000]))
    n_words = sum(len(t.split()) for t in texts[:2000])
    print(f"{n_tok / n_words:.3f} tokens / word on 2000 train rows")
    t = texts[0][:200]
    enc = tok.encode(t)
    print(f"round trip ok={tok.decode(enc.ids) == t}\n  {t!r}\n  {enc.tokens[:30]}")


if __name__ == "__main__":
    main()
