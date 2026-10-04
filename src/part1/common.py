"""Config, special tokens and small helpers shared by the Q1 scripts."""

import torch
import yaml

# Fixed ids 0..5 (first entries of the trained vocab).
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<vi>", "<ja>", "<2en>"]
PAD, BOS, EOS, VI, JA, TO_EN = range(len(SPECIAL_TOKENS))
LANG_TOKEN = {"vi": VI, "ja": JA}
LANG_ID = {"vi": 0, "ja": 1}   # index used in cached pair arrays / routing stats
LANGS = ["vi", "ja"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_variant(cfg: dict, name: str) -> dict:
    for v in cfg["variants"]:
        if v["name"] == name:
            return v
    raise SystemExit(f"unknown variant {name!r}; choose from {[v['name'] for v in cfg['variants']]}")


def model_config(cfg: dict, variant: dict, vocab_size: int):
    from src.part1.model import TransformerConfig

    fields = {**cfg["model"], **{k: v for k, v in variant.items() if k != "name"}}
    return TransformerConfig(vocab_size=vocab_size, n_ctx=cfg["data"]["max_len"], **fields)


def load_tokenizer(cfg: dict):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(cfg["tokenizer"]["path"])
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special token {t} should have id {i}"
    return tok


def get_device(cfg: dict, override: str | None = None) -> torch.device:
    """--device beats the config's `device`; falls back to CPU if CUDA is unavailable."""
    name = override or cfg.get("device", "cuda:0")
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable, using CPU instead of {name}")
        name = "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    return device
