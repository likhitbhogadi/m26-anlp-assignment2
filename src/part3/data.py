"""ROCStories loading: 1st sentence is the prompt, the remaining 4 sentences are the reference."""

import torch
from datasets import load_dataset


def load_samples(dataset: str, split: str, n_samples: int, seed: int) -> list[dict]:
    """Randomly sample `n_samples` stories (fixed seed) and return them in index order."""
    ds = load_dataset(dataset, split=split)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_samples].sort().values.tolist()
    rows = ds.select(idx)
    return [
        {"idx": i, "prompt": r["prompt"], "reference": r["continuation"]}
        for i, r in zip(idx, rows)
    ]


def tokenize_samples(samples: list[dict], tokenizer) -> list[dict]:
    """Adds `prompt_ids` and `ref_ids`. The reference gets a leading space because it
    directly follows the prompt sentence in the story."""
    for s in samples:
        s["prompt_ids"] = tokenizer(s["prompt"]).input_ids
        s["ref_ids"] = tokenizer(" " + s["reference"]).input_ids
    return samples


def batches(samples: list[dict], batch_size: int):
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


def collate_left_pad(prompt_ids: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pads prompts so every row's next token sits in the last column."""
    max_len = max(len(p) for p in prompt_ids)
    input_ids = torch.full((len(prompt_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompt_ids), max_len), dtype=torch.long)
    for row, p in enumerate(prompt_ids):
        input_ids[row, max_len - len(p) :] = torch.tensor(p)
        attention_mask[row, max_len - len(p) :] = 1
    return input_ids.to(device), attention_mask.to(device)
