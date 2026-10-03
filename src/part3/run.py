"""Runs every decoding experiment in the config on the same 1000 ROCStories prompts.

Usage:
    uv run python -m src.part3.run --config src/part3/config.yaml
    uv run python -m src.part3.run --config src/part3/config.yaml --only greedy,beam_w2

Writes per experiment `<output_dir>/<name>.jsonl` (raw generations) and upserts a row
into `<output_dir>/results.csv` (metrics + timing), so experiments can be run in chunks.
"""

import argparse
import csv
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.part3 import data, decoding, metrics
from src.part3.logging_utils import setup_run_log

N_LOGGED_GENERATIONS = 50


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


def load_model(cfg: dict, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]))
    return model.to(device).eval(), tokenizer


def prepare_samples(cfg: dict, tokenizer, n_samples: int, n_shots: int | None = None) -> list[dict]:
    """The same `n_samples` stories for every experiment; `n_shots` (default: the config's)
    only changes the example-story prefix in front of each prompt."""
    n_shots = cfg.get("n_shots", 0) if n_shots is None else n_shots
    samples = data.load_samples(cfg["dataset"], cfg["split"], cfg["n_samples"], cfg["data_seed"])
    prefix = data.build_shot_prefix(cfg["dataset"], n_shots, cfg["data_seed"])
    if prefix:
        print(f"prompt prefix ({n_shots}-shot, {len(tokenizer(prefix).input_ids)} tokens):\n{prefix!r}")
    return data.tokenize_samples(samples[:n_samples], tokenizer, prefix)


def decode_batch(model, exp: dict, cfg: dict, input_ids, attention_mask, eos_id, pad_id, generator):
    common = dict(max_new_tokens=cfg["max_new_tokens"], eos_id=eos_id, pad_id=pad_id)
    strategy = exp["strategy"]
    if strategy == "greedy":
        return decoding.greedy_decode(model, input_ids, attention_mask, **common)
    if strategy == "top_k":
        return decoding.top_k_decode(model, input_ids, attention_mask, **common, k=exp["k"],
                                     temperature=exp.get("temperature", 1.0), generator=generator)
    if strategy == "top_p":
        return decoding.top_p_decode(model, input_ids, attention_mask, **common, p=exp["p"],
                                     temperature=exp.get("temperature", 1.0), generator=generator)
    if strategy == "beam":
        return decoding.beam_search(model, input_ids, attention_mask, **common, num_beams=exp["num_beams"],
                                    length_penalty=exp.get("length_penalty", cfg["length_penalty"]))
    raise ValueError(f"unknown strategy {strategy!r}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_experiment(model, tokenizer, samples, exp, cfg, device, wandb_run=None) -> tuple[dict, list[dict]]:
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    batch_size = cfg["beam_batch_size"] if exp["strategy"] == "beam" else cfg["batch_size"]

    # Untimed warm-up (CUDA context, kernels, allocator) on one small batch.
    warm = data.collate_left_pad([s["prompt_ids"] for s in samples[:batch_size]], pad_id, device)
    decode_batch(model, exp, {**cfg, "max_new_tokens": 8}, *warm, eos_id, pad_id, None)

    generator = torch.Generator(device=device).manual_seed(exp.get("seed", cfg["sample_seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records, total_time = [], 0.0
    for b, batch in enumerate(data.batches(samples, batch_size)):
        input_ids, attention_mask = data.collate_left_pad([s["prompt_ids"] for s in batch], pad_id, device)
        _sync(device)
        t0 = time.perf_counter()
        outs = decode_batch(model, exp, cfg, input_ids, attention_mask, eos_id, pad_id, generator)
        _sync(device)
        batch_time = time.perf_counter() - t0
        total_time += batch_time
        if wandb_run is not None:
            wandb_run.log({"batch/time_s": batch_time, "batch/idx": b})

        for s, o in zip(batch, outs):
            records.append({
                "idx": s["idx"],
                "prompt": s["prompt"],
                "reference": s["reference"],
                "generation": tokenizer.decode(o["token_ids"], skip_special_tokens=True),
                "gen_ids": o["token_ids"],
                "ref_ids": s["ref_ids"],
                "logprobs": [round(lp, 5) for lp in o["logprobs"]],
                "ppl": metrics.sample_perplexity(o["logprobs"]),
                "acc": metrics.positional_accuracy(o["token_ids"], s["ref_ids"]),
            })
        print(f"  [{exp['name']}] batch {b + 1}: {batch_time:.2f}s", flush=True)

    total_tokens = sum(len(r["gen_ids"]) for r in records)
    summary = {
        "name": exp["name"],
        "strategy": exp["strategy"],
        "n_shots": exp.get("n_shots", cfg.get("n_shots", 0)),
        **{k: v for k, v in exp.items() if k not in ("name", "strategy")},
        **metrics.aggregate(records),  # corpus PPL uses the 5-decimal logprobs; negligible
        "n_samples": len(records),
        "batch_size": batch_size,
        "total_time_s": total_time,
        "time_per_sample_ms": 1000 * total_time / len(records),
        "time_per_token_ms": 1000 * total_time / total_tokens,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    return summary, records


def save_records(records: list[dict], path: str):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def upsert_results(summary: dict, path: str):
    """Replaces the row with the same experiment name, keeping the others."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["name"] != summary["name"]]
    rows.append(summary)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part3/config.yaml")
    parser.add_argument("--only", default=None, help="comma-separated experiment names")
    parser.add_argument("--n-samples", type=int, default=None, help="override for quick tests")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", default=None, help="e.g. cuda:1; overrides the config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_run_log(cfg["log_dir"], "run", cfg)
    experiments = cfg["experiments"]
    if args.only:
        wanted = set(args.only.split(","))
        experiments = [e for e in experiments if e["name"] in wanted]
        missing = wanted - {e["name"] for e in experiments}
        if missing:
            raise SystemExit(f"unknown experiments: {sorted(missing)}")

    device = get_device(cfg, args.device)
    model, tokenizer = load_model(cfg, device)
    n_samples = args.n_samples or cfg["n_samples"]
    samples_by_shots = {}  # n_shots -> tokenized samples (an experiment may override n_shots)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
    run_config = {k: v for k, v in cfg.items() if k not in ("experiments", "wandb")}

    for exp in experiments:
        print(f"== {exp['name']}")
        n_shots = exp.get("n_shots", cfg.get("n_shots", 0))
        if n_shots not in samples_by_shots:
            samples_by_shots[n_shots] = prepare_samples(cfg, tokenizer, n_samples, n_shots)
            with open(os.path.join(cfg["output_dir"], "sample_indices.json"), "w") as f:
                json.dump([s["idx"] for s in samples_by_shots[n_shots]], f)
            print(f"{n_samples} samples from {cfg['dataset']}[{cfg['split']}], {n_shots}-shot prompts")
        samples = samples_by_shots[n_shots]
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], group=cfg["wandb"]["group"],
                name=exp["name"], tags=["part3", exp["strategy"]],
                config={**run_config, **exp, "n_shots": n_shots, "n_samples": len(samples)},
                reinit="finish_previous",
            )
        summary, records = run_experiment(model, tokenizer, samples, exp, cfg, device, wandb_run)

        save_records(records, os.path.join(cfg["output_dir"], f"{exp['name']}.jsonl"))
        upsert_results(summary, os.path.join(cfg["output_dir"], "results.csv"))
        print("  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))

        if wandb_run is not None:
            wandb_run.summary.update(summary)
            table = wandb.Table(columns=["idx", "prompt", "reference", "generation", "ppl", "acc"])
            for r in records[:N_LOGGED_GENERATIONS]:
                table.add_data(r["idx"], r["prompt"], r["reference"], r["generation"], r["ppl"], r["acc"])
            wandb_run.log({"generations": table})
            wandb_run.finish()


if __name__ == "__main__":
    main()
