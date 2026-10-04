"""Uploads each trained Q1 variant to a public HuggingFace model repo.

Usage (after `huggingface-cli login`):
    uv run python -m src.part1.upload_hf                       # every variant with a checkpoint
    uv run python -m src.part1.upload_hf --variants dense,moe_4e_top1

Repo `<hf_user>/<hf.repo_prefix>-<variant>` gets model.safetensors, config.json, the shared
tokenizer.json, all test translations and a model card with the test metrics (results.csv) and the WandB run link.
Prints a markdown list of the repo + run links for the README / report.
"""

import argparse
import csv
import os
import shutil
import tempfile

from huggingface_hub import HfApi

from src.part1.common import load_config

CARD = """---
language: [en, vi, ja]
tags: [translation, mixture-of-experts, anlp-assignment]
---
# {repo}

Decoder-only transformer ({total:.1f}M total / {active:.1f}M active params) trained from scratch for
vi→en and ja→en translation on `belumind/en-vi-ja-curated-500k-triplets` (30M training tokens).
FFN variant: **{variant}** ({desc}).

Sequence format: `<bos> <vi|ja> source <2en> target <eos>`, with the shared BPE tokenizer in
`tokenizer.json`. Model code: `src/part1/model.py` in the assignment repository.

| | vi→en | ja→en | both |
|---|---|---|---|
| test perplexity | {ppl_vi} | {ppl_ja} | {ppl_all} |
| test BLEU (greedy, sacrebleu 13a) | {bleu_vi} | {bleu_ja} | {bleu_all} |

`test_translations.csv`: every greedy test translation with its reference and sentence BLEU.

Training log: {wandb}
"""


def describe(v: dict) -> str:
    if not v.get("use_moe"):
        return "standard 2-layer MLP"
    shared = f"{v['n_shared']} shared + " if v.get("n_shared") else ""
    return f"MoE, {shared}{v['n_experts']} routed experts of width {v['d_expert']}, top-{v['top_k']}"


def wandb_url(cfg: dict, run_name: str) -> str:
    try:
        import wandb

        api = wandb.Api()
        project = f"{cfg['wandb']['entity'] or api.default_entity}/{cfg['wandb']['project']}"
        runs = [r for r in api.runs(project, filters={"display_name": run_name}) if "final" in r.tags]
        return runs[0].url if runs else f"https://wandb.ai/{project}"
    except Exception as e:  # noqa: BLE001 - the link is a nicety, never block the upload
        print(f"  (no WandB link for {run_name}: {e})")
        return "see the WandB project linked in the README"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/part1/config.yaml")
    parser.add_argument("--variants", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    api = HfApi()
    user = api.whoami()["name"]
    results = {}
    with open(os.path.join(cfg["output_dir"], "results.csv")) as f:
        for r in csv.DictReader(f):
            results[(r["variant"], r["lang"])] = r

    wanted = set(args.variants.split(",")) if args.variants else None
    links = []
    for v in cfg["variants"]:
        name = v["name"]
        ckpt = os.path.join(cfg["checkpoint_dir"], name)
        if (wanted and name not in wanted) or not os.path.exists(os.path.join(ckpt, "model.safetensors")):
            continue
        repo = f"{user}/{cfg['hf']['repo_prefix']}-{name}"
        run_url = wandb_url(cfg, name)
        res = lambda lang, k: results[(name, lang)][k] if (name, lang) in results else "n/a"
        with tempfile.TemporaryDirectory() as tmp:
            for fname in ["model.safetensors", "config.json"]:
                shutil.copy(os.path.join(ckpt, fname), tmp)
            shutil.copy(cfg["tokenizer"]["path"], tmp)
            trans = os.path.join(cfg["output_dir"], f"translations_{name}.csv")
            if os.path.exists(trans):  # every greedy test translation (too large for git)
                shutil.copy(trans, os.path.join(tmp, "test_translations.csv"))
            with open(os.path.join(tmp, "README.md"), "w") as f:
                f.write(CARD.format(
                    repo=repo, variant=name, desc=describe(v), wandb=run_url,
                    total=float(res("all", "total_params")) / 1e6, active=float(res("all", "active_params")) / 1e6,
                    **{f"ppl_{l}": res(l, "test_ppl") for l in ["vi", "ja", "all"]},
                    **{f"bleu_{l}": res(l, "bleu") for l in ["vi", "ja", "all"]},
                ))
            api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
            api.upload_folder(folder_path=tmp, repo_id=repo, repo_type="model",
                              commit_message=f"Q1 {name}: weights, config, tokenizer, model card")
        links.append(f"- `{name}`: [HF](https://huggingface.co/{repo}) · [WandB]({run_url})")
        print(f"uploaded {repo}")
    print("\n" + "\n".join(links))


if __name__ == "__main__":
    main()
