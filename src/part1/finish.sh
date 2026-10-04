#!/usr/bin/env bash
# Q1 finishing run, one shot on GPU 0 (~20 min). Retrains moe_4e_top1 with a checkpoint (its
# identical sweep twin was --no-save). In parallel on the same GPU: dense full eval + routing-only
# re-evals of the 3 already-evaluated MoE variants (they only lack the per-token routing stats).
# Then evaluates moe_4e_top1, rebuilds every plot/table, uploads checkpoints to HF.
# Usage: bash src/part1/finish.sh 2>&1 | tee /scratch/likhit/anlp-a2/q1/finish.out
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/scratch/likhit/hf
export CUDA_VISIBLE_DEVICES=0
D="--device cuda:0"

echo "== 0. login checks"
hf_ok() { uv run python -c "from huggingface_hub import HfApi; print('HF user:', HfApi().whoami()['name'])" 2>/dev/null; }
hf_ok || echo "WARNING: not logged in to HF -> upload (step 6) will be skipped; log in meanwhile with: HF_HOME=$HF_HOME uv run hf auth login"
uv run python -c "import wandb; print('WandB entity:', wandb.Api().default_entity)"

echo "== 1. remove smoke-test leftovers; retag the two interrupted WandB runs"
rm -f src/part1/outputs/*smoke*
rm -rf /scratch/likhit/anlp-a2/q1/checkpoints/smoke_moe_4e_top2
uv run python - <<'EOF'
import csv, os, wandb
for f in ["results.csv", "routing.csv"]:
    p = f"src/part1/outputs/{f}"
    if os.path.exists(p):
        rows = [r for r in csv.DictReader(open(p)) if not r["variant"].startswith("smoke")]
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
api = wandb.Api()
for rid in ["w45qlhmb", "ahf33u52"]:
    r = api.run(f"likhitbhogadi-iiit-hyderabad/anlp-a2-q1/{rid}")
    r.tags = [t for t in r.tags if t not in ("final", "aborted")] + ["aborted"]
    r.notes = "interrupted; superseded by the complete run with the same name"
    r.update(); print("retagged", r.name, r.id, r.tags)
EOF

echo "== 2. background: dense eval + routing-only re-evals (same GPU, ~10 min)"
(
  uv run python -m src.part1.evaluate --variant dense $D
  for v in moe_4e_top2 moe_shared1_3e_top1 moe_4e_top2_active; do
    uv run python -m src.part1.evaluate --variant $v --routing-only $D
  done
) > /scratch/likhit/anlp-a2/q1/finish_evals.out 2>&1 &
EVALS=$!

echo "== 3. foreground: retrain moe_4e_top1 (~15 min)"
uv run python -m src.part1.train --variant moe_4e_top1 --tags final $D
wait $EVALS || { echo "background evals failed, see /scratch/likhit/anlp-a2/q1/finish_evals.out"; exit 1; }
tail -n 3 /scratch/likhit/anlp-a2/q1/finish_evals.out

echo "== 4. evaluate moe_4e_top1 (~3 min)"
uv run python -m src.part1.evaluate --variant moe_4e_top1 $D

echo "== 5. plots + tables"
uv run python -m src.part1.analyze
uv run python -m src.part1.analyze --runs sweep_dense_lr3e-4,sweep_dense_lr6e-4,sweep_dense_lr1.2e-3,sweep_moe_4e_top1_lr3e-4,sweep_moe_4e_top1_lr6e-4,sweep_moe_4e_top1_lr1.2e-3 --out-name lr_sweep_curves.png

echo "== 6. upload checkpoints + translations to HF (public repos)"
if hf_ok; then
  uv run python -m src.part1.upload_hf | tee src/part1/outputs/hf_links.md
else
  echo "skipped: log in (HF_HOME=$HF_HOME uv run hf auth login), then: HF_HOME=$HF_HOME uv run python -m src.part1.upload_hf | tee src/part1/outputs/hf_links.md"
fi
echo "== done"
