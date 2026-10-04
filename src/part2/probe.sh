#!/usr/bin/env bash
# Q2 low-compute LR probe: 3-4 peak LRs x 4 optimizers, 500 steps (~0.2x dataset) each, warmup + cosine
# compressed to the probe length, val loss only (no BLEU, no checkpoint). Logged to WandB with tag lr-probe.
# Usage: bash src/part2/probe.sh cuda:0
set -euo pipefail
DEVICE=${1:-cuda:0}
STEPS=500
run() { uv run python -m src.part2.train --optimizer "$1" --lr "$2" --max-steps $STEPS --no-bleu --no-save \
          --run-name "probe_$1_lr$2" --tags lr-probe --device "$DEVICE"; }
for lr in 3e-4 6e-4 1.2e-3 2.4e-3; do run adamw $lr; done   # 3e-4 added after 6e-4 won at the edge
for lr in 6e-4 1.2e-3 2.4e-3; do run mars  $lr; done
for lr in 1.2e-4 2.4e-4 4.8e-4; do run lion  $lr; done
for lr in 1e-2 2e-2 4e-2 8e-2; do run muon  $lr; done   # 8e-2 added after 4e-2 won at the edge
