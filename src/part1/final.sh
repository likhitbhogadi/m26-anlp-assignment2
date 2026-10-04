#!/usr/bin/env bash
# Q1 final runs: the 5 FFN variants at the LR in config.yaml (6e-4, chosen from the LR sweep on
# dense + moe_4e_top1), same 30M training tokens and seed; then test PPL / BLEU / routing for each,
# then all plots and tables. Tag: final.
# Usage: bash src/part1/final.sh cuda:0
set -euo pipefail
DEVICE=${1:-cuda:0}
VARIANTS="dense moe_4e_top1 moe_4e_top2 moe_shared1_3e_top1 moe_4e_top2_active"
for v in $VARIANTS; do
  uv run python -m src.part1.train --variant $v --tags final --device "$DEVICE"
done
for v in $VARIANTS; do
  uv run python -m src.part1.evaluate --variant $v --device "$DEVICE"
done
uv run python -m src.part1.analyze
