#!/usr/bin/env bash
# Q2 final runs: one full 1x-dataset run per optimizer at the LR in config.yaml (chosen from the
# LR probe), with val loss + test BLEU every 0.1x of the dataset, checkpoint saved. Tag: final.
# Usage: bash src/part2/final.sh cuda:0
set -euo pipefail
DEVICE=${1:-cuda:0}
for opt in adamw mars lion muon; do
  uv run python -m src.part2.train --optimizer $opt --run-name "final_$opt" --tags final --device "$DEVICE"
done
