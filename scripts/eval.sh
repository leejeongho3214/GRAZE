#!/usr/bin/env bash
# Score results/graze_<fold>.pkl: per-frame metrics (GPU) and the simulation metric.
#   bash scripts/eval.sh obj0 [cuda:0]
set -euo pipefail
FOLD=${1:?usage: scripts/eval.sh <obj0|obj1|obj2> [device]}
DEVICE=${2:-cuda:0}
cd "$(dirname "$0")/.."
ROOT=$PWD
eval/run_eval_gpu.sh --device "$DEVICE" \
  --input "$ROOT/results/graze_${FOLD}.pkl" \
  --new-metric-csv-output "$ROOT/results/graze_${FOLD}.csv" \
  --new-metric-md-output "$ROOT/results/graze_${FOLD}.md"
(cd eval/hot3d/visualize-code && python sim_displacement.py \
  --input "$ROOT/results/graze_${FOLD}.pkl" --workers 8 \
  --out-csv "$ROOT/results/sim_${FOLD}.csv" --out-md "$ROOT/results/sim_${FOLD}.md")
