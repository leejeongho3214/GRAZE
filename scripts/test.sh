#!/usr/bin/env bash
# Generate 10 samples per test clip (seeds 0-9) with MANO fitting + contact refinement.
#   bash scripts/test.sh obj0 [checkpoint] [extra Hydra overrides]
set -euo pipefail
FOLD=${1:?usage: scripts/test.sh <obj0|obj1|obj2> [checkpoint] [overrides]}; shift
CKPT=${1:-checkpoints/graze_${FOLD}.pth}; [[ $# -gt 0 ]] && shift
cd "$(dirname "$0")/.."
mkdir -p results
python graze/test.py fold="$FOLD" \
  texthom.exp.weight_path="$CKPT" \
  texthom.exp.save_name="results/graze_${FOLD}" \
  texthom.exp.seed=0 texthom.exp.num_test_seeds=10 "$@"
# -> results/graze_<fold>.pkl
