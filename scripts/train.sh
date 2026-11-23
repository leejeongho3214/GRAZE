#!/usr/bin/env bash
# Train GRAZE on one HOT3D-Grasp object fold (about 14 h on one GPU for 4000 epochs).
#   bash scripts/train.sh obj0 [extra Hydra overrides]
set -euo pipefail
FOLD=${1:?usage: scripts/train.sh <obj0|obj1|obj2> [overrides]}; shift
cd "$(dirname "$0")/.."
python graze/train.py fold="$FOLD" "$@"
# -> outputs/graze_<fold>/epoch_4000.pth (the released checkpoints are epoch 4000)
