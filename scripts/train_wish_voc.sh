#!/bin/bash
# Single-GPU VOC training wrapper for WISH
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hetero/vendor/Mask2Former:$PROJECT_ROOT/hetero/vendor/segment-anything:$PYTHONPATH"
export DETECTRON2_DATASETS="$PROJECT_ROOT/data"

CONFIG="hetero/config/wish_voc2012.yaml"
SAVE_DIR="outputs/hetero_voc2012"
NUM_GPUS="${NUM_GPUS:-1}"
mkdir -p "$SAVE_DIR"

python -m hetero.engine.train_wish \
    --config-file "$CONFIG" \
    --num-gpus "$NUM_GPUS" \
    OUTPUT_DIR "$SAVE_DIR" \
    "$@"

echo "WISH VOC training done. Checkpoints in $SAVE_DIR"
