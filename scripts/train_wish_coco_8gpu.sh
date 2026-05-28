#!/bin/bash
# ──────────────────────────────────────────────────────────────
#  WISH — paper-spec 8-GPU run (bs 16, lr 1e-4, 50 epochs)
#  Backbone: Mask2Former R50 + frozen SAM ViT-B
#  Dataset: COCO 2017 instance segmentation
# ──────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hetero/vendor/Mask2Former:$PROJECT_ROOT/hetero/vendor/segment-anything:$PYTHONPATH"

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"

export DETECTRON2_DATASETS="$PROJECT_ROOT/data"
if [ ! -e "$PROJECT_ROOT/data/coco" ]; then
    ln -s "$PROJECT_ROOT/data/coco2017" "$PROJECT_ROOT/data/coco"
fi

CONFIG="hetero/config/wish_coco.yaml"
SAVE_DIR="outputs/hetero_v2"
NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$SAVE_DIR"

echo "============================================="
echo "  WISH (paper-spec) | $NUM_GPUS GPUs"
echo "  Config : $CONFIG"
echo "  Save   : $SAVE_DIR"
echo "  bs=16  lr=1e-4  max_iter=368750  warmup_iters=30000"
echo "============================================="

python -m hetero.engine.train_wish \
    --config-file "$CONFIG" \
    --num-gpus "$NUM_GPUS" \
    OUTPUT_DIR "$SAVE_DIR" \
    "$@"

echo "WISH training done. Checkpoints in $SAVE_DIR"
