#!/bin/bash
# ──────────────────────────────────────────────────────────────
#  WISH (CVPR 2025) — Weakly Supervised Instance Segmentation
#  with Heterogeneous Labels (tag / point / box)
#  Backbone: Mask2Former R50 + frozen SAM ViT-B
#  Dataset: COCO 2017 instance segmentation
# ──────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv (see hetero/NOTE.md).
source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hetero/vendor/Mask2Former:$PROJECT_ROOT/hetero/vendor/segment-anything:$PYTHONPATH"

# Keep logs clean: silence Python warnings and TF/cuDNN INFO chatter.
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"

# Reduce CUDA fragmentation (matcher allocates large per-step workspaces).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Detectron2 reads $DETECTRON2_DATASETS/coco/{train2017,val2017,annotations}.
# The local data lives at data/coco2017 — expose it under data/coco for d2.
export DETECTRON2_DATASETS="$PROJECT_ROOT/data"
if [ ! -e "$PROJECT_ROOT/data/coco" ]; then
    ln -s "$PROJECT_ROOT/data/coco2017" "$PROJECT_ROOT/data/coco"
fi

CONFIG="hetero/config/wish_coco.yaml"
SAVE_DIR="outputs/hetero_v2"
NUM_GPUS="${NUM_GPUS:-1}"
mkdir -p "$SAVE_DIR"

echo "============================================="
echo "  WISH | Mask2Former R50 + SAM ViT-B"
echo "  Config : $CONFIG"
echo "  Save   : $SAVE_DIR"
echo "  GPUs   : $NUM_GPUS"
echo "============================================="

# Single-GPU friendly defaults; override IMS_PER_BATCH / BASE_LR / MAX_ITER
# from the command line for full 8-GPU runs (paper uses bs16, lr 1e-4, 50ep).
python -m hetero.engine.train_wish \
    --config-file "$CONFIG" \
    --num-gpus "$NUM_GPUS" \
    OUTPUT_DIR "$SAVE_DIR" \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.0000125 \
    "$@"

echo "WISH training done. Checkpoints in $SAVE_DIR"
