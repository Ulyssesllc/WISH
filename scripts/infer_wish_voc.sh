#!/bin/bash
# Single-image inference + visualization on Pascal VOC 2012
set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD:$PWD/hetero/vendor/Mask2Former:$PWD/hetero/vendor/segment-anything:$PYTHONPATH"
export DETECTRON2_DATASETS="$PWD/data"
export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"

[ -e data/voc ] || ln -s PASCAL\ VOC\ 2012 data/voc

IMAGE=${1:?"Usage: $0 <image_path> [options]"}
WEIGHTS=${WEIGHTS:-outputs/hetero_voc2012/model_final.pth}
SCORE_THRESHOLD=${SCORE_THRESHOLD:-0.5}

echo "Inference on: $IMAGE"
echo "Weights: $WEIGHTS"

python -m hetero.tools.infer_wish \
    --config-file hetero/config/wish_voc2012.yaml \
    --weights "$WEIGHTS" \
    --image "$IMAGE" \
    --score-threshold "$SCORE_THRESHOLD" \
    "$@"
