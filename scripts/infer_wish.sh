#!/bin/bash
# WISH — single-image visualization
set -e

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <image_path> [extra args]" >&2
    exit 1
fi

IMAGE="$1"; shift

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hetero/vendor/Mask2Former:$PROJECT_ROOT/hetero/vendor/segment-anything:$PYTHONPATH"

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"

CONFIG="hetero/config/wish_coco.yaml"
WEIGHTS="${WEIGHTS:-outputs/hetero_v2/model_final.pth}"

python -m hetero.tools.infer_wish \
    --config-file "$CONFIG" \
    --weights "$WEIGHTS" \
    --image "$IMAGE" \
    "$@"
