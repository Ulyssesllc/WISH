#!/bin/bash
# Evaluate WISH on VOC2012 val
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/hetero/vendor/Mask2Former:$PROJECT_ROOT/hetero/vendor/segment-anything:$PYTHONPATH"
export DETECTRON2_DATASETS="$PROJECT_ROOT/data"

WEIGHTS="${WEIGHTS:-outputs/hetero_voc2012/model_final.pth}"
CONFIG="hetero/config/wish_voc2012.yaml"

python -m hetero.engine.eval_wish \
    --config-file "$CONFIG" \
    --weights "$WEIGHTS" \
    "$@"
