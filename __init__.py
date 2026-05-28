"""hetero — WISH (Weakly Supervised Instance Segmentation, Heterogeneous labels).

Public submodules: models, data, engine, config, tools.

Side effect: prepends `hetero/vendor/Mask2Former` and `hetero/vendor/segment-anything`
to sys.path so vendor packages (`mask2former`, `segment_anything`, `train_net`)
import as top-level modules from any entry point.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_M2F_ROOT = _REPO_ROOT / "hetero" / "vendor" / "Mask2Former"
_SAM_ROOT = _REPO_ROOT / "hetero" / "vendor" / "segment-anything"

for _p in (_M2F_ROOT, _SAM_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Match detectron2 expectations for dataset path resolution (same defaults as
# hetero/models/mask2former_r50.py). Set only if unset so callers can override.
os.environ.setdefault("DETECTRON2_DATASETS", str(_REPO_ROOT / "data"))
