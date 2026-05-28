"""Mask2Former R50 (COCO panoptic) — cfg, model, and train loader for hetero pipelines."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import nn

# Repo layout: hetero/models/mask2former_r50.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_M2F_ROOT = _REPO_ROOT / "hetero" / "vendor" / "Mask2Former"
if str(_M2F_ROOT) not in sys.path:
    sys.path.insert(0, str(_M2F_ROOT))


def setup_detectron2_env(
    datasets_root: Optional[Path] = None,
    cuda_home: str = "/usr/local/cuda-12.5",
) -> None:
    """Set env vars expected by detectron2 / Mask2Former (call once per process)."""
    os.environ.setdefault(
        "DETECTRON2_DATASETS",
        str(datasets_root or (_REPO_ROOT / "data")),
    )
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ["PATH"] = f"{cuda_home}/bin:" + os.environ.get("PATH", "")


# Must run before mask2former import: dataset paths are fixed at registration time.
setup_detectron2_env()

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import CfgNode, get_cfg  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

from mask2former import add_maskformer2_config  # noqa: E402
from train_net import Trainer  # noqa: E402  # vendor train_net (on sys.path)

DEFAULT_CONFIG = (
    _M2F_ROOT / "configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml"
)
DEFAULT_ZOO_WEIGHTS = (
    _M2F_ROOT / "checkpoints/maskformer2_R50_coco_panoptic.pkl"
)


def build_mask2former_r50_cfg(
    config_file: Path | str = DEFAULT_CONFIG,
    opts: Optional[Sequence[str]] = None,
    *,
    weights: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    freeze: bool = True,
) -> CfgNode:
    """Build detectron2 cfg for Mask2Former R50 COCO panoptic."""
    setup_detectron2_env()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_file))
    if weights is not None:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(weights)
    if output_dir is not None:
        cfg.defrost()
        cfg.OUTPUT_DIR = str(output_dir)
    if opts:
        cfg.merge_from_list(list(opts))
    if freeze:
        cfg.freeze()
    return cfg


def build_mask2former_r50(
    cfg: Optional[CfgNode] = None,
    *,
    weights: Optional[str | Path] = None,
    device: str = "cuda",
    load_weights: bool = True,
    eval_mode: bool = False,
) -> nn.Module:
    """
    Instantiate MaskFormer (R50) and optionally load checkpoint.

    Training: call with eval_mode=False; forward expects detectron2 batched dicts
    from Trainer.build_train_loader(cfg).
    """
    if cfg is None:
        cfg = build_mask2former_r50_cfg(weights=weights)
    model = build_model(cfg)
    if load_weights and cfg.MODEL.WEIGHTS:
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(torch.device(device))
    model.train(not eval_mode)
    return model


def build_mask2former_r50_train_loader(cfg: CfgNode):
    """COCO panoptic train loader (mapper from config, e.g. coco_panoptic_lsj)."""
    return Trainer.build_train_loader(cfg)


def build_mask2former_r50_training_bundle(
  opts: Optional[Sequence[str]] = None,
  weights: Optional[str | Path] = DEFAULT_ZOO_WEIGHTS,
  device: str = "cuda",
) -> Tuple[CfgNode, nn.Module, Any]:
    """One-shot: cfg + model (weights loaded) + train DataLoader."""
    cfg = build_mask2former_r50_cfg(opts=opts, weights=weights)
    model = build_mask2former_r50(cfg, load_weights=True, device=device, eval_mode=False)
    loader = build_mask2former_r50_train_loader(cfg)
    return cfg, model, loader