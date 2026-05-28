"""WISH-specific detectron2 cfg keys.

Call `add_wish_config(cfg)` AFTER `add_maskformer2_config(cfg)` and before
`cfg.merge_from_file(...)`. All keys live under `MODEL.WISH.*` and
`INPUT.WEAK_LABELS.*` so the vendor Mask2Former namespace stays untouched.
"""
from __future__ import annotations

from detectron2.config import CfgNode as CN


def add_wish_config(cfg) -> None:
    cfg.MODEL.WISH = CN()

    # Heads
    cfg.MODEL.WISH.PROMPT_DIM = 256  # SAM prompt latent dim
    cfg.MODEL.WISH.PROMPT_HEAD_LAYERS = 3
    cfg.MODEL.WISH.PROMPT_HEAD_HIDDEN = 256

    # Matching costs (paper Eq. 14: alpha=2, beta=5, gamma=5)
    cfg.MODEL.WISH.CLASS_WEIGHT = 2.0
    cfg.MODEL.WISH.PROMPT_WEIGHT = 5.0
    cfg.MODEL.WISH.SAM_MASK_WEIGHT = 5.0

    # Auxiliary loss weights (Eq. 20)
    cfg.MODEL.WISH.CAM_WEIGHT = 1.0
    cfg.MODEL.WISH.SELF_WEIGHT = 1.0

    # Mask sub-cost composition mirrors Mask2Former (sigmoid CE + dice)
    cfg.MODEL.WISH.MASK_DICE_WEIGHT = 5.0
    cfg.MODEL.WISH.MASK_BCE_WEIGHT = 5.0

    # Schedule
    cfg.MODEL.WISH.WARMUP_ITERS = 30000  # paper: 3 epochs VOC / 30k iters COCO

    # SAM
    cfg.MODEL.WISH.SAM = CN()
    cfg.MODEL.WISH.SAM.TYPE = "vit_b"
    cfg.MODEL.WISH.SAM.WEIGHTS = (
        "hetero/vendor/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    )
    cfg.MODEL.WISH.SAM.MULTIMASK = True  # always return 3 candidates
    cfg.MODEL.WISH.SAM.FREEZE = True

    # Downsample factor applied to stored SAM GT masks before they enter the
    # matcher/criterion. point_sample uses normalized coords, so this only
    # affects memory, not correctness. 1 = full resolution.
    cfg.MODEL.WISH.SAM.TARGET_DOWNSAMPLE = 4

    # CAM head (Sec. 4.3)
    cfg.MODEL.WISH.CAM = CN()
    cfg.MODEL.WISH.CAM.IN_FEATURE = "res5"  # backbone level fed to H_cam
    cfg.MODEL.WISH.CAM.PEAK_THRESHOLD = 0.3  # tau
    cfg.MODEL.WISH.CAM.PEAK_FILTER_SIZE = 3  # local-max filter window
    cfg.MODEL.WISH.CAM.DEDUP_IOU = 0.5  # peaks merged if SAM-mask IoU > this
    cfg.MODEL.WISH.CAM.MAX_PEAKS_PER_CLASS = 32

    # Heterogeneous label routing
    cfg.INPUT.WEAK_LABELS = CN()
    cfg.INPUT.WEAK_LABELS.MODE = "hetero"  # "hetero" | "tag" | "point" | "box"
    cfg.INPUT.WEAK_LABELS.TAG_RATIO = 0.34
    cfg.INPUT.WEAK_LABELS.POINT_RATIO = 0.33
    cfg.INPUT.WEAK_LABELS.BOX_RATIO = 0.33
    cfg.INPUT.WEAK_LABELS.SEED = 0  # deterministic per-image label-type assignment
