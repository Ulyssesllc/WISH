"""Dataset mapper for Pascal VOC with heterogeneous weak labels.

Similar to HeteroDatasetMapper but handles VOC-specific issues:
- VOC datasets don't have masks by default, only bounding boxes
- Adds dummy segmentation to make parent mapper work
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch

from detectron2.config import configurable

from hetero.data.hetero_dataset_mapper import HeteroDatasetMapper


class VOCHeteroDatasetMapper(HeteroDatasetMapper):
    """VOC mapper + weak-label derivation. Works around missing masks."""

    @configurable
    def __init__(self, *args, weak_mode: str = "hetero", **kwargs):
        super().__init__(*args, weak_mode=weak_mode, **kwargs)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = HeteroDatasetMapper.from_config(cfg, is_train=is_train)
        ret["weak_mode"] = cfg.INPUT.WEAK_LABELS.MODE
        return ret

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        d = copy.deepcopy(dataset_dict)
        
        # For VOC: add dummy segmentation to each annotation so parent doesn't fail
        # The dummy mask will be a rectangle matching the bbox
        if "annotations" in d:
            for anno in d["annotations"]:
                if "segmentation" not in anno and "bbox" in anno:
                    # Create a dummy polygon from bbox
                    x, y, w, h = anno["bbox"]
                    # Polygon in COCO format: [x1, y1, x2, y2, x3, y3, x4, y4]
                    anno["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        
        # Now call parent which should work
        return super().__call__(d)



