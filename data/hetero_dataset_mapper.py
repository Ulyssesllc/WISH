"""Dataset mapper emitting heterogeneous weak labels per image.

Output dict per item (extends vendor COCO instance mapper):
  image:             Tensor[C,H,W]
  height, width:     int
  instances:         d2 Instances (used at eval time / kept for reference;
                                  training does not rely on dense masks)
  weak_label_type:   "tag" | "point" | "box"
  weak_labels:       dict with the populated key:
                       - "tag":   labels [k_present_classes]
                       - "point": labels [k], points (x,y) [k,2]
                       - "box":   labels [k], boxes XYXY    [k,4]
  image_classes:     LongTensor [k_present] (always derivable; feeds L_cam)
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import torch

from detectron2.config import configurable
from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    COCOInstanceNewBaselineDatasetMapper,
)


class HeteroDatasetMapper(COCOInstanceNewBaselineDatasetMapper):
    """Vendor mapper + weak-label derivation from `dataset_dict["weak_label_type"]`."""

    @configurable
    def __init__(self, *args, weak_mode: str = "hetero", **kwargs):
        super().__init__(*args, **kwargs)
        self.weak_mode = weak_mode

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train=is_train)
        ret["weak_mode"] = cfg.INPUT.WEAK_LABELS.MODE
        return ret

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        d = copy.deepcopy(dataset_dict)
        out = super().__call__(d)  # gives us out["instances"], geometric aug applied

        # Determine label type (per-image, from registered dataset metadata)
        if self.weak_mode == "hetero":
            wtype = d.get("weak_label_type", "box")
        else:
            wtype = self.weak_mode

        out["weak_label_type"] = wtype
        out["weak_labels"] = self._derive_weak_labels(out["instances"], wtype)
        out["image_classes"] = torch.unique(out["instances"].gt_classes)
        return out

    @staticmethod
    def _derive_weak_labels(instances, wtype: str) -> Dict[str, torch.Tensor]:
        """Pull the weak label from instance GT. NOTE: for the heterogeneous
        protocol the mask GT is never consumed downstream — only the box, point,
        or set-of-classes derived here is used. We strip masks before returning."""
        if wtype == "tag":
            return {"labels": torch.unique(instances.gt_classes)}
        if wtype == "box":
            return {
                "labels": instances.gt_classes,
                "boxes": instances.gt_boxes.tensor,  # XYXY
            }
        if wtype == "point":
            # Use mask centroid as the point prompt (paper assumes a point
            # inside each instance; centroid is the standard proxy).
            # Vendor M2F mapper stores gt_masks as a raw Tensor [k, H, W]
            # (via convert_coco_poly_to_mask), not a BitMasks object.
            masks = instances.gt_masks
            if hasattr(masks, "tensor"):
                masks = masks.tensor
            k = masks.shape[0] if masks.ndim == 3 else 0
            if k == 0:
                return {
                    "labels": instances.gt_classes,
                    "points": torch.zeros(0, 2, dtype=torch.float32),
                }
            ys, xs = [], []
            for m in masks:
                idx = m.nonzero(as_tuple=False)
                if idx.numel() == 0:
                    ys.append(torch.tensor(0.0))
                    xs.append(torch.tensor(0.0))
                else:
                    ys.append(idx[:, 0].float().mean())
                    xs.append(idx[:, 1].float().mean())
            pts = torch.stack([torch.stack(xs), torch.stack(ys)], dim=-1)
            return {"labels": instances.gt_classes, "points": pts}
        raise ValueError(f"unknown weak_label_type: {wtype}")
