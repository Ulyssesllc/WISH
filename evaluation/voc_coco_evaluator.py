"""Pascal VOC evaluator with mAPval/mAPtest metric naming.

Extends InstanceSegEvaluator to output metrics differently based on eval set:
- VOC val/test: AP, AP50, AP75, APs, APm, APl
- COCO val: mAPval, mAPval50, mAPval75, etc.
- COCO test: mAPtest, mAPtest50, mAPtest75, etc.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

from mask2former.evaluation.instance_evaluation import InstanceSegEvaluator


class VOCInstanceSegEvaluator(InstanceSegEvaluator):
    """VOC-specific evaluator that outputs AP/AP50/AP75 metrics."""

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """Override to keep standard AP naming for VOC (not mAP)."""
        # Call parent to get standard results dict
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        # For VOC, keep metric names as-is: AP, AP50, AP75, APs, APm, APl
        # (parent already returns these)
        return results


class COCOInstanceSegEvaluator(InstanceSegEvaluator):
    """COCO-specific evaluator that outputs mAPval/mAPtest/etc. metric naming."""

    def __init__(self, *args, eval_type: str = "val", **kwargs):
        """
        Args:
            eval_type (str): "val" or "test" — determines metric prefix (mAPval or mAPtest)
        """
        super().__init__(*args, **kwargs)
        self.eval_type = eval_type

    @classmethod
    def from_config(cls, cfg, is_train: bool = False, eval_type: str = "val"):
        ret = super().from_config(cfg, is_train=is_train)
        ret["eval_type"] = eval_type
        return ret

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """Override to rename metrics with mAPval/mAPtest prefix for COCO."""
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        # Rename metrics: AP → mAPval/mAPtest, AP50 → mAPval50/mAPtest50, etc.
        metric_prefix = f"mAP{self.eval_type}"
        renamed_results = {}

        for metric_name, value in results.items():
            if metric_name == "AP":
                renamed_results[metric_prefix] = value
            elif metric_name.startswith("AP"):
                # AP50 → mAPval50, AP75 → mAPval75, APs → mAPvals, etc.
                suffix = metric_name[2:]  # Remove "AP" prefix
                renamed_results[f"{metric_prefix}{suffix}"] = value
            else:
                renamed_results[metric_name] = value

        self._logger.info(
            f"Renamed metrics for COCO {self.eval_type}: {list(renamed_results.keys())}"
        )
        return renamed_results
