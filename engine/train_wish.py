"""WISH training entrypoint (detectron2 DefaultTrainer subclass).

  python -m hetero.engine.train_wish --config-file hetero/config/wish_coco.yaml \
      --num-gpus 8

Key responsibilities beyond vendor Trainer:
  - Register `coco_2017_train_hetero` before build_train_loader runs.
  - Use HeteroDatasetMapper for the train loader.
  - Push the current iteration into model._iter each step (for warmup gating).
"""
from __future__ import annotations

# Silence Python warnings before any heavy import.
from hetero.engine._logging import quiet_loggers, silence_third_party
silence_third_party()

import copy
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

# Vendor sys.path glue is set up when hetero.models is imported.
import hetero.models  # noqa: F401  (registers WISH meta-arch + decoder)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping

from mask2former import add_maskformer2_config, InstanceSegEvaluator

from hetero.config import add_wish_config
from hetero.data import HeteroDatasetMapper, register_coco_hetero, register_voc_hetero
from hetero.data.voc_dataset_mapper import VOCHeteroDatasetMapper


class WISHTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader
        # Use VOC-specific mapper if training on VOC
        is_voc = any("voc" in name.lower() for name in cfg.DATASETS.TRAIN)
        mapper_cls = VOCHeteroDatasetMapper if is_voc else HeteroDatasetMapper
        mapper = mapper_cls(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Mirror vendor Mask2Former optimizer build so that
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE='full_model' is honored (the
        stock detectron2 enum doesn't include that value)."""
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}

        norm_module_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm, torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hp = copy.copy(defaults)
                if "backbone" in module_name:
                    hp["lr"] = hp["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in param_name or "absolute_pos_embed" in param_name:
                    hp["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hp["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hp["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hp})

        def maybe_full_model_clip(optim_cls):
            clip_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_val > 0.0
            )

            class FullModelGradientClipping(optim_cls):
                def step(self, closure=None):
                    all_p = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_p, clip_val)
                    super().step(closure=closure)

            return FullModelGradientClipping if enable else optim_cls

        opt_type = cfg.SOLVER.OPTIMIZER
        if opt_type == "SGD":
            optimizer = maybe_full_model_clip(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif opt_type == "ADAMW":
            optimizer = maybe_full_model_clip(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {opt_type}")
        if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                f"no evaluator for dataset {dataset_name} with type {evaluator_type}"
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    def run_step(self):
        # Push iter into model for warmup gate.
        m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(m, "_iter"):
            m._iter.fill_(self.iter)
        super().run_step()


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_wish_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    quiet_loggers()

    # Register the hetero wrapper for the train set referenced by the cfg.
    for name in cfg.DATASETS.TRAIN:
        if name.endswith("_hetero"):
            base = name.replace("_hetero", "")
            if "voc" in name.lower():
                register_voc_hetero(
                    name=name,
                    base_name=base,
                    seed=cfg.INPUT.WEAK_LABELS.SEED,
                    ratios=(
                        cfg.INPUT.WEAK_LABELS.TAG_RATIO,
                        cfg.INPUT.WEAK_LABELS.POINT_RATIO,
                        cfg.INPUT.WEAK_LABELS.BOX_RATIO,
                    ),
                )
            else:
                register_coco_hetero(
                    name=name,
                    base_name=base,
                    seed=cfg.INPUT.WEAK_LABELS.SEED,
                    ratios=(
                        cfg.INPUT.WEAK_LABELS.TAG_RATIO,
                        cfg.INPUT.WEAK_LABELS.POINT_RATIO,
                        cfg.INPUT.WEAK_LABELS.BOX_RATIO,
                    ),
                )
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = WISHTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return WISHTrainer.test(cfg, model)
    trainer = WISHTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
