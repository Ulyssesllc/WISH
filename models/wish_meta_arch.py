"""WISHMaskFormer: Mask2Former meta-arch wired for heterogeneous weak supervision.

Differences vs vendor MaskFormer:
  * Owns a frozen SAMWrapper.
  * Owns a CAMHead operating on cfg.MODEL.WISH.CAM.IN_FEATURE.
  * prepare_targets(...) builds {labels, prompts, sam_masks} per image by
    invoking SAM on each weak label (tags first converted via cam_head.peaks_to_points).
  * Loss = L_seg (WISH criterion) + L_cam + L_self.
  * Iterations < cfg.MODEL.WISH.WARMUP_ITERS only return L_cam (Sec. 5.1.2).
  * Inference path = vendor instance_inference (no SAM decoder at test time, per Fig. 3).
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList

from mask2former.maskformer_model import MaskFormer

from hetero.models.cam_head import CAMHead, PeakConfig, loss_cam, loss_self, peaks_to_points
from hetero.models.sam_wrapper import SAMWrapper
from hetero.models.wish_matcher import WISHMatcher
from hetero.models.wish_criterion import WISHSetCriterion


@META_ARCH_REGISTRY.register()
class WISHMaskFormer(MaskFormer):
    """Subclass — keeps inference path identical to MaskFormer."""

    @configurable
    def __init__(
        self,
        *,
        sam: SAMWrapper,
        cam_head: CAMHead,
        cam_in_feature: str,
        peak_cfg: PeakConfig,
        cam_weight: float,
        self_weight: float,
        warmup_iters: int,
        num_classes: int,
        sam_target_downsample: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sam = sam
        self.cam_head = cam_head
        self.cam_in_feature = cam_in_feature
        self.peak_cfg = peak_cfg
        self.cam_weight = cam_weight
        self.self_weight = self_weight
        self.warmup_iters = warmup_iters
        self.num_classes_wish = num_classes
        self.sam_target_downsample = int(sam_target_downsample)
        # Updated by Trainer hook each step (see hetero.engine.train_wish).
        self.register_buffer("_iter", torch.zeros(1, dtype=torch.long), persistent=False)

    @classmethod
    def from_config(cls, cfg):
        base = MaskFormer.from_config(cfg)
        # Swap criterion + matcher for WISH variants.
        matcher = WISHMatcher(
            cost_class=cfg.MODEL.WISH.CLASS_WEIGHT,
            cost_prompt=cfg.MODEL.WISH.PROMPT_WEIGHT,
            cost_mask=cfg.MODEL.WISH.SAM_MASK_WEIGHT,
            cost_dice=cfg.MODEL.WISH.MASK_DICE_WEIGHT,
            cost_bce=cfg.MODEL.WISH.MASK_BCE_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {
            "loss_ce": cfg.MODEL.WISH.CLASS_WEIGHT,
            "loss_prompt": cfg.MODEL.WISH.PROMPT_WEIGHT,
            "loss_mask": cfg.MODEL.WISH.MASK_BCE_WEIGHT,
            "loss_dice": cfg.MODEL.WISH.MASK_DICE_WEIGHT,
        }
        if cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION:
            aux = {}
            for i in range(cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1):
                aux.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux)
        base["criterion"] = WISHSetCriterion(
            num_classes=base["sem_seg_head"].num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        sam = SAMWrapper(
            sam_type=cfg.MODEL.WISH.SAM.TYPE,
            weights=cfg.MODEL.WISH.SAM.WEIGHTS,
            freeze=cfg.MODEL.WISH.SAM.FREEZE,
            multimask=cfg.MODEL.WISH.SAM.MULTIMASK,
        )
        cam_head = _build_cam_head_lazy(cfg)
        peak_cfg = PeakConfig(
            threshold=cfg.MODEL.WISH.CAM.PEAK_THRESHOLD,
            filter_size=cfg.MODEL.WISH.CAM.PEAK_FILTER_SIZE,
            dedup_iou=cfg.MODEL.WISH.CAM.DEDUP_IOU,
            max_peaks_per_class=cfg.MODEL.WISH.CAM.MAX_PEAKS_PER_CLASS,
        )
        return {
            **base,
            "sam": sam,
            "cam_head": cam_head,
            "cam_in_feature": cfg.MODEL.WISH.CAM.IN_FEATURE,
            "peak_cfg": peak_cfg,
            "cam_weight": cfg.MODEL.WISH.CAM_WEIGHT,
            "self_weight": cfg.MODEL.WISH.SELF_WEIGHT,
            "warmup_iters": cfg.MODEL.WISH.WARMUP_ITERS,
            "num_classes": base["sem_seg_head"].num_classes,
            "sam_target_downsample": cfg.MODEL.WISH.SAM.TARGET_DOWNSAMPLE,
        }

    # -----------------------------------------------------------------------
    def forward(self, batched_inputs):
        if not self.training:
            return super().forward(batched_inputs)  # vendor instance inference path

        raw_images = [x["image"].to(self.device) for x in batched_inputs]
        norm_images = [(x - self.pixel_mean) / self.pixel_std for x in raw_images]
        images = ImageList.from_tensors(norm_images, self.size_divisibility)
        features = self.backbone(images.tensor)

        # ---- CAM head + L_cam ------------------------------------------------
        cam_feat = features[self.cam_in_feature]
        cam, cam_logits = self.cam_head(cam_feat)
        cam_gt = self._image_level_targets(batched_inputs).to(cam_logits.device)
        L_cam = loss_cam(cam_logits, cam_gt) * self.cam_weight

        # Warmup phase: only L_cam (paper Sec. 5.1.2)
        if int(self._iter.item()) < self.warmup_iters:
            return {"loss_cam": L_cam}

        # ---- Build heterogeneous targets (z, SAM masks) ---------------------
        targets = self._prepare_wish_targets(batched_inputs, cam, images, raw_images)

        # ---- Main Mask2Former forward + WISH criterion ----------------------
        outputs = self.sem_seg_head(features)
        losses = self.criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        # ---- L_self (Eq. 19) ------------------------------------------------
        sem_pred = self._merge_predicted_masks_per_class(outputs)  # [B, C, H, W]
        losses["loss_self"] = loss_self(cam, sem_pred) * self.self_weight
        losses["loss_cam"] = L_cam
        return losses

    # -----------------------------------------------------------------------
    def _image_level_targets(self, batched_inputs) -> torch.Tensor:
        """[B, C] multi-hot of present class ids (derivable from any weak-label type)."""
        B = len(batched_inputs)
        C = self.num_classes_wish
        out = torch.zeros(B, C, dtype=torch.float32)
        for b, inp in enumerate(batched_inputs):
            ids = inp.get("image_classes", None)
            if ids is None or ids.numel() == 0:
                continue
            valid = ids[(ids >= 0) & (ids < C)].long()
            out[b, valid] = 1.0
        return out

    def _prepare_wish_targets(
        self,
        batched_inputs,
        cam: torch.Tensor,
        images: ImageList,
        raw_images: List[torch.Tensor],
    ) -> List[Dict]:
        """Per image, build the WISH target dict.

        For each image:
          - If weak label is 'box' or 'point' -> SAM.encode_prompts directly.
          - If weak label is 'tag'           -> peaks_to_points then SAM.encode_prompts.
          - Run SAM.decode_masks for 3 candidate masks; pad to images.tensor size.
        """
        device = self.device
        padded_h, padded_w = images.tensor.shape[-2:]
        targets: List[Dict] = []

        for b, inp in enumerate(batched_inputs):
            wtype = inp.get("weak_label_type", "box")
            weak = inp.get("weak_labels", {})
            chw = raw_images[b]
            H, W = int(chw.shape[-2]), int(chw.shape[-1])

            # Convert CHW -> HWC. Detectron2's `input_format` controls whether
            # the incoming tensor is BGR or RGB. Only flip channels when the
            # configured input format is BGR (vendor default).
            hwc = chw.permute(1, 2, 0).contiguous()
            if hwc.shape[-1] == 3 and getattr(self, "input_format", "BGR") == "BGR":
                hwc = hwc.flip(-1)
            hwc_u8 = hwc.to(torch.uint8)

            feats, in_size, orig_size = self.sam.encode_image(hwc_u8)

            if wtype == "tag":
                present = weak.get("labels", torch.zeros(0, dtype=torch.long)).to(device)
                # Crop CAM to the unpadded image region; detach so peak finding
                # does not hold onto the L_cam autograd graph through SAM calls.
                cam_b = cam[b].detach()
                full = F.interpolate(
                    cam_b[None], size=(padded_h, padded_w),
                    mode="bilinear", align_corners=False,
                )[0]
                cam_crop = full[:, :H, :W].contiguous()
                wlabel = peaks_to_points(
                    cam=cam_crop,
                    present_classes=present,
                    cfg=self.peak_cfg,
                    sam=self.sam,
                    image_features=feats,
                    input_size=in_size,
                    original_size=orig_size,
                )
            elif wtype == "box":
                wlabel = {
                    "type": "box",
                    "labels": weak["labels"].to(device).long(),
                    "boxes": weak["boxes"].to(device).float(),
                }
            elif wtype == "point":
                wlabel = {
                    "type": "point",
                    "labels": weak["labels"].to(device).long(),
                    "points": weak["points"].to(device).float(),
                }
            else:
                raise ValueError(f"unknown weak_label_type: {wtype}")

            k = int(wlabel["labels"].shape[0])
            if k == 0:
                targets.append({
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                    "prompts": torch.zeros(0, self.sam.prompt_dim, device=device),
                    "sam_masks": torch.zeros(0, 3, padded_h, padded_w, device=device),
                })
                continue

            prompts = self.sam.encode_prompts(wlabel, orig_size)
            masks_orig, _ = self.sam.decode_masks(feats, wlabel, in_size, orig_size)
            # Pad SAM masks from (H, W) to (padded_h, padded_w): right & bottom only.
            pad_right = max(padded_w - W, 0)
            pad_bottom = max(padded_h - H, 0)
            sam_masks = F.pad(masks_orig, (0, pad_right, 0, pad_bottom), value=0.0)
            # Downsample stored SAM GT to cut matcher/criterion `point_sample`
            # memory (matcher runs 10x per step over [k,3,H,W]). point_sample
            # uses normalized coords, so resolution need not match pred_masks.
            ds = self.sam_target_downsample
            if ds > 1:
                sam_masks = F.avg_pool2d(sam_masks.float(), kernel_size=ds, stride=ds)

            targets.append({
                "labels": wlabel["labels"],
                "prompts": prompts,
                "sam_masks": sam_masks,
            })

        return targets

    def _merge_predicted_masks_per_class(self, outputs) -> torch.Tensor:
        """Eq. 19 input: M̂_c = OR over queries q whose argmax(ŷ_cls)=c.

        Returns a soft per-class map in [0,1] (max over sigmoid masks within
        each class assignment), differentiable through the mask side.
        """
        logits = outputs["pred_logits"]            # [B, Nq, C+1]
        masks = outputs["pred_masks"]              # [B, Nq, H, W]
        B, Nq = logits.shape[:2]
        H, W = masks.shape[-2:]
        C = self.num_classes_wish

        cls_argmax = logits[..., :C].argmax(dim=-1)  # [B, Nq], drops no-obj
        mask_sig = masks.sigmoid()                    # [B, Nq, H, W]

        merged = mask_sig.new_zeros(B, C, H, W)
        for b in range(B):
            for c in range(C):
                sel = (cls_argmax[b] == c)
                if sel.any():
                    merged[b, c] = mask_sig[b, sel].max(dim=0).values
        return merged


def _build_cam_head_lazy(cfg):
    """Build CAMHead. Channel count taken from ResNet stage-channel table."""
    _RESNET_CH = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}
    in_ch = _RESNET_CH[cfg.MODEL.WISH.CAM.IN_FEATURE]
    return CAMHead(in_channels=in_ch, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES)