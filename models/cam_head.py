"""H_cam: 1x1 conv on backbone feature -> CAMs (Sec. 4.3).

Two responsibilities:
  - Forward: produce A in R^{C x H x W} and image-level logits y_cls^cam (GAP).
  - Tag-to-point conversion (Eq. 18) using local-max peaks + SAM-IoU dedup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from hetero.models.sam_wrapper import SAMWrapper, WeakLabel


class CAMHead(nn.Module):
    """1x1 conv producing class-activation maps A in R^{C x H x W}."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feature: [B, in_channels, H, W] -> (cam [B, C, H, W], logits [B, C])."""
        cam = self.conv(feature)
        logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)  # GAP
        return cam, logits


@dataclass
class PeakConfig:
    threshold: float = 0.3
    filter_size: int = 3
    dedup_iou: float = 0.5
    max_peaks_per_class: int = 32


@torch.no_grad()
def peaks_to_points(
    cam: torch.Tensor,            # [C, H, W] for a single image (any spatial res)
    present_classes: torch.Tensor,  # [k_img] class ids that exist in this image
    cfg: PeakConfig,
    sam: SAMWrapper,
    image_features: torch.Tensor,
    input_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> WeakLabel:
    """Eq. 18: convert tags Y_t -> tag-derived points Y_{t->p}.

    Steps (per present class c):
      1. Apply local-max filter on cam[c]; keep peaks with normalized cam[c] >= threshold.
      2. For each peak, query SAM with that point -> (3 masks, iou_pred).
      3. Greedy dedup: merge peaks whose SAM-mask IoU > dedup_iou, keep the one
         with highest stability (SAM iou_pred).
      4. Points are returned in original image resolution.
    """
    device = cam.device
    H_orig, W_orig = original_size

    # Upsample CAM to original-image resolution, ReLU + per-channel max-normalize.
    cam_up = F.interpolate(
        cam[None], size=(H_orig, W_orig), mode="bilinear", align_corners=False
    )[0]
    cam_up = F.relu(cam_up)
    max_per_c = cam_up.flatten(1).max(dim=-1).values.clamp_min(1e-6)
    cam_norm = cam_up / max_per_c[:, None, None]

    all_pts: list[list[float]] = []
    all_labels: list[int] = []

    for c in present_classes.detach().cpu().tolist():
        if c < 0 or c >= cam_norm.shape[0]:
            continue
        ch = cam_norm[c]  # [H, W]
        ks = max(cfg.filter_size, 1)
        pad = ks // 2
        pooled = F.max_pool2d(ch[None, None], kernel_size=ks, stride=1, padding=pad)[0, 0]
        is_peak = (ch == pooled) & (ch >= cfg.threshold)
        ys, xs = is_peak.nonzero(as_tuple=True)
        if ys.numel() == 0:
            continue

        scores = ch[ys, xs]
        if scores.numel() > cfg.max_peaks_per_class:
            top = scores.topk(cfg.max_peaks_per_class).indices
            ys, xs, scores = ys[top], xs[top], scores[top]

        # Run SAM per peak, store best-of-3 mask and stability.
        peak_masks: list[torch.Tensor] = []
        peak_stab: list[float] = []
        for y, x in zip(ys.tolist(), xs.tolist()):
            pt = torch.tensor([[float(x), float(y)]], dtype=torch.float32, device=device)
            m, iou = sam.mask_for_point(image_features, pt, input_size, original_size)
            best = int(iou.argmax().item())
            peak_masks.append(m[best].bool())
            peak_stab.append(float(iou[best].item()))

        # Greedy dedup by IoU (descending stability).
        order = sorted(range(len(peak_stab)), key=lambda i: -peak_stab[i])
        kept: list[int] = []
        for i in order:
            mi = peak_masks[i]
            dup = False
            for j in kept:
                mj = peak_masks[j]
                inter = (mi & mj).sum().float()
                union = (mi | mj).sum().float().clamp_min(1.0)
                if (inter / union).item() > cfg.dedup_iou:
                    dup = True
                    break
            if not dup:
                kept.append(i)

        for i in kept:
            all_pts.append([float(xs[i].item()), float(ys[i].item())])
            all_labels.append(int(c))

    if not all_pts:
        return {
            "type": "tag",
            "labels": torch.zeros(0, dtype=torch.long, device=device),
            "points": torch.zeros(0, 2, dtype=torch.float32, device=device),
        }
    return {
        "type": "tag",
        "labels": torch.as_tensor(all_labels, dtype=torch.long, device=device),
        "points": torch.as_tensor(all_pts, dtype=torch.float32, device=device),
    }


def loss_cam(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_cam (Eq. 17): multi-label BCE over image-level class presence."""
    return F.binary_cross_entropy_with_logits(logits, target.float(), reduction="mean")


def loss_self(cam: torch.Tensor, semantic_pred: torch.Tensor) -> torch.Tensor:
    """L_self (Eq. 19): BCE between CAM and OR-merged predicted masks per class.

    cam:            [B, C, h, w]  raw logits from H_cam
    semantic_pred:  [B, C, H, W]  built in wish_meta_arch by OR-merging matched
                                  predicted masks; values already in [0, 1].

    Only CAM is updated by this loss; the merged prediction is detached so it
    does not feed back into the mask head (paper intent: self-improve CAMs).
    """
    target = semantic_pred.detach().clamp(0.0, 1.0)
    if cam.shape[-2:] != target.shape[-2:]:
        cam = F.interpolate(cam, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(cam, target)