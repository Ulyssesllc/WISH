"""Frozen SAM wrapper used by WISH for prompt encoding + pseudo-mask generation.

Two services to the rest of the pipeline:
  1. `encode_prompts(weak_labels)` -> z in R^{k x 256}  (paper Eq. 9)
  2. `decode_masks(image_emb, prompt_emb)` -> (3, H, W) candidate masks (Sec. 3.2)
Both are pure inference; the SAM module is frozen by default (cfg.MODEL.WISH.SAM.FREEZE).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import hashlib

import numpy as np
import torch
from torch import nn

# vendor sys.path side-effect already handled in hetero.models.mask2former_r50
from segment_anything import sam_model_registry  # noqa: E402
from segment_anything.utils.transforms import ResizeLongestSide  # noqa: E402


# A single per-image weak-label record handed in from the dataset mapper.
# Exactly one of {points, boxes} is populated for the spatial prompts; class tags
# convert to points upstream (cam_head.peaks_to_points).
WeakLabel = Dict[str, torch.Tensor]
# Schema:
#   "type":  str in {"tag", "point", "box"}
#   "labels": LongTensor [k]   instance class ids (contiguous)
#   "points": FloatTensor [k, 2]  (x, y) in original image px  -- type in {tag,point}
#   "boxes":  FloatTensor [k, 4]  XYXY in original image px    -- type == "box"


class SAMWrapper(nn.Module):
    """Thin wrapper around `segment_anything.Sam` with frozen weights by default."""

    def __init__(
        self,
        sam_type: str = "vit_b",
        weights: str | Path = "hetero/vendor/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
        freeze: bool = True,
        multimask: bool = True,
    ) -> None:
        super().__init__()
        self.sam = sam_model_registry[sam_type](checkpoint=str(weights))
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.multimask = multimask
        if freeze:
            for p in self.sam.parameters():
                p.requires_grad_(False)
            self.sam.eval()
        # Simple in-memory cache mapping image content hash -> (features_cpu, input_size, original_size)
        # This avoids re-encoding the same image multiple times per training step
        # (the dataset mapper / dataloader tend to reuse images across iterations
        # when using the same pinned cache). Keys are SHA1 of the uint8 image bytes.
        self._image_cache: Dict[str, Tuple[object, Tuple[int, int], Tuple[int, int]]] = {}

    @property
    def device(self) -> torch.device:
        return next(self.sam.parameters()).device

    @property
    def prompt_dim(self) -> int:
        return self.sam.prompt_encoder.embed_dim

    # ------- image embedding (called once per image; cache externally) -------
    @torch.no_grad()
    def encode_image(
        self, image_rgb_uint8: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """image_rgb_uint8: (H, W, 3) uint8 on any device.
        Returns (features [1,C,h,w], input_size, original_size)."""
        H, W = int(image_rgb_uint8.shape[0]), int(image_rgb_uint8.shape[1])
        image_np = image_rgb_uint8.detach().cpu().numpy().astype(np.uint8)
        # Cache by image content to avoid repeated SAM image encoding.
        key = hashlib.sha1(image_np.tobytes()).hexdigest()
        cached = self._image_cache.get(key)
        if cached is not None:
            # cached[0] may be a tensor or a nested structure of tensors stored on CPU.
            features_cpu, input_size, orig_size = cached
            # Move features to current device before returning.
            def _to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(self.device)
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_device(o) for o in obj)
                return obj

            features = _to_device(features_cpu)
            return features, input_size, (H, W)
        # ResizeLongestSide -> (h_r, w_r, 3) uint8 numpy
        resized = self.transform.apply_image(image_np)
        x = torch.as_tensor(resized, device=self.device).permute(2, 0, 1).contiguous()[None]
        input_size = tuple(int(s) for s in x.shape[-2:])  # before padding
        padded = self.sam.preprocess(x.float())
        features = self.sam.image_encoder(padded)
        # Store a CPU copy in cache (detached) to keep GPU memory available.
        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_cpu(o) for o in obj)
            return obj

        features_cpu = _to_cpu(features)
        self._image_cache[key] = (features_cpu, input_size, (H, W))
        return features, input_size, (H, W)

    # ------- prompts -> latents (Eq. 9) -------
    @torch.no_grad()
    def encode_prompts(
        self,
        weak_labels: WeakLabel,
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Project each per-image weak label to its SAM prompt latent z_j.

        Returns: FloatTensor [k, prompt_dim].  For 'box' uses both corner embeddings
        pooled (mean) to a single vector per instance; for 'point'/'tag' uses the
        positive-point embedding directly.
        """
        wtype = weak_labels.get("type", "point")
        device = self.device

        if wtype == "box":
            boxes_np = weak_labels["boxes"].detach().cpu().numpy().astype(np.float32)
            if boxes_np.shape[0] == 0:
                return torch.zeros(0, self.prompt_dim, device=device)
            boxes_t = self.transform.apply_boxes(boxes_np, original_size)  # numpy
            boxes_t = torch.as_tensor(boxes_t, dtype=torch.float, device=device)
            sparse, _ = self.sam.prompt_encoder(points=None, boxes=boxes_t, masks=None)
            # sparse: [k, 2, D] (two corner tokens) -> mean over corners
            return sparse.mean(dim=1)

        # point / tag (already converted to points upstream)
        pts = weak_labels["points"].detach().cpu().numpy().astype(np.float32)
        if pts.shape[0] == 0:
            return torch.zeros(0, self.prompt_dim, device=device)
        coords_t = self.transform.apply_coords(pts, original_size)  # numpy
        # batch=k, n_points=1
        coords_t = torch.as_tensor(coords_t, dtype=torch.float, device=device)[:, None, :]
        labels_t = torch.ones(coords_t.shape[0], 1, dtype=torch.int, device=device)
        sparse, _ = self.sam.prompt_encoder(
            points=(coords_t, labels_t), boxes=None, masks=None
        )
        # sparse: [k, 2, D] (point token + 1 padding "not-a-box" token) -> take the point token
        return sparse[:, 0, :]

    # ------- prompts -> 3 candidate masks (Eq. 13 supervision) -------
    @torch.no_grad()
    def decode_masks(
        self,
        image_features: torch.Tensor,
        weak_labels: WeakLabel,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run SAM mask decoder for each weak label.

        Returns:
          masks:   FloatTensor [k, 3, H, W] in {0,1} at *original* image resolution
          iou_pred: FloatTensor [k, 3]
        """
        wtype = weak_labels.get("type", "point")
        device = self.device

        if wtype == "box":
            boxes_np = weak_labels["boxes"].detach().cpu().numpy().astype(np.float32)
            k = boxes_np.shape[0]
            if k == 0:
                return (
                    torch.zeros(0, 3, *original_size, device=device),
                    torch.zeros(0, 3, device=device),
                )
            boxes_t = torch.as_tensor(
                self.transform.apply_boxes(boxes_np, original_size),
                dtype=torch.float,
                device=device,
            )
            sparse, dense = self.sam.prompt_encoder(points=None, boxes=boxes_t, masks=None)
        else:
            pts = weak_labels["points"].detach().cpu().numpy().astype(np.float32)
            k = pts.shape[0]
            if k == 0:
                return (
                    torch.zeros(0, 3, *original_size, device=device),
                    torch.zeros(0, 3, device=device),
                )
            coords_t = torch.as_tensor(
                self.transform.apply_coords(pts, original_size),
                dtype=torch.float,
                device=device,
            )[:, None, :]
            labels_t = torch.ones(k, 1, dtype=torch.int, device=device)
            sparse, dense = self.sam.prompt_encoder(
                points=(coords_t, labels_t), boxes=None, masks=None
            )

        low_res, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_features,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=self.multimask,
        )
        masks = self.sam.postprocess_masks(low_res, input_size, original_size)
        masks = (masks > self.sam.mask_threshold).float()
        return masks, iou_pred

    # ------- tag-only helper: get SAM mask for a single point (peak dedup) -------
    @torch.no_grad()
    def mask_for_point(
        self,
        image_features: torch.Tensor,
        point_xy: torch.Tensor,  # [1, 2]
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Used by cam_head.peaks_to_points for IoU-based dedup + stability ranking.

        Returns (mask [3,H,W], iou_pred [3]).
        """
        device = self.device
        pts = point_xy.detach().cpu().numpy().astype(np.float32)  # [1, 2]
        coords_t = torch.as_tensor(
            self.transform.apply_coords(pts, original_size),
            dtype=torch.float,
            device=device,
        )[None]  # [1, 1, 2]
        labels_t = torch.ones(1, 1, dtype=torch.int, device=device)
        sparse, dense = self.sam.prompt_encoder(
            points=(coords_t, labels_t), boxes=None, masks=None
        )
        low_res, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_features,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        masks = self.sam.postprocess_masks(low_res, input_size, original_size)
        masks = (masks > self.sam.mask_threshold).float()
        return masks[0], iou_pred[0]