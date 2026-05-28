from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry


class SAMInference:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if checkpoint_path is None:
            checkpoint_path = str(
                Path(__file__).resolve().parents[1]
                / "checkpoints"
                / "sam_vit_b_01ec64.pth"
            )

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.predictor = SamPredictor(self.model)

        self.output_dir = Path(output_dir)

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        self.predictor.set_image(image, image_format=image_format)

    def predict(
        self,
        image: Optional[np.ndarray] = None,
        image_format: str = "RGB",
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        save_prefix: str = "mask",
        save_png: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        point_coords: Nx2 in (X, Y) pixels; point_labels: length N (1 fg, 0 bg)
        box: length-4 XYXY pixels
        mask_input: 1x256x256 low-res mask logits or 256x256 mask
        """
        if image is not None:
            self.set_image(image, image_format=image_format)
        if not self.predictor.is_image_set:
            raise RuntimeError("Call set_image or pass image before predict.")

        if point_coords is not None:
            point_coords = np.asarray(point_coords, dtype=np.float32)
            if point_coords.ndim == 1:
                point_coords = point_coords[None, :]
        if point_labels is not None:
            point_labels = np.asarray(point_labels, dtype=np.int32)
            if point_labels.ndim == 0:
                point_labels = point_labels[None]
        if point_coords is not None and point_labels is None:
            raise ValueError("point_labels must be provided with point_coords.")
        if box is not None:
            box = np.asarray(box, dtype=np.float32)
        if mask_input is not None:
            mask_input = np.asarray(mask_input, dtype=np.float32)
            if mask_input.ndim == 2:
                mask_input = mask_input[None, :, :]

        masks, scores, low_res = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

        self.save_masks(masks, prefix=save_prefix, save_png=save_png)
        return masks, scores, low_res

    def get_image_embedding(self, as_numpy: bool = False) -> torch.Tensor | np.ndarray:
        emb = self.predictor.get_image_embedding()
        if as_numpy:
            return emb.detach().cpu().numpy()
        return emb

    def get_embedding_at_point(
        self, point_xy: Tuple[float, float], as_numpy: bool = True
    ) -> torch.Tensor | np.ndarray:
        if not self.predictor.is_image_set:
            raise RuntimeError("Call set_image before get_embedding_at_point.")
        coords = np.array([point_xy], dtype=np.float32)
        coords = self.predictor.transform.apply_coords(coords, self.predictor.original_size)
        x_resized, y_resized = coords[0]

        emb = self.predictor.get_image_embedding()
        feat_h, feat_w = emb.shape[-2:]
        img_size = self.predictor.model.image_encoder.img_size

        x_idx = int(np.clip(x_resized / img_size * feat_w, 0, feat_w - 1))
        y_idx = int(np.clip(y_resized / img_size * feat_h, 0, feat_h - 1))

        vec = emb[0, :, y_idx, x_idx]
        if as_numpy:
            return vec.detach().cpu().numpy()
        return vec

    def save_masks(self, masks: np.ndarray, prefix: str = "mask", save_png: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks):
            if mask.dtype == np.bool_:
                mask_to_save = mask.astype(np.uint8)
            else:
                mask_to_save = mask
            np.save(self.output_dir / f"{prefix}_{i}.npy", mask_to_save)
            if save_png:
                if mask.dtype not in (np.bool_, np.uint8):
                    raise ValueError("save_png requires binary masks.")
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise ImportError("save_png=True requires pillow.") from exc
                Image.fromarray(mask_to_save.astype(np.uint8) * 255).save(
                    self.output_dir / f"{prefix}_{i}.png"
                )
