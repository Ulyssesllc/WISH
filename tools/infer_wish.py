"""WISH inference: uses the WISH mask head (Fig. 3a), NOT the SAM decoder.

  python -m hetero.tools.infer_wish --config-file hetero/config/wish_coco.yaml \
      --weights outputs/hetero_v2/model_final.pth --image path/to/img.jpg
"""
from __future__ import annotations

from hetero.engine._logging import quiet_loggers, silence_third_party
silence_third_party()

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import hetero.models  # noqa: F401  (registers WISH meta-arch + decoder)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config

from hetero.config import add_wish_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument(
        "--output",
        default=None,
        help="output path; defaults to <cfg.OUTPUT_DIR>/vis/<image_stem>.png",
    )
    p.add_argument("--score-threshold", type=float, default=0.5)
    return p.parse_args()


def _palette(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(n, 3), dtype=np.int32)


def main():
    args = parse_args()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_wish_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    quiet_loggers()

    model = build_model(cfg).eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = next(model.parameters()).device

    image_path = Path(args.image)
    out_path = (
        Path(args.output)
        if args.output is not None
        else Path(cfg.OUTPUT_DIR) / "vis" / f"{image_path.stem}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    H, W = image_bgr.shape[:2]
    tensor = torch.as_tensor(image_bgr.astype("float32").transpose(2, 0, 1)).to(device)

    with torch.no_grad():
        outputs = model([{"image": tensor, "height": H, "width": W}])

    inst = outputs[0]["instances"].to("cpu")
    keep = inst.scores >= args.score_threshold
    inst = inst[keep]

    vis = image_bgr.copy()
    colors = _palette(max(len(inst), 1))
    for i in range(len(inst)):
        m = inst.pred_masks[i].numpy().astype(bool)
        if not m.any():
            continue
        color = colors[i % len(colors)]
        vis[m] = (0.5 * vis[m] + 0.5 * color).clip(0, 255).astype("uint8")
        # contour for boundary clarity
        cnts, _ = cv2.findContours(
            m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, cnts, -1, color.tolist(), 2)

    cv2.imwrite(str(out_path), vis)
    print(f"Wrote {len(inst)} instances (score>={args.score_threshold}) to {out_path}")


if __name__ == "__main__":
    main()