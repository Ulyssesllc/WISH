#!/usr/bin/env python3
"""Test script to run SAM inference with box prompt and visualize output."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sam_infer.utils import SAMInference


def visualize_mask(image: np.ndarray, masks: np.ndarray, output_path: Path) -> None:
    """Overlay masks on image and save."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create colored overlay
    overlay = image.copy().astype(float)
    colors = [
        [0, 255, 0],      # Green for mask 0
        [255, 0, 0],      # Red for mask 1
        [0, 0, 255],      # Blue for mask 2
    ]
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array(color) * 0.5
    
    overlay = overlay.astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved visualized output to {output_path}")


def main():
    # Paths
    script_dir = Path(__file__).parent
    image_path = Path("/home/linhdang/workspace/minhbao_workspace/SSL/WSIS/Vig-Box2Seg/hetero/vendor/segment-anything/test_image.png")
    box_file = script_dir / "test_box.txt"
    output_dir = script_dir / "outputs"
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    if not box_file.exists():
        print(f"Box file not found: {box_file}")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image.shape}")
    
    # Read box
    with open(box_file) as f:
        box_str = f.read().strip()
    box = np.array([float(x) for x in box_str.split()], dtype=np.float32)
    print(f"Box (XYXY): {box}")
    
    # Run inference
    
    ITER = 100
    for _ in range(ITER):
        print("Initializing SAMInference...")
        infer = SAMInference(output_dir=str(output_dir))
        
        print("Running inference...")
        masks, scores, low_res = infer.predict(
            image=image,
            box=box,
            multimask_output=True,
            save_prefix="mask_box",
            save_png=True,
        )
        
        print(f"Masks shape: {masks.shape}")
        print(f"IoU scores: {scores}")
    
    # Visualize
    print("Visualizing output...")
    visualize_mask(image, masks, output_dir / "visualization_box.png")
    
    print("✓ Inference complete!")
    print(f"  Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()