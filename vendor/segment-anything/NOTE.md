
**Codebase Map**
- Public API and builders live in __init__.py and build_sam.py, exposing `sam_model_registry` plus `build_sam_*` that assemble the model parts.
- The core model wrapper `Sam` (image encoder -> prompt encoder -> mask decoder) is in sam.py.
- The ViT image encoder and neck are in image_encoder.py.
- Prompt embedding for points, boxes, and masks is in prompt_encoder.py.
- Mask generation and IoU scoring are in mask_decoder.py and the TwoWayTransformer is in transformer.py.
- Inference helpers and preprocessing are in predictor.py, automatic_mask_generator.py, and transforms.py.

**Inference and Prompts**
- Image input: `SamPredictor.set_image()` expects HWC uint8 [0,255] in RGB or BGR and handles resizing internally. predictor.py
- Point prompts: `point_coords` is Nx2 (X,Y pixels) with `point_labels` length N (1=foreground, 0=background). predictor.py
- Box prompts: `box` is length-4 XYXY pixels. predictor.py
- Mask prompts: `mask_input` is a low-res 1x256x256 mask/logits, typically from a previous call’s `low_res_logits`. predictor.py
- Batched/torch path: `predict_torch()` expects prompts already resized to the model frame (same transform as `ResizeLongestSide`). predictor.py, transforms.py
- Whole-image proposals: `SamAutomaticMaskGenerator.generate()` runs a grid of points and returns many mask records with segmentation, bbox, and scores. automatic_mask_generator.py

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="path/to/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

predictor.set_image(image)  # HWC uint8 RGB
masks, scores, low_res = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),
    box=np.array([x0, y0, x1, y1]),
    mask_input=None,
    multimask_output=True,
)
```

**Extracting Embeddings**
- After `set_image`, call `get_image_embedding()` to retrieve the image encoder output (shape 1xCxHxW, typically C=256, H=W=64). predictor.py
- In the raw model path, `Sam.forward()` computes `image_embeddings` via `image_encoder` before prompt encoding and decoding. sam.py
- The encoder implementation is in image_encoder.py; for earlier ViT block activations you would add hooks or return intermediates there.
- Prompt embeddings can be obtained directly by calling `sam.prompt_encoder(points, boxes, masks)` once prompts are in model coordinates. prompt_encoder.py

**More than 3 Masks**
- Default `num_multimask_outputs=3` is set when the model is built, and `multimask_output=True` returns three proposals per prompt. build_sam.py, mask_decoder.py, predictor.py
- `multimask_output=False` forces a single mask per prompt. predictor.py
- More than 3 proposals from a single prompt is not supported by the shipped checkpoints; you would need to rebuild the decoder with a larger `num_multimask_outputs` and train or adapt weights to match the new token/head shapes. mask_decoder.py, build_sam.py
- If you want "all masks," use `SamAutomaticMaskGenerator` and tune `points_per_side` plus filtering thresholds (`pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`) to keep more proposals. automatic_mask_generator.py

If you want, tell me your exact prompt type (points only, box only, mask refinement, or mixed) and I can tailor a minimal inference snippet.

## Local inference helper
- New helper in sam_infer/utils.py provides SAMInference for the ViT-B checkpoint at checkpoints/sam_vit_b_01ec64.pth.
- Supports point, box, and mask prompts; saves masks to outputs/ as .npy files (optional .png).
- Embedding access is available via get_image_embedding() and get_embedding_at_point((x, y)).