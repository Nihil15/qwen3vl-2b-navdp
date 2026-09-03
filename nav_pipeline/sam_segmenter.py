"""SAM 2.1 (hiera-small) box-prompted segmentation.

Sits between Grounding DINO and NavDP: the DINO bbox prompts SAM, and the
resulting instance mask gives
  * a clean object depth (median over mask pixels only — no background
    contamination like the shrunk-bbox heuristic), and
  * a mask-area stop criterion that is robust to elongated/partial boxes.

Loads facebook/sam2.1-hiera-small (~39 M params) from the local HF cache.
"""

from typing import Optional

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor


class Sam2Segmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda:0"):
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def segment_box(self, rgb: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """rgb: HxWx3 uint8; box: [x0,y0,x1,y1] pixels -> bool mask HxW or None."""
        inputs = self.processor(
            images=rgb,
            input_boxes=[[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]  # (num_objects, num_masks, H, W)
        mask = masks[0, 0].numpy().astype(bool)
        if mask.sum() == 0:
            return None
        return mask


def mask_median_depth(depth_m: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Median of valid depth inside the mask (meters)."""
    vals = depth_m[mask]
    vals = vals[np.isfinite(vals) & (vals > 0.1)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Mask centroid as [u, v] pixels."""
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    """Tight [x0, y0, x1, y1] box around the mask's true pixels."""
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
