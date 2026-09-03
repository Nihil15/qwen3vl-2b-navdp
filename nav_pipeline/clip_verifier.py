"""CLIP open-vocabulary verification for a SAM-segmented detection crop.

Grounding DINO's box can be a confident false positive (wrong object, same
rough shape/context). Once SAM turns that box into a tight instance mask,
CLIP scores the masked crop against the target phrase versus a handful of
generic "not the target" prompts -- a cheap second opinion the rover can use
to refuse to commit to a goal instead of walking up to the wrong object.

Loads openai/clip-vit-base-patch32 from the local HF cache.
"""

from typing import List, Optional

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


class ClipVerifier:
    # Generic "this crop is not the target" anchors. Softmax over
    # [target, *negatives] turns a raw, hard-to-calibrate cosine similarity
    # into a probability that's comparable across crop sizes/content.
    NEGATIVE_PROMPTS = ("background", "an empty scene", "the floor or ground", "a wall")

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_id)
        # force safetensors: torch 2.5.x here is below the 2.6 floor transformers
        # now requires for torch.load-based (pickle) checkpoint loading
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def verify(self, rgb: np.ndarray, box: np.ndarray, text: str) -> float:
        """Crop `box` ([x0,y0,x1,y1] px) from rgb; return P(target | crop) in [0, 1]."""
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, rgb.shape[1]), min(y1, rgb.shape[0])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        crop = rgb[y0:y1, x0:x1]
        prompts: List[str] = [f"a photo of a {text}"] + [f"a photo of {p}" for p in self.NEGATIVE_PROMPTS]
        inputs = self.processor(images=crop, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        return float(probs[0].item())
