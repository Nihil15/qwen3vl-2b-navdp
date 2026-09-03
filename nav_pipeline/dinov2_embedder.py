"""DINOv2 instance-appearance embeddings for same-class object re-identification.

CLIP's embeddings are global/class-level -- good at "is this a chair" (see
clip_verifier.py's verify(), which stays on CLIP for exactly that job), weak
at "is this THIS chair": empirically, solid red vs. solid blue crops scored
0.93 cosine similarity on CLIP, an unusably high floor for telling same-class
instances apart. DINOv2 is a self-supervised ViT with pixel-level
correspondence and is the standard swap for instance-level re-ID/retrieval --
it separates same-category objects that CLIP conflates.

Loads facebook/dinov2-small from the local HF cache.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


class Dinov2Embedder:
    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cuda:0"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(device).eval()

    @torch.no_grad()
    def embed(self, rgb: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """L2-normalized DINOv2 CLS-token embedding of `box`'s crop, or None if degenerate."""
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, rgb.shape[1]), min(y1, rgb.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgb[y0:y1, x0:x1]
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        feat = out.last_hidden_state[:, 0]  # CLS token
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()
