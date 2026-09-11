"""SigLIP2 appearance embeddings for cross-session entity memory.

Drop-in twin of dinov2_embedder.py's Dinov2Embedder (same embed(rgb, box)
signature, same lazy-nothing/eager-at-construction load style) but with
SigLIP2's image tower instead of DINOv2 -- deliberately the SAME encoder
family fact3r-map's own entity memory uses (nav_pipeline/fact3r_live_memory.py,
fact3r_entity_recall.py, MASt3R-SLAM/fact3r-map/fact3r/semantics/goal_memory.py's
own SigLIP2 TextEncoder), so an embedding banked here is directly comparable
in spirit to -- and could later be pointed at -- that same model family if
this ever needs to interop with a real fact3r-map map instead of just this
process's own goal-memory JSONL.

Why a SEPARATE embedder from Dinov2Embedder rather than reusing it for this
job too: this repo already picked DINOv2 over CLIP for DINO-mode's live
same-tick re-identification (dinov2_embedder.py's own docstring: CLIP's
global/class-level embedding conflated solid-red vs. solid-blue crops at
0.93 cosine similarity). SigLIP2 is used here instead for the SEPARATE job
of cross-session goal-memory appearance verification specifically because
that is what fact3r-map's own entity/goal memory already standardizes on
project-wide (see the modules listed above) -- keeping this one entry point
on the same family means a future shared on-disk memory format doesn't need
a second embedding column.

Loads google/siglip2-base-patch16-224 from the local HF cache.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class Siglip2Embedder:
    def __init__(self, model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda:0"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def _feats(out):
        """`SiglipModel.get_{image,text}_features` returns a bare tensor on
        recent transformers, but on some versions (and whenever `AutoModel`
        resolves the siglip2 checkpoint through a generic head) it returns a
        `ModelOutput` instead -- pull the pooled (B, D) vector out of it so the
        L2-normalize below has a tensor to work on either way."""
        import torch

        if torch.is_tensor(out):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        lhs = getattr(out, "last_hidden_state", None)
        if lhs is not None:
            return lhs[:, -1] if lhs.dim() == 3 else lhs
        return out[0]

    @torch.no_grad()
    def embed(self, rgb: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """L2-normalized SigLIP2 image-tower embedding of `box`'s crop, or None if degenerate."""
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, rgb.shape[1]), min(y1, rgb.shape[0])
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgb[y0:y1, x0:x1]
        inputs = self.processor(images=[crop], return_tensors="pt").to(self.device)
        feats = self._feats(self.model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts) -> np.ndarray:
        """L2-normalized SigLIP2 text-tower embeddings, one row per text.

        Signature/semantics match MASt3R-SLAM/fact3r-map/fact3r/semantics/
        goal_memory.py's TextEncoder Protocol exactly (structural typing --
        this class needs no inheritance from it to satisfy it), so a
        Siglip2Embedder instance can be passed straight into that module's
        find_memory_hit() for its tier-1 paraphrase-similarity fallback
        ("go back to the mug" vs "the coffee mug"), the same mechanism
        resolve_semantic_goal_verified.py's offline locate stage uses.
        SigLIP was trained with texts padded to the model's fixed max
        length (unlike CLIP's variable-length padding), so this pads to
        max_length rather than the shortest-batch-member default.
        """
        inputs = self.processor(text=list(texts), padding="max_length", return_tensors="pt").to(self.device)
        feats = self._feats(self.model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Both embeddings are already L2-normalized by embed(), so a plain
        dot product IS the cosine similarity -- kept as a named helper so
        call sites read as intent, not arithmetic."""
        return float(np.dot(a, b))
