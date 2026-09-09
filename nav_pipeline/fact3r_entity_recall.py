"""Pixel-goal recall: SAM2 mask proposals + SigLIP2 re-identification against
a small multi-view memory bank -- fact3r-map's own entity-memory shape (up to
a handful of diverse embeddings per entity, matched by best-of-bank
similarity; see fact3r-map's README section on "appearance-aware UOT": each
entity keeps at most eight diverse SigLIP2 views), applied to ONE entity
learned live during navigation instead of an offline multi-entity map.

This is fact3r-map's own building blocks (SAM2 class-agnostic proposals,
SigLIP2 appearance embeddings -- the same pair ``run_fact3r_live.py`` uses
for its causal mapper) wired directly into NavDP's existing pixel-goal path,
instead of into a 3D map or a full separate-process entity index. The full
``fact3r.*`` package needs Python 3.10+ (``dataclass(slots=True)``) and
can't import under this pipeline's Python-3.9 env, so this reimplements the
two primitives directly rather than depending on it -- see
FACT3R_INTEGRATION.md / memory notes for the offline-map alternative
(``fact3r_goal_source.py`` + ``resolve_semantic_goal.py``) when a real
pre-built MASt3R map is available instead of a live single-entity memory.

``Fact3rEntityRecallGuide`` mirrors ``qwen_pixel_goal.QwenVLPixelGoal``'s
public surface exactly (``ground_candidates`` / ``ground`` / lazy
``_ensure_loaded``), so it drops straight into
``pipeline.DinoNavDPPipeline`` by swapping ``pipe._qwen_pixel_guide`` --
nothing in pipeline.py changes. That gets the recall leg the exact same
TRACK / belief-coasting / SEARCH-and-reacquire machinery Qwen's live
grounding already has, for free.

Two-phase use:

* **Leg 1 (live, Qwen driving) -- building the memory:** call
  :meth:`remember` every time some other process flags a pixel as worth
  remembering (e.g. a secondary Qwen "is there anything else here?" probe,
  called repeatedly as the robot continues past the object at different
  distances/angles). Each call SAM2-segments the frame, finds the proposal
  under that pixel, SigLIP2-encodes the crop, and -- if it's meaningfully
  different from views already banked (``diversity_max_similarity``) --
  adds it to ``self.memory``, capped at ``max_views``. No depth, no
  odometry, no 3D point anywhere in this memory.
* **Leg 2 ("go to B") -- recalling from that memory:** install this guide
  as ``pipe._qwen_pixel_guide`` and drive with
  ``pipe.step(..., instruction="go to B", ...)`` exactly as leg 1 drove on
  Qwen. Every throttled tick (the same ``qwen_instruction_period_s`` gate
  Qwen uses), this proposes masks in the CURRENT frame, embeds each, scores
  it against the BEST-matching memory view (not a single stored embedding),
  and returns the best-scoring proposal above ``match_threshold`` as a
  ``PixelGoal`` -- or an empty list when nothing matches, which the
  pipeline's own belief/SEARCH fallback already handles exactly like a lost
  Qwen grounding (spin to reacquire, resume TRACK the moment recognized).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .qwen_pixel_goal import PixelGoal

# The `sam2` package isn't pip-installed in every env this pipeline runs in
# (it's an editable checkout elsewhere) -- same SAM2_ROOT-on-sys.path pattern
# MARS/mars-habitatsim/sam_vla/perception/sam2_custom_head.py uses. Only
# needed if `import sam2` doesn't already resolve.
_SAM2_ROOT = os.environ.get("SAM2_ROOT", "/home/gpu/Desktop/pineapple/packages/sam2")
if _SAM2_ROOT and _SAM2_ROOT not in sys.path:
    sys.path.insert(0, _SAM2_ROOT)


def _xywh_to_xyxy(box) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x + w), float(y + h)


@dataclass
class MemoryView:
    """One banked SigLIP2 observation of the recall target."""

    embedding: np.ndarray
    label: str = ""
    step: int = -1  # caller-supplied tick index, for the report/log only


class Fact3rEntityRecallGuide:
    def __init__(
        self,
        device: str = "cuda:0",
        discovery_model: str = "facebook/sam2.1-hiera-small",
        siglip_model: str = "google/siglip2-base-patch16-224",
        match_threshold: float = 0.72,
        points_per_side: int = 24,
        points_per_batch: int = 64,
        min_mask_region_area: int = 400,
        max_views: int = 5,
        diversity_max_similarity: float = 0.97,
    ):
        self.device = device
        self.discovery_model = discovery_model
        self.siglip_model = siglip_model
        self.match_threshold = float(match_threshold)
        self.points_per_side = int(points_per_side)
        self.points_per_batch = int(points_per_batch)
        self.min_mask_region_area = int(min_mask_region_area)
        self.max_views = int(max_views)
        self.diversity_max_similarity = float(diversity_max_similarity)

        self._sam = None
        self._siglip = None
        self._siglip_proc = None

        self.memory: List[MemoryView] = []  # the multi-view recall memory ("the map" for this entity)
        self.target_label: str = ""
        self.last_match_score: Optional[float] = None       # debug/logging only
        self.last_match_view_index: Optional[int] = None     # which banked view won, for the report

    @property
    def views_stored(self) -> int:
        return len(self.memory)

    # -- lazy load, same eager-at-construction pattern QwenVLPixelGoal uses -- #
    def _ensure_loaded(self) -> None:
        if self._sam is not None:
            return
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from transformers import AutoModel, AutoProcessor

        self._sam = SAM2AutomaticMaskGenerator.from_pretrained(
            self.discovery_model,
            device=self.device,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=self.min_mask_region_area,
        )
        self._siglip = AutoModel.from_pretrained(self.siglip_model).to(self.device).eval()
        self._siglip_proc = AutoProcessor.from_pretrained(self.siglip_model)

    def _propose(self, rgb: np.ndarray) -> list:
        import torch

        image = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if str(self.device).startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return self._sam.generate(image)
        with torch.inference_mode():
            return self._sam.generate(image)

    def _embed_crop(self, rgb: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = box_xyxy
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = rgb[y0:y1, x0:x1]
        pil = Image.fromarray(np.ascontiguousarray(crop))
        inputs = self._siglip_proc(images=[pil], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._siglip.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()

    def _best_mask_at_pixel(self, rgb: np.ndarray, pixel: Tuple[float, float]):
        anns = self._propose(rgb)
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        best = None
        for ann in anns:
            mask = ann["segmentation"]
            if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]:
                if best is None or ann["area"] < best["area"]:  # smallest containing mask = tightest fit
                    best = ann
        if best is None:
            # no proposal's mask covers the pixel exactly (gaps between SAM2
            # proposals are common) -- fall back to the nearest bbox centroid,
            # within a small radius, rather than giving up the whole notice.
            closest, closest_d = None, 1e9
            for ann in anns:
                x0, y0, x1, y1 = _xywh_to_xyxy(ann["bbox"])
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                d = ((cx - u) ** 2 + (cy - v) ** 2) ** 0.5
                if d < closest_d:
                    closest, closest_d = ann, d
            if closest is not None and closest_d <= 60.0:
                best = closest
        return best

    # -- leg 1: bank one more view of the recall target into memory -- #
    def remember(self, rgb: np.ndarray, pixel: Tuple[float, float], label: str = "", step: int = -1) -> bool:
        """SAM2-segment `rgb`, take the tightest proposal containing `pixel`,
        SigLIP2-embed it, and bank it into ``self.memory`` if it's diverse
        enough versus views already stored (a near-duplicate of an existing
        view is skipped, same "diverse views bank" idea fact3r-map's own
        entity memory uses). Call this repeatedly as the object stays in
        view -- one call banks at most one view, not the whole memory.
        Returns False only when this call found nothing to embed at all
        (caller should just try again next tick); returns True whether the
        view was newly banked or skipped as a near-duplicate."""
        self._ensure_loaded()
        best = self._best_mask_at_pixel(rgb, pixel)
        if best is None:
            return False
        emb = self._embed_crop(rgb, _xywh_to_xyxy(best["bbox"]))
        if emb is None:
            return False
        if self.memory:
            nearest = max(float(np.dot(emb, v.embedding)) for v in self.memory)
            if nearest >= self.diversity_max_similarity:
                return True  # a real observation, just not a new one -- not an error
        self.memory.append(MemoryView(embedding=emb, label=label, step=step))
        if len(self.memory) > self.max_views:
            self.memory.pop(0)  # oldest view evicted first
        self.target_label = label
        return True

    # -- leg 2: same call signature as QwenVLPixelGoal.ground_candidates -- #
    def ground_candidates(self, rgb: np.ndarray, instruction: str, max_candidates: int = 1) -> List[PixelGoal]:
        self._ensure_loaded()
        if not self.memory:
            return []
        anns = self._propose(rgb)
        scored = []
        for ann in anns:
            x0, y0, x1, y1 = _xywh_to_xyxy(ann["bbox"])
            emb = self._embed_crop(rgb, (x0, y0, x1, y1))
            if emb is None:
                continue
            sims = [float(np.dot(emb, v.embedding)) for v in self.memory]
            best_i = int(np.argmax(sims))
            scored.append((sims[best_i], best_i, (x0 + x1) / 2.0, (y0 + y1) / 2.0))
        if not scored:
            self.last_match_score = None
            self.last_match_view_index = None
            return []
        scored.sort(key=lambda t: -t[0])
        self.last_match_score = scored[0][0]
        self.last_match_view_index = scored[0][1]
        out = [
            PixelGoal(u=u, v=v, confidence=sim, in_view=True)
            for sim, _view_i, u, v in scored[:max_candidates]
            if sim >= self.match_threshold
        ]
        return out

    def ground(self, rgb: np.ndarray, instruction: str = "") -> Optional[PixelGoal]:
        candidates = self.ground_candidates(rgb, instruction, max_candidates=1)
        return candidates[0] if candidates else None
