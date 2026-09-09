"""Live fact3r-style entity memory with METRIC POSITIONS.

The distinction that matters, and the one `fact3r_entity_recall.py` gets
wrong for this task: that module stores what an object *looks like* and
re-identifies it when it comes back into view. If the target is NOT visible
at recall time -- the actual case here, "go to B" issued after driving away
from B and parking at A -- appearance matching can never fire, because there
is nothing in frame to match against.

So this stores what fact3r's own entity map stores: an appearance descriptor
bank AND a metric position, accumulated over every observation. At recall
time the text query picks the entity by appearance, and what comes back is a
POSITION, which NavDP can drive to blind (`pipeline.step(external_goal=...)`,
state "GOTO") with the target never entering the camera.

Built from fact3r-map's own primitives -- SAM2 class-agnostic proposals +
SigLIP2 embeddings, the same pair `run_fact3r_live.py` uses -- reimplemented
in-process because the `fact3r.*` package needs Python 3.10+ and this
pipeline's env is 3.9. Association is nearest-embedding above a threshold
(fact3r proper uses unbalanced optimal transport over geometry + appearance;
this is the same idea at single-entity scale).

Per observation, for each SAM2 proposal:
    mask -> median depth -> centroid pixel -> pixel_depth_to_point()  (ego frame)
         -> local_to_odom(pose)                                       (map frame)
    crop -> SigLIP2 embedding
and that (position, embedding) pair is either merged into the nearest
matching entity or opens a new one.
"""

from __future__ import annotations

import math
import os
import re as _re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SAM2_ROOT = os.environ.get("SAM2_ROOT", "/home/gpu/Desktop/pineapple/packages/sam2")
if _SAM2_ROOT and _SAM2_ROOT not in sys.path:
    sys.path.insert(0, _SAM2_ROOT)


def local_to_odom(pose: Sequence[float], fwd: float, left: float) -> Tuple[float, float]:
    """Ego [fwd,left] at `pose`=(x,y,theta) -> odom xy. Inverse of
    Fact3rGoalSource.tick, same convention throughout this pipeline."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(th), math.sin(th)
    return x + (fwd * c - left * s), y + (fwd * s + left * c)


def parse_verification(answer: str) -> dict:
    """Parse the verifier's reply into decision/confidence/predicted_object.

    Tolerant on purpose. Qwen wraps its JSON in a ```json fence and writes a
    chatty `reason`, so a strict `raw_decode` on a token-limited generation
    throws on a truncated object and the candidate is silently discarded --
    which cost this run its best table (0.97 m from ground truth, thrown away
    with reason "```json" while a 4.47 m distractor was accepted). A verdict
    that was clearly stated and then cut off mid-sentence should still count,
    so fall back to reading the fields out individually.
    """
    import json as _json

    text = answer.strip()
    if "```" in text:                       # unwrap ```json ... ``` fences
        parts = text.split("```")
        for part in parts:
            cleaned = part[4:] if part.lower().startswith("json") else part
            if "{" in cleaned:
                text = cleaned
                break

    start = text.find("{")
    if start >= 0:
        try:
            payload, _ = _json.JSONDecoder().raw_decode(text[start:])
            return {
                "decision": str(payload.get("decision", "uncertain")).strip().lower(),
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
                "predicted_object": str(payload.get("predicted_object", ""))[:60],
                "reason": str(payload.get("reason", ""))[:200],
            }
        except ValueError:
            pass                            # truncated or malformed -> field scrape

    dec = _re.search(r'"decision"\s*:\s*"?(yes|no|uncertain)', text, _re.I)
    conf = _re.search(r'"confidence"\s*:\s*([01]?\.?\d+)', text)
    obj = _re.search(r'"predicted_object"\s*:\s*"([^"]*)"', text)
    reason = _re.search(r'"reason"\s*:\s*"([^"]*)', text)
    if dec is None:
        return {"decision": "uncertain", "confidence": 0.0, "predicted_object": "",
                "reason": f"unparseable reply: {answer.strip()[:120]}"}
    return {
        "decision": dec.group(1).lower(),
        "confidence": float(conf.group(1)) if conf else 0.0,
        "predicted_object": obj.group(1)[:60] if obj else "",
        "reason": ((reason.group(1)[:200] + " [truncated]") if reason else "recovered from truncated JSON"),
    }


# Lookalike pairs the verification prompt itself asks the model to
# disambiguate using scene context (see `Fact3rLiveMemory.verify`'s prompt).
# Measured failure mode this guards against: given a context-padded crop
# showing a bed headboard, Qwen's OWN reason text correctly said "the
# context shows it is part of a bed ... indicating it is a bed headboard"
# and then the `decision` field said "yes" anyway (query was "the
# cabinet") -- the reasoning was right, the structured decision field
# didn't follow it. This is a plain keyword override on that specific,
# observed contradiction: if the target query is one of these categories
# and the model's OWN reason names the lookalike (or its telltale context,
# e.g. "mattress"/"blanket" for a headboard), the decision is forced to
# "no" regardless of what the JSON said, since the model's own words
# already argued against its answer.
_LOOKALIKE_TELLTALES = {
    "cabinet": ["headboard", "footboard", "part of a bed", "bed frame", "mattress", "blanket"],
    "wardrobe": ["headboard", "footboard", "part of a bed", "bed frame", "mattress", "blanket"],
    "dresser": ["headboard", "footboard", "part of a bed", "bed frame", "mattress", "blanket"],
    "nightstand": ["dresser", "wardrobe", "cabinet"],
    "shelf": ["windowsill", "window sill"],
}

# The mirror failure: decision="no" while the reason explicitly states the
# PROMPT'S OWN yes-criterion has been met. Measured real case (query "a
# table"): reason said "...the entity is clearly a table, not a bed or
# other furniture. The object is a table, and no better-fitting lookalike
# exists in the context" -- a near-verbatim echo of the prompt's own
# "decision=yes means this entity itself is the target AND no better-
# fitting lookalike is visible" -- with decision still "no".
#
# Deliberately narrow, unlike a generic "does the reason mention the query
# word" check: matching on the model's own criterion-language, not on the
# query's head noun, avoids the obvious false-positive trap of a query word
# appearing inside a genuinely different reason ("is a table LAMP" contains
# "is a table" as a substring; "no better-fitting lookalike" does not have
# that problem since it isn't a phrase a genuine "no" reason has any reason
# to produce -- a real "no" explains what the object actually IS instead).
# Regex, not a literal substring: the measured real case actually misspelled
# it ("no better-fiting lookalike", one 't') -- Qwen-2B is not a reliable
# speller, so `fit+ing` tolerates 1+ repeats of the doubled consonant rather
# than requiring an exact match that a trivial typo would silently miss.
_YES_CRITERION_ECHO_RE = _re.compile(r"no better[- ]fit+ing lookalike")


def _enforce_reason_decision_consistency(query: str, verdict: dict, raw_text: str | None = None) -> dict:
    """`raw_text`, if given, is the model's full undecoded/untruncated reply
    -- used ONLY for the "no" echo-check below. `parse_verification()`
    truncates `reason` to 200 chars for storage, and the model's own prose
    routinely runs past that (it's asked for "one short sentence" but rarely
    complies) -- measured real miss: the actual yes-criterion echo sat in the
    LAST sentence of a ~350-char reason, past the 200-char cut, so checking
    only `verdict['reason']` silently never saw it. The 'yes' lookalike-
    telltale check below is untouched (unchanged from before raw_text
    existed) since its own measured cases had the telltale word near the
    front of the reason, within the truncation window."""
    decision = verdict.get("decision")
    reason_l = verdict.get("reason", "").lower()
    if decision == "yes":
        query_l = query.lower()
        for target, telltales in _LOOKALIKE_TELLTALES.items():
            if target not in query_l:
                continue
            for telltale in telltales:
                if telltale in reason_l:
                    verdict = dict(verdict)
                    verdict["decision"] = "no"
                    verdict["reason"] = (
                        f"[overridden: reason names '{telltale}', contradicting its own 'yes' for "
                        f"'{target}'] {verdict['reason']}")
                    return verdict
        return verdict
    if decision == "no":
        scan_text = (raw_text.lower() if raw_text is not None else reason_l)
        m = _YES_CRITERION_ECHO_RE.search(scan_text)
        if m:
            verdict = dict(verdict)
            verdict["decision"] = "yes"
            verdict["reason"] = (
                f"[overridden: reason states the yes-criterion ('{m.group(0)}') was met, "
                f"contradicting its own 'no'] {verdict['reason']}")
            return verdict
    return verdict


@dataclass
class LiveEntity:
    entity_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    crops: List[np.ndarray] = field(default_factory=list)   # TIGHT mask bbox -- SigLIP embedding + association
    # WIDER, padded crop around the same bbox, index-aligned with `crops` --
    # for VLM verification only (see `verify`'s docstring for why: a tight
    # crop of a bed headboard reads as plausibly "a dark rectangular cabinet"
    # with nothing in frame to rule that out; the surrounding room usually
    # settles it). Kept SEPARATE from `crops` rather than widening that crop
    # outright -- padding it would dilute the SigLIP embedding used for
    # cross-observation association and text-query matching with background
    # clutter, which is a different, already-working job.
    context_crops: List[np.ndarray] = field(default_factory=list)
    # Quality of each BANKED view (index-aligned with embeddings/crops/
    # context_crops, not with `qualities` below, which covers every raw
    # observation including ones that never made the bank) -- lets a later,
    # clearer sighting evict the worst currently-banked view instead of the
    # bank being frozen at whatever the first max_views_per_entity glimpses
    # happened to look like. See LiveEntityMemory.observe()'s docstring for
    # the measured case this fixes.
    view_qualities: List[float] = field(default_factory=list)
    first_step: int = -1
    last_step: int = -1
    # Per-observation reliability, index-aligned with `positions`: how many
    # valid-depth pixels the mask contributed, and the median depth at
    # observation time. Both feed `position`'s weighting -- see its
    # docstring for why an unweighted median under-uses what's available.
    qualities: List[float] = field(default_factory=list)
    depths: List[float] = field(default_factory=list)

    @property
    def observations(self) -> int:
        return len(self.positions)

    @property
    def position(self) -> Tuple[float, float]:
        """Quality-weighted mean over observations.

        A flat median treats a 6 m, half-occluded, 40-pixel glimpse the same
        as a 1.5 m, fully-visible, 4000-pixel one -- but pixel-to-metric
        unprojection error grows with depth (a 1-pixel centroid wobble is a
        few mm at 1.5 m and tens of cm at 6 m) and a small/partial mask's
        centroid is a poor proxy for the object's true centroid (it's the
        centroid of whatever sliver was visible, not the object). Measured
        case this exists for: an entity with 3 far, partial observations
        recalled 3.45 m from the true object with the plain median.

        Weight each observation `n_valid_px / depth^2` (visibility x inverse-
        square range) and take the weighted mean. Falls back to the plain
        median when quality data isn't available (e.g. entities restored
        from an older run's JSON) so this stays backward compatible."""
        arr = np.asarray(self.positions, dtype=np.float64)
        if len(self.qualities) != len(self.positions) or len(self.depths) != len(self.positions):
            return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        q = np.asarray(self.qualities, dtype=np.float64)
        d = np.clip(np.asarray(self.depths, dtype=np.float64), 0.3, None)
        w = q / (d * d)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        w = w / w.sum()
        return float(np.dot(w, arr[:, 0])), float(np.dot(w, arr[:, 1]))

    @property
    def position_spread(self) -> float:
        """Std of observed positions (m) -- how self-consistent this entity's
        geometry is; a fragmented or mis-associated entity reads high.
        Deliberately unweighted (unlike `position`): this is a sanity check
        on raw observation scatter, not the best-estimate position, so it
        should not hide a wildly-disagreeing far observation just because
        it would be down-weighted in the estimate above."""
        if len(self.positions) < 2:
            return 0.0
        arr = np.asarray(self.positions, dtype=np.float64)
        return float(np.linalg.norm(arr.std(axis=0)))

    @property
    def position_spread_weighted(self) -> float:
        """Deviation of observations from `position` (the weighted mean),
        weighted by the SAME quality/depth weights `position` uses.

        Exists because the plain `position_spread` above -- deliberately
        unweighted, as a strict sanity check -- can veto a position that is
        actually accurate. Measured case: 10 observations, one or two close
        clean ones plus several far/partial ones; `position` (weighted mean)
        landed 0.40 m from the true object, but `position_spread` (raw,
        unweighted) was 4.75 m -- comfortably over a 2.5 m trust gate, so a
        correct position never reached Qwen for verification at all.

        This does NOT replace `position_spread` as a gate on its own --
        combine as `min(position_spread, position_spread_weighted)` so an
        entity still fails if it's genuinely fragmented (two comparable-
        weight clusters far apart read as high spread here too, since this
        still measures deviation from ONE weighted mean); it only forgives
        the specific case of a few low-weight outliers dragging the raw
        number up despite the mean itself being tight and reliable."""
        arr = np.asarray(self.positions, dtype=np.float64)
        if len(arr) < 2 or len(self.qualities) != len(arr) or len(self.depths) != len(arr):
            return self.position_spread
        q = np.asarray(self.qualities, dtype=np.float64)
        d = np.clip(np.asarray(self.depths, dtype=np.float64), 0.3, None)
        w = q / (d * d)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return self.position_spread
        w = w / w.sum()
        mean = np.array(self.position, dtype=np.float64)
        diffs = arr - mean
        weighted_var = np.sum(w[:, None] * diffs ** 2, axis=0)
        return float(np.linalg.norm(np.sqrt(weighted_var)))

    def similarity(self, embedding: np.ndarray) -> float:
        return max(float(np.dot(embedding, e)) for e in self.embeddings)


class Fact3rLiveMemory:
    def __init__(
        self,
        device: str = "cuda:0",
        discovery_model: str = "facebook/sam2.1-hiera-small",
        siglip_model: str = "google/siglip2-base-patch16-224",
        associate_threshold: float = 0.90,
        associate_position_gate_m: float = 3.0,
        associate_position_gate_min_obs: int = 5,
        max_views_per_entity: int = 6,
        points_per_side: int = 16,
        points_per_batch: int = 64,
        min_mask_region_area: int = 900,
        depth_min: float = 0.4,
        depth_max: float = 8.0,
        context_pad_frac: float = 0.6,
    ):
        self.device = device
        self.discovery_model = discovery_model
        self.siglip_model = siglip_model
        self.associate_threshold = float(associate_threshold)
        self.associate_position_gate_m = float(associate_position_gate_m)
        self.associate_position_gate_min_obs = int(associate_position_gate_min_obs)
        self.max_views_per_entity = int(max_views_per_entity)
        self.points_per_side = int(points_per_side)
        self.points_per_batch = int(points_per_batch)
        self.min_mask_region_area = int(min_mask_region_area)
        self.depth_min, self.depth_max = float(depth_min), float(depth_max)
        # How much surrounding scene context to pad the VLM-verification
        # crop with, as a fraction of the tight mask's larger side (0.6 ->
        # a bbox roughly 2.2x wider/taller than the object itself). See
        # LiveEntity.context_crops' docstring for why this exists.
        self.context_pad_frac = float(context_pad_frac)

        self._sam = None
        self._siglip = None
        self._proc = None
        self.entities: List[LiveEntity] = []

    # -- models ------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._sam is not None:
            return
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from transformers import AutoModel, AutoProcessor

        self._sam = SAM2AutomaticMaskGenerator.from_pretrained(
            self.discovery_model, device=self.device,
            points_per_side=self.points_per_side, points_per_batch=self.points_per_batch,
            pred_iou_thresh=0.88, stability_score_thresh=0.94,
            min_mask_region_area=self.min_mask_region_area,
        )
        self._siglip = AutoModel.from_pretrained(self.siglip_model).to(self.device).eval()
        self._proc = AutoProcessor.from_pretrained(self.siglip_model)

    def _propose(self, rgb: np.ndarray) -> list:
        import torch

        image = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if str(self.device).startswith("cuda"):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                return self._sam.generate(image)
        with torch.inference_mode():
            return self._sam.generate(image)

    def _embed_images(self, crops) -> Optional[np.ndarray]:
        import torch
        from PIL import Image

        if not crops:
            return None
        pil = [Image.fromarray(np.ascontiguousarray(c)) for c in crops]
        inputs = self._proc(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self._siglip.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        self._ensure_loaded()
        inputs = self._proc(text=[text], return_tensors="pt", padding="max_length",
                            truncation=True).to(self.device)
        with torch.no_grad():
            feats = self._siglip.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()

    # -- mapping ------------------------------------------------------ #
    @staticmethod
    def select_association_target(entities: List["LiveEntity"], emb: np.ndarray,
                                   ox: float, oy: float, threshold: float,
                                   position_gate_m: float, position_gate_min_obs: int
                                   ) -> Optional["LiveEntity"]:
        """Which existing entity (if any) a new proposal (embedding + world
        position) belongs to -- appearance-best, gated on position once a
        candidate entity has enough observations for its position to have
        settled (see observe()'s call site for the full rationale: pure
        appearance-only association let one real entity absorb 495
        observations across a 3.57m spread in one live run). Pulled out of
        observe() as a pure function so it's testable without SAM2/SigLIP2/
        torch (same reasoning as this module's other dataclass-only tests)."""
        best, best_sim = None, -1.0
        for ent in entities:
            sim = ent.similarity(emb)
            if sim <= best_sim:
                continue
            if ent.observations >= position_gate_min_obs:
                ex, ey = ent.position
                if math.hypot(ox - ex, oy - ey) > position_gate_m:
                    continue
            best, best_sim = ent, sim
        return best if (best is not None and best_sim >= threshold) else None

    def observe(self, rgb: np.ndarray, depth: np.ndarray, pose, intrinsics, step: int = -1) -> int:
        """One mapping tick: SAM2 -> per-proposal (metric position, embedding)
        -> associate into entities. Returns how many proposals were absorbed."""
        from .goal_utils import pixel_depth_to_point

        self._ensure_loaded()
        fx, fy, cx, cy = intrinsics
        anns = self._propose(rgb)

        crops, context_crops, metas = [], [], []
        H, W = depth.shape[:2]
        for ann in anns:
            mask = np.asarray(ann["segmentation"], dtype=bool)
            values = depth[mask]
            values = values[np.isfinite(values) & (values > self.depth_min) & (values < self.depth_max)]
            if values.size < 30:
                continue                       # no usable geometry -> no position -> skip
            d = float(np.median(values))
            rows, cols = np.nonzero(mask)
            u, v = float(cols.mean()), float(rows.mean())
            local = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
            ox, oy = local_to_odom(pose, float(local[0]), float(local[1]))

            x0, y0, w, h = ann["bbox"]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(x0 + w)), min(H, int(y0 + h))
            if x1 <= x0 or y1 <= y0:
                continue
            crops.append(rgb[y0:y1, x0:x1])
            # Wider, padded crop of the SAME region for VLM verification only
            # (see LiveEntity.context_crops' docstring) -- surrounding scene
            # is what tells a bed headboard from a cabinet; the tight mask
            # crop above deliberately excludes it so SigLIP's embedding
            # stays about the object, not the room.
            pad = int(self.context_pad_frac * max(x1 - x0, y1 - y0))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            context_crops.append(rgb[cy0:cy1, cx0:cx1])
            # quality = how many valid-depth pixels the mask contributed --
            # a big, unoccluded, close-up view scores high; a sliver seen at
            # the edge of the frame or half-behind furniture scores low.
            # Paired with `d` (this observation's own depth), LiveEntity.position
            # uses both to down-weight far/partial glimpses instead of
            # trusting every observation equally.
            metas.append((ox, oy, float(values.size), d))

        embeddings = self._embed_images(crops)
        if embeddings is None:
            return 0

        for emb, (ox, oy, quality, d), crop, ccrop in zip(embeddings, metas, crops, context_crops):
            # Appearance alone can't tell two genuinely different real
            # objects apart if they merely look similar (several tables/
            # counters/shelves in one house often do) -- measured live
            # (2026-09-05): one entity accumulated 495 observations and a
            # 3.57m position spread through pure appearance-only association,
            # no merge_nearby_duplicates() involvement at all (that post-hoc
            # pass already gates on position; this live path didn't).
            # select_association_target additionally gates on position once
            # a candidate has enough observations for it to have settled --
            # see its own docstring for why not from the first observation.
            best = self.select_association_target(
                self.entities, emb, ox, oy, self.associate_threshold,
                self.associate_position_gate_m, self.associate_position_gate_min_obs)
            if best is not None:
                best.positions.append((ox, oy))
                best.qualities.append(quality)
                best.depths.append(d)
                if len(best.embeddings) < self.max_views_per_entity:
                    best.embeddings.append(emb)
                    best.crops.append(np.ascontiguousarray(crop))
                    best.context_crops.append(np.ascontiguousarray(ccrop))
                    best.view_qualities.append(quality)
                elif best.view_qualities and quality > min(best.view_qualities):
                    # Bank is full of BETTER-than-nothing views, not
                    # necessarily good ones -- a well-observed entity (tens of
                    # observations) can still be stuck showing Qwen only its
                    # first few, worst-angle glimpses otherwise. Evict the
                    # single worst-quality banked view for this better one.
                    worst = int(np.argmin(best.view_qualities))
                    best.embeddings[worst] = emb
                    best.crops[worst] = np.ascontiguousarray(crop)
                    best.context_crops[worst] = np.ascontiguousarray(ccrop)
                    best.view_qualities[worst] = quality
                best.last_step = step
            else:
                ent = LiveEntity(entity_id=f"live-{len(self.entities):04d}",
                                 embeddings=[emb], positions=[(ox, oy)],
                                 crops=[np.ascontiguousarray(crop)],
                                 context_crops=[np.ascontiguousarray(ccrop)],
                                 qualities=[quality], depths=[d],
                                 view_qualities=[quality],
                                 first_step=step, last_step=step)
                self.entities.append(ent)
        return len(metas)

    def merge_nearby_duplicates(self, position_radius: float = 0.6,
                                appearance_threshold: float = 0.75,
                                max_cluster_diameter: float = 1.2) -> int:
        """Post-hoc fragment merge: the same physical object, seen from
        different angles/lighting/distances across a longer walk, routinely
        fails `observe()`'s live per-tick `associate_threshold` (0.90) and
        splits into several entities. Measured case: one hallway console
        with basket-drawers fragmented into 4 separate entities, each
        independently offered to Qwen as a 'cabinet' candidate, none of
        which was the true one -- the shortlist was crowded with duplicates
        of the WRONG object rather than reaching the real one.

        The live threshold has to stay conservative: early on, an object has
        few observations and a noisy position, so appearance alone must
        carry high confidence to associate safely. This method runs once,
        after mapping is done, when every entity's position has had time to
        settle, and can safely use a MUCH looser appearance bar because it
        additionally requires the two entities' positions to already agree
        (within `position_radius`) -- the two-signal test (geometry AND
        appearance) fact3r's own UOT association uses, just applied as a
        one-shot pass here instead of live per-tick.

        max_cluster_diameter guards against TRANSITIVE over-merging: plain
        union-find only checks each PAIR against position_radius, so a chain
        (i near j, j near k, k near l, ...) can span an arbitrarily large
        total distance even though every single hop was individually close
        -- measured live (2026-09-05, MP3D scene, query 'a table'): one
        merged entity ended up with 930 observations and a 3.40m position
        spread (vs. <90 observations and <2m spread for every other entity
        that run), then got accepted by the weighted-spread gate anyway
        (weighted spread 1.12m, since a large enough observation count can
        drag the WEIGHTED mean tight even when the raw scatter reflects
        several genuinely different real objects folded together) and was
        3.24m off the true target once verified -- almost certainly several
        distinct flat/reflective surfaces (a real house has several) chained
        together rather than one physical table. Candidate unions are now
        tried in ascending distance order, and a union is skipped (not
        merged) if it would push the resulting group's own max INTERNAL
        pairwise distance past max_cluster_diameter -- so a long chain of
        marginally-close fragments can no longer silently span meters, while
        genuinely tight duplicate clusters (the common, intended case) merge
        exactly as before.

        Returns how many entities were merged away (4 fragments -> 1 entity
        returns 3)."""
        n = len(self.entities)
        parent = list(range(n))
        positions = [np.array(e.position, dtype=np.float64) for e in self.entities]
        members: dict[int, list[int]] = {i: [i] for i in range(n)}

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def cluster_diameter(idxs: list[int]) -> float:
            maxd = 0.0
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    d = float(np.linalg.norm(positions[idxs[a]] - positions[idxs[b]]))
                    if d > maxd:
                        maxd = d
            return maxd

        candidates: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d > position_radius:
                    continue
                ei, ej = self.entities[i], self.entities[j]
                sim = max((float(np.dot(a, b)) for a in ei.embeddings for b in ej.embeddings),
                         default=-1.0)
                if sim >= appearance_threshold:
                    candidates.append((d, i, j))
        candidates.sort(key=lambda c: c[0])   # closest pairs merge first

        for _, i, j in candidates:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            combined = members[ri] + members[rj]
            if cluster_diameter(combined) > max_cluster_diameter:
                continue   # would over-extend the cluster -- leave these separate
            parent[rj] = ri
            members[ri] = combined
            del members[rj]

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged: List[LiveEntity] = []
        n_merged_away = 0
        for members in groups.values():
            base = self.entities[members[0]]
            if len(members) > 1:
                n_merged_away += len(members) - 1
                for idx in members[1:]:
                    other = self.entities[idx]
                    base.positions.extend(other.positions)
                    base.qualities.extend(other.qualities)
                    base.depths.extend(other.depths)
                    base.embeddings.extend(other.embeddings)
                    base.crops.extend(other.crops)
                    base.context_crops.extend(other.context_crops)
                    base.view_qualities.extend(other.view_qualities)
                    base.first_step = min(base.first_step, other.first_step)
                    base.last_step = max(base.last_step, other.last_step)
                # keep the view budget sane -- verify()/similarity() only
                # ever look at the first few anyway (see their own slicing).
                # Keep the BEST-quality views of the merged pool, not just
                # the first `cap` in concatenation order (same reasoning as
                # observe()'s per-tick eviction above).
                cap = self.max_views_per_entity
                if len(base.view_qualities) == len(base.embeddings):
                    order = sorted(range(len(base.view_qualities)),
                                   key=lambda k: -base.view_qualities[k])[:cap]
                    base.embeddings = [base.embeddings[k] for k in order]
                    base.crops = [base.crops[k] for k in order]
                    base.context_crops = [base.context_crops[k] for k in order]
                    base.view_qualities = [base.view_qualities[k] for k in order]
                else:
                    base.embeddings, base.crops, base.context_crops = (
                        base.embeddings[:cap], base.crops[:cap], base.context_crops[:cap])
            merged.append(base)

        self.entities = merged
        return n_merged_away

    def reidentify(self, rgb: np.ndarray, depth: np.ndarray, pose, intrinsics,
                   target: "LiveEntity", threshold: Optional[float] = None
                   ) -> Optional[Tuple[Tuple[float, float], float]]:
        """Is `target` (a specific, already-recalled entity) back in view?

        Different question from `verify` -- and a stronger one for this
        purpose. `verify` asks a VLM "does this crop look like a cabinet?",
        an independent judgment that can be fooled by any lookalike (a bed
        headboard also "looks like" a cabinet from a tight, context-free
        crop -- measured, see LiveEntity.context_crops' docstring). This
        instead asks "does anything in THIS frame match `target`'s OWN
        stored appearance?" -- SigLIP re-identification against the actual
        recalled object's embedding bank, not a text label. It can't be
        fooled by a differently-shaped lookalike that merely fits the query
        text; it can only fire on something that looks like this specific
        previously-observed object.

        Read-only (does not mutate `self.entities` or associate anything) --
        this is a targeted check for one entity, not a mapping tick.

        Returns ((x, y) in odom frame, similarity) for the best-matching
        proposal above `threshold` (default: `self.associate_threshold`,
        the same bar used to associate two observations as one entity), or
        None if nothing in frame matches closely enough."""
        from .goal_utils import pixel_depth_to_point

        if not target.embeddings:
            return None
        self._ensure_loaded()
        threshold = self.associate_threshold if threshold is None else threshold
        fx, fy, cx, cy = intrinsics
        anns = self._propose(rgb)

        crops, metas = [], []
        H, W = depth.shape[:2]
        for ann in anns:
            mask = np.asarray(ann["segmentation"], dtype=bool)
            values = depth[mask]
            values = values[np.isfinite(values) & (values > self.depth_min) & (values < self.depth_max)]
            if values.size < 30:
                continue
            d = float(np.median(values))
            rows, cols = np.nonzero(mask)
            u, v = float(cols.mean()), float(rows.mean())
            local = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
            ox, oy = local_to_odom(pose, float(local[0]), float(local[1]))
            x0, y0, w, h = ann["bbox"]
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(W, int(x0 + w)), min(H, int(y0 + h))
            if x1 <= x0 or y1 <= y0:
                continue
            crops.append(rgb[y0:y1, x0:x1])
            metas.append((ox, oy))

        embeddings = self._embed_images(crops)
        if embeddings is None:
            return None

        best_xy, best_sim = None, threshold
        for emb, (ox, oy) in zip(embeddings, metas):
            sim = target.similarity(emb)
            if sim > best_sim:
                best_xy, best_sim = (ox, oy), sim
        return (best_xy, best_sim) if best_xy is not None else None

    # -- recall ------------------------------------------------------- #
    def query(self, text: str, min_observations: int = 2, min_score: float = 0.0):
        """Text -> (entity, position, score) for the best-matching entity that
        has enough observations to trust its geometry. Nothing needs to be in
        view: the answer is a stored position."""
        self._ensure_loaded()
        prompts = [text, f"a photo of {text}", f"{text} in an indoor scene"]
        text_emb = np.mean([self.embed_text(p) for p in prompts], axis=0)
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-12)

        ranked = []
        for ent in self.entities:
            if ent.observations < min_observations:
                continue
            score = ent.similarity(text_emb)
            ranked.append((score, ent))
        if not ranked:
            return None
        ranked.sort(key=lambda t: -t[0])
        score, ent = ranked[0]
        if score < min_score:
            return None
        return ent, ent.position, float(score)

    # -- stage 2: VLM verification ------------------------------------ #
    def _verify_call(self, text: str, images: list, qwen, max_new_tokens: int = 256):
        """One verification call over exactly the given image(s) (raw numpy
        crops, not yet PIL-converted). Returns (verdict_dict, raw_decoded_text)
        -- the raw text is needed by the caller for
        `_enforce_reason_decision_consistency`'s echo-check, which scans past
        `parse_verification`'s 200-char truncation. Called once per view by
        `verify()`'s ensemble below, and previously (before that existed)
        once over the whole view bank at once -- see `verify()`'s own
        docstring for why per-view calls replaced that.

        SigLIP alone is not enough and is not what fact3r does: its raw
        cosine scores sit around 0.1 and separate candidates by hundredths,
        so a two-observation fragment can outrank a well-observed real
        object. fact3r's retrieval is two-stage -- SigLIP shortlists, a VLM
        adjudicates -- and this is that second stage, using fact3r's own
        deliberately strict prompt (see
        fact3r/semantics/vlm_verification.py::build_verification_prompt).

        `qwen` is any object exposing `_ensure_loaded`, `_model` and
        `_processor` -- i.e. the pipeline's already-loaded QwenVLPixelGoal,
        so verification costs no extra VRAM.

        Uses `entity.context_crops` (padded with surrounding scene, not just
        the tight SAM2 mask) when available, falling back to `entity.crops`
        for entities from before that field existed. Measured why this
        matters: on the tight crop alone, a bed headboard (an ornate
        wood-panel arch) was called "a cabinet" at confidence 0.95 -- nothing
        in frame contradicted it. The same object with the room around it
        (a bed frame underneath, a nightstand beside it) is not something a
        model would call a cabinet; the surrounding scene is often the ONLY
        thing that disambiguates two furniture types with a similar frontal
        silhouette (headboard/cabinet, nightstand/dresser, etc.).

        The headboard-override clause is deliberately narrowed to require
        the object be ATTACHED to the bed frame itself, not merely visible
        in the same room as one. Measured failure the narrower wording
        fixes: a real table 0.55m from the true recall target was rejected
        at confidence 0.99 with reason "the surrounding context shows a room
        with a bed and a doorway... it is not a table, but a bed headboard"
        -- a real table that happened to share a room with a bed, not one
        attached to it, confidently misclassified by the broader "bed
        visible -> headboard" wording this replaced."""
        import json as _json

        import torch
        from PIL import Image

        if not images:
            return {"decision": "uncertain", "confidence": 0.0, "predicted_object": "",
                    "reason": "no stored crops"}, ""
        qwen._ensure_loaded()
        images = [Image.fromarray(np.asarray(c, dtype=np.uint8)).convert("RGB").resize((224, 224))
                  for c in images]
        prompt = (
            "You are verifying an object-memory retrieval for a mobile robot. "
            "All supplied images are different observations of ONE persistent 3D "
            "entity, cropped from the robot's camera WITH some surrounding scene "
            "left in for context -- the entity is the main/central object in each "
            "crop, not necessarily the only thing visible. Judge the entity from "
            "those crops. Reject thin borders, image-edge strips, shadows, walls, "
            "floor patches, and incomplete fragments.\n\n"
            f"Target query: {_json.dumps(text.strip())}\n\n"
            "Before deciding, use the surrounding context to rule out lookalikes: "
            "furniture types are often confused when only the front panel is "
            "visible (e.g. a bed headboard vs a cabinet/wardrobe door, a "
            "nightstand vs a small dresser, a windowsill vs a shelf). Only call "
            "this object a headboard/footboard if it is ITSELF physically "
            "attached to or built into the bed frame (touching, or directly "
            "behind/at the head or foot of, the mattress) -- NOT merely because "
            "a bed happens to be visible elsewhere in the same room. A table, "
            "nightstand, dresser, or cabinet standing separately from the bed "
            "is judged on its own appearance even when a bed shares the crop. "
            "Use evidence across the views. Small size alone is not a "
            "rejection, but answer uncertain if the crop is too incomplete or "
            "ambiguous, and answer no (not yes) if a plausible lookalike fits the "
            "context at least as well as the target query does. decision=yes "
            "means this entity itself is the target AND no better-fitting "
            "lookalike is visible. If this object is a close functional subtype "
            "or synonym of the target category rather than a genuinely different "
            "kind of furniture (e.g. a nightstand or side table when asked for "
            "'a table', a loveseat when asked for 'a sofa'), that still counts as "
            "decision=yes -- name the specific subtype in predicted_object rather "
            "than rejecting it. This is different from a lookalike: a lookalike "
            "is a DIFFERENT kind of object that merely resembles the target from "
            "some angle (e.g. a headboard resembling a cabinet); a subtype is "
            "still a real, usable instance of what was asked for. decision=no "
            "means it is a genuinely different, unrelated object; name it in "
            "predicted_object. Return ONLY one JSON object "
            "with exactly these fields:\n"
            '{"decision":"yes|no|uncertain","confidence":0.0,'
            '"predicted_object":"short noun phrase","reason":"one short sentence"}'
        )
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        chat = qwen._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen._processor(text=[chat], images=images, return_tensors="pt").to(qwen._model.device)
        with torch.no_grad():
            # no_repeat_ngram_size scoped to THIS call only (not the
            # pipeline's other Qwen .generate() sites, which have no observed
            # problem): measured a real degenerate loop here -- the model
            # restated "not a table, but a bed headboard" ~6x within 256
            # tokens, still emitting confidence 0.99 on the garbled result.
            # Tried repetition_penalty=1.3 first -- WORSE: it broke JSON
            # output entirely on the same real crops (malformed keys, a
            # rambling incoherent wall of text, unparseable). A soft per-
            # token penalty punishes legitimate repeated structure (JSON
            # braces/quotes, common words) along with the actual loop.
            # no_repeat_ngram_size=4 is a hard, surgical block on repeated
            # 4-grams only -- it can't touch single-token JSON syntax and
            # only kicks in on genuine phrase-level repetition.
            out = qwen._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       no_repeat_ngram_size=4)
        answer = qwen._processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = _enforce_reason_decision_consistency(text, parse_verification(answer), raw_text=answer)
        return verdict, answer

    def verify(self, text: str, entity: "LiveEntity", qwen, max_views: int = 6,
              max_new_tokens: int = 256) -> dict:
        """One holistic call over the whole view bank at once.

        A per-view INDEPENDENT-voting ensemble was tried and directly
        rejected here (2026-09-06), not just considered: hypothesis was that
        one occluded/ambiguous view in a batch can veto an otherwise well-
        evidenced entity (a real table 0.72m from the true recall target was
        called "no" once with a self-contradictory-favorable reason).
        Tested head-to-head on that EXACT real entity's saved crops: the
        holistic call (all 4 views together) correctly answered "yes" --
        but decomposing into 4 independent single-image calls gave 4/4
        "no" votes (one even misread it as "a wooden chair"). The theory
        had it backwards: the views corroborating EACH OTHER is what lets
        the model identify the object at all; each crop alone is too weak,
        so per-view voting throws away the exact cross-view evidence this
        task depends on. Confirmed deterministic (reran the holistic call
        twice, identical "yes" both times) so this wasn't a fluke. Keeping
        the single holistic call; do not reintroduce per-view voting
        without new evidence it helps on some case where THIS approach
        fails, since it demonstrably regresses on the case it was meant to
        fix."""
        views = (entity.context_crops or entity.crops)[:max_views]
        if not views:
            return {"decision": "uncertain", "confidence": 0.0, "predicted_object": "",
                    "reason": "no stored crops"}
        verdict, _ = self._verify_call(text, views, qwen, max_new_tokens)
        return verdict

    def query_verified(self, text: str, qwen, min_observations: int = 3, top: int = 6,
                       min_confidence: float = 0.5, max_position_spread: float = 2.0):
        """fact3r's real two-stage retrieval: SigLIP shortlist -> Qwen3-VL
        verification -> the stored POSITION of the best accepted entity.

        Returns (entity, position, record, shortlist_records). Entities whose
        observed positions disagree by more than `max_position_spread` are
        dropped before the VLM sees them: a fragmented or mis-associated
        entity has no trustworthy position to drive to, so verifying its
        appearance would be beside the point.

        Gate uses min(position_spread, position_spread_weighted) -- passes
        if EITHER reads trustworthy. Measured case for why the plain
        (unweighted) spread alone isn't enough: an entity whose weighted
        `position` landed 0.40 m from the true object had a raw spread of
        4.75 m (a couple of low-quality outlier observations dragging it up)
        and was dropped before Qwen ever got a look. `position_spread_weighted`
        still catches genuine fragmentation (two comparable-weight clusters
        read as high spread there too), so this isn't just loosening the
        gate -- see that property's docstring."""
        shortlist = self.ranked(text, min_observations=min_observations, top=top)
        records, accepted = [], []
        for score, ent in shortlist:
            trust_spread = min(ent.position_spread, ent.position_spread_weighted)
            rec = {"entity_id": ent.entity_id, "siglip_score": float(score),
                   "observations": ent.observations,
                   "position_xy": list(ent.position),
                   "position_spread_m": ent.position_spread,
                   "position_spread_weighted_m": ent.position_spread_weighted}
            if trust_spread > max_position_spread:
                rec["vlm"] = {"decision": "skipped", "confidence": 0.0,
                              "reason": f"position spread {ent.position_spread:.2f} m "
                                        f"(weighted {ent.position_spread_weighted:.2f} m) exceeds "
                                        f"{max_position_spread:.2f} m -- no trustworthy position"}
            else:
                rec["vlm"] = self.verify(text, ent, qwen)
                if rec["vlm"]["decision"] == "yes" and rec["vlm"]["confidence"] >= min_confidence:
                    accepted.append((rec["vlm"]["confidence"], float(score), ent, rec))
            records.append(rec)
        if not accepted:
            return None, None, None, records
        accepted.sort(key=lambda t: (-t[0], -t[1]))
        _, _, ent, rec = accepted[0]
        return ent, ent.position, rec, records

    def ranked(self, text: str, min_observations: int = 2, top: int = 5):
        """Same scoring, but returns the whole shortlist for reporting."""
        self._ensure_loaded()
        prompts = [text, f"a photo of {text}", f"{text} in an indoor scene"]
        text_emb = np.mean([self.embed_text(p) for p in prompts], axis=0)
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-12)
        out = [(ent.similarity(text_emb), ent) for ent in self.entities
               if ent.observations >= min_observations]
        out.sort(key=lambda t: -t[0])
        return out[:top]
