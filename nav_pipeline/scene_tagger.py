"""Periodic open-vocabulary object inventory of the current frame.

Unlike clip_verifier.ClipVerifier (which scores one known crop against one
known target phrase), this answers "what's in the frame right now" with no
target in mind: it runs Grounding DINO against a broad vocabulary list and
counts detections per label, e.g. {"bottle": 1, "chair": 2, "monitor": 1}.
Meant to be sampled at a low, fixed rate (see PipelineConfig.scene_tag_period_s)
and logged for later route-finding -- not part of the per-tick control path.

Reuses the GroundingDinoDetector already loaded for goal detection; tagging
is just a different text prompt against the same model.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .dino_detector import GroundingDinoDetector

DEFAULT_VOCAB: List[str] = [
    "bottle", "cup", "mug", "chair", "table", "desk", "monitor", "laptop",
    "keyboard", "mouse", "door", "doorway", "window", "backpack", "bag",
    "box", "plant", "potted plant", "person", "shoe", "trash can", "sofa",
    "couch", "shelf", "book", "lamp", "wall outlet", "stairs", "cabinet",
]


def _clean_label(label: str) -> str:
    label = label.strip().lower().rstrip(".")
    for article in ("a ", "an ", "the "):
        if label.startswith(article):
            label = label[len(article):]
    return label.strip()


@dataclass
class SceneTagEntry:
    t: float
    objects: Dict[str, int]


class SceneTagger:
    def __init__(self, detector: GroundingDinoDetector, vocab: List[str] = None):
        self.detector = detector
        self.vocab = list(vocab) if vocab else list(DEFAULT_VOCAB)
        self._vocab_set = set(self.vocab)
        self._prompt = " ".join(f"{v}." for v in self.vocab)

    def _match_vocab(self, raw: str) -> str:
        if raw in self._vocab_set:
            return raw
        # For a multi-phrase prompt, Grounding DINO's returned span can
        # cross a period boundary and merge two adjacent vocab phrases into
        # one string (e.g. "door doorway" for a box that matched both
        # "door." and "doorway."). Bucket under the longest vocab word it
        # contains so counts stay attributable to a single known label
        # instead of fragmenting into merged-text buckets.
        candidates = [v for v in self.vocab if v in raw]
        return max(candidates, key=len) if candidates else raw

    def tag(self, rgb: np.ndarray) -> Dict[str, int]:
        """Return counts per vocabulary label found in this frame."""
        dets = self.detector.detect(rgb, self._prompt)
        counts: Dict[str, int] = {}
        for d in dets:
            raw = _clean_label(d.label)
            if not raw:
                continue
            label = self._match_vocab(raw)
            counts[label] = counts.get(label, 0) + 1
        return counts
