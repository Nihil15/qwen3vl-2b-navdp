"""Grounding DINO open-vocabulary detector wrapper.

Loads IDEA-Research/grounding-dino-base from the local HF cache
(HF_HOME=/mnt/bigdisk/hf_cache) and returns the best box for a free-text
target phrase, e.g. "a red chair." -> (bbox, score).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class Detection:
    box: np.ndarray  # [x0, y0, x1, y1] pixels in the input image
    score: float
    label: str

    @property
    def center(self):
        return np.array([(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0])


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boxes. (Containment-based dedup was tried and reverted --
    it also deletes genuinely distinct objects that happen to occlude one
    another in the 2D projection, e.g. a box sitting on a chair, or a chair
    in front of a door: high containment, but two real separate objects.
    Plain IoU only fires on near-identical boxes, which is what an actual
    duplicate/mislabeled detection of ONE object looks like.)"""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _class_agnostic_nms(dets: List["Detection"], iou_threshold: float) -> List["Detection"]:
    """Greedily drop lower-score boxes that heavily overlap a kept one.

    HF's Grounding DINO post-processing does no suppression at all: with a
    long multi-phrase prompt (scene tagging's ~28-word vocab) the model
    routinely emits two-plus boxes -- same or different labels -- on the
    SAME physical object (observed live: one chair scored both "chair" 0.70
    and "table" 0.53 at 85% box overlap). Class-agnostic (label-blind) NMS
    is deliberate: a high-IoU pair is overwhelmingly one mislabeled/duplicate
    instance, not two genuinely different co-located objects -- those (e.g.
    a box sitting on a desk) have low box overlap even when adjacent. This
    directly fixes hallucinated/inflated per-class counts in scene_tagger.
    """
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List["Detection"] = []
    for d in dets:
        if all(_overlap(d.box, k.box) < iou_threshold for k in kept):
            kept.append(d)
    return kept


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        # 0.5 let real duplicates through: a live rover frame with one chair
        # at the frame edge produced two boxes on that SAME chair at IoU
        # 0.476 (score 0.376 vs 0.352, inflating the chair count to 5 for 4
        # real chairs) -- just under 0.5. Meanwhile genuinely distinct
        # co-located objects in the same frame (a chair pushed under the
        # table, a cardboard box beside a chair) sat at IoU 0.34 and 0.29.
        # 0.4 sits in the gap: catches the measured duplicate, keeps both
        # measured distinct pairs.
        nms_iou_threshold: float = 0.4,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def _normalize_prompt(text: str) -> str:
        # Grounding DINO expects lowercase phrases terminated by periods.
        text = text.strip().lower()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def detect(self, image: np.ndarray, text: str) -> List[Detection]:
        """image: HxWx3 uint8 RGB. Returns detections sorted by score desc."""
        pil = Image.fromarray(image)
        prompt = self._normalize_prompt(text)
        inputs = self.processor(images=pil, text=prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]
        dets = [
            Detection(box=np.array(b, dtype=np.float32), score=float(s), label=str(lbl))
            for b, s, lbl in zip(
                results["boxes"].cpu().numpy(),
                results["scores"].cpu().numpy(),
                results.get("text_labels", results.get("labels", [])),
            )
        ]
        dets = _class_agnostic_nms(dets, self.nms_iou_threshold)
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

    def detect_best(self, image: np.ndarray, text: str) -> Optional[Detection]:
        dets = self.detect(image, text)
        return dets[0] if dets else None
