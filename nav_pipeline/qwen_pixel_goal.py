"""Qwen2.5-VL instruction-grounded pixel goal for NavDP trajectory following.

Free-text navigation instructions ("walk through the doorway and stop near
the desk") aren't Grounding DINO's job -- DINO finds a named OBJECT, not a
described maneuver. This lets a frozen Qwen2.5-VL-7B-Instruct ground the
instruction directly to a 2D pixel (u, v) in the current frame -- the next
waypoint to head toward. pipeline.py turns that pixel into a 3D goal point
via depth (goal_utils.pixel_depth_to_point) and hands it to NavDP exactly
like a DINO detection would -- NavDP samples trajectories against it and the
pipeline's existing trajectory-selection/obstacle-guard machinery follows
it, completely unchanged.

Sibling of MARS/mars-habitatsim/navdp/navdp/extensions/system2_pixel_goal.py
(the Habitat-sim version, which renders the pixel into a goal-MASK channel
for a different NavDP variant's own point-conditioning input). Kept as a
separate module rather than shared code: different conda env/subproject,
and this one feeds the real-rover pipeline's depth-derived 3D goal path
instead of a mask channel.

Distinct from qwen_search_guide.py: that module only ever steers the
SEARCH state while chasing a DINO *text target* (a bearing, not a point,
and only a fallback when DINO can't see the target). This module is the
PRIMARY goal source for a free-text *instruction* -- there is no DINO
target in this mode at all.

Between Qwen calls (throttled -- see QwenInstructionGate), pipeline.py does
NOT hold a stale pixel: it lets GoalBelief propagate the last 3D goal by
ego-motion, the same belief-coasting machinery a lost DINO detection already
falls back to. That is more physically correct than re-using a fixed pixel
column across ticks where the rover has since turned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

DEFAULT_PROMPT = (
    "You are guiding a mobile ground robot. Instruction: \"{instruction}\". "
    "Look at this image and point to the single best next waypoint pixel to "
    "move toward in order to follow the instruction. Reply with only one "
    "pixel coordinate as (x, y)."
)

# Distractor-robustness fix: asking for ONE point forces a guess the instant
# there's real ambiguity (e.g. two similar doors) -- there's no way to tell
# "confidently the only door" apart from "arbitrarily picked one of two"
# from a single (x, y) alone. Asking for several ranked, confidence-scored
# candidates instead lets pipeline.py score each against BOTH goal
# continuity (does this look like what we're already heading toward) and
# obstacle cost (see PipelineConfig.qwen_max_candidates and _step_inner's
# instruction branch) -- closer to the "semantic score minus collision
# cost" combination arxiv 2605.19420 ("Beyond Waypoints: Dual-Heatmap
# Grounding") uses a purpose-trained heatmap network for, adapted here to a
# frozen general VLM that can only be prompted for a handful of discrete
# points, not a dense field.
MULTI_CANDIDATE_PROMPT = (
    "You are guiding a mobile ground robot. Instruction: \"{instruction}\". "
    "Look at this image and identify up to {max_candidates} candidate "
    "waypoint pixels that could satisfy the instruction. If there is more "
    "than one plausible match (e.g. two similar doors or openings), list "
    "each one separately instead of picking just one. List your best "
    "candidate first. Reply with ONE candidate per line, each formatted "
    "exactly as: x, y, confidence -- where confidence is your certainty "
    "from 0 (unsure) to 1 (certain) that this candidate is correct."
)


def parse_pixel_coordinate(text: str, image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Pull the first (x, y) pixel pair out of a VLM answer.

    Handles bare ``(x, y)``, ``x, y``, JSON-ish ``[x, y]`` and Qwen box tags.
    If the numbers look normalized (<=1) they are scaled by the image size; if
    they look like Qwen's 0-1000 grounding scale they are rescaled too.
    """
    w, h = image_size
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:  # normalized
        return x * (w - 1), y * (h - 1)
    if x > w or y > h:  # likely Qwen 0-1000 grounding scale
        return x / 1000.0 * (w - 1), y / 1000.0 * (h - 1)
    return x, y


def parse_pixel_candidates(text: str, image_size: Tuple[int, int],
                           max_candidates: int = 3) -> List["PixelGoal"]:
    """Pull up to max_candidates (x, y[, confidence]) groups out of a VLM
    answer, one per line (see MULTI_CANDIDATE_PROMPT). Confidence defaults
    to a rank-based decay (1.0, 0.7, 0.5, ...) when a line omits it or the
    model didn't follow the format -- still usable, just less informative
    than a real per-candidate confidence. Same normalization rules as
    parse_pixel_coordinate, applied per line independently.
    """
    w, h = image_size
    default_confs = [1.0, 0.7, 0.5, 0.35, 0.25]
    results: List[PixelGoal] = []
    # A numbered list ("1. 300, 200, 0.95") would otherwise have its
    # leading "1." misread as x itself, shifting every value over by one
    # (caught by testing before this shipped) -- strip a SHORT (1-2 digit,
    # so a real 3-digit pixel coordinate can never match) leading list
    # marker before grabbing numbers, rather than requiring strict
    # comma-separation (which broke on label-prefixed replies like
    # "x=300, y=200, confidence=0.8" -- also caught by testing).
    leading_marker = re.compile(r"^\s*[\(\[]?\d{1,2}[.\):\]]\s+")
    lines = [ln for ln in re.split(r"[\n;]", text) if ln.strip()] or [text]
    for line in lines:
        line = leading_marker.sub("", line, count=1)
        nums = re.findall(r"-?\d+\.?\d*", line)
        if len(nums) < 2:
            continue
        x, y = float(nums[0]), float(nums[1])
        raw_conf = float(nums[2]) if len(nums) >= 3 else None
        conf = raw_conf if raw_conf is not None and 0.0 <= raw_conf <= 1.0 else None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x, y = x * (w - 1), y * (h - 1)
        elif x > w or y > h:
            x, y = x / 1000.0 * (w - 1), y / 1000.0 * (h - 1)
        if conf is None:
            conf = default_confs[len(results)] if len(results) < len(default_confs) else 0.2
        results.append(PixelGoal(float(x), float(y), float(conf), in_view=(0 <= x < w and 0 <= y < h)))
        if len(results) >= max_candidates:
            break
    return results


@dataclass
class PixelGoal:
    u: float          # column (x) in image pixels
    v: float          # row (y) in image pixels
    confidence: float
    in_view: bool = True


class QwenVLPixelGoal:
    """Frozen Qwen2.5-VL-7B-Instruct grounder: (rgb, instruction) -> PixelGoal.

    Inference only -- no gradients, no finetuning. load_in_4bit=True
    (default, needs bitsandbytes) measured ~6.2GB VRAM / ~0.7-1.0s per call
    on a 3090 Ti alongside the rest of nav_pipeline's stack; fp16 needs
    ~16.6GB / ~0.4-0.9s. See qwen_search_guide.QwenVLSearchGuide for the
    same numbers measured on that sibling module -- this one loads an
    equivalent model the same way.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_new_tokens: int = 48,
        max_new_tokens_multi: int = 160,
        prompt_template: Optional[str] = None,
        multi_prompt_template: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.max_new_tokens_multi = int(max_new_tokens_multi)
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self.multi_prompt_template = multi_prompt_template or MULTI_CANDIDATE_PROMPT
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        # Model-class-agnostic load so a Qwen3-VL id (needs
        # Qwen3VLForConditionalGeneration) works alongside the original
        # Qwen2.5-VL default: AutoModelForImageTextToText reads whichever
        # class the checkpoint's own config names. Fall back to the explicit
        # 2.5-VL class on a transformers too old to expose the Auto mapping.
        try:
            from transformers import AutoModelForImageTextToText as _VLMClass
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLMClass

        kwargs = {"torch_dtype": torch.float16}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        self._model = _VLMClass.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def ground(self, rgb: np.ndarray, instruction: str) -> Optional[PixelGoal]:
        """Single best point -- thin convenience wrapper, kept for callers
        that don't need candidate scoring (e.g. qwen_search_guide's sibling
        use case has no obstacle/continuity scoring to apply). Prefer
        ground_candidates() wherever a distractor could plausibly be in
        view -- see this module's docstring on why."""
        candidates = self.ground_candidates(rgb, instruction, max_candidates=1)
        return candidates[0] if candidates else None

    def ground_candidates(self, rgb: np.ndarray, instruction: str,
                          max_candidates: int = 3) -> List[PixelGoal]:
        """Up to max_candidates ranked, confidence-scored waypoint pixels
        for `instruction` -- see MULTI_CANDIDATE_PROMPT and this module's
        docstring. Caller (pipeline.py) scores each against goal continuity
        and obstacle cost and picks the winner; this method does no
        scoring of its own, just grounding."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        prompt = self.multi_prompt_template.format(instruction=instruction, max_candidates=max_candidates)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens_multi, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(gen, skip_special_tokens=True)
        return parse_pixel_candidates(answer, image_size=(w, h), max_candidates=max_candidates)


class QwenInstructionGate:
    """Wall-clock throttle: is it time to call Qwen again? Same pattern as
    sam_period_s/scene_tag_period_s/qwen_search_period_s elsewhere in this
    pipeline -- 7B inference is seconds, not one pipeline tick. Between due
    ticks, pipeline.py lets GoalBelief coast the goal by ego-motion instead
    of holding a stale pixel here (see this module's docstring)."""

    def __init__(self, period_s: float = 1.5):
        self.period_s = float(period_s)
        self._last_t = 0.0

    def due(self, now: float) -> bool:
        if now - self._last_t < self.period_s:
            return False
        self._last_t = now
        return True

    def reset(self) -> None:
        self._last_t = 0.0
