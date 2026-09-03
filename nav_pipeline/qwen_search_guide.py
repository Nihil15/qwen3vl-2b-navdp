"""Qwen2.5-VL "ghost pixel" search guidance for the real-rover DINO+NavDP pipeline.

While Grounding DINO can't currently see pipeline.py's text target (SEARCH
state), a frozen Qwen2.5-VL-7B-Instruct is asked to point at the single pixel
it thinks is the best direction to look/move toward next -- a doorway,
opening, hallway -- given the current frame and the target phrase. Straight
ahead maps to image center, right of center to steering right, and so on.
That pixel becomes a steering bearing for the SEARCH spin, replacing the
fixed spin-toward-last-known-side when a suggestion is available.

Grounding DINO keeps running every tick regardless (see pipeline.py's
_step_inner, which calls self.detector.detect(...) unconditionally). The
instant DINO reacquires the target, pipeline.py leaves the SEARCH branch
entirely and this module is not consulted again -- there is no explicit
"switch Qwen off" step because it never sits in the live TRACK/GOTO/STOP
path to begin with.

Sibling of MARS/mars-habitatsim/navdp/navdp/extensions/system2_pixel_goal.py
(the Habitat-sim version, which renders a goal MASK for NavDP's own
point-conditioning channel). Deliberately not shared code: that module lives
in a separate conda env/subproject, and this one only needs a STEERING
BEARING, not a full goal-mask channel -- once DINO reacquires the target,
goal creation goes back through DINO's own bbox + depth, same as always.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEFAULT_PROMPT = (
    "You are guiding a mobile ground robot searching for \"{target}\", which "
    "is not currently visible in this image. Point to the single pixel "
    "location most likely to lead toward \"{target}\" -- e.g. a doorway, "
    "hallway, opening, or corridor it could be through or behind. If nothing "
    "in view suggests a direction, point at the image center. Reply with "
    "only one pixel coordinate as (x, y)."
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


@dataclass
class SearchDirection:
    bearing: float       # radians, +left (matches pipeline.py's goal-bearing convention)
    confidence: float
    raw: str = ""


class QwenSearchGuide:
    """Interface: (rgb, target_text) -> Optional[SearchDirection]."""

    def suggest(self, rgb: np.ndarray, target_text: str) -> Optional[SearchDirection]:  # pragma: no cover
        raise NotImplementedError


class StubQwenSearchGuide(QwenSearchGuide):
    """Deterministic stand-in for validating the SEARCH wiring without loading
    the 7B -- mirrors system2_pixel_goal.StubPixelGoal's role in the sim side."""

    def __init__(self, bearing: float = 0.0, confidence: float = 1.0):
        self.bearing = float(bearing)
        self.confidence = float(confidence)

    def suggest(self, rgb: np.ndarray, target_text: str) -> Optional[SearchDirection]:
        return SearchDirection(self.bearing, self.confidence, raw="stub")


class QwenVLSearchGuide(QwenSearchGuide):
    """Frozen Qwen2.5-VL-7B-Instruct grounder -- inference only, no finetuning.

    Converts the model's pointed pixel column into a bearing (radians, +left)
    by mapping pixel columns linearly across the caller's horizontal FOV:
    center column = straight ahead (0), right edge = -fov/2, left edge =
    +fov/2. load_in_4bit (default, needs bitsandbytes) measured ~6.2GB VRAM
    on the internnav env / 3090 Ti, ~0.7-1.0s/call once loaded -- vs fp16's
    ~16.6GB / ~0.4-0.9s/call. Set False only if bitsandbytes isn't
    available; fp16 is the fallback, not the other way around.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        max_new_tokens: int = 48,
        horizontal_fov_deg: float = 90.0,
        prompt_template: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.horizontal_fov_rad = float(np.radians(horizontal_fov_deg))
        self.prompt_template = prompt_template or DEFAULT_PROMPT
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {"torch_dtype": torch.float16}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def suggest(self, rgb: np.ndarray, target_text: str) -> Optional[SearchDirection]:
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        prompt = self.prompt_template.format(target=target_text)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(gen, skip_special_tokens=True)
        uv = parse_pixel_coordinate(answer, image_size=(w, h))
        if uv is None:
            return None
        u, _v = uv
        frac = float(np.clip(u / max(w - 1, 1), 0.0, 1.0))
        bearing = (0.5 - frac) * self.horizontal_fov_rad
        return SearchDirection(bearing, 1.0, raw=answer)


class QwenSearchScheduler:
    """Throttles the VLM (seconds per call) to a wall-clock period and holds
    the last suggested bearing between calls. pipeline.py's SEARCH branch
    calls step() every tick; only every `period_s` does the model actually
    run -- same throttle pattern already used for SAM/CLIP (sam_period_s) and
    the scene tagger (scene_tag_period_s) elsewhere in this pipeline.
    """

    def __init__(self, guide: QwenSearchGuide, period_s: float = 2.0):
        self.guide = guide
        self.period_s = float(period_s)
        self._last_t = 0.0
        self._last_bearing: Optional[float] = None

    def step(self, now: float, rgb: np.ndarray, target_text: str) -> Optional[float]:
        if (now - self._last_t) < self.period_s and self._last_bearing is not None:
            return self._last_bearing
        direction = self.guide.suggest(rgb, target_text)
        self._last_t = now
        if direction is not None:
            self._last_bearing = direction.bearing
        return self._last_bearing

    def reset(self) -> None:
        self._last_t = 0.0
        self._last_bearing = None
