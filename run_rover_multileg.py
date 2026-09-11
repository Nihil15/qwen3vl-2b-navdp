#!/usr/bin/env python3
"""Headless multi-subtask real-rover runner (ESP32 --rover).

Sequences a compound instruction of MIXED subtask types on the physical rover:

  * ``move:<d>``   -- drive straight <d> metres, closed-loop on the ESP32
                      encoder+IMU-fused pose (nav_pipeline/odometry_logger.py).
                      No vision, no NavDP -- a plain P-controller on odometry so
                      the distance is exact and independent of the camera.
  * ``turn:<deg>`` -- rotate <deg> in place (negative = right / clockwise),
                      closed-loop on the IMU-fused heading (theta). Waits for
                      the BNO08x magnetometer to calibrate first and ABORTS the
                      run if it never does -- an uncalibrated-mag turn drifts.
  * ``object:<phrase>`` -- the existing Qwen3-VL-2B pixel-goal -> depth -> NavDP
                      crossmodal loop (nav_pipeline/zenoh_node.py::_tick), driven
                      to a self-declared STOP.

One second of rest (zero velocity) between every subtask.

This is a thin orchestrator: it instantiates
``nav_pipeline.zenoh_node.DinoNavDPZenohNode`` for ALL the Zenoh plumbing
(camera/depth/rpm subscribers, cmd_vel/waypoints/explanation publishers,
odometry, spin-stall watchdog, heartbeat) and drives its members directly
instead of calling ``.spin()``. Nothing in nav_pipeline/ is modified.

Run via LAUNCH/launch_rover_multileg.sh (brings the Pi up first). Direct:

  python run_rover_multileg.py --pi-ip 10.86.180.125 \
      --subtask move:2.5 --subtask turn:-90 --subtask "object:the air cooler"

or free text (split on "then"/commas, each clause classified):

  python run_rover_multileg.py --pi-ip 10.86.180.125 \
      --instruction "go straight up to 2.5 meter, turn right, go to the air cooler"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# nav_pipeline pulls in torch/transformers/navdp -- imported lazily in main()
# so --dry-run stays instant and doesn't need the heavy env.
GuardConfig = DinoNavDPPipeline = PipelineConfig = DinoNavDPZenohNode = serialize_string = None
LiveOccupancy = parse_string = thin_waypoints = None
# fact3r-map's "entity memory" (SigLIP2 appearance embeddings) integrated
# into the live rover's goal_memory: an OPT-IN twin of dinov2_embedder.py's
# Dinov2Embedder (see nav_pipeline/siglip2_embedder.py), not the SAM2+SigLIP2
# Fact3rEntityRecallGuide -- that one replaces the whole live grounder for an
# in-session memory with no disk persistence; this instead adds ONE extra
# field (`appearance_embedding`) onto the existing cross-session goal_memory
# JSONL record, so a "go back to X" recall gets an appearance sanity check
# on top of the existing text-label + odometry-point match, without changing
# what already drives (DINO still grounds, Qwen still supervises).
Siglip2Embedder = None
# fact3r-style LIVE ENTITY MEMORY (nav_pipeline/fact3r_live_memory.py):
# SAM2 class-agnostic proposals + SigLIP2 embeddings + a metric odom-frame
# position per object, accumulated over every frame while driving -- so a
# later "go to X" can recall the POSITION of an X that was only ever seen in
# passing. Opt-in (loads SAM2) via --fact3r-live-memory. intrinsics_from_fov
# is pulled alongside for the fallback when no camera_info has landed yet.
Fact3rLiveMemory = None
intrinsics_from_fov = None
# fact3r-map's own goal-memory storage/normalization (see
# MASt3R-SLAM/fact3r-map/fact3r/semantics/goal_memory.py) -- reused directly
# for its JSONL load/append + text-normalization utilities, which have zero
# heavy deps (json/pathlib/numpy only). NOT its SigLIP paraphrase-similarity
# tier or its entity/BEV-cell record schema -- those belong to the offline
# SAM2 entity-mapping pipeline, which this live DINO-box path has no
# equivalent of; a live record here is just {query_text, world_xy, ts}.
# Optional: a missing/moved MASt3R-SLAM checkout just disables recall,
# never breaks the rest of the runner.
_goal_memory_load = _goal_memory_append = _goal_memory_normalize = _goal_memory_find_hit = None

# LiveOccupancy grid values (see nav_pipeline/live_occupancy.py) -- copied
# here as plain ints so the occupancy-image encoder below works even before
# _load_nav_pipeline() has run (used only for their numeric value, never
# imported eagerly).
_OCC_UNKNOWN, _OCC_FREE, _OCC_OCCUPIED = -1, 0, 1


def _load_nav_pipeline():
    global GuardConfig, DinoNavDPPipeline, PipelineConfig, DinoNavDPZenohNode, serialize_string
    global LiveOccupancy, parse_string, thin_waypoints
    from nav_pipeline.obstacle_guard import GuardConfig as _GC
    from nav_pipeline.pipeline import DinoNavDPPipeline as _P, PipelineConfig as _PC
    from nav_pipeline.zenoh_node import (
        DinoNavDPZenohNode as _N, parse_string as _ps, serialize_string as _ss,
    )
    from nav_pipeline.live_occupancy import LiveOccupancy as _LO, thin_waypoints as _tw
    GuardConfig, DinoNavDPPipeline, PipelineConfig, DinoNavDPZenohNode, serialize_string = _GC, _P, _PC, _N, _ss
    LiveOccupancy, parse_string = _LO, _ps
    thin_waypoints = _tw

    global Siglip2Embedder
    from nav_pipeline.siglip2_embedder import Siglip2Embedder as _SE
    Siglip2Embedder = _SE

    global Fact3rLiveMemory, intrinsics_from_fov
    try:
        from nav_pipeline.fact3r_live_memory import Fact3rLiveMemory as _FLM
        Fact3rLiveMemory = _FLM
    except Exception as e:  # noqa: BLE001 -- missing SAM2_ROOT / torch just disables the feature
        print(f"[fact3r] live entity memory unavailable ({e}) -- --fact3r-live-memory disabled")
    try:
        from nav_pipeline.goal_utils import intrinsics_from_fov as _iff
        intrinsics_from_fov = _iff
    except Exception:
        intrinsics_from_fov = None

    global _goal_memory_load, _goal_memory_append, _goal_memory_normalize, _goal_memory_find_hit
    fact3r_map_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MASt3R-SLAM", "fact3r-map")
    if fact3r_map_root not in sys.path:
        sys.path.insert(0, fact3r_map_root)
    try:
        from fact3r.semantics.goal_memory import (
            append_memory as _gma, load_memory as _gml, normalize_query as _gmn, find_memory_hit as _gmf,
        )
        _goal_memory_load, _goal_memory_append, _goal_memory_normalize = _gml, _gma, _gmn
        # find_memory_hit's tier-1 fallback (SigLIP2 text-paraphrase
        # similarity, see resolve_semantic_goal_verified.py's locate-stage
        # step 2) needs a TextEncoder -- only usable once --goal-memory-
        # appearance loads a Siglip2Embedder (it satisfies that Protocol
        # directly, see siglip2_embedder.py). Without it, _recall_and_
        # approach() falls back to tier-0 exact match only (unchanged
        # pre-existing behavior).
        _goal_memory_find_hit = _gmf
    except Exception as e:
        print(f"[memory] fact3r-map goal_memory not available ({e}) -- recall disabled, "
              f"expected MASt3R-SLAM as a sibling of {os.path.dirname(os.path.abspath(__file__))}")


# ======================================================================
#  Instruction -> ordered subtask plan
# ======================================================================
# Clause splitter -- copied verbatim from run_vlnce_episode_test.py's
# split_instruction_clauses (importing that module would pull in habitat_sim,
# which need not exist in the internnav env this runs under). Splits on
# "then" AND on bare commas/periods so "go straight 2.5 m, turn right, go to
# the cooler" becomes three clauses.
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|\.|;|\bthen\b|\band\b)\s+", flags=re.I)

_MOVE_RE = re.compile(
    r"(?:go|move|drive|walk|head)?\s*(?:straight|forward|ahead|on)?\s*"
    r"(?:up\s*to|upto|for|about|around)?\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|metre|metres|cm|centimeter|centimeters)\b",
    flags=re.I,
)
# "back"/"backward"/"reverse" WITH a distance ("go back 1.5 meters") is a
# negative-distance move -- but "back" alone, followed by "to" ("go back
# to the fan"), is goal_memory's own navigation-prefix idiom for recalling
# an OBJECT, not a direction (see MASt3R-SLAM/fact3r-map/fact3r/semantics/
# goal_memory.py's _NAVIGATION_PREFIXES) -- the negative lookahead below
# excludes exactly that case, same distinction _classify_clause already
# makes structurally: this only ever fires alongside a real _MOVE_RE
# distance match, an object-recall clause never has one.
_BACKWARD_RE = re.compile(r"\b(?:back(?:ward)?|reverse)\b(?!\s+to\b)", flags=re.I)
_TURN_RE = re.compile(
    r"\bturn\b\s*(left|right|around)?\s*(?:by\s*)?"
    r"([0-9]+(?:\.[0-9]+)?)?\s*(?:deg|degree|degrees|°)?",
    flags=re.I,
)
# DINO/CLIP grounding wants a bare noun phrase ("white air cooler"), not a
# verb phrase ("go to the white air cooler") -- CLIP's zero-shot text
# similarity in particular saw its own score crater on a real, clearly-
# visible target because of this leftover prefix (live-observed 2026-09-08:
# a confirmed-in-view "chair" scoring 0.02-0.6 against target_text=="go to
# chair", under both the CLIP and appearance-lock floors, forcing the
# pipeline into SEARCH's fixed scan for almost the whole approach -- read
# by the operator as "continuously rotating"/"deflecting" even with the
# object plainly on camera). Strip it once here so every caller (--subtask,
# a single free-text --instruction, or one clause of a multi-leg one) gets
# a clean phrase into node.target.
_GOTO_PREFIX_RE = re.compile(
    r"^(?:go|come|get|walk|drive|move|head|navigate|proceed|return)\s+"
    r"(?:back\s+)?(?:up\s*to|to|towards?|toward|near)\s+"
    r"|^(?:go|come|walk|drive|head)\s+back\s+"
    r"|^(?:find|locate|approach|reach)\s+",
    flags=re.I,
)


def _strip_goto_prefix(text: str) -> str:
    stripped = _GOTO_PREFIX_RE.sub("", text.strip(), count=1).strip()
    # a leading article left behind ("the white air cooler") is also dead
    # weight in the text embedding -- drop it too.
    stripped = re.sub(r"^(?:the|a|an)\s+", "", stripped, flags=re.I).strip()
    return stripped or text.strip()   # never return an empty phrase


class Subtask:
    __slots__ = ("kind", "arg", "raw")

    def __init__(self, kind: str, arg, raw: str):
        self.kind = kind          # "move" | "turn" | "object"
        self.arg = arg            # float metres (move) | float deg, +ccw/-cw (turn) | str (object)
        self.raw = raw

    def __repr__(self):
        if self.kind == "move":
            return f"move:{self.arg:.2f}m"
        if self.kind == "turn":
            return f"turn:{self.arg:+.0f}deg"
        return f'object:"{self.arg}"'


def _parse_explicit(spec: str) -> Subtask:
    kind, _, val = spec.partition(":")
    kind = kind.strip().lower()
    val = val.strip()
    if kind == "move":
        return Subtask("move", float(val), spec)
    if kind == "turn":
        return Subtask("turn", float(val), spec)
    if kind in ("object", "obj", "goto", "go"):
        return Subtask("object", _strip_goto_prefix(val), spec)
    raise SystemExit(f"--subtask must be move:<m> | turn:<deg> | object:<phrase>, got {spec!r}")


def _classify_clause(clause: str) -> Subtask:
    c = clause.strip().strip(" .,")
    low = c.lower()

    if "turn" in low or "rotate" in low:
        m = _TURN_RE.search(low)
        direction = (m.group(1) or "right").lower() if m else "right"
        deg = float(m.group(2)) if (m and m.group(2)) else (180.0 if direction == "around" else 90.0)
        sign = 1.0 if direction == "left" else -1.0   # theta is CCW+, so right turn is negative
        return Subtask("turn", sign * deg, clause)

    m = _MOVE_RE.search(low)
    if m and "turn" not in low:
        dist = float(m.group(1))
        if m.group(2).lower().startswith("c"):   # cm
            dist /= 100.0
        if _BACKWARD_RE.search(low):
            dist = -dist
        return Subtask("move", dist, clause)

    return Subtask("object", _strip_goto_prefix(c), clause)


def _split_compound(clause: str) -> List[str]:
    """A run-on clause like "go straight 2.5 m turn right" that has BOTH a
    move-distance and a turn keyword -> split at "turn" so neither half is
    lost. One level of splitting is enough for the instructions this project
    uses; explicit --subtask avoids the guesswork entirely."""
    low = clause.lower()
    if _MOVE_RE.search(low) and re.search(r"\bturn\b", low):
        i = low.index("turn")
        return [clause[:i].strip(" .,"), clause[i:].strip(" .,")]
    return [clause]


def plan_from_instruction(instruction: str) -> List[Subtask]:
    """Free text -> ordered subtasks. Standalone (no argparse Namespace) so
    --serve mode can call this fresh for every new instruction it receives,
    not just once at startup."""
    clauses = [p.strip() for p in _CLAUSE_SPLIT_RE.split(instruction) if p.strip()]
    parts: List[str] = []
    for c in clauses:
        parts.extend(p for p in _split_compound(c) if p)
    return [_classify_clause(c) for c in parts] or [Subtask("object", _strip_goto_prefix(instruction), instruction)]


def resolve_plan(args) -> List[Subtask]:
    if args.subtask:
        return [_parse_explicit(s) for s in args.subtask]
    if args.instruction:
        return plan_from_instruction(args.instruction)
    raise SystemExit("give a plan: repeat --subtask move:2.5 / turn:-90 / object:... OR --instruction \"...\"")


# ======================================================================
#  geometry helpers
# ======================================================================
def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def body_delta(dx: float, dy: float, theta: float) -> Tuple[float, float]:
    """World-frame vector (dx, dy) -> (forward, left) in the body frame at heading theta (CCW+)."""
    c, s = math.cos(theta), math.sin(theta)
    return dx * c + dy * s, -dx * s + dy * c


def body_to_world(fwd: float, left: float, x: float, y: float, theta: float) -> Tuple[float, float]:
    """Inverse of body_delta: a body-frame (forward, left) point, at world pose (x, y, theta),
    -> its absolute world (x, y). Used to check whether NavDP's own goal-point trajectory for the
    CURRENT live detection is anywhere near a remembered goal_memory coordinate (see go_to_object's
    geo-consistency check) -- same rotation as body_delta, just not negated."""
    c, s = math.cos(theta), math.sin(theta)
    return x + fwd * c - left * s, y + fwd * s + left * c


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# Small, dependency-free stopword list for _shares_key_word -- just enough
# to strip the connective words a "go to X" clause/retarget phrase is built
# from, so what's left is the actual noun(s) being compared.
_STOPWORDS = frozenset(
    "a an the go to towards near by next of at in on go-to and or".split())


def _shares_key_word(a: str, b: str) -> bool:
    """True if `a` and `b` share at least one non-stopword token, or one is
    a substring of the other (handles "fan" vs "ceiling fan" for free).
    Used to gate the supervisor's DINO retarget (see object_leg's retarget
    branch) -- a legitimate paraphrase/landmark-narrowing of the object
    actually being searched for shares a word with it; an unrelated object
    Qwen noticed instead (e.g. "fan" -> "door") does not."""
    a_norm, b_norm = a.strip().lower(), b.strip().lower()
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        return True
    a_words = {w for w in a_norm.split() if w not in _STOPWORDS}
    b_words = {w for w in b_norm.split() if w not in _STOPWORDS}
    return bool(a_words & b_words)


def _extract_json(text: str) -> Optional[dict]:
    """First balanced {...} block in a VLM reply -> dict, or None."""
    i = text.find("{")
    while i != -1:
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        break
        i = text.find("{", i + 1)
    return None


# ======================================================================
#  Qwen task supervisor  (grounding is Grounding-DINO; Qwen only judges)
# ======================================================================
SUPERVISOR_PROMPT = """You are the SUPERVISOR + NAVIGATOR of a ground robot doing one navigation subtask.

SUBTASK: "{clause}"
The robot's object detector is currently locked on the phrase: "{target}"
Detector state this moment: {dino_state}
Images: [1] the robot's current forward camera view. {crop_note}{history_note}

Decide, from the camera view:

1. target  -- the SHORT noun phrase (2-5 words, lowercase) the detector should
              look for to make progress. For a "go to X" subtask this is X. For a
              "go through / past / between X" subtask, pick the landmark to head
              toward (e.g. "go through the doorway" -> "doorway"; "pass between the
              desks" -> "gap between desks" or "desk"). Keep it stable unless the
              detector is lost or locked on the wrong thing, or the subtask needs
              a more distinguishing phrase (e.g. two matching objects -> "white
              air cooler near the window").

2. status  -- one of:
     "searching"     the target/landmark is NOT clearly visible yet
     "approaching"   it IS visible ahead and the robot should keep driving to it
     "arrived"       DONE: for "go to X" the robot is close and centered on X;
                     for "through/past/between X" the robot has now PASSED X
                     (X is beside or behind it, the opening is cleared)
     "wrong_object"  the detector is locked onto something that is NOT the target

3. action  -- steering hint for THIS moment, one of "forward", "left", "right",
              "stop". Used only while the detector has no lock. "left"/"right" =
              turn that way to bring the target/opening toward centre; "forward" =
              path ahead is clear toward it; "stop" = arrived or blocked.

4. reason  -- one short sentence.

Reply with ONLY a JSON object:
{{"target": "...", "status": "...", "action": "...", "reason": "..."}}"""


RECALL_VERIFY_PROMPT = """You are verifying whether a robot has re-found the SAME physical object \
it found before, not just something else with a similar name.

Remembered object: "{phrase}"
{age_note}

Images: [1] the robot's current camera view. [2] a close crop of the object the \
detector is currently locked onto, which the robot is proposing as a re-find of \
the remembered object.

Judge from crop [2] (using [1] for surrounding context if it helps) whether this \
plausibly IS the same physical instance -- consider colour, shape, material, and \
visible surroundings, not just the category name. A different chair is NOT a match \
for "the chair" even though the category matches.

Reply with ONLY a JSON object:
{{"decision": "yes"|"no"|"uncertain", "confidence": 0.0-1.0, "reason": "one short sentence"}}"""


class TaskSupervisor:
    """Throttled Qwen-VL call that watches a DINO-grounded object subtask and
    returns {target, status, reason}. Does NOT produce coordinates -- grounding
    is Grounding-DINO's job; Qwen only supplies task-level judgement (what to
    look for, whether we've arrived, whether the lock is wrong)."""

    def __init__(self, model_id: str, load_in_4bit: bool, device: str = "cuda:0",
                 max_new_tokens: int = 200):
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as _VLM
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLM
        import importlib.util
        use_4bit = self.load_in_4bit
        if use_4bit and importlib.util.find_spec("bitsandbytes") is None:
            print("[supervisor] bitsandbytes not available -- falling back to fp16 load")
            use_4bit = False
        kwargs = {"torch_dtype": torch.float16}
        if use_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        print(f"[supervisor] loading {self.model_id} (4bit={use_4bit}) ...")
        self._model = _VLM.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def check(self, rgb, clause: str, target: str, dino_state: str, crop=None,
              history_frames: Optional[list] = None) -> Optional[dict]:
        """history_frames: NavDP's own short-term visual memory (see
        DinoNavDPPipeline.get_recent_frames), oldest first -- the same
        couple of recent frames NavDP's diffusion policy itself has been
        conditioning on, given to Qwen too so "is progress being made"
        judgments aren't made from a single frozen frame every call. Purely
        additive: omit (default) for the exact previous single/dual-image
        behavior."""
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image
            images = [Image.fromarray(rgb.astype("uint8"))]
            crop_note = ""
            if crop is not None and getattr(crop, "size", 0) and min(crop.shape[:2]) >= 8:
                images.append(Image.fromarray(crop.astype("uint8")))
                crop_note = f"[{len(images)}] a close crop of what the detector is currently locked on -- judge if it is the right object."
            history_note = ""
            if history_frames:
                start = len(images) + 1
                for f in history_frames:
                    images.append(Image.fromarray(f.astype("uint8")))
                end = len(images)
                span = f"[{start}]" if start == end else f"[{start}-{end}]"
                history_note = (f" {span} NavDP's own recent frame(s), OLDEST first, lower "
                                f"resolution than [1] -- use these ONLY to judge motion/progress "
                                f"since a moment ago (getting closer? passed it? still searching the "
                                f"same view?), not for fine detail.")
            prompt = SUPERVISOR_PROMPT.format(clause=clause, target=target, dino_state=dino_state,
                                              crop_note=crop_note, history_note=history_note)
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=images, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            reply = self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            v = _extract_json(reply)
            if not v or "status" not in v:
                return {"target": target, "status": "searching", "action": "forward",
                        "reason": f"unparsed: {reply[:80]}"}
            v.setdefault("target", target)
            v.setdefault("reason", "")
            v["action"] = str(v.get("action", "forward")).strip().lower()
            if v["action"] not in ("forward", "left", "right", "stop"):
                v["action"] = "forward"
            v["status"] = str(v["status"]).strip().lower()
            v["target"] = str(v["target"]).strip().lower() or target
            return v
        except Exception as e:
            print(f"[supervisor] call failed: {e}")
            return None

    def verify_recall(self, rgb, crop, phrase: str, age_s: float) -> Optional[dict]:
        """Locate-stage step 3 (resolve_semantic_goal_verified.py's
        Qwen3VLVerifier.verify_many), adapted to a live single-candidate
        tick instead of an offline multi-view evidence render: is the
        object DINO is currently locked onto plausibly the SAME physical
        instance goal_memory recalled for `phrase`, not just the same
        category? Reuses THIS SAME already-loaded model/processor -- the
        one already doing navigation supervision via check() -- rather than
        a second Qwen3VLVerifier instance, so this costs zero extra VRAM
        and zero extra model load time. Returns the same strict-JSON
        decision/confidence/reason shape the offline verifier uses (minus
        predicted_object/confusable_with/supporting_frames, which only mean
        something across multiple rendered evidence views)."""
        if crop is None or getattr(crop, "size", 0) == 0 or min(crop.shape[:2]) < 8:
            return None
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image
            images = [Image.fromarray(rgb.astype("uint8")), Image.fromarray(crop.astype("uint8"))]
            age_note = (f"It was last confirmed seen {age_s:.0f}s ago at a different position -- "
                        f"the robot has since driven back hoping to find it again.")
            prompt = RECALL_VERIFY_PROMPT.format(phrase=phrase, age_note=age_note)
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=images, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                # Was a hardcoded 120 (vs. check()'s self.max_new_tokens,
                # default 200) -- live-observed 2026-09-10: this recurringly
                # truncated the model mid-"reason" string before it reached
                # the closing '}', so _extract_json's balanced-brace scan
                # never found a match and every one of these fell through to
                # "uncertain (confidence 0.00) -- unparsed: ..." even when
                # the model had actually already decided yes/0.95 -- reads
                # as "Qwen isn't supervising correctly" when it's really
                # just a truncated response. Match check()'s budget.
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            reply = self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            v = _extract_json(reply)
            if not v or "decision" not in v:
                return {"decision": "uncertain", "confidence": 0.0, "reason": f"unparsed: {reply[:80]}"}
            v["decision"] = str(v["decision"]).strip().lower()
            if v["decision"] not in ("yes", "no", "uncertain"):
                v["decision"] = "uncertain"
            try:
                v["confidence"] = float(v.get("confidence", 0.0))
            except (TypeError, ValueError):
                v["confidence"] = 0.0
            v.setdefault("reason", "")
            return v
        except Exception as e:
            print(f"[supervisor] recall verification failed: {e}")
            return None


# ======================================================================
#  Runner
# ======================================================================
class PlanRunner:
    def __init__(self, node: DinoNavDPZenohNode, args, supervisor: "Optional[TaskSupervisor]" = None,
                 occupancy=None, fact3r_mem=None):
        self.node = node
        self.args = args
        self.supervisor = supervisor
        # fact3r-style live entity map (SAM2 + SigLIP2, metric positions) --
        # built + eager-loaded in main() when --fact3r-live-memory, else None.
        # _observe_entities() feeds it every few seconds during move/object
        # legs; _fact3r_recall() queries it at the start of each object leg.
        self.fact3r_mem = fact3r_mem
        self._last_fact3r_obs_t = 0.0
        self._fact3r_obs_count = 0
        # Passive feed of the entity map (id + world xy + obs) for the GUI's
        # "fact3r entities" column -- published after each mapping tick and
        # whenever a recall marks one. None viewers = no cost.
        self.fact3r_pub = node.session.declare_publisher("multileg/fact3r") if fact3r_mem is not None else None
        self._fact3r_recalled_id: Optional[str] = None
        # SAM2 automatic-mask generation is ~0.5-1.5 s of GPU per observe() --
        # run it on a background worker so the control loop never blocks on it
        # (live-observed 2026-09-10: the rover stuttered and the camera/status
        # feed hitched every --fact3r-observe-period-s because the tick that
        # fired the observe ran SAM2 inline). _observe_entities() just hands
        # the worker a copied frame when it's idle; _fact3r_lock guards
        # mem.entities against the recall/publish paths reading mid-mutation.
        self._fact3r_lock = threading.RLock()   # reentrant: _fact3r_recall holds it and calls _publish_fact3r
        self._fact3r_job: Optional[tuple] = None
        self._fact3r_worker_busy = False
        self._fact3r_stop = threading.Event()
        self._fact3r_thread: Optional[threading.Thread] = None
        self._last_depthfill_t = 0.0   # throttle for _fill_depth_holes()
        self._last_obs_theta = 0.0     # yaw-rate gate for _observe_entities()
        self._last_obs_theta_t = 0.0
        if fact3r_mem is not None:
            self._fact3r_thread = threading.Thread(target=self._fact3r_worker, daemon=True)
            self._fact3r_thread.start()
        self.results: List[dict] = []
        self._abort = False
        self._exit_requested = False   # set by request_abort; --serve checks this to fully stop
        self._last_w = 0.0
        self._leg_path_start = 0.0   # node.odom.total_dist_m snapshot at the current subtask's
        # start -- see _begin_leg_path()/_end_leg_path(); ACTUAL path length (not straight-line
        # start->end), also shown live in _publish_status while the leg runs, for the GUI's
        # per-subtask path-length column / an "experiment log" the GUI can tag and save.
        self._origin_to_plan_start_m = 0.0   # straight-line distance from the session's true
        # origin (0,0, set once by reset_pose()) to the pose this RUN's first subtask starts
        # from -- for the very first plan of a --serve session this is ~the odom self-check's
        # own return error; for a later instruction in the same session it's how far the rover
        # has already travelled since session start. Set once per run() call, in write_results()
        # + every _publish_status payload.
        self._plan: List[Subtask] = []
        self._current_index = -1
        # Passive status feed for multileg_gui.py -- a live viewer subscribes
        # to this; the runner works identically with zero viewers attached.
        self.status_pub = node.session.declare_publisher("multileg/status")
        self._last_status_pub = 0.0
        # Live BEV occupancy (see nav_pipeline/live_occupancy.py) -- carved
        # from the same depth+odometry every tick already touches, persists
        # across subtasks AND across --serve instructions (the whole point:
        # the map accumulates over a session instead of restarting per plan).
        self.occupancy = occupancy
        self.occupancy_pub = node.session.declare_publisher("multileg/occupancy") if occupancy else None
        self._last_occ_pub = 0.0
        self._last_occ_theta: Optional[float] = None   # for the turning-fast noise gate
        self._last_occ_theta_t = 0.0
        self._last_memory_event: Optional[dict] = None   # for the GUI's Recall & Remember panel
        self._occ_origin = (0.0, 0.0)   # set once, in run(), at the session's launch pose
        self._pose_reset_done = False   # gates the ONE-TIME pose/self-check in run() (see there)
        # fact3r-map entity-memory appearance check (see Siglip2Embedder's
        # module docstring) -- lazily constructed on first actual use (an
        # extra ~400MB model, opt-in via --goal-memory-appearance) so a run
        # that never recalls anything never pays for it.
        self.appearance_embedder = None
        self._recalled_appearance_embedding: Optional[np.ndarray] = None  # set by _recall_and_approach,
        # consumed once by the next object_leg's DINO loop, then cleared
        self._recalled_hit_age_s: Optional[float] = None   # goes with the embedding above
        self._recalled_world_xy: Optional[Tuple[float, float]] = None  # goes with the embedding
        # above too -- the remembered (gx, gy) itself, so go_to_object's first live re-detection
        # can check whether NavDP's own goal-point trajectory is geometrically consistent with it
        # (not just whether the crop LOOKS right -- see the geo-consistency check there)

    def _dbg(self, msg: str):
        """Step trace for --debug-steps -- prints exactly what the object leg
        is doing / calling at each decision point."""
        if getattr(self.args, "debug_steps", False):
            print(f"  [STEP] {msg}", flush=True)

    def _begin_leg_path(self):
        """Call once at the start of a move/turn/object subtask: snapshots the
        session-long odometer (node.odom.total_dist_m, only ever reset at
        session start) so _end_leg_path() -- or _publish_status's live field
        -- can report THIS leg's actual path length (integrated |v|*dt, not
        just straight-line start->end -- an object leg that searches/creeps
        covers far more ground than its net displacement)."""
        self._leg_path_start = self.node.odom.total_dist_m

    def _end_leg_path(self) -> float:
        return round(max(0.0, self.node.odom.total_dist_m - self._leg_path_start), 3)

    def _leg_odometry_csv(self) -> Optional[str]:
        """Filename (not full path -- the directory is odometry_log_dir,
        already in result.json / --odometry-log-dir) of the CSV OdometryLogger
        has been writing this whole subtask -- the raw (t, x, y, theta, v, w,
        ...) trace path_length_m was integrated from. Valid right up until the
        NEXT start_new_goal() call, i.e. safe to read at the end of
        drive_straight/turn_in_place/go_to_object before it returns."""
        p = self.node.odom.path
        return os.path.basename(p) if p else None

    def _publish_status(self, kind: str, desc: str, detector: Optional[dict] = None,
                         supervisor: Optional[dict] = None, force: bool = False,
                         memory: Optional[dict] = None):
        now = time.time()
        if not force and now - self._last_status_pub < 0.1:   # cap ~10 Hz, cheap JSON
            return
        self._last_status_pub = now
        if memory is not None:
            self._last_memory_event = memory   # sticky: keeps showing in the GUI between ticks
        try:
            payload = json.dumps(dict(
                ts=now,
                plan=[repr(s) for s in self._plan],
                current_index=self._current_index,
                current_kind=kind,
                current_desc=desc,
                results=self.results,
                current_path_length_m=self._end_leg_path(),   # live, updates while the leg runs
                origin_to_plan_start_m=self._origin_to_plan_start_m,
                pose=[round(self.node.odom.x, 3), round(self.node.odom.y, 3),
                      round(math.degrees(self.node.odom.theta), 1)],
                detector=detector,
                supervisor=supervisor,
                memory=self._last_memory_event,
            ))
            self.status_pub.put(serialize_string(payload))
        except Exception:
            pass

    def _integrate_occupancy(self):
        """Carve the live grid from whatever real depth+intrinsics the node
        already has (same staleness gate _tick uses), then publish a cropped
        PNG for multileg_gui.py's BEV panel, throttled. Called every tick
        from move/object subtask loops (not turn -- see turn_in_place's own
        comment); cheap no-op when disabled, when depth/intrinsics aren't
        fresh, or while the rover is actively rotating fast (SEARCH's
        creep-scan can spin almost as fast as an explicit turn -- same
        depth/pose desync problem, gated here since that loop has no
        turn-shaped place to skip the call from)."""
        if self.occupancy is None:
            return
        node = self.node
        now = time.time()
        theta = node.odom.theta
        dtheta = abs(wrap_angle(theta - self._last_occ_theta)) if self._last_occ_theta is not None else 0.0
        dt = now - self._last_occ_theta_t if self._last_occ_theta_t else 0.0
        turning_fast = dt > 0 and dtheta / dt > 0.3   # rad/s
        self._last_occ_theta, self._last_occ_theta_t = theta, now
        if turning_fast:
            return
        with node.lock:
            depth = node.latest_depth
            depth_age = now - node.latest_depth_t if node.latest_depth_t else 1e9
            intr = node.latest_intrinsics
            intr_age = now - node.latest_intrinsics_t if node.latest_intrinsics_t else 1e9
        if depth is not None and depth_age < node.DEPTH_STALE_S \
                and intr is not None and intr_age < node.INTRINSICS_STALE_S:
            try:
                self.occupancy.integrate(depth, (node.odom.x, node.odom.y, theta), intr)
            except Exception as e:
                print(f"[occupancy] integrate failed: {e}")
        if now - self._last_occ_pub >= self.args.occupancy_publish_period_s:
            self._last_occ_pub = now
            self._publish_occupancy_image()

    # -- depth hole-fill (glossy/dark/thin surfaces) ---------------
    def _fill_depth_holes(self, min_period_s: float = 0.0) -> bool:
        """Splice monocular metric depth into no-return pixels of the live
        RealSense frame, in place on node.latest_depth, so the obstacle guard
        and fact3r's SAM2 observe() both see the surfaces (a monitor screen,
        glass, black objects) the D435i can't range. Every valid RealSense
        pixel is kept; only 0/NaN/out-of-range holes are filled. Returns
        whether it filled this call. No-op unless --depth-fill-holes."""
        a = self.args
        if not getattr(a, "depth_fill_holes", False):
            return False
        node = self.node
        depther = getattr(node.pipe, "depther", None)
        if depther is None:
            return False
        now = time.time()
        if min_period_s and now - self._last_depthfill_t < min_period_s:
            return False
        with node.lock:
            rgb = node.latest_rgb
            depth = node.latest_depth
            depth_age = now - node.latest_depth_t if node.latest_depth_t else 1e9
        if rgb is None or depth is None or depth_age > node.DEPTH_STALE_S:
            return False
        if depth.shape[:2] != rgb.shape[:2]:
            return False
        invalid = ~(np.isfinite(depth) & (depth > 0.15) & (depth < 8.0))
        H, W = depth.shape[:2]
        cor = invalid[int(H * 0.35):int(H * 0.97), int(W * 0.30):int(W * 0.70)]
        frac = float(cor.mean()) if cor.size else 0.0
        if frac < a.depth_fill_min_hole_frac:
            return False
        try:
            mono = depther.estimate(rgb)
        except Exception as e:
            self._dbg(f"[depth-fill] monocular estimate failed ({e}) -- skipping")
            return False
        if mono is None or mono.shape[:2] != depth.shape[:2]:
            return False
        filled = np.asarray(depth, dtype=np.float32).copy()
        filled[invalid] = np.asarray(mono, dtype=np.float32)[invalid]
        with node.lock:
            if node.latest_depth is depth:   # camera callback hasn't replaced it since
                node.latest_depth = filled
        self._last_depthfill_t = now
        self._dbg(f"[depth-fill] {frac * 100:.0f}% of the forward corridor had NO RealSense "
                  f"return -> filled from monocular depth (glossy/dark/thin surfaces like a "
                  f"monitor screen are now visible to the obstacle guard + fact3r)")
        return True

    # -- fact3r live entity memory ------------------------------------
    def _observe_entities(self, during_metric: bool = False, during_turn: bool = False):
        """One throttled fact3r mapping tick: SAM2 class-agnostic proposals
        on the current frame -> per-object (metric odom xy, SigLIP2
        embedding) -> associate into self.fact3r_mem so a LATER object leg
        can recall an object seen here even if the rover never drove to it.

        Same depth/intrinsics staleness gate as _integrate_occupancy, plus a
        wall-clock throttle (--fact3r-observe-period-s) because SAM2
        automatic masking is ~0.3-1 s/call -- far too slow for the 3-10 Hz
        control loops this is called from. Cheap no-op when the feature is
        off, depth isn't fresh, or the throttle hasn't elapsed."""
        mem = self.fact3r_mem
        if mem is None:
            return
        a = self.args
        if during_metric and not a.fact3r_observe_during_metric:
            return
        if during_turn and not getattr(a, "fact3r_observe_during_turn", False):
            return
        now = time.time()
        if now - self._last_fact3r_obs_t < a.fact3r_observe_period_s:
            return
        node = self.node
        # Skip a tick while the rover is yawing fast -- streaming latency
        # misaligns the depth frame from the pose it's paired with, so the
        # unprojected positions smear. dtheta/dt over the gap since the last
        # check (any call site); the 0.5 rad/s bar clears a normal slow
        # turn:/GOTO turn but drops a search spin or a turn's fast opening.
        _th = node.odom.theta
        _prev_th, _prev_tt = self._last_obs_theta, self._last_obs_theta_t
        self._last_obs_theta, self._last_obs_theta_t = _th, now
        if _prev_tt and (now - _prev_tt) < 2.0:
            yaw_rate = abs(wrap_angle(_th - _prev_th)) / max(now - _prev_tt, 1e-3)
            if yaw_rate > 0.5:
                return
        with node.lock:
            rgb = node.latest_rgb
            depth = node.latest_depth
            depth_age = now - node.latest_depth_t if node.latest_depth_t else 1e9
            intr = node.latest_intrinsics
            intr_age = now - node.latest_intrinsics_t if node.latest_intrinsics_t else 1e9
        if rgb is None or depth is None or depth_age > node.DEPTH_STALE_S:
            return
        if depth.shape[:2] != rgb.shape[:2]:
            return
        if intr is None or intr_age > node.INTRINSICS_STALE_S:
            if intrinsics_from_fov is None:
                return
            H, W = rgb.shape[:2]
            intr = intrinsics_from_fov(W, H, a.fov)
        # Hand the frame to the background worker instead of running SAM2
        # here. If the worker is still on the previous frame, skip -- the
        # next throttle window will try again; dropping mapping frames is
        # fine, blocking the drive loop is not.
        if self._fact3r_worker_busy or self._fact3r_job is not None:
            return
        self._last_fact3r_obs_t = now
        self._fact3r_job = (
            np.ascontiguousarray(rgb).copy(), np.ascontiguousarray(depth).copy(),
            (node.odom.x, node.odom.y, node.odom.theta), tuple(intr))

    def _fact3r_worker(self):
        """Drains _fact3r_job: runs mem.observe() (the slow SAM2 pass) off the
        control loop, then republishes the entity feed."""
        mem = self.fact3r_mem
        while not self._fact3r_stop.is_set():
            job = self._fact3r_job
            if job is None:
                time.sleep(0.05)
                continue
            self._fact3r_job = None
            self._fact3r_worker_busy = True
            rgb, depth, pose, intr = job
            try:
                with self._fact3r_lock:
                    n = mem.observe(rgb, depth, pose, intr, step=self._fact3r_obs_count)
                self._fact3r_obs_count += 1
                if n or self._fact3r_obs_count <= 3 or self._fact3r_obs_count % 10 == 0:
                    print(f"  [fact3r] map tick {self._fact3r_obs_count}: +{n} proposal(s) "
                          f"-> {len(mem.entities)} entities")
                self._publish_fact3r()
            except Exception as e:
                print(f"  [fact3r] observe failed (ignoring): {e}")
            finally:
                self._fact3r_worker_busy = False

    def _publish_fact3r(self):
        """Push the current entity map (id, world xy, obs, median depth) to
        'multileg/fact3r' for the GUI column. Cheap JSON; no-op with no map."""
        mem = self.fact3r_mem
        if mem is None or self.fact3r_pub is None:
            return
        try:
            ents = []
            with self._fact3r_lock:
                for e in list(mem.entities):
                    x, y = e.position
                    md = float(np.median(e.depths)) if getattr(e, "depths", None) else None
                    ents.append({"id": e.entity_id, "xy": [round(x, 2), round(y, 2)],
                                 "obs": e.observations,
                                 "depth_m": (round(md, 1) if md is not None else None),
                                 "spread_m": round(e.position_spread, 2)})
            ents.sort(key=lambda r: (-r["obs"]))
            payload = json.dumps({"entities": ents, "count": len(ents),
                                  "recalled_id": self._fact3r_recalled_id,
                                  "ts": time.time()})
            self.fact3r_pub.put(serialize_string(payload))
        except Exception as e:
            print(f"  [fact3r] status publish failed (ignoring): {e}")

    def _fact3r_recall(self, phrase: str, label: str) -> Optional[Tuple[float, float]]:
        """Query the live entity map for `phrase`. Returns an odom-frame
        (x, y) to drive to, or None. Two-stage (SigLIP2 shortlist -> Qwen
        supervisor verifies the crop -> stored position) when a supervisor
        is loaded and --fact3r-recall-verify; SigLIP-only text match
        otherwise."""
        a = self.args
        mem = self.fact3r_mem
        self._fact3r_recalled_id = None
        if mem is None or not a.fact3r_recall or not mem.entities:
            return None
        try:
            # hold the map lock for the whole query so the background observe
            # worker can't mutate entities mid-merge / mid-verify
            with self._fact3r_lock:
                if a.fact3r_recall_merge:
                    merged = mem.merge_nearby_duplicates()
                    if merged:
                        print(f"  [{label}] [fact3r] merged {merged} fragmented duplicate(s) "
                              f"-> {len(mem.entities)} entities")
                if a.fact3r_recall_verify and self.supervisor is not None:
                    ent, pos, rec, shortlist = mem.query_verified(
                        phrase, self.supervisor,
                        min_observations=a.fact3r_recall_min_obs,
                        min_confidence=a.fact3r_recall_min_confidence,
                        max_position_spread=a.fact3r_recall_max_spread_m,
                        max_median_depth_m=a.fact3r_recall_max_depth_m,
                        verbose=a.debug_steps)
                    if ent is None:
                        tops = ", ".join(
                            f"{r['entity_id']}~{r['siglip_score']:.2f}"
                            f"[{r.get('vlm', {}).get('decision', '?')}]"
                            for r in (shortlist or [])[:5])
                        print(f"  [{label}] [fact3r] nothing passed Qwen verification for "
                              f"'{phrase}' (shortlist: {tops or 'empty -- all entities filtered out'})")
                        return None
                    print(f"  [{label}] [fact3r] '{phrase}' -> {ent.entity_id} at "
                          f"({pos[0]:+.2f},{pos[1]:+.2f})  {ent.observations} obs, "
                          f"siglip {rec['siglip_score']:.2f}, vlm conf "
                          f"{rec['vlm'].get('confidence', 0.0):.2f}")
                    self._fact3r_recalled_id = ent.entity_id
                    self._publish_fact3r()
                    return float(pos[0]), float(pos[1])
                self._dbg("no supervisor / --no-fact3r-recall-verify -> SigLIP-only text match "
                          f"mem.query(\"{phrase}\", min_obs={a.fact3r_recall_min_obs}, "
                          f"min_score={a.fact3r_recall_min_score})")
                hit = mem.query(phrase, min_observations=a.fact3r_recall_min_obs,
                                min_score=a.fact3r_recall_min_score,
                                max_median_depth_m=a.fact3r_recall_max_depth_m)
                if hit is None:
                    self._dbg("mem.query returned None (best SigLIP cosine below --fact3r-recall-min-score)")
                    return None
                ent, pos, score = hit
                print(f"  [{label}] [fact3r] '{phrase}' -> {ent.entity_id} at "
                      f"({pos[0]:+.2f},{pos[1]:+.2f})  siglip {score:.2f}, {ent.observations} obs")
                self._fact3r_recalled_id = ent.entity_id
                self._publish_fact3r()
                return float(pos[0]), float(pos[1])
        except Exception as e:
            print(f"  [{label}] [fact3r] recall failed ({e})")
            return None

    def _drive_to_point(self, gx: float, gy: float, phrase: str, label: str,
                        timeout_s: float) -> float:
        """Blind drive to an odom-frame point: turn to face it (NavDP GOTO is
        forward-biased -- see _recall_and_approach's face-turn note), then
        NavDP body-frame GOTO with its own reactive obstacle guard until
        within --goal-memory-arrive-m, a stall, or timeout. Keeps carving
        the occupancy grid + entity map on the way. Returns the final
        straight-line distance left to (gx, gy) -- the caller decides
        whether that counts as arrival. This is the fact3r-recall analogue
        of _recall_and_approach (which stays the JSONL goal-memory path with
        its own A* routing / drift gates / appearance checks)."""
        node = self.node
        a = self.args
        period = 1.0 / a.metric_hz
        t0 = time.time()
        stall_xy, stall_t = (node.odom.x, node.odom.y), t0
        faced = False
        was_turning = False   # reset the stall window the instant GOTO actually
        # starts translating -- otherwise a slow face-turn + settle (can be
        # ~10s at goal_memory_face_max_angular) eats the whole stall budget and
        # the very first GOTO tick reads as "stalled" before the rover has
        # moved (the memory approach then hands to search having driven ~0 m).
        start_d = math.hypot(gx - node.odom.x, gy - node.odom.y)
        print(f"  [{label}] [fact3r] approaching ({gx:+.2f},{gy:+.2f}) "
              f"-- {start_d:.1f} m straight-line, blind GOTO until {a.goal_memory_arrive_m:.1f} m out "
              f"(or it stalls) -- then search")
        self._publish_status("object", phrase, force=True,
                             memory={"event": "fact3r_recall", "phrase": phrase,
                                     "world_xy": [round(gx, 3), round(gy, 3)]})
        while not self._abort and time.time() - t0 < timeout_s:
            x, y, th = node.odom.x, node.odom.y, node.odom.theta
            d = math.hypot(gx - x, gy - y)
            if d < a.goal_memory_arrive_m:
                print(f"  [{label}] [fact3r] within {d:.1f} m of the stored position "
                      f"-- memory exhausted, handing to DINO")
                break
            fwd, lat = body_delta(gx - x, gy - y, th)
            face_err = math.atan2(lat, fwd)
            self._fill_depth_holes()   # so NavDP GOTO's obstacle guard sees monitor-shaped holes
            with node.lock:
                rgb, depth, intr = node.latest_rgb, node.latest_depth, node.latest_intrinsics
            if not faced and abs(face_err) > math.radians(25):
                w = clamp(a.turn_kp * face_err, -a.goal_memory_face_max_angular,
                          a.goal_memory_face_max_angular)
                if abs(w) < a.turn_w_floor:
                    w = math.copysign(a.turn_w_floor, w)
                node.publish_cmd(0.0, w)
                was_turning = True
                time.sleep(period)
                continue
            if not faced:
                faced = True
                node.publish_cmd(0.0, 0.0)
                if a.goal_memory_turn_to_drive_settle_s > 0 and not self._abort:
                    time.sleep(a.goal_memory_turn_to_drive_settle_s)
                continue
            if abs(face_err) > math.radians(70):
                faced = False        # drifted off NavDP's forward cone -- re-face
                was_turning = True
                continue
            if rgb is None:
                time.sleep(period)
                continue
            if was_turning:
                stall_xy, stall_t = (x, y), time.time()   # GOTO just started -- fresh stall window
                was_turning = False
            try:
                gres = node.pipe.step(rgb, "", depth=depth, pose=(x, y, th), intrinsics=intr,
                                      external_goal=np.array([fwd, lat, 0.0], dtype=np.float32))
                node.publish_cmd(gres.linear, gres.angular)
            except Exception as e:
                print(f"  [{label}] [fact3r] GOTO step failed ({e}) -- handing to DINO")
                break
            self._integrate_occupancy()
            self._observe_entities()
            self._publish_status("object", f"{phrase} (fact3r approach)")
            if math.hypot(x - stall_xy[0], y - stall_xy[1]) >= a.goal_memory_stall_min_move_m:
                stall_xy, stall_t = (x, y), time.time()
            elif time.time() - stall_t > a.goal_memory_stall_after_s:
                print(f"  [{label}] [fact3r] stalled ({a.goal_memory_stall_after_s:.0f}s "
                      f"< {a.goal_memory_stall_min_move_m:.2f}m) -- handing to DINO, {d:.1f} m short")
                break
            time.sleep(period)
        node.publish_cmd(0.0, 0.0)
        if not self._abort and a.goal_memory_face_settle_s > 0:
            time.sleep(a.goal_memory_face_settle_s)
        node.odom._history.clear()
        return math.hypot(gx - node.odom.x, gy - node.odom.y)

    def clear_map(self):
        """Reset the live occupancy grid to all-UNKNOWN in place -- used by
        the GUI's 'Clear map' button (published as '__clear_map__' on the
        same instruction topic --serve already listens on)."""
        if self.occupancy is None:
            return
        self.occupancy.grid.fill(-1)   # UNKNOWN, see live_occupancy.py
        print("[occupancy] map cleared")

    def _publish_occupancy_image(self):
        try:
            import cv2
            half = self.args.occupancy_range_m
            cx, cy = self._occ_origin
            r0, c0 = self.occupancy.to_cell(cx - half, cy - half)
            r1, c1 = self.occupancy.to_cell(cx + half, cy + half)
            h, w = self.occupancy.grid.shape
            r0, r1 = max(0, r0), min(h, r1)
            c0, c1 = max(0, c0), min(w, c1)
            if r1 <= r0 or c1 <= c0:
                return
            sub = self.occupancy.grid[r0:r1, c0:c1]
            # UNKNOWN(-1)->mid grey, FREE(0)->white, OCCUPIED(1)->black --
            # matches the values live_occupancy.py itself defines.
            img = np.where(sub == 1, 0, np.where(sub == 0, 255, 128)).astype(np.uint8)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                self.occupancy_pub.put(bytes(buf))
        except Exception as e:
            print(f"[occupancy] publish failed: {e}")

    # -- fact3r-map goal memory ------------------------------------------
    def _ensure_appearance_embedder(self):
        """Lazily build the SigLIP2 appearance embedder (fact3r-map entity-
        memory integration -- see Siglip2Embedder's module docstring).
        Returns None (never raises) if --goal-memory-appearance wasn't
        passed, or if loading fails for any reason -- appearance checking is
        a bonus signal on top of the existing text+odometry memory, never a
        requirement for it to keep working."""
        if not self.args.goal_memory_appearance:
            return None
        if self.appearance_embedder is not None:
            return self.appearance_embedder
        if Siglip2Embedder is None:
            return None
        try:
            print("[memory] loading SigLIP2 for goal-memory appearance verification "
                  f"({self.args.siglip_model_id}) ...")
            self.appearance_embedder = Siglip2Embedder(model_id=self.args.siglip_model_id, device=self.args.device)
        except Exception as e:
            print(f"[memory] SigLIP2 appearance embedder failed to load ({e}) -- "
                  "continuing with text+odometry-only memory")
            self.appearance_embedder = None
        return self.appearance_embedder

    def _remember_arrival(self, phrase: str, x: float, y: float,
                          det_box: Optional[np.ndarray] = None, rgb: Optional[np.ndarray] = None):
        if not self.args.goal_memory or _goal_memory_append is None:
            return
        try:
            ts = time.time()
            record = {
                "query_text": phrase, "world_xy": [round(float(x), 3), round(float(y), 3)], "ts": ts,
                # Drift-risk snapshot -- see _recall_and_approach's gate.
                # Session-long odometers, NOT reset per goal, so a later
                # recall can measure how much unanchored dead reckoning has
                # happened since THIS specific arrival, not just elapsed
                # wall-clock time.
                "odom_enc_rot_rad": round(float(self.node.odom.total_enc_rot_rad), 3),
                "odom_dist_m": round(float(self.node.odom.total_dist_m), 3),
            }
            # fact3r-map entity-memory integration: bank ONE SigLIP2 view of
            # the confirmed target alongside the text+point record, so a
            # later recall can appearance-check what it finds against what
            # was actually here -- see object_leg()'s call site for why
            # det_box/rgb can legitimately be None (arrival declared on a
            # tick DINO had just lost the box, e.g. close_loss) -- degrades
            # gracefully to the pre-existing text+odometry-only record.
            embedder = self._ensure_appearance_embedder()
            if embedder is not None and det_box is not None and rgb is not None:
                try:
                    emb = embedder.embed(rgb, det_box)
                    if emb is not None:
                        record["appearance_embedding"] = [round(float(v), 5) for v in emb]
                        record["appearance_model"] = self.args.siglip_model_id
                except Exception as e:
                    print(f"  [memory] appearance embed failed (recording text+point only): {e}")
            _goal_memory_append(Path(self.args.goal_memory_path), record)
            has_emb = "appearance_embedding" in record
            print(f"  [memory] recorded confirmed arrival at '{phrase}' -> ({x:+.2f},{y:+.2f})"
                  f"{' + appearance view' if has_emb else ''}")
            self._publish_status("object", phrase, force=True,
                                 memory={"event": "remembered", "phrase": phrase,
                                         "world_xy": [round(float(x), 3), round(float(y), 3)], "ts": ts})
        except Exception as e:
            print(f"  [memory] write failed: {e}")

    def _recall_and_approach(self, phrase: str, label: str) -> bool:
        """Look up a past confirmed arrival for `phrase` and, if found,
        drive toward it via NavDP's obstacle-avoided GOTO (pipe.step's
        external_goal, body-frame -- the same mechanism home_gui.py's
        "Go Home" uses) before the normal DINO search loop ever starts.
        Vision still has the final word: this only gets the rover close,
        capped by distance/time, then falls straight through to the
        existing DINO+supervisor loop for the confirmed approach. Returns
        whether a memory hit was found and used (not whether it fully
        closed the distance)."""
        a = self.args
        self._recalled_appearance_embedding = None   # reset -- stale from a previous leg otherwise
        self._recalled_hit_age_s = None
        self._recalled_world_xy = None
        if not a.goal_memory or _goal_memory_load is None or _goal_memory_normalize is None:
            self._dbg("_recall_and_approach: JSONL goal-memory unavailable (--no-goal-memory or "
                      "MASt3R-SLAM checkout missing) -> no hit")
            return False
        try:
            records = _goal_memory_load(Path(a.goal_memory_path))
        except Exception:
            return False
        records = [r for r in records if "world_xy" in r]
        self._dbg(f"_recall_and_approach: {len(records)} arrival record(s) on disk "
                  f"({a.goal_memory_path}); trying tier-0 exact then tier-1 SigLIP2 paraphrase "
                  f"(>={a.goal_memory_paraphrase_min_sim}) for \"{phrase}\"")
        # Locate-stage step 2 (resolve_semantic_goal_verified.py): memory
        # check happens with tier-0 exact match first, tier-1 SigLIP2
        # paraphrase-similarity fallback second. Tier 1 needs a TextEncoder
        # (SigLIP2's text tower) -- reuse the SAME appearance embedder
        # --goal-memory-appearance already loads for the image side, rather
        # than a second model. Without that flag, this degrades to tier-0
        # only, exactly the pre-existing behavior.
        embedder = self._ensure_appearance_embedder()
        hit = None
        if embedder is not None and _goal_memory_find_hit is not None:
            try:
                hit = _goal_memory_find_hit(phrase, records, embedder,
                                            min_similarity=a.goal_memory_paraphrase_min_sim)
            except Exception as e:
                print(f"  [{label}] [memory] paraphrase lookup failed ({e}) -- falling back to exact match")
        if hit is None:
            target_norm = _goal_memory_normalize(phrase)
            for r in reversed(records):
                if _goal_memory_normalize(str(r.get("query_text", ""))) == target_norm:
                    hit = r
                    break
            self._dbg(f"tier-1 paraphrase: no hit. tier-0 exact match on '{target_norm}': "
                      f"{'HIT' if hit else 'no hit'}")
        else:
            self._dbg(f"tier-1 SigLIP2 paraphrase: HIT on stored '{hit.get('query_text', '?')}'")
        if hit is None:
            return False
        gx, gy = hit["world_xy"]
        self._recalled_world_xy = (float(gx), float(gy))   # for go_to_object's geo-consistency check
        hit_ts = float(hit.get("ts", time.time()))
        age_s = time.time() - hit_ts
        print(f"  [{label}] [memory] recalling '{phrase}' from a confirmed arrival "
              f"{age_s:.0f}s ago at ({gx:+.2f},{gy:+.2f}) -- approaching before search")
        # fact3r-map entity-memory integration: if this record banked a
        # SigLIP2 view (see _remember_arrival), hand it to object_leg()'s
        # own DINO loop below -- once handed off from this blind approach to
        # a real DINO detection, that loop appearance-checks the FIRST live
        # detection against this embedding and folds the result into the
        # Qwen supervisor's context, instead of trusting the text label +
        # odometry point alone the way this function's approach itself must
        # (there's nothing to visually compare against yet while driving
        # blind toward a remembered point with no live detection).
        stored_emb = hit.get("appearance_embedding")
        if stored_emb and embedder is not None:
            self._recalled_appearance_embedding = np.asarray(stored_emb, dtype=np.float32)
            self._recalled_hit_age_s = age_s
            print(f"  [{label}] [memory] recalled view also carries an appearance signature -- "
                  f"will sanity-check the first live re-detection against it")
        self._publish_status("object", phrase, force=True,
                             memory={"event": "recalled", "phrase": phrase,
                                     "world_xy": [gx, gy], "ts": hit_ts})
        node = self.node
        # Drift-risk gate: world_xy was accurate WHEN BANKED, but pose is
        # continuous dead reckoning for the rest of the --serve session (see
        # OdometryLogger's module docstring) with no absolute correction --
        # error only ever grows, and grows fastest from UNANCHORED rotation
        # (wheel slip during in-place turns/searches whenever the IMU isn't
        # mag-calibrated or was auto-declared frozen, see
        # OdometryLogger.total_enc_rot_rad) plus raw distance travelled
        # (wheel-radius/track-width calibration error, total_dist_m). Both
        # are session-long odometers, snapshotted into the record at
        # _remember_arrival() time and compared against their CURRENT value
        # here -- this is "how much unanchored dead reckoning has happened
        # since", a much better trust signal than age_s alone (a rover that
        # sat still for 10 minutes hasn't drifted; one that ran three more
        # search spins in the last 10 seconds has). Live-observed 2026-09-10:
        # position/recall accuracy degrading task over task in a single
        # --serve session, first task fine -- exactly the symptom of
        # uncorrected dead-reckoning error compounding across every leg run
        # since boot, not a per-leg bug. Older records without these fields
        # (banked before this existed) get 0 -- i.e. no gate, prior behavior.
        rot_since = max(0.0, node.odom.total_enc_rot_rad - float(hit.get("odom_enc_rot_rad", node.odom.total_enc_rot_rad)))
        dist_since = max(0.0, node.odom.total_dist_m - float(hit.get("odom_dist_m", node.odom.total_dist_m)))
        # Effective cap on the blind approach below -- normally the full
        # --goal-memory-approach-timeout-s. Originally this gate `return
        # True`d outright on a drift-risk hit (ZERO attempt); then a flat
        # --goal-memory-drift-quick-timeout-s (5s). Both were wrong in the
        # same direction: live-observed 2026-09-10, EVERY recall in a long
        # test session tripped this gate (repeated retries make total_dist_m
        # balloon fast, so "distance driven since banking" exceeds 10m
        # almost immediately and stays there for the rest of the session --
        # this isn't really "drift", it's normal repeated testing), and the
        # actual remembered points were 5-9m away -- at this rig's real
        # creep speed (~0.05-0.15 m/s, see the PRED log lines) a flat 5s
        # covers under a metre, nowhere near enough to even finish the
        # turn-to-face, let alone start closing real distance. The result
        # was worse than no gate at all: never enough time to get near the
        # target, so DINO handed off to SEARCH from far away with a wrong
        # heading and no positional bias, then found unrelated objects
        # instead (see the geo-consistency mismatches: 3.5-6.1m off).
        # Fix: scale the cap to the actual straight-line distance at a
        # nominal creep speed (--goal-memory-quick-timeout-speed-mps)
        # instead of a magic constant divorced from geometry -- a genuinely
        # far target still gets close to the full timeout (this gate then
        # only trims the excess beyond what the distance itself needs), a
        # genuinely near one still fails fast.
        approach_timeout_s = a.goal_memory_approach_timeout_s
        if (a.goal_memory_max_drift_rot_rad > 0 and rot_since > a.goal_memory_max_drift_rot_rad) or \
           (a.goal_memory_max_drift_dist_m > 0 and dist_since > a.goal_memory_max_drift_dist_m):
            straight_m = math.hypot(gx - node.odom.x, gy - node.odom.y)
            needed_s = straight_m / max(a.goal_memory_quick_timeout_speed_mps, 0.01)
            approach_timeout_s = min(approach_timeout_s,
                                     max(a.goal_memory_drift_quick_timeout_s, needed_s))
            print(f"  [{label}] [memory] {math.degrees(rot_since):.0f}deg unanchored turning + "
                  f"{dist_since:.1f}m driven since this arrival was banked -- position may have "
                  f"drifted, so capping the blind approach at {approach_timeout_s:.0f}s ({straight_m:.1f}m "
                  f"straight-line at ~{a.goal_memory_quick_timeout_speed_mps:.2f}m/s) instead of the "
                  f"full {a.goal_memory_approach_timeout_s:.0f}s before falling back to search")
        period = 1.0 / a.metric_hz
        # A* over the live-carved BEV occupancy grid (nav_pipeline/
        # live_occupancy.py), same building block the offline sim_bridge
        # return-path uses (A* centerline -> waypoint-by-waypoint NavDP) --
        # previously built but never called from this live path (all memory
        # approaches drove a raw straight line at the remembered point,
        # leaving NavDP's own reactive obstacle guard as the only thing
        # standing between the rover and anything it had already mapped in
        # the way). A "go back to X" leg is exactly the case this helps
        # most: the grid cells between here and there were very likely
        # observed on the ORIGINAL trip that reached X in the first place.
        # Route-finding only changes what point each tick steers TOWARD --
        # the final-arrival check below still measures against the true
        # remembered (gx, gy), and if no route is found this degrades
        # exactly to the prior straight-line behavior, not a hard failure.
        route: Optional[List[Tuple[float, float]]] = None
        route_idx = 0
        if self.occupancy is not None and thin_waypoints is not None:
            try:
                x0, y0 = node.odom.x, node.odom.y
                path, info = self.occupancy.astar((x0, y0), (gx, gy))
                if path is None:
                    # Strict pass refuses UNKNOWN cells -- exactly the common
                    # case here (only the outbound trip's OWN path is
                    # confirmed-free, not the whole room), so retry once
                    # allowing them, per live_occupancy.py's own documented
                    # "route of last resort" calling convention.
                    path, info = self.occupancy.astar((x0, y0), (gx, gy), allow_unknown=True)
                if path and len(path) >= 2:
                    straight_m = math.hypot(gx - x0, gy - y0)
                    length_m = info.get("length_m", 0.0)
                    # Sanity check on the route itself, not just whether one
                    # was found: A* is only as good as the occupancy grid it
                    # searched, and that grid was carved from odometry poses
                    # over the WHOLE session (see LiveOccupancy.integrate) --
                    # if odometry had already drifted by the time an earlier
                    # leg carved a given region, free space gets baked in at
                    # the wrong world location relative to where the rover
                    # actually is NOW, and A* then returns a real, internally
                    # -consistent-with-the-map route that points somewhere
                    # genuinely wrong in the real world. Live-observed
                    # 2026-09-10: a recalled red bucket dead ahead via a
                    # direct ~6.4m line instead got an 18.1m/27-waypoint
                    # route (2.8x indirection) whose first leg turned the
                    # rover CCW/anticlockwise -- the user, looking at the
                    # actual room, confirmed the bucket was CW from here. A
                    # route this much longer than the straight line is
                    # itself the warning sign, independent of diagnosing
                    # WHY the map disagrees with reality this time -- refuse
                    # it and fall back to the straight-line target (still
                    # gx, gy directly, not a possibly-bad intermediate
                    # waypoint) rather than trusting an indirect route blind.
                    if straight_m > 0.5 and length_m > straight_m * a.goal_memory_max_route_indirection:
                        print(f"  [{label}] [memory] A* route ({length_m:.1f}m) is "
                              f"{length_m / straight_m:.1f}x the {straight_m:.1f}m straight-line "
                              f"distance -- too indirect to trust (occupancy grid likely stale/"
                              f"drifted relative to the current pose) -- falling back to a straight "
                              f"line at the remembered point instead")
                    else:
                        route = thin_waypoints(path, spacing=a.goal_memory_waypoint_spacing_m)
                        print(f"  [{label}] [memory] A* route: {len(route)} waypoints, "
                              f"{length_m:.1f}m over {info.get('cells', 0)} cells")
                else:
                    reason = info.get("reason", "no path") if isinstance(info, dict) else "no path"
                    print(f"  [{label}] [memory] A* found no route ({reason}) -- "
                          f"falling back to a straight line at the remembered point")
            except Exception as e:
                print(f"  [{label}] [memory] A* routing failed ({e}) -- falling back to a straight line")
        t_start = time.time()
        face_latch_sign: Optional[float] = None   # latched turn direction while facing a behind-us point -- see below
        last_assist_probe_t = 0.0   # throttle for the DINO assist probe below
        assist_period = 1.0 / a.goal_memory_dino_assist_hz if a.goal_memory_dino_assist_hz > 0 else 0.0
        # Stall watchdog -- ported from run_rover_occluded_recall.py's
        # _approach_point (the same A*-waypoint-following shape), which had
        # this from the start; this function never got the same treatment.
        # Live-observed 2026-09-10: TWO consecutive recalls in one session
        # (red bucket, air cooler) both got a real A* route, drove a few
        # tenths of a meter, then went to v=0/w=0 and sat there completely
        # frozen for the rest of --goal-memory-approach-timeout-s (30s) with
        # no recovery of any kind -- burning the full timeout every time and
        # requiring a manual abort. Whatever the underlying cause on a given
        # run (bad waypoint, obstacle-guard veto with nothing to route
        # around, momentarily lost camera frame), sitting motionless for the
        # rest of the window is never the right response -- bail to DINO
        # search as soon as real progress stops, instead of only on the
        # full timeout.
        stall_xy = (node.odom.x, node.odom.y)
        stall_t = t_start
        was_turning = False   # so the stall anchor resets the instant GOTO actually starts,
        # rather than inheriting however long the (legitimately stationary) face-turn just took
        goto_start_t: Optional[float] = None   # set the instant GOTO actually begins (see below) --
        # the DINO assist probe's grace period (its gate further down) is measured from here, not
        # from t_start, so a turn that takes a few seconds doesn't eat into the grace period meant
        # for the DRIVE phase.
        while not self._abort and time.time() - t_start < approach_timeout_s:
            now = time.time()
            x, y, th = node.odom.x, node.odom.y, node.odom.theta
            if math.hypot(gx - x, gy - y) < a.goal_memory_arrive_m:
                print(f"  [{label}] [memory] within {a.goal_memory_arrive_m:.1f}m of the "
                      f"remembered point -- memory exhausted, handing off to DINO")
                break
            # Steer toward the current route waypoint if A* gave one,
            # advancing once close enough -- the LAST waypoint should equal
            # (gx, gy) itself (thin_waypoints always keeps the final point),
            # so this converges on the same target the no-route path uses.
            tx, ty = gx, gy
            if route:
                while route_idx < len(route) - 1 and math.hypot(route[route_idx][0] - x,
                                                                 route[route_idx][1] - y) < a.goal_memory_waypoint_advance_m:
                    route_idx += 1
                tx, ty = route[route_idx]
            self._fill_depth_holes()   # so NavDP GOTO's obstacle guard sees monitor-shaped holes
            with node.lock:
                rgb, depth, intr = node.latest_rgb, node.latest_depth, node.latest_intrinsics
            if rgb is None:
                time.sleep(period)
                continue
            # Heading error to the current steering target -- computed here
            # (ahead of both the DINO assist probe below and the face-turn
            # branch further down) since the assist probe needs it as a
            # geometric gate.
            desired_th = math.atan2(ty - y, tx - x)
            face_err = wrap_angle(desired_th - th)
            # DINO assist: everything else in this loop is pure odometry
            # steering toward a point banked at arrival time -- nothing
            # corrects course if that point has drifted (dead reckoning over
            # the time since, plus the original arrival's own odometry
            # error) by more than --goal-memory-arrive-m, and until now this
            # leg only handed off on distance/timeout regardless of whether
            # the target was already back in view. Run a cheap detect_best()
            # probe (NOT a full pipe.step() -- that would run the whole
            # TRACK/AVOID/appearance-relock state machine meant for a live
            # search, and would fight this loop's own external_goal steering
            # for the publish_cmd this tick) at a slow, throttled cadence,
            # and hand off once it sees the real object rather than
            # continuing to trust a possibly-stale remembered point.
            #
            # Two guards before trusting a hit, both added after a
            # live-observed false trigger (2026-09-10, remembered bucket at
            # bearing=172deg): a bare detect_best() score of 0.38 -- barely
            # past the raw box_threshold detect_best() itself already
            # applied -- triggered an instant handoff at t=0, before ANY
            # turning had happened. CLIP re-id then rejected it (0.019) and
            # Qwen confirmed it was a frosted glass door, but only after the
            # deliberate turn-to-face was abandoned and ~15s were burned
            # SEARCHing from a heading pointed nowhere near the actual
            # target.
            #   (1) Geometric gate: skip the probe entirely while
            #       |face_err| is outside NavDP's own ~70deg forward cone
            #       (same bound the face-turn branch below uses). If the
            #       remembered point isn't roughly in front of us, the real
            #       target CANNOT be in frame -- a "detection" then is
            #       necessarily something else, full stop, no score would
            #       make it legitimate.
            #   (2) Confidence: a stricter score floor than the raw
            #       box_threshold (--goal-memory-dino-assist-min-score), AND
            #       -- when this recall banked a SigLIP2 view (--goal-
            #       memory-appearance) -- the SAME appearance cross-check
            #       go_to_object() runs on the first post-handoff detection,
            #       just run here BEFORE committing to hand off instead of
            #       after.
            #   (3) Grace period: even the ~70deg cone gate above let the
            #       probe fire the INSTANT face_err first dipped under 70deg
            #       -- which is BEFORE the face-turn branch below actually
            #       exits (that needs face_err under 25deg, hysteresis) --
            #       so DINO could hijack mid-turn, and the moment it exited
            #       cleanly, GOTO got zero time to actually drive before the
            #       next assist tick grabbed control. Live-observed
            #       2026-09-10: "after rotation the detection comes
            #       immediately" instead of driving toward the target for a
            #       while first. Fixed by gating on face_latch_sign being
            #       fully cleared (i.e. actually in GOTO, not just inside
            #       the cone) AND a minimum dwell in GOTO
            #       (--goal-memory-assist-grace-s) before the probe is even
            #       eligible -- let the blind drive make a real attempt,
            #       with the probe as a correction/early-exit once it's had
            #       a chance, not a replacement for it. goto_start_t itself
            #       is set below, past the face-turn branch -- NOT here:
            #       this point in the tick runs BEFORE that branch decides
            #       whether to actually turn, so setting it on face_latch_
            #       sign alone (without also checking face_err) could stamp
            #       a start time on a tick that's about to turn anyway.
            if (a.goal_memory_dino_assist and assist_period > 0
                    and now - last_assist_probe_t >= assist_period
                    and face_latch_sign is None
                    and goto_start_t is not None
                    and now - goto_start_t >= a.goal_memory_assist_grace_s
                    and getattr(node.pipe, "detector", None) is not None):
                last_assist_probe_t = now
                try:
                    assist_det = node.pipe.detector.detect_best(rgb, phrase)
                except Exception as e:
                    assist_det = None
                    print(f"  [{label}] [memory] DINO assist probe failed (ignoring): {e}")
                accept = assist_det is not None and assist_det.score >= a.goal_memory_dino_assist_min_score
                if accept and self._recalled_appearance_embedding is not None and self.appearance_embedder is not None:
                    sim = None
                    try:
                        emb = self.appearance_embedder.embed(rgb, assist_det.box)
                        if emb is not None:
                            sim = Siglip2Embedder.cosine(emb, self._recalled_appearance_embedding)
                    except Exception as e:
                        print(f"  [{label}] [memory] DINO assist appearance check failed (ignoring): {e}")
                    if sim is not None:
                        accept = sim >= a.goal_memory_appearance_min_sim
                        print(f"  [{label}] [memory] DINO assist candidate '{phrase}' (score "
                              f"{assist_det.score:.2f}) appearance sim={sim:.2f} (floor "
                              f"{a.goal_memory_appearance_min_sim:.2f}) -- "
                              f"{'accepted' if accept else 'rejected, staying blind'}")
                if accept:
                    print(f"  [{label}] [memory] DINO confirmed '{phrase}' (score {assist_det.score:.2f}) "
                          f"during blind approach, {math.hypot(gx - x, gy - y):.1f}m from the remembered "
                          f"point -- handing off early instead of continuing blind")
                    node.publish_cmd(0.0, 0.0)
                    if not self._abort and a.goal_memory_face_settle_s > 0:
                        time.sleep(a.goal_memory_face_settle_s)
                    node.odom._history.clear()
                    return True
            fwd, lat = body_delta(tx - x, ty - y, th)
            # NavDP's diffusion trajectories are forward-biased (trained on
            # ego-forward motion) -- fed a remembered point that's actually
            # BEHIND the rover (outside a ~70deg forward cone), none of the
            # sampled trajectories make real progress toward it, so
            # pipe.step's GOTO branch has no good option and the rover
            # visibly stalls/creeps sideways instead of turning around.
            # Live-observed 2026-09-09: a remembered air cooler at
            # fwd=-0.92m never got approached, just timed out. Fix: turn in
            # place to face the point first, THEN hand off to NavDP's
            # obstacle-aware GOTO once it's inside a cone NavDP can plan
            # through.
            #
            # This turn is a P-controller on the WRAPPED heading error to
            # the point (same mechanism as turn_in_place), NOT bang-bang on
            # atan2(lat, fwd). The old branch saturated |w| for the whole
            # state, so the only thing that varied was the SIGN of the
            # bearing -- and with the point near dead astern, odometry noise
            # flips that sign tick to tick, flipping the commanded turn
            # direction with it. Live-observed 2026-09-09: read as the rover
            # twitching/reversing mid-turn instead of committing to one way
            # around, until the spin-stall watchdog killed the leg (14.5rad
            # turned, 0.01m travelled, 0 DINO infers). Two guards here:
            #   (1) latch the turn DIRECTION on entry and hold it until the
            #       state is exited, so wrap_angle(+/-pi) noise can't flip it;
            #   (2) hysteresis -- enter past 70deg, only leave once well
            #       inside (25deg), so noise at the boundary can't chatter
            #       us in and out (each re-entry re-picks the direction).
            # (desired_th/face_err already computed above, ahead of the
            # DINO assist probe, which needs them as its geometric gate.)
            if face_latch_sign is not None and abs(face_err) < math.radians(25.0):
                face_latch_sign = None   # faced it -- fall through to NavDP GOTO
            if face_latch_sign is not None or abs(face_err) > math.radians(70.0):
                if face_latch_sign is None:
                    face_latch_sign = math.copysign(1.0, face_err if face_err != 0.0 else 1.0)
                w = face_latch_sign * min(abs(a.turn_kp * face_err), a.goal_memory_face_max_angular)
                node.publish_cmd(0.0, self._slew(w))
                self._integrate_occupancy()
                self._publish_status("object", f"{phrase} (memory approach: turning to face, "
                                               f"{math.degrees(face_err):+.0f}deg)")
                was_turning = True
                goto_start_t = None   # re-entered turning (e.g. a new route waypoint swung the
                # bearing back out) -- the NEXT GOTO phase earns its own fresh assist-probe grace
                # period rather than inheriting whatever's left of a much-earlier one.
                time.sleep(period)
                continue
            if was_turning or goto_start_t is None:
                # Covers both real transitions: just finished turning
                # (was_turning), and the never-needed-to-turn case (target
                # already inside the cone on tick 1, goto_start_t still
                # unset). Per user request 2026-09-10: pause a beat here
                # after an actual turn -- --goal-memory-turn-to-drive-
                # settle-s (default 1s) -- before driving, so a fresh frame
                # lands post-rotation instead of NavDP planning off a
                # still-blurred one from mid-spin.
                if was_turning and a.goal_memory_turn_to_drive_settle_s > 0:
                    node.publish_cmd(0.0, 0.0)
                    self._publish_status("object", f"{phrase} (memory approach: settling before drive)")
                    time.sleep(a.goal_memory_turn_to_drive_settle_s)
                    now = time.time()   # the settle pause itself must not look like a stall or
                    # eat into the assist probe's grace period below
                stall_xy, stall_t = (x, y), now   # GOTO just started -- give it its own fresh window
                goto_start_t = now
                was_turning = False
            # Stall watchdog -- ONLY once past the face-turn branch above
            # (which legitimately commands zero translation for as long as
            # a near-180deg turn takes, up to several seconds -- gating this
            # on GOTO ticks only avoids mistaking that deliberate stillness
            # for a stall). Ported from run_rover_occluded_recall.py's
            # _approach_point (same A*-waypoint-following shape), which had
            # this from the start; this function never got it. Live-observed
            # 2026-09-10: TWO consecutive recalls in one session (red
            # bucket, air cooler) each got a real A* route, drove a few
            # tenths of a meter, then sat at v=0/w=0 for the ENTIRE rest of
            # --goal-memory-approach-timeout-s (30s) with no recovery,
            # requiring a manual abort both times. Whatever the underlying
            # cause on a given run (bad waypoint, obstacle-guard veto with
            # nothing to route around, a momentarily lost camera frame),
            # sitting motionless for the rest of the window is never the
            # right response -- bail to DINO as soon as real progress stops.
            if math.hypot(x - stall_xy[0], y - stall_xy[1]) >= a.goal_memory_stall_min_move_m:
                stall_xy, stall_t = (x, y), now
            elif now - stall_t > a.goal_memory_stall_after_s:
                print(f"  [{label}] [memory] stalled ({a.goal_memory_stall_after_s:.0f}s with < "
                      f"{a.goal_memory_stall_min_move_m:.2f}m net movement), "
                      f"{math.hypot(gx - x, gy - y):.1f}m still from the remembered point -- "
                      f"giving up on the blind approach and handing off to DINO instead of waiting "
                      f"out the rest of the timeout")
                node.publish_cmd(0.0, 0.0)
                if not self._abort and a.goal_memory_face_settle_s > 0:
                    time.sleep(a.goal_memory_face_settle_s)
                node.odom._history.clear()
                return True
            try:
                gres = node.pipe.step(rgb, "", depth=depth, pose=(x, y, th), intrinsics=intr,
                                      external_goal=np.array([fwd, lat, 0.0], dtype=np.float32))
                node.publish_cmd(gres.linear, gres.angular)
            except Exception as e:
                print(f"  [{label}] [memory] approach step failed: {e} -- handing off to DINO")
                break
            self._integrate_occupancy()
            self._observe_entities()
            self._publish_status("object", f"{phrase} (memory approach)")
            time.sleep(period)
        else:
            print(f"  [{label}] [memory] approach timed out ({approach_timeout_s:.0f}s) -- handing off to DINO")
        node.publish_cmd(0.0, 0.0)
        # Hold still briefly before the DINO search loop takes over: (1) let
        # motion blur from the turn-around clear and a clean frame land
        # before the first re-detection attempt, and (2) reset the
        # spin-stall window -- object_leg() clears it at the START of the
        # leg, but this approach may have just spent up to
        # --goal-memory-approach-timeout-s turning in place, so without this
        # the watchdog can be pre-tripped the instant search begins (the
        # 2026-09-09 air-cooler leg died this way: 0 DINO infers, spin_stall
        # in step 0).
        if not self._abort and a.goal_memory_face_settle_s > 0:
            time.sleep(a.goal_memory_face_settle_s)
        node.odom._history.clear()
        return True

    # -- lifecycle ------------------------------------------------------
    def request_abort(self, *_):
        """OS signal handler (SIGINT/SIGTERM) -- always ends the process:
        aborts whatever plan is running (if any) AND tells a --serve loop to
        exit rather than wait for the next instruction. For an in-band
        'stop just this plan, keep the server up' request, see soft_abort()."""
        if self._abort:
            print("\n[abort] second interrupt -- exiting hard")
            self._safe_stop()
            os._exit(130)
        print("\n[abort] interrupt received -- stopping rover, finishing up")
        self._abort = True
        self._exit_requested = True

    def soft_abort(self):
        """Abort only the in-progress plan -- used by --serve's '__stop__'
        sentinel. Does NOT set _exit_requested, so the server loop goes
        straight back to waiting for the next instruction."""
        if not self._abort:
            print("\n[abort] soft-stop received -- ending the current plan, server stays up")
        self._abort = True

    def _safe_stop(self):
        try:
            self.node.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.node.publish_cmd(0.0, 0.0)
        except Exception:
            pass

    def wait_for_inputs(self, timeout_s: float) -> bool:
        """Block until a camera frame AND an IMU/rpm sample have arrived."""
        t0 = time.time()
        cam_ok = rpm_ok = False
        while time.time() - t0 < timeout_s and not self._abort:
            cam_ok = self.node._frame_count > 0
            rpm_ok = (self.node.odom.last_imu_heading_deg is not None
                      or self.node.odom.last_imu_calib is not None)
            if cam_ok and rpm_ok:
                print(f"[inputs] camera + /rover/rpm live after {time.time() - t0:.1f}s "
                      f"(imu calib=[{self.node.odom.decode_calib(self.node.odom.last_imu_calib)}])")
                return True
            time.sleep(0.5)
        print(f"[inputs] TIMEOUT after {timeout_s:.0f}s -- camera={'ok' if cam_ok else 'MISSING'} "
              f"rpm/imu={'ok' if rpm_ok else 'MISSING'}")
        return False

    def settle(self, seconds: float):
        if seconds > 0:
            print(f"  -- resting {seconds:.1f}s before the next subtask --")
        self.node.publish_cmd(0.0, 0.0)
        end = time.time() + max(0.0, seconds)
        while time.time() < end and not self._abort:
            self.node.publish_cmd(0.0, 0.0)
            time.sleep(0.1)

    # -- metric subtasks ---------------------------------------------------
    def drive_straight(self, signed_dist: float, tol: float, timeout: float, label: str) -> dict:
        a = self.args
        odom = self.node.odom
        odom.start_new_goal(label, reset_pose=False)
        self._begin_leg_path()
        time.sleep(0.2)
        x0, y0, th0 = odom.x, odom.y, odom.theta
        gx = x0 + signed_dist * math.cos(th0)
        gy = y0 + signed_dist * math.sin(th0)
        period = 1.0 / a.metric_hz
        hold = 0
        t_start = time.time()
        outcome = "ok"
        print(f"  [{label}] target {signed_dist:+.2f} m  from ({x0:+.2f},{y0:+.2f},{math.degrees(th0):+.0f}deg)")
        while not self._abort:
            x, y, th = odom.x, odom.y, odom.theta
            fwd, lat = body_delta(gx - x, gy - y, th)
            remaining = fwd
            self._publish_status("move", f"{signed_dist:+.2f}m (remaining {remaining:+.2f}m)")
            if self.args.fact3r_observe_during_metric:
                self._fill_depth_holes(min_period_s=1.5)   # for fact3r mapping only (move legs are vision-free)
            self._integrate_occupancy()
            self._observe_entities(during_metric=True)
            if abs(remaining) < tol:
                hold += 1
                self.node.publish_cmd(0.0, 0.0)
                if hold >= 3:
                    break
                if time.time() - t_start > timeout:
                    outcome = "timeout"
                    break
                time.sleep(period)
                continue
            hold = 0
            v = clamp(a.move_kp * remaining, -a.max_linear, a.max_linear)
            if 0.0 < abs(v) < a.move_v_floor:
                v = math.copysign(a.move_v_floor, v)
            # steer: chase the goal point going forward, hold launch heading in reverse
            aim = math.atan2(gy - y, gx - x) if remaining > 0 else th0
            w = clamp(a.move_hold_kp * wrap_angle(aim - th), -a.metric_max_angular, a.metric_max_angular)
            w = self._slew(w)
            self.node.publish_cmd(v, w)
            if time.time() - t_start > timeout:
                outcome = "timeout"
                break
            time.sleep(period)
        self.node.publish_cmd(0.0, 0.0)
        self._last_w = 0.0
        x, y, th = odom.x, odom.y, odom.theta
        travelled, _ = body_delta(x - x0, y - y0, th0)
        res = dict(label=label, kind="move", target_m=signed_dist, travelled_m=round(travelled, 3),
                   error_m=round(abs(travelled - signed_dist), 3), path_length_m=self._end_leg_path(),
                   odometry_csv=self._leg_odometry_csv(),
                   start=[round(x0, 3), round(y0, 3), round(th0, 4)],
                   end=[round(x, 3), round(y, 3), round(th, 4)],
                   dur_s=round(time.time() - t_start, 1), theta_src_final=odom.theta_source,
                   outcome="abort" if self._abort else outcome)
        print(f"  [{label}] done: travelled {travelled:+.2f} m (err {res['error_m']:.2f} m) [{res['outcome']}]")
        return res

    def turn_in_place(self, signed_deg: float, tol_deg: float, timeout: float, label: str) -> dict:
        a = self.args
        odom = self.node.odom

        # gate on magnetometer calibration -- an uncalibrated-mag turn drifts
        t0 = time.time()
        last_msg = 0.0
        while not odom.is_imu_calibrated():
            if self._abort:
                return dict(label=label, kind="turn", target_deg=signed_deg, outcome="abort")
            if time.time() - t0 > a.imu_calib_wait_s:
                if a.imu_calib_timeout == "encoder-only":
                    # The magnetometer isn't coming up (chronic indoors). The
                    # fused rotation vector / gyro is fine for a short RELATIVE
                    # turn -- _imu_theta rejects the uncalibrated MAG anyway, so
                    # theta free-runs on wheel-diff + gyro dead reckoning
                    # (theta_src=enc) whether or not we flip the flag; flip it
                    # so the turn stops printing "waiting" and just goes.
                    print(f"  [{label}] MAG still {odom.decode_calib(odom.last_imu_calib)} after "
                          f"{a.imu_calib_wait_s:.0f}s -- proceeding on encoder/gyro dead reckoning "
                          f"(--imu-calib-timeout=encoder-only; expect ~2-4 deg turn error)")
                    odom.encoder_only = True
                    odom.auto_detect_frozen_imu = False
                    break
                print(f"  [{label}] IMU never reached mag-calib >= {odom.imu_min_mag_calib} in "
                      f"{a.imu_calib_wait_s:.0f}s -- ABORTING THIS SUBTASK (calib=[{odom.decode_calib(odom.last_imu_calib)}])")
                self._safe_stop()
                self._abort = True
                return dict(label=label, kind="turn", target_deg=signed_deg, outcome="imu_uncalibrated")
            if time.time() - last_msg > 5.0:
                print(f"  [{label}] waiting for IMU mag calibration "
                      f"(calib=[{odom.decode_calib(odom.last_imu_calib)}]) ... "
                      f"drive/rotate the rover by hand or wait")
                last_msg = time.time()
            time.sleep(0.5)

        odom.start_new_goal(label, reset_pose=False)
        self._begin_leg_path()
        time.sleep(0.2)
        th0 = odom.theta
        target = wrap_angle(th0 + math.radians(signed_deg))
        period = 1.0 / a.metric_hz
        enc_ticks = tot_ticks = 0
        t_start = time.time()
        outcome = "ok"
        print(f"  [{label}] target {signed_deg:+.0f} deg  from {math.degrees(th0):+.0f} deg "
              f"(theta_src={odom.theta_source})")
        while not self._abort:
            th = odom.theta
            err = wrap_angle(target - th)
            tot_ticks += 1
            if odom.theta_source != "imu":
                enc_ticks += 1
            self._publish_status("turn", f"{signed_deg:+.0f}deg (remaining {math.degrees(err):+.0f}deg)")
            # NOT integrating occupancy here on purpose: a depth frame can be
            # up to DEPTH_STALE_S old relative to the pose it's paired with,
            # and while actively rotating that lag turns into real angular
            # smearing -- carved rays land at the wrong bearing and scatter
            # speckle across the grid. Confirmed as the dominant noise source
            # in the BEV panel. Moves and object legs (near-zero angular
            # velocity, or NavDP's own much slower turns) don't have this
            # problem badly enough to need the same gate.
            if abs(err) < math.radians(tol_deg):
                # Stop on the FIRST in-tolerance tick -- don't hold/re-verify
                # and re-enter the w=... branch below if drift nudges it back
                # out during a settle wait. Per-user request: a rough turn
                # that stops promptly is preferred over one that keeps
                # correcting to nail the angle; some error is fine.
                self.node.publish_cmd(0.0, 0.0)
                break
            w = clamp(a.turn_kp * err, -a.turn_max_angular, a.turn_max_angular)
            if abs(w) < a.turn_w_floor:
                w = math.copysign(a.turn_w_floor, w)
            w = self._slew(w)
            self.node.publish_cmd(0.0, w)
            # map objects the rover rotates past (its own yaw-rate gate drops
            # any tick where the spin is still too fast to trust the pose)
            self._observe_entities(during_turn=True)
            if time.time() - t_start > timeout:
                outcome = "timeout"
                break
            time.sleep(period)
        self.node.publish_cmd(0.0, 0.0)
        self._last_w = 0.0
        thf = odom.theta
        turned = math.degrees(wrap_angle(thf - th0))
        res = dict(label=label, kind="turn", target_deg=signed_deg, turned_deg=round(turned, 1),
                   error_deg=round(math.degrees(wrap_angle(target - thf)), 1),
                   path_length_m=self._end_leg_path(), odometry_csv=self._leg_odometry_csv(),
                   enc_fallback_frac=round(enc_ticks / max(tot_ticks, 1), 2),
                   start_theta_deg=round(math.degrees(th0), 1), end_theta_deg=round(math.degrees(thf), 1),
                   dur_s=round(time.time() - t_start, 1), theta_src_final=odom.theta_source,
                   outcome="abort" if self._abort else outcome)
        flag = "  <-- heading rode encoder, low confidence" if res["enc_fallback_frac"] > 0.3 else ""
        print(f"  [{label}] done: turned {turned:+.1f} deg (err {res['error_deg']:+.1f} deg) [{res['outcome']}]{flag}")
        return res

    def _slew(self, w: float) -> float:
        lim = self.args.angular_slew_max
        if lim > 0:
            w = clamp(w, self._last_w - lim, self._last_w + lim)
        self._last_w = w
        return w

    # -- object subtask -------------------------------------------------
    def go_to_object(self, phrase: str, timeout: float, label: str,
                     raw_clause: Optional[str] = None) -> dict:
        node = self.node
        a = self.args
        raw_clause = raw_clause or phrase
        self._dbg(f"object leg {label}: raw clause = \"{raw_clause}\"  | regex-stripped seed phrase = \"{phrase}\"")
        # Qwen reviews the RAW instruction clause (not the regex-cleaned
        # `phrase`) and decides the actual noun phrase Grounding-DINO should
        # search for -- the regex stripper (_strip_goto_prefix) only exists
        # as the seed/fallback below when no supervisor is loaded at all
        # (--no-supervisor); whenever Qwen IS available, it -- not a hand-
        # written pattern -- is what the raw instruction is handed to.
        if self.supervisor is not None:
            self._dbg(f"calling supervisor.check(frame, clause=\"{raw_clause}\", seed=\"{phrase}\", "
                      f"state='not started yet') to pick the DINO target phrase from the frame")
            with node.lock:
                rgb0 = node.latest_rgb
            if rgb0 is not None:
                v0 = self.supervisor.check(rgb0, raw_clause, phrase,
                                           "not started yet -- no detection attempted")
                self._dbg(f"supervisor.check -> {v0}")
                if v0 and v0.get("target"):
                    cand = str(v0["target"]).strip()
                    if cand and cand.lower() not in ("...", "none", "n/a", ""):
                        print(f'  [{label}] Qwen interprets "{raw_clause}" -> target \'{cand}\''
                              + (f'  (reason: {v0.get("reason", "")[:120]})' if a.debug_steps else ''))
                        phrase = cand
            else:
                self._dbg("no camera frame yet -- keeping the regex seed phrase")
        else:
            self._dbg("no supervisor loaded -- DINO target phrase = the regex-stripped seed (no VLM interpretation)")
        node.target = phrase
        self._dbg(f"==> DINO will search for: \"{phrase}\"   (this is Detection.label's source; "
                  f"the box label you see is DINO's matched span of THIS phrase)")
        node._goal_reached = False
        node._stop_streak = 0
        node.last_result = None
        self._last_memory_event = None   # don't let a previous leg's recall notice linger
        node.pipe.reset()
        # Do NOT inherit the previous object leg's spin-delta window -- otherwise
        # a leg that ended in the spin-stall watchdog leaves the next leg
        # tripping it on tick 1 (observed: chair/door legs DOA, steps=0).
        node.odom._history.clear()
        node.odom.start_new_goal(label, reset_pose=False)
        self._begin_leg_path()
        node.pub_explain.put(serialize_string(f"SUBTASK go-to: {phrase}"))
        infer0 = node._infer_count
        period = 1.0 / a.predict_hz
        t_start = time.time()
        last_status = t_start
        last_sup = 0.0
        last_det_t = 0.0          # wall-clock of the last tick DINO had a detection
        last_goal_dist = None     # dist-to-goal on the last tick DINO had a lock
        min_goal_dist = None      # closest DINO ever brought us
        last_approach_t = 0.0     # last time the supervisor said approaching/arrived
        approach_streak = 0
        last_action = "forward"
        last_action_t = 0.0
        scan_sign = -1.0
        creep_anchor = None       # (x, y) where the current creep started
        sup_log: List[dict] = []
        last_sup_v: Optional[dict] = None   # most recent supervisor verdict (for the status feed)
        arrived_by = None
        print(f'  [{label}] DINO+NavDP -> "{phrase}"'
              f'{" | Qwen supervisor/navigator" if self.supervisor is not None else ""}  (timeout {timeout:.0f}s)')

        # fact3r live entity-map recall FIRST: an object seen in passing on an
        # earlier leg (never navigated to, so no goal_memory JSONL record) can
        # still be recalled here as a stored position and driven to blind. If
        # that gets us close, skip the JSONL recall below; the DINO+supervisor
        # loop still makes the confirmed approach either way.
        used_fact3r = False
        self._dbg(f"MEMORY CHECK 1/2 -- fact3r live entity map (SAM2+SigLIP2): "
                  f"enabled={a.fact3r_recall}  map_loaded={self.fact3r_mem is not None}  "
                  f"entities={len(self.fact3r_mem.entities) if self.fact3r_mem is not None else 0}")
        if a.fact3r_recall and self.fact3r_mem is not None and not self._abort:
            self._dbg(f"calling _fact3r_recall(\"{phrase}\") -> merge_nearby_duplicates + "
                      f"{'query_verified (SigLIP shortlist -> Qwen crop verify)' if (a.fact3r_recall_verify and self.supervisor) else 'query (SigLIP text match only)'}")
            _fxy = self._fact3r_recall(phrase, label)
            if _fxy is not None:
                self._dbg(f"fact3r HIT at {tuple(round(v,2) for v in _fxy)} -- driving there BLIND "
                          f"(NavDP GOTO) before any DINO search; DINO still confirms at the end")
                _left = self._drive_to_point(_fxy[0], _fxy[1], phrase, label,
                                             a.goal_memory_approach_timeout_s)
                used_fact3r = _left <= a.goal_memory_arrive_m * 2.0
                print(f"  [{label}] [fact3r] approach ended {_left:.1f} m from the stored "
                      f"position ({'good enough -- DINO takes over' if used_fact3r else 'still far -- trying JSONL recall / DINO'})")
            else:
                self._dbg("fact3r MISS (no entity accepted) -- falling through to JSONL goal-memory")
        _try_jsonl = a.goal_memory_approach and not used_fact3r
        _jsonl_note = (f"querying _recall_and_approach('{phrase}')" if _try_jsonl
                       else "skipped (fact3r already got us there)")
        self._dbg(f"MEMORY CHECK 2/2 -- JSONL goal-memory (past confirmed DINO arrivals): {_jsonl_note}")
        used_memory = (self._recall_and_approach(phrase, label)
                       if _try_jsonl else False)
        if not used_fact3r and not used_memory:
            self._dbg("NO memory hit from either store -> DIRECT DETECTION: Grounding-DINO "
                      f"detect('{phrase}') + Qwen supervisor loop starts now")
        # Consume once -- _recall_and_approach() re-sets this fresh at the
        # start of the NEXT object leg's own call, but clear it here too so
        # nothing leaks if this leg's supervisor retargets DINO onto a
        # DIFFERENT phrase mid-leg (see the retarget branch below): the
        # remembered embedding was for the ORIGINAL recalled phrase only.
        recalled_embedding = self._recalled_appearance_embedding
        self._recalled_appearance_embedding = None
        recalled_world_xy = self._recalled_world_xy
        self._recalled_world_xy = None
        last_det_box = None       # last DINO box seen, for _remember_arrival's appearance embed
        appearance_sim: Optional[float] = None      # set once, first live re-detection after recall
        appearance_checked = False
        geo_checked = False       # set once, first live re-detection after recall (see below)

        while not self._abort and not node._goal_reached:
            self._fill_depth_holes()   # before _tick so the guard sees monitor-shaped holes
            node._tick()
            now = time.time()
            res = node.last_result
            det = getattr(res, "detection", None) if res is not None else None
            st = getattr(res, "state", "?") if res is not None else "?"
            gp = getattr(res, "goal_point", None) if res is not None else None
            self._publish_status(
                "object", phrase,
                detector=dict(
                    state=st, target=node.target, detected=det is not None,
                    score=(round(float(det.score), 3) if det is not None else None),
                    box=(np.asarray(det.box, dtype=float).round(1).tolist() if det is not None else None),
                    goal_point=(np.asarray(gp, dtype=float).round(3).tolist() if gp is not None else None),
                    ambiguous=bool(getattr(res, "ambiguous", False)) if res is not None else False,
                ),
                supervisor=last_sup_v)
            self._integrate_occupancy()
            self._observe_entities()
            if det is not None:
                if a.debug_steps and last_det_t == 0.0:
                    _b = np.asarray(det.box, dtype=float).round(0).tolist()
                    self._dbg(f"Grounding-DINO FIRST LOCK for '{node.target}': "
                              f"label='{getattr(det, 'label', '?')}' score={getattr(det, 'score', 0):.2f} "
                              f"box={_b}  (this label is DINO's matched span of the phrase '{node.target}', "
                              f"NOT a fact3r/SAM annotation)")
                last_det_t = now
                last_det_box = det.box
                creep_anchor = None
                if gp is not None:
                    last_goal_dist = float((gp[0] ** 2 + gp[1] ** 2) ** 0.5)
                    min_goal_dist = last_goal_dist if min_goal_dist is None else min(min_goal_dist, last_goal_dist)
                # fact3r-map entity-memory integration: the FIRST live
                # re-detection after a memory-recalled approach gets checked
                # against the banked SigLIP2 view -- once, not every tick
                # (an embed+compare call per tick would compete with
                # predict_hz for GPU time for no added benefit; appearance
                # doesn't change tick to tick). Feeds into the supervisor's
                # dstate context below rather than silently vetoing anything
                # -- the DINO box + supervisor judgment still have the final
                # say, this is a sanity signal, not a hard gate.
                if recalled_embedding is not None and not appearance_checked:
                    appearance_checked = True
                    with node.lock:
                        rgb_now = node.latest_rgb
                    if rgb_now is not None:
                        try:
                            emb = self.appearance_embedder.embed(rgb_now, det.box)
                            if emb is not None:
                                appearance_sim = Siglip2Embedder.cosine(emb, recalled_embedding)
                                flag = "" if appearance_sim >= a.goal_memory_appearance_min_sim else \
                                    "  ** LOW -- this may be a DIFFERENT object than the one remembered **"
                                print(f"  [{label}] [memory] appearance check vs. recalled '{phrase}': "
                                      f"sim={appearance_sim:.2f} (floor {a.goal_memory_appearance_min_sim:.2f}){flag}")
                        except Exception as e:
                            print(f"  [{label}] [memory] appearance check failed (ignoring): {e}")
                        # Locate-stage step 3, live-adapted (see TaskSupervisor.
                        # verify_recall's docstring): SigLIP similarity alone is
                        # a coarse gate (same category, different instance can
                        # still score reasonably high) -- the SAME Qwen model
                        # already supervising this leg's navigation also judges
                        # the crop against the remembered phrase. A confident
                        # "no" forces DINO to drop this lock and re-search
                        # instead of quietly approaching the wrong object,
                        # mirroring the offline pipeline's accept-threshold
                        # philosophy (decision==yes, confidence>=floor) rather
                        # than only logging a warning.
                        if self.supervisor is not None:
                            try:
                                b = np.asarray(det.box, dtype=int)
                                pad = 12
                                y0, y1 = max(b[1] - pad, 0), min(b[3] + pad, rgb_now.shape[0])
                                x0, x1 = max(b[0] - pad, 0), min(b[2] + pad, rgb_now.shape[1])
                                vcrop = rgb_now[y0:y1, x0:x1].copy() if y1 > y0 and x1 > x0 else None
                            except Exception:
                                vcrop = None
                            verdict = self.supervisor.verify_recall(
                                rgb_now, vcrop, phrase, self._recalled_hit_age_s or 0.0)
                            if verdict is not None:
                                print(f"  [{label}] [memory] Qwen recall verification: "
                                      f"{verdict['decision']} (confidence {verdict['confidence']:.2f}) "
                                      f"-- {verdict.get('reason', '')[:80]}")
                                if (verdict["decision"] == "no"
                                        and verdict["confidence"] >= a.goal_memory_min_vlm_confidence):
                                    print(f"  [{label}] [memory] Qwen rejects the recalled match -- "
                                          f"dropping this DINO lock and searching fresh")
                                    node.pipe.reset()
                                    node._stop_streak = 0
                                    last_det_t = 0.0   # so close-loss/approach-lost logic doesn't
                                    last_goal_dist = None   # trust the just-rejected lock's history
                                    min_goal_dist = None
                # Geo-consistency check -- independent of the appearance
                # check above (runs even with no banked SigLIP2 view, and
                # even when the appearance check above already ran/rejected
                # something): NavDP's own goal_point for the CURRENT live
                # detection is a real 3D point (depth-derived), not just a
                # 2D box -- project it into world coordinates and compare
                # against the remembered goal_memory coordinate. Per user
                # request 2026-09-10: DINO can lock onto a same-category but
                # physically DIFFERENT object (a second red bucket across
                # the room, say) that still passes the appearance/Qwen
                # checks above (same category, so a plausible crop) while
                # NavDP's own goal point is nowhere near where memory says
                # the real one should be -- geometry catches that case
                # appearance alone can't. Once, not every tick (same
                # reasoning as the appearance check above); action on
                # mismatch is the SAME confirm-then-correct idea, just
                # steering by memory instead of dropping the lock: trust
                # world_xy over a detection that's geometrically
                # inconsistent with it, for this tick.
                if recalled_world_xy is not None and not geo_checked and gp is not None:
                    geo_checked = True
                    wx, wy = body_to_world(float(gp[0]), float(gp[1]),
                                           node.odom.x, node.odom.y, node.odom.theta)
                    geo_dist = math.hypot(wx - recalled_world_xy[0], wy - recalled_world_xy[1])
                    mismatch = geo_dist > a.goal_memory_geo_mismatch_m
                    flag = "  ** MISMATCH -- steering by memory coordinate this tick instead **" if mismatch else ""
                    print(f"  [{label}] [memory] geo-consistency check: NavDP's goal point projects to "
                          f"({wx:+.2f},{wy:+.2f}), {geo_dist:.1f}m from the remembered "
                          f"({recalled_world_xy[0]:+.2f},{recalled_world_xy[1]:+.2f}) "
                          f"(floor {a.goal_memory_geo_mismatch_m:.1f}m){flag}")
                    if mismatch:
                        try:
                            with node.lock:
                                rgb_ov, depth_ov, intr_ov = node.latest_rgb, node.latest_depth, node.latest_intrinsics
                            if rgb_ov is not None:
                                fwd_ov, lat_ov = body_delta(recalled_world_xy[0] - node.odom.x,
                                                            recalled_world_xy[1] - node.odom.y, node.odom.theta)
                                ov = node.pipe.step(rgb_ov, "", depth=depth_ov, pose=(node.odom.x, node.odom.y, node.odom.theta),
                                                    intrinsics=intr_ov,
                                                    external_goal=np.array([fwd_ov, lat_ov, 0.0], dtype=np.float32))
                                node.publish_cmd(ov.linear, ov.angular)   # overrides node._tick()'s own
                                # publish above, which was computed off the now-distrusted detection
                        except Exception as e:
                            print(f"  [{label}] [memory] geo-mismatch override step failed (ignoring): {e}")

            # -- ARRIVAL-ON-CLOSE-LOSS: DINO lost the lock while it was close
            #    (box fills the frame when you're on top of the object). Do NOT
            #    creep past it. Fire immediately, every tick, not on the 4 s
            #    supervisor cadence.
            ready = now - t_start >= a.supervisor_min_arrive_s
            if (ready and det is None and 0.4 < now - last_det_t < 4.0
                    and last_goal_dist is not None and last_goal_dist <= a.close_arrive_m):
                print(f"  [{label}] DINO lost the lock at {last_goal_dist:.2f} m -- on top of the "
                      f"target, calling it arrived")
                arrived_by = "close_loss"
                break

            # -- lost-lock recovery: creep + supervisor-hinted turn instead of
            #    the pipeline's blind in-place SEARCH spin. BOUNDED: never creep
            #    more than search_creep_max_m without re-acquiring, or the rover
            #    just drives off across the room (observed: 10 m past the cooler).
            if (not getattr(a, "object_search", True) and st in ("SEARCH", "?")
                    and not node._goal_reached):
                # detection-only mode: no creep, no scan. Hold still; if DINO
                # stays lost past the grace window (measured from the last
                # lock, or from leg start if it never locked), end the leg.
                node.publish_cmd(0.0, 0.0)
                lost_for = now - (last_det_t if last_det_t > 0.0 else t_start)
                if lost_for > a.search_off_grace_s:
                    if min_goal_dist is not None and min_goal_dist <= a.close_arrive_m + 0.6:
                        print(f"  [{label}] search OFF: DINO lost {lost_for:.0f}s "
                              f"(had it to {min_goal_dist:.2f} m earlier) -- calling it arrived")
                        arrived_by = "close_loss"
                    else:
                        print(f"  [{label}] search OFF: DINO never (re)acquired '{node.target}' "
                              f"({min_goal_dist} m closest) -- ending leg")
                        arrived_by = "creep_exhausted"
                    break
            elif (self.supervisor is not None and st in ("SEARCH", "?")
                    and now - last_det_t > 1.0 and not node._goal_reached):
                if creep_anchor is None:
                    creep_anchor = (node.odom.x, node.odom.y)
                crept = math.hypot(node.odom.x - creep_anchor[0], node.odom.y - creep_anchor[1])
                if crept >= a.search_creep_max_m:
                    node.publish_cmd(0.0, 0.0)
                    if min_goal_dist is not None and min_goal_dist <= a.close_arrive_m + 0.6:
                        print(f"  [{label}] crept {crept:.1f} m with no re-acquire; DINO had it to "
                              f"{min_goal_dist:.2f} m earlier -- calling it arrived")
                        arrived_by = "close_loss"
                    else:
                        print(f"  [{label}] crept {crept:.1f} m with no re-acquire and never got close "
                              f"({min_goal_dist} m) -- giving up on this leg")
                        arrived_by = "creep_exhausted"
                    break
                if last_action == "left":
                    aw = a.search_turn
                elif last_action == "right":
                    aw = -a.search_turn
                elif last_action == "stop":
                    aw = 0.0
                else:
                    if now - last_action_t > 6.0:
                        scan_sign = -scan_sign
                        last_action_t = now
                    aw = a.search_turn * 0.5 * scan_sign
                node.publish_cmd(a.search_creep_v, self._slew(aw))

            # -- Qwen task supervisor (throttled) --------------------------
            if self.supervisor is not None and now - last_sup >= a.supervisor_period_s:
                last_sup = now
                with node.lock:
                    rgb = node.latest_rgb
                score = f"{det.score:.2f}" if det is not None else "none"
                d2g = (f"{float((res.goal_point[0] ** 2 + res.goal_point[1] ** 2) ** 0.5):.2f}m"
                       if res is not None and getattr(res, "goal_point", None) is not None else "n/a")
                amb = getattr(res, "ambiguous", False) if res is not None else False
                dstate = (f"detector_state={st} detection_conf={score} dist_to_target={d2g} "
                          f"stop_streak={node._stop_streak}/{node.stop_confirm_count}"
                          f"{' AMBIGUOUS(2+ matches, needs a distinguishing phrase)' if amb else ''} "
                          f"elapsed={now - t_start:.0f}s")
                if appearance_sim is not None:
                    ok = appearance_sim >= a.goal_memory_appearance_min_sim
                    dstate += (f" remembered_appearance_sim={appearance_sim:.2f}"
                               f"({'matches' if ok else 'MISMATCH -- possibly a different object'})")
                # NavDP's currently-chosen trajectory endpoint (see
                # DinoNavDPPipeline.get_recent_frames' sibling: this is
                # NUMERIC NavDP context, cheap to add as text vs. an image).
                # chosen[:, 0]=forward, [:, 1]=left, body frame at this tick.
                traj = getattr(res, "trajectory", None) if res is not None else None
                if traj is not None and len(traj) > 0:
                    tf, tl = float(traj[-1, 0]), float(traj[-1, 1])
                    dstate += f" navdp_planned_path=({tf:+.1f}m fwd, {tl:+.1f}m left)"
                # Rolling text history of the supervisor's OWN past few
                # verdicts this leg -- each check() call is otherwise
                # completely stateless (a fresh judgment from one snapshot),
                # so without this Qwen has no way to know "I've said
                # searching 3 times in a row" vs. "I just started".
                if sup_log:
                    recent = "; ".join(f"t={s['t']:.0f}s:{s['status']}" for s in sup_log[-3:])
                    dstate += f" recent_supervisor_history=[{recent}]"
                history_frames = (node.pipe.get_recent_frames(n=a.supervisor_history_frames)
                                  if a.supervisor_history_frames > 0 else None)
                crop = None
                if det is not None and rgb is not None:
                    try:
                        b = np.asarray(det.box, dtype=int)
                        pad = 12
                        y0, y1 = max(b[1] - pad, 0), min(b[3] + pad, rgb.shape[0])
                        x0, x1 = max(b[0] - pad, 0), min(b[2] + pad, rgb.shape[1])
                        if y1 > y0 and x1 > x0:
                            crop = rgb[y0:y1, x0:x1].copy()
                    except Exception:
                        crop = None
                if rgb is not None:
                    v = self.supervisor.check(rgb, phrase, node.target, dstate, crop=crop,
                                              history_frames=history_frames)
                    if v is not None:
                        v["t"] = round(now - t_start, 1)
                        sup_log.append(v)
                        last_sup_v = v
                        last_action = v.get("action", "forward")
                        last_action_t = now
                        print(f"  [{label}] [supervisor t={v['t']:.0f}s] {v['status']} act={last_action} "
                              f"target='{v['target']}' -- {v.get('reason', '')[:80]}")
                        if v["status"] in ("approaching", "arrived"):
                            approach_streak += 1
                            last_approach_t = now
                        else:
                            approach_streak = 0
                        # 1) explicit 'arrived', DINO saw the target reasonably recently
                        if v["status"] == "arrived" and ready and now - last_det_t < 8.0:
                            arrived_by = "supervisor"
                            break
                        # 2) was approaching within the last 10 s and DINO has now
                        #    lost the lock -- trust the approach history (the
                        #    'searching' verdict that follows a close loss must NOT
                        #    erase this the way a reset streak did). BUT only if
                        #    DINO actually got close at some point -- this check
                        #    used to be time-only (no distance gate), which let it
                        #    declare "arrived" 3-4+ m out just because the
                        #    supervisor said "approaching" a few ticks earlier and
                        #    then DINO lost grounding -- the rover visibly stopped
                        #    mid-approach, short of the real object, every time.
                        # Was `max(close_arrive_m*2.0, 3.0)` (3.0 m by default)
                        # -- live-observed calling a *3.0 m-out* DINO loss
                        # "arrived", stopping the rover 2+ m short of the real
                        # object every time the supervisor said "approaching"
                        # a few ticks earlier. Tightened to match the sibling
                        # close-loss gate right above (close_arrive_m + 0.6,
                        # ~2.1 m by default) -- still some slack beyond the
                        # hard close_arrive_m radius for the box-fills-frame
                        # case, not a 3 m "close enough".
                        close_enough = (min_goal_dist is not None
                                       and min_goal_dist <= a.close_arrive_m + 0.6)
                        if (ready and now - last_approach_t < 10.0 and st in ("SEARCH", "?")
                                and now - last_det_t > 3.0):
                            if close_enough:
                                print(f"  [{label}] was approaching (closest {min_goal_dist:.2f}m), "
                                      f"DINO lost the lock -- calling it arrived")
                                arrived_by = "supervisor_approach_lost"
                                break
                            else:
                                cd = f"{min_goal_dist:.2f}m" if min_goal_dist is not None else "n/a"
                                print(f"  [{label}] was approaching but never got close (closest "
                                      f"{cd}) -- NOT calling it arrived, continuing to search")
                        # retarget ONLY when DINO is genuinely lost or ambiguous --
                        # never yank the phrase while it is tracking cleanly
                        lost = now - last_det_t > a.supervisor_retarget_lost_s
                        # `lost` alone is a WEAK signal to retarget on: not seeing
                        # the target for supervisor_retarget_lost_s (default 6s)
                        # is completely normal for the first stretch of an
                        # ordinary SEARCH -- the object may just not be in frame
                        # yet, exactly like a real "go to fan" leg where the fan
                        # never was in view to begin with. Live-observed
                        # 2026-09-09: with nothing else gating it, Qwen (seeing
                        # no fan, something else salient instead) proposed
                        # 'door' as the new target, and the WHOLE leg drove
                        # there instead -- for an object the user never asked
                        # for. `amb` needs no such gate (DINO itself already
                        # found 2+ candidates OF THE SAME CLASS as `phrase`,
                        # Qwen is only disambiguating among them). For `lost`,
                        # require the proposed target to share a word with the
                        # ORIGINAL requested phrase (not node.target, which may
                        # have already drifted from an earlier retarget -- this
                        # is the anti-drift anchor, see _shares_key_word) --
                        # "white air cooler" -> "air cooler near window" passes,
                        # "fan" -> "door" does not.
                        on_topic = amb or _shares_key_word(phrase, v["target"])
                        if (v["target"] and v["target"] != node.target and len(v["target"]) <= 40
                                and (lost or amb) and on_topic):
                            print(f"  [{label}] supervisor retargets DINO ({'lost' if lost else 'ambiguous'}): "
                                  f"'{node.target}' -> '{v['target']}'")
                            self._dbg(f"supervisor RETARGET: node.target '{node.target}' -> '{v['target']}' "
                                      f"(trigger={'lost' if lost else 'ambiguous'}, on_topic vs requested "
                                      f"'{phrase}'={on_topic}, reason=\"{v.get('reason', '')[:120]}\") "
                                      f"-- DINO now searches for the NEW phrase")
                            node.target = v["target"]
                            node.pipe.reset()
                            node._stop_streak = 0
                        elif v["target"] and v["target"] != node.target and lost and not on_topic:
                            print(f"  [{label}] supervisor suggested '{v['target']}' while lost, but it "
                                  f"doesn't share anything with the requested '{phrase}' -- ignoring, "
                                  f"keeping DINO searching for '{node.target}'")
                        elif v["status"] == "wrong_object" and now - t_start >= a.supervisor_min_arrive_s:
                            print(f"  [{label}] supervisor: wrong lock -- resetting DINO acquisition")
                            node.pipe.reset()
                            node._stop_streak = 0

            if now - t_start > timeout:
                break
            if now - last_status > 10.0:
                print(f"  [{label}] .. t={now - t_start:4.0f}s pose=({node.odom.x:+.2f},{node.odom.y:+.2f},"
                      f"{math.degrees(node.odom.theta):+.0f}deg) det={'y' if det is not None else 'n'} "
                      f"frames={node._frame_count} infers={node._infer_count - infer0}")
                last_status = now
            time.sleep(period)

        node.publish_cmd(0.0, 0.0)
        real_stop = node._goal_reached and node._stop_streak >= node.stop_confirm_count
        if arrived_by == "supervisor":
            outcome = "reached_supervisor"
        elif arrived_by == "supervisor_approach_lost":
            outcome = "reached_supervisor_approx"
        elif arrived_by == "close_loss":
            outcome = "reached_close_loss"     # DINO lost the lock on top of the target
        elif arrived_by == "creep_exhausted":
            outcome = "lost"                   # never got close, stopped creeping
        elif self._abort:
            outcome = "abort"
        elif real_stop:
            outcome = "reached_dino"        # genuine STOP-confirmed arrival
        elif node._goal_reached:
            outcome = "spin_stall"          # watchdog stopped it -- NOT an arrival
        else:
            outcome = "timeout"
        if outcome in ("reached_dino", "reached_close_loss"):
            # A real STOP-confirm (reached_dino) is trustworthy. So is
            # reached_close_loss: DINO tracked the target down to within
            # close_arrive_m (default 1.5 m, live-observed 0.88 m) and only
            # then lost the box -- that's the frame-filling/edge-clip failure
            # mode of being ON TOP of the object (see the reid-debug note
            # above), not a guess. The remaining heuristic outcomes
            # (supervisor / supervisor_approach_lost) are judgment calls made
            # from farther out and stay excluded -- writing THEM would let
            # one bad guess poison every future "go back to X" for this
            # phrase.
            with node.lock:
                rgb_at_arrival = node.latest_rgb
            self._remember_arrival(phrase, node.odom.x, node.odom.y,
                                   det_box=last_det_box, rgb=rgb_at_arrival)
        res = dict(label=label, kind="object", instruction=phrase, final_target=node.target,
                   grounder=("Qwen-pixel" if node.pipe.cfg.use_qwen_instruction else "DINO"),
                   outcome=outcome, used_memory=bool(used_memory or used_fact3r),
                   used_fact3r=used_fact3r, path_length_m=self._end_leg_path(),
                   odometry_csv=self._leg_odometry_csv(),
                   end=[round(node.odom.x, 3), round(node.odom.y, 3), round(node.odom.theta, 4)],
                   steps=node._infer_count - infer0, dur_s=round(time.time() - t_start, 1),
                   theta_src_final=node.odom.theta_source, supervisor=sup_log)
        print(f"  [{label}] done: {outcome}  end=({node.odom.x:+.2f},{node.odom.y:+.2f}) "
              f"steps={res['steps']}")
        return res

    # -- self-check ------------------------------------------------------
    def odom_selfcheck(self) -> bool:
        d = self.args.selfcheck_dist_m
        tol = self.args.selfcheck_tol_m
        print(f"\n=== odometry self-check: forward {d:.2f} m then back (tol {tol*100:.0f} cm) ===")
        fwd = self.drive_straight(d, self.args.move_tol_m, self.args.move_timeout_s, "selfcheck_fwd")
        self.results.append(fwd)
        if self._abort:
            return False
        ex0, ey0, _ = fwd["end"]
        if abs(fwd["error_m"]) > tol or abs(ey0) > tol:
            print(f"  self-check FAIL: forward err {fwd['error_m']:.2f} m, lateral {ey0:+.2f} m  (> {tol:.2f} m)")
            return False
        self.settle(1.0)
        back = self.drive_straight(-d, self.args.move_tol_m, self.args.move_timeout_s, "selfcheck_back")
        self.results.append(back)
        if self._abort:
            return False
        bx, by, bth = back["end"]
        home_err = math.hypot(bx, by)
        print(f"  returned to ({bx:+.2f},{by:+.2f}) -- {home_err*100:.0f} cm from origin")
        if home_err > 2.0 * tol:
            print(f"  self-check FAIL: did not return to origin ({home_err:.2f} m > {2*tol:.2f} m)")
            return False
        self.node.odom.reset_pose()
        print("  self-check PASS -- pose re-zeroed at physical start\n")
        return True

    # -- results ------------------------------------------------------
    def write_results(self, aborted_at: Optional[str] = None, reason: Optional[str] = None):
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.args.log_root, f"rover_multileg_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        payload = dict(
            timestamp=ts, pi_ip=self.args.pi_ip,
            plan=[repr(s) for s in self._plan],
            aborted_at=aborted_at, abort_reason=reason,
            origin_to_plan_start_m=self._origin_to_plan_start_m,
            results=self.results,
            odometry_log_dir=os.path.abspath(self.args.odometry_log_dir),
            config=dict(qwen_model_id=self.args.qwen_model_id, qwen_fp16=self.args.qwen_fp16,
                        qwen_appearance_lock=self.args.qwen_appearance_lock,
                        stop_distance=self.args.stop_distance, max_linear=self.args.max_linear,
                        metric_max_angular=self.args.metric_max_angular,
                        turn_max_angular=self.args.turn_max_angular, predict_hz=self.args.predict_hz,
                        metric_hz=self.args.metric_hz),
        )
        path = os.path.join(out_dir, "result.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[results] {path}")
        return path

    def _cleanup(self, close_odom: bool = True):
        # close_odom=False for the end-of-plan call in --serve mode: the
        # process (and this same OdometryLogger) keeps running to serve the
        # next instruction, so closing its CSV file here would just force
        # every /rover/rpm sample in the idle gap to hit a closed file (see
        # OdometryLogger.close()'s note) until the next subtask's
        # start_new_goal() reopens it. Every other caller (single-shot CLI
        # run, a real abort/exit, or --serve's own final shutdown) still
        # closes it -- see call sites.
        try:
            self.node.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.node.publish_cmd(0.0, 0.0)
            if close_odom:
                self.node.odom.close()
        except Exception:
            pass
        # Persist the fact3r entity map ONLY if --fact3r-memory-persist (default
        # OFF -- per user request the map is per-session: it lives in this
        # --serve process and dies with it / when the GUI is closed, like the
        # occupancy grid). Called per-plan in --serve as well, so a persisted
        # map is also checkpointed after every instruction.
        if self.fact3r_mem is not None and getattr(self.args, "fact3r_memory_persist", False):
            try:
                n = self.fact3r_mem.save(self.args.fact3r_memory_path)
                print(f"[fact3r] saved {n} entities -> {self.args.fact3r_memory_path}")
            except Exception as e:
                print(f"[fact3r] save failed: {e}")
        # Final shutdown only (not the per-plan --serve call): stop the
        # background SAM2 observe worker.
        if close_odom and self._fact3r_thread is not None:
            self._fact3r_stop.set()

    # -- orchestrator hooks (overridden by run_rover_memfirst_dino.py) -----
    def plan_from_instruction(self, text: str) -> "List[Subtask]":
        """Free text -> ordered subtasks. Default = the regex decomposer;
        the Qwen orchestrator subclass replaces this with a VLM plan call
        (regex as fallback)."""
        return plan_from_instruction(text)

    def _on_subtask_done(self, st: "Subtask", res: dict, remaining: "List[Subtask]",
                         retries: int) -> str:
        """Called after each subtask in run(). Return 'continue' (default),
        'retry' (re-run this subtask, bounded by --orchestrator-max-retries),
        or 'abort' (stop the plan). Base impl always continues."""
        return "continue"

    # -- main ------------------------------------------------------
    def run(self, plan: List[Subtask]):
        self._plan = plan
        self.results = []   # don't let a PREVIOUS --serve instruction's entries leak into this
        # plan's per-index outcome lookup (multileg_gui.py matches results by "{i}_" label prefix)
        a = self.args

        if not self.wait_for_inputs(a.input_wait_s):
            print("[fatal] no camera and/or /rover/rpm -- is the Pi up? "
                  "(rover-camera / rover-agent / rover-zenoh). Aborting.")
            self._cleanup()
            sys.exit(4)

        # Pose (and the occupancy map, if enabled) is reset/calibrated ONCE
        # per process, not once per plan -- in --serve mode this is what lets
        # several instructions compose in one consistent frame and share one
        # growing map, instead of each instruction re-zeroing the world under
        # the previous one's carved cells.
        if not self._pose_reset_done:
            self.node.odom.reset_pose()
            self._occ_origin = (0.0, 0.0)
            self._pose_reset_done = True
            print("[frame] odometry origin set at current pose (theta = 0)")
            if a.odom_selfcheck:
                if not self.odom_selfcheck():
                    self._safe_stop()
                    self.write_results(aborted_at="odom_selfcheck", reason="odometry_error")
                    self._cleanup()
                    sys.exit(3)
            self.settle(a.rest_s)
        else:
            print("[frame] continuing in the existing odometry frame + occupancy map")

        self._origin_to_plan_start_m = round(math.hypot(self.node.odom.x, self.node.odom.y), 3)
        print(f"[odometry] this plan's first subtask starts {self._origin_to_plan_start_m:.2f} m "
              f"from the session origin (0, 0)")

        i = 0
        retries = 0
        while i < len(plan) and not self._abort:
            st = plan[i]
            self._current_index = i
            self._publish_status(st.kind, str(st), force=True)
            print(f"\n=== subtask {i + 1}/{len(plan)} : {st!r} ===")
            if st.kind == "move":
                res = self.drive_straight(st.arg, a.move_tol_m, a.move_timeout_s, f"{i}_move")
            elif st.kind == "turn":
                res = self.turn_in_place(st.arg, a.turn_tol_deg, a.turn_timeout_s, f"{i}_turn")
            else:
                res = self.go_to_object(st.arg, a.object_timeout_s, f"{i}_object", raw_clause=st.raw)
            self.results.append(res)
            # workflow-supervisor hook (default 'continue'; the Qwen orchestrator
            # in run_rover_memfirst_dino.py can say retry / abort)
            action = "continue"
            if not self._abort:
                try:
                    action = self._on_subtask_done(st, res, plan[i + 1:], retries)
                except Exception as e:
                    print(f"[orchestrator] _on_subtask_done failed ({e}) -- continuing")
            if action == "abort":
                print(f"[orchestrator] stopping the plan after subtask {i + 1}/{len(plan)}")
                self._abort = True
                self.settle(a.rest_s)
                break
            if action == "retry" and retries < getattr(a, "orchestrator_max_retries", 0):
                retries += 1
                print(f"[orchestrator] retrying subtask {i + 1}/{len(plan)} (attempt {retries + 1})")
                self.results.pop()   # drop this attempt's result; the retry reuses the same label
                self.settle(a.rest_s)
                continue
            retries = 0
            i += 1
            self.settle(a.rest_s)

        self._current_index = len(plan)   # tells the GUI the plan is finished/aborted
        self._publish_status("done", "all subtasks finished" if not self._abort else "aborted", force=True)
        self._safe_stop()
        self.write_results(aborted_at="interrupt" if self._abort else None)
        self._cleanup(close_odom=not self.args.serve)
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 64)
        print(f"{'subtask':<16}{'target':>12}{'result':>14}{'error':>12}{'outcome':>10}")
        print("-" * 64)
        for r in self.results:
            if r["kind"] == "move":
                tgt, got, err = f"{r['target_m']:+.2f} m", f"{r.get('travelled_m', 0):+.2f} m", f"{r.get('error_m', 0):.2f} m"
            elif r["kind"] == "turn":
                tgt, got, err = f"{r['target_deg']:+.0f} deg", f"{r.get('turned_deg', 0):+.0f} deg", f"{r.get('error_deg', 0):+.0f} deg"
            else:
                tgt, got, err = r["instruction"][:12], "-", "-"
            print(f"{r['label']:<16}{tgt:>12}{got:>14}{err:>12}{r['outcome']:>10}")
        print("=" * 64)


# ======================================================================
#  CLI
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Headless mixed-subtask real-rover runner (Qwen3-VL-2B + NavDP + odometry)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # plan
    p.add_argument("--subtask", action="append", default=[], metavar="move:2.5|turn:-90|object:...",
                   help="ordered subtask (repeatable). Right turn = negative degrees.")
    p.add_argument("--instruction", type=str, default=None,
                   help="free-text alternative to --subtask; split on then/commas, each clause classified")
    p.add_argument("--dry-run", action="store_true", help="print the resolved plan + config and exit (no Zenoh, no models)")
    p.add_argument("--serve", action="store_true",
                   help="load every model ONCE, then loop forever accepting new free-text "
                        "instructions over Zenoh topic 'multileg/instruction' (std_msgs/String) -- "
                        "no --subtask/--instruction needed at launch. Publish '__stop__' on that "
                        "same topic to abort just the current plan (server stays up, keep sending "
                        "instructions); '__shutdown__' to abort (if running) and exit cleanly. "
                        "Ctrl-C always fully exits the process, running or idle.")
    p.add_argument("--yes", action="store_true", help="skip the confirm-the-plan prompt")

    # Qwen orchestrator (implemented in run_rover_memfirst_dino.py; a no-op in
    # the base runner, whose plan_from_instruction / _on_subtask_done hooks
    # just do the regex split + always-continue). When on: the WHOLE
    # instruction goes to the Qwen supervisor, which (1) decomposes it into
    # move/turn/object subtasks, (2) decides per object leg whether to recall
    # a remembered location or search fresh, (3) after each subtask says
    # continue / retry / abort.
    p.add_argument("--qwen-orchestrator", dest="qwen_orchestrator",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="let the Qwen supervisor plan + supervise the whole instruction "
                        "(run_rover_memfirst_dino.py only; regex decomposer is the fallback).")
    p.add_argument("--orchestrator-max-retries", type=int, default=1,
                   help="max times the workflow supervisor may retry a single subtask before the "
                        "plan moves on / aborts.")

    # transport / rover
    p.add_argument("--pi-ip", type=str, default=None, help="Pi IP for the Zenoh peer; omit for multicast scouting")
    p.add_argument("--predict-hz", type=float, default=3.0, help="object-subtask inference loop rate")
    p.add_argument("--metric-hz", type=float, default=10.0, help="move/turn control loop rate")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    p.add_argument("--log-root", type=str, default="logs")
    p.add_argument("--input-wait-s", type=float, default=25.0, help="max wait for first camera frame + /rover/rpm")

    # motion caps
    p.add_argument("--max-linear", type=float, default=0.15)
    p.add_argument("--max-angular", type=float, default=1.2, help="pipeline (object-leg) angular cap, rad/s")
    p.add_argument("--metric-max-angular", type=float, default=0.50, help="heading-hold cap during a straight move, rad/s")
    p.add_argument("--turn-max-angular", type=float, default=0.35,
                   help="in-place turn cap, rad/s. Lowered 2026-09-10 (0.72 -> 0.35): a fast "
                        "instructed turn (turn:-90) swept the camera past objects so detection / "
                        "fact3r mapping missed them, and left a motion-blurred frame for the next "
                        "leg's first detection. ~20 deg/s gives vision time to keep up.")
    p.add_argument("--angular-slew-max", type=float, default=0.08, help="rad/s change per tick (0 disables) -- gentler ramp into/out of turns")
    p.add_argument("--navdp-angular-gain", type=float, default=1.0,
                   help="multiplier on the angular command NavDP's chosen trajectory produces "
                        "(_command_from_trajectory). <1 = softer, less twitchy yawing while "
                        "following NavDP waypoints (still clipped to +/-max-angular; near-straight "
                        "waypoints command 0); 1.0 = unchanged; 0 = NavDP sets speed only, heading "
                        "comes from the goal-bearing servo.")
    p.add_argument("--fov", type=float, default=55.5, help="camera horizontal FOV deg (D435i bootstrap; camera_info overrides)")

    # depth hole-fill -- the RealSense D435i returns NO valid depth on glossy /
    # dark / thin / IR-absorbing surfaces (a monitor screen, glass, a black
    # object). Those pixels stay 0/NaN, so the depth-based obstacle guard sees
    # "clear" there (-> collision) AND fact3r's SAM2 proposal for that object
    # gets <30 valid-depth pixels and is dropped (-> never mapped, can't be
    # recalled). Splicing the monocular metric-depth estimate into just those
    # holes fixes both, and keeps every real RealSense pixel untouched.
    p.add_argument("--depth-fill-holes", dest="depth_fill_holes",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="fill no-return depth pixels with the monocular metric-depth estimate "
                        "before the obstacle guard + fact3r mapping see the frame.")
    p.add_argument("--depth-fill-min-hole-frac", type=float, default=0.04,
                   help="only run the (costly) monocular fill when this fraction of the forward "
                        "corridor has no RealSense return.")

    # metric-controller gains / floors / tolerances
    p.add_argument("--move-kp", type=float, default=0.8)
    p.add_argument("--move-hold-kp", type=float, default=1.2)
    p.add_argument("--move-v-floor", type=float, default=0.05, help="min |v| that beats the ESP32 wheel deadband")
    p.add_argument("--move-tol-m", type=float, default=0.10)
    p.add_argument("--move-timeout-s", type=float, default=45.0)
    p.add_argument("--turn-kp", type=float, default=1.5)
    p.add_argument("--turn-w-floor", type=float, default=0.20, help="min |w| that beats the wheel deadband (last-validated 0.18-0.20)")
    p.add_argument("--turn-tol-deg", type=float, default=3.0)
    p.add_argument("--turn-timeout-s", type=float, default=25.0)
    p.add_argument("--imu-calib-wait-s", type=float, default=25.0,
                   help="max wait for the BNO08x MAG sub-score to reach --imu-min-mag-calib "
                        "before a turn: leg. The magnetometer frequently never calibrates indoors "
                        "(motor/rebar interference), so on timeout the leg does NOT abort by "
                        "default -- see --imu-calib-timeout. Was 120 s (2026-09-09); lowered since "
                        "waiting longer for something that isn't coming just delays the fallback.")
    p.add_argument("--imu-calib-timeout", choices=["encoder-only", "abort"], default="encoder-only",
                   help="what a turn: leg does when --imu-calib-wait-s elapses with the MAG score "
                        "still below --imu-min-mag-calib. 'encoder-only' (default): run the turn on "
                        "wheel-differential + gyro dead reckoning (theta_src=enc) -- live-measured "
                        "~1.6-4 deg error on a 90 deg turn, fine for this task. 'abort': the old "
                        "behaviour, ends the whole plan (--serve stays up).")
    p.add_argument("--imu-min-mag-calib", type=int, default=3)
    p.add_argument("--encoder-only", action="store_true",
                   help="disable IMU heading fusion entirely -- theta is pure wheel-differential "
                        "dead reckoning. Use when the IMU heading channel is dead/frozen (confirmed "
                        "via: spin the rover, watch imu_heading_deg on /rover/rpm never change). "
                        "Also skips the turn subtask's IMU-calibration wait. Redundant with (but a "
                        "faster start than) --auto-encoder-only's own detection.")
    p.add_argument("--auto-encoder-only", dest="auto_encoder_only", action="store_true", default=True,
                   help="on by default: self-detect a dead/frozen IMU heading channel live (see "
                        "nav_pipeline/odometry_logger.py's _check_imu_alive -- reported heading "
                        "unchanged across IMU_FREEZE_ROT_THRESH_RAD of REAL wheel-encoder-attested "
                        "rotation while the mag calib byte claims it's trustworthy) and switch to "
                        "encoder-only dead reckoning automatically, mid-run, the first time it fires. "
                        "No manual --encoder-only needed for a fault that only shows up after the "
                        "rover starts turning -- catches it within one bad turn instead of a whole run.")
    p.add_argument("--no-auto-encoder-only", dest="auto_encoder_only", action="store_false",
                   help="disable the self-detection above -- trust the IMU (or --encoder-only's "
                        "manual override) for the whole run regardless of what it does")

    # object subtask
    p.add_argument("--rest-s", type=float, default=1.0, help="zero-velocity pause between subtasks")
    p.add_argument("--object-timeout-s", type=float, default=180.0)
    p.add_argument("--stop-distance", type=float, default=0.5)
    p.add_argument("--stop-confirm-count", type=int, default=3)
    p.add_argument("--qwen-model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--qwen-fp16", action="store_true", default=True,
                   help="load Qwen in fp16 (default for the 2B model); --no-qwen-fp16 for 4-bit")
    p.add_argument("--no-qwen-fp16", dest="qwen_fp16", action="store_false")
    p.add_argument("--qwen-instruction-period-s", type=float, default=1.5)
    p.add_argument("--qwen-goal-consistency-m", type=float, default=1.5)
    p.add_argument("--qwen-max-candidates", type=int, default=3)
    p.add_argument("--qwen-appearance-lock", action="store_true", default=True,
                   help="multi-view DINOv2 identity lock on the Qwen goal (default ON here); --no-qwen-appearance-lock to disable")
    p.add_argument("--no-qwen-appearance-lock", dest="qwen_appearance_lock", action="store_false")
    p.add_argument("--wheel-deadband-correction", action="store_true", default=True,
                   help="ESP32 6WD only -- boost a command that would differential-mix to a stalled wheel")
    p.add_argument("--no-wheel-deadband-correction", dest="wheel_deadband_correction", action="store_false")
    p.add_argument("--no-belief-goal", action="store_true")

    # object grounding: Grounding-DINO (default) vs the Qwen pixel-goal path
    p.add_argument("--object-grounder", choices=["dino", "qwen"], default="dino",
                   help="dino: Grounding-DINO box + DINOv2 appearance lock -> NavDP (robust identity + "
                        "real STOP gate). qwen: the Qwen3-VL pixel-goal path (jittery on real frames).")
    p.add_argument("--dino-target-threshold", type=float, default=None,
                   help="override PipelineConfig.detect_score_min (Grounding-DINO box score floor)")
    p.add_argument("--dino-appearance-min-sim", type=float, default=0.65,
                   help="PipelineConfig.appearance_min_similarity -- LOWER (default 0.85) so DINO "
                        "keeps the lock as the box grows during close approach instead of dropping "
                        "the target at ~1.5 m and forcing a heuristic arrival")
    p.add_argument("--dino-appearance-iou-override", type=float, default=0.45,
                   help="PipelineConfig.appearance_iou_override -- box-IoU above this bypasses the "
                        "appearance check entirely (default 0.6); lower = more forgiving on approach")
    p.add_argument("--dino-clip-min-sim", type=float, default=0.5,
                   help="PipelineConfig.clip_min_similarity -- SAM-mask/raw-box CLIP-vs-text floor. "
                        "Live-observed 2026-09-08: a real, supervisor-confirmed 'chair' scoring "
                        "0.02-0.6 across a whole approach (rarely clearing the 0.5 default), which "
                        "starves TRACK of any accepted detection and leaves the pipeline stuck in "
                        "SEARCH's fixed left-right scan for most of the run -- reads as 'continuously "
                        "rotating'/'deflecting' even with the object plainly on camera. Lower (e.g. "
                        "0.35-0.4) if a real object's own scores are landing consistently below this.")
    # Qwen task supervisor (grounding stays DINO's job; Qwen only judges)
    p.add_argument("--supervisor", dest="supervisor", action="store_true", default=True,
                   help="run a throttled Qwen-VL supervisor over the object leg: picks the DINO target "
                        "phrase, calls 'arrived', flags a wrong lock. Default ON with --object-grounder dino.")
    p.add_argument("--no-supervisor", dest="supervisor", action="store_false")
    p.add_argument("--supervisor-model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="VLM for task supervision (a stronger judge than the 2B pixel model; called rarely)")
    p.add_argument("--supervisor-4bit", dest="supervisor_4bit", action="store_true", default=True)
    p.add_argument("--no-supervisor-4bit", dest="supervisor_4bit", action="store_false")
    p.add_argument("--supervisor-period-s", type=float, default=4.0)
    p.add_argument("--supervisor-min-arrive-s", type=float, default=6.0,
                   help="ignore a supervisor 'arrived' verdict before this many seconds into the leg")
    p.add_argument("--supervisor-retarget-lost-s", type=float, default=6.0,
                   help="only let the supervisor change the DINO phrase after DINO has had no "
                        "detection for this long (or the detection is ambiguous)")
    p.add_argument("--supervisor-history-frames", type=int, default=0,
                   help="give the Qwen supervisor N of NavDP's own recent buffered frames "
                        "(DinoNavDPPipeline.get_recent_frames, 224x224, lower-res than the live "
                        "frame) alongside the current view each check() call, so 'is progress being "
                        "made' judgments aren't made from one frozen snapshot. 0 (default) = off -- "
                        "each extra image adds real VLM latency at the 4s supervisor cadence; the "
                        "cheap additions (NavDP's planned-path endpoint, the supervisor's own recent "
                        "verdict history) are always on regardless of this flag.")
    p.add_argument("--search-creep-v", type=float, default=0.06,
                   help="forward speed while DINO has no lock -- replaces the pipeline's blind "
                        "in-place SEARCH spin so the view keeps changing (0 = spin in place)")
    p.add_argument("--search-turn", type=float, default=0.18,
                   help="turn rate (rad/s) for a supervisor left/right steering hint / scan while "
                        "lost -- kept slow so the target isn't swept past faster than the detector "
                        "runs (2026-09-10: 0.25 -> 0.18)")
    p.add_argument("--search-creep-max-m", type=float, default=1.2,
                   help="stop the leg if it creeps this far with no DINO re-acquire -- stops the "
                        "rover driving off across the room past a target it just lost")
    p.add_argument("--object-search", dest="object_search", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="when DINO has no lock in an object leg, do the creep+scan recovery. "
                        "--no-object-search = detection-only mode: the rover holds position and, "
                        "if DINO stays lost for --search-off-grace-s, the leg ends (as an arrival "
                        "if it had been close, else 'lost'). run_rover_memfirst_dino.py also forces "
                        "this OFF for the detection close-in that follows a memory approach whose "
                        "frame-check failed.")
    p.add_argument("--search-off-grace-s", type=float, default=5.0,
                   help="with --no-object-search, how long DINO may stay lost before the leg ends.")
    p.add_argument("--close-arrive-m", type=float, default=1.5,
                   help="if DINO loses the lock while the goal was within this distance, treat it "
                        "as arrival (box fills the frame when you are on top of the object)")

    # fact3r-map goal memory ("go back to X") -- see _load_nav_pipeline()'s
    # comment for exactly which parts of goal_memory.py this reuses
    p.add_argument("--goal-memory", dest="goal_memory", action="store_true", default=True,
                   help="on a confirmed DINO arrival, record {phrase, world_xy} to a live goal-"
                        "memory JSONL; on a later object leg for the SAME phrase, approach that "
                        "remembered point via NavDP GOTO before falling back to DINO search")
    p.add_argument("--no-goal-memory", dest="goal_memory", action="store_false")
    p.add_argument("--goal-memory-approach", dest="goal_memory_approach", action="store_true",
                   default=True,
                   help="actually DRIVE toward a remembered point before DINO search starts, "
                        "giving memory first priority over a fresh detection-only search. Was "
                        "default OFF (opt-in, only multileg_gui.py's Run button turned it on via "
                        "extra_args -- which only takes effect on a FRESH --serve spawn, so a "
                        "stale server or the headless launch_rover_multileg.sh silently never "
                        "attempted recall at all, live-observed 2026-09-10: zero '[memory]' lines, "
                        "straight to DINO, even with a matching record on disk). Per user request "
                        "2026-09-10: default ON everywhere -- recording a confirmed arrival (see "
                        "--goal-memory) was always pointless without this actually being used.")
    p.add_argument("--no-goal-memory-approach", dest="goal_memory_approach", action="store_false")
    p.add_argument("--goal-memory-path", type=str, default="logs/live_goal_memory.jsonl")
    p.add_argument("--goal-memory-approach-timeout-s", type=float, default=30.0)
    p.add_argument("--goal-memory-arrive-m", type=float, default=1.5,
                   help="stop the memory-guided blind approach once within this distance of the "
                        "remembered point and hand off to object search. Lower it to make the "
                        "recall drive further on memory before searching (the stall watchdog + "
                        "timeout still catch 'can't get any closer'). run_rover_memfirst_dino.py "
                        "defaults it to 0.5 -- memory used to its maximum extent, then search.")
    p.add_argument("--goal-memory-max-drift-rot-rad", type=float, default=2.0 * math.pi,
                   help="drift-risk gate: if this much UNANCHORED rotation (wheel-diff heading "
                        "while the IMU wasn't mag-calibrated/trusted, see OdometryLogger."
                        "total_enc_rot_rad) has accumulated since this arrival was banked, skip the "
                        "blind memory-guided drive (position too likely drifted) and go straight to "
                        "DINO search -- the appearance signature, if any, is still kept for a sanity "
                        "check on the first live re-detection. Default: one full turn's worth "
                        "(2*pi). 0 disables this gate (always trust world_xy, prior behavior).")
    p.add_argument("--goal-memory-max-drift-dist-m", type=float, default=10.0,
                   help="drift-risk gate: same as --goal-memory-max-drift-rot-rad but for raw "
                        "distance driven since banking (OdometryLogger.total_dist_m) -- bounds "
                        "error from wheel-radius/track-width calibration and slip during normal "
                        "driving, not just in-place turning. 0 disables this gate.")
    p.add_argument("--goal-memory-drift-quick-timeout-s", type=float, default=5.0,
                   help="FLOOR on the drift-risk gate's timeout cap below -- the actual cap is "
                        "max(this, straight-line distance / --goal-memory-quick-timeout-speed-mps), "
                        "so a genuinely close recall still fails fast even under this floor. Per user "
                        "request 2026-09-10: a flat cap (this used to BE the cap, not just its floor) "
                        "left a several-metres-away recall with under a metre's worth of drive time "
                        "-- not even enough to finish turning -- so it always fell through to a blind "
                        "SEARCH with no positional bias.")
    p.add_argument("--goal-memory-quick-timeout-speed-mps", type=float, default=0.08,
                   help="nominal creep speed used to size the drift-risk gate's timeout cap from the "
                        "straight-line distance to the remembered point -- conservative (this rig's "
                        "observed TRACK speed is ~0.05-0.15 m/s) so a far-but-legitimate recall gets "
                        "close to the full --goal-memory-approach-timeout-s rather than an arbitrary "
                        "short cut. Live-observed 2026-09-10: EVERY recall in a long test session "
                        "tripped the drift gate (repeated retries inflate the session-long distance "
                        "odometer fast, not genuine drift) while real remembered points were 5-9m "
                        "away -- a flat 5s cap covered under a metre of that.")
    p.add_argument("--goal-memory-stall-after-s", type=float, default=6.0,
                   help="if the blind memory approach makes less than --goal-memory-stall-min-"
                        "move-m of net progress for this long while still short of --goal-memory-"
                        "arrive-m, give up and hand off to DINO instead of waiting out the rest of "
                        "--goal-memory-approach-timeout-s. Live-observed 2026-09-10: with no stall "
                        "detection at all, a bad A* waypoint or an obstacle-guard veto with nowhere "
                        "to go left the rover sitting at v=0/w=0 for the ENTIRE remaining timeout, "
                        "twice in one session, both requiring a manual abort. Ported from "
                        "run_rover_occluded_recall.py's _approach_point, which already had this for "
                        "the same A*-waypoint-following shape.")
    p.add_argument("--goal-memory-stall-min-move-m", type=float, default=0.15,
                   help="net-movement floor for the stall watchdog above.")
    p.add_argument("--goal-memory-max-route-indirection", type=float, default=2.5,
                   help="reject an A* route (fall back to the straight-line target instead) if its "
                        "length is more than this many times the straight-line distance to the "
                        "remembered point -- a route this indirect is a sign the occupancy grid it "
                        "was searched over disagrees with the CURRENT pose (e.g. carved from "
                        "already-drifted odometry earlier in the session), not a genuine detour. "
                        "Live-observed 2026-09-10: an 18.1m/2.8x route sent the rover turning "
                        "anticlockwise to start when the remembered point was actually clockwise "
                        "from where it stood.")
    p.add_argument("--goal-memory-face-max-angular", type=float, default=0.28,
                   help="angular cap (rad/s) for the in-place 'turn to face the remembered point' "
                        "step before NavDP GOTO takes over (2026-09-10: 0.42 -> 0.28). Kept below "
                        "--turn-max-angular: a faster in-place spin smears the rolling-shutter "
                        "camera and starves the DINO re-detection that has to happen the moment this "
                        "hands off.")
    p.add_argument("--goal-memory-face-settle-s", type=float, default=0.5,
                   help="after the approach ends, hold still this long (let motion blur clear and a "
                        "clean frame arrive) and reset the spin-stall window before DINO search "
                        "starts -- otherwise the watchdog is already tripped by the turn-around.")
    p.add_argument("--goal-memory-turn-to-drive-settle-s", type=float, default=4.0,
                   help="same idea as --goal-memory-face-settle-s but mid-approach: after the "
                        "turn-to-face step finishes and before GOTO starts driving, hold still this "
                        "long so a fresh, unblurred frame lands post-rotation instead of NavDP's "
                        "first GOTO step planning off one still mid-spin. Was 1.0s; raised +3s (to "
                        "4.0s) per user request 2026-09-10 -- streaming/detection latency was still "
                        "high enough that 1s wasn't leaving time for a clean, low-latency frame to "
                        "actually arrive before GOTO started planning. 0 disables (drive immediately, "
                        "prior behavior).")
    p.add_argument("--goal-memory-dino-assist", dest="goal_memory_dino_assist", action="store_true",
                   default=True, help="while blindly approaching a remembered point, probe with "
                        "Grounding-DINO at --goal-memory-dino-assist-hz and hand off the INSTANT the "
                        "target is actually seen, instead of only on distance/timeout -- the blind "
                        "leg is pure odometry and a stale/drifted remembered point can be off by more "
                        "than --goal-memory-arrive-m; a real detection is always more correct than "
                        "dead reckoning.")
    p.add_argument("--no-goal-memory-dino-assist", dest="goal_memory_dino_assist", action="store_false")
    p.add_argument("--goal-memory-dino-assist-hz", type=float, default=2.0,
                   help="throttle for the assist probe above -- deliberately well below --metric-hz: "
                        "this is a cheap early-exit trigger, not the steering signal (go_to_object's "
                        "own full detect+track+supervisor loop takes over completely once triggered), "
                        "so it shouldn't compete with the blind approach's own control-loop rate for "
                        "GPU time.")
    p.add_argument("--goal-memory-assist-grace-s", type=float, default=3.0,
                   help="minimum time actually spent in GOTO (post face-turn) before the DINO "
                        "assist probe is even eligible to fire -- lets the blind drive make a real "
                        "attempt at closing distance first, with the probe as a correction/early-"
                        "exit afterward rather than a replacement for it. Live-observed 2026-09-10: "
                        "without this, the probe could fire the instant face_err dipped under the "
                        "70deg cone gate -- BEFORE the face-turn branch even finished (that needs "
                        "under 25deg, hysteresis) -- so detection could hijack mid-turn, or grab "
                        "control on literally the first GOTO tick with zero driving in between.")
    p.add_argument("--goal-memory-dino-assist-min-score", type=float, default=0.55,
                   help="score floor for the assist probe's OWN early-handoff decision -- deliberately "
                        "stricter than the raw Grounding-DINO box_threshold (~0.35, --dino-target-"
                        "threshold) that detect_best() itself already applied. Live-observed "
                        "2026-09-10: a bare 0.38 hit (barely past that floor) triggered an early "
                        "handoff while the rover was still facing 172deg away from the remembered "
                        "point -- CLIP re-id then rejected it (0.019) and Qwen confirmed it was a "
                        "frosted glass door, but only after the deliberate turn-to-face was already "
                        "abandoned and ~15s were burned SEARCHing from the wrong heading. See the "
                        "geometric gate + appearance cross-check in _recall_and_approach for the other "
                        "half of this fix.")
    p.add_argument("--goal-memory-waypoint-spacing-m", type=float, default=0.6,
                   help="A* route thinning: keep waypoints roughly this far apart (thin_waypoints' "
                        "own default) -- raw cell-by-cell A* output is 0.1m apart, far too dense to "
                        "steer to")
    p.add_argument("--goal-memory-waypoint-advance-m", type=float, default=0.5,
                   help="advance to the next A* waypoint once within this distance of the current "
                        "one (the LAST waypoint is always the true remembered point itself, still "
                        "gated by --goal-memory-arrive-m separately)")
    # fact3r-map ENTITY memory integration -- see nav_pipeline/siglip2_embedder.py's
    # module docstring for why SigLIP2 specifically (same encoder family
    # fact3r-map's own entity/goal memory already standardizes on). Opt-in
    # (extra ~400MB model, only relevant once --goal-memory-approach is also
    # on -- there's nothing to visually compare against on a pure text+xy
    # recall with no live re-detection yet).
    p.add_argument("--goal-memory-appearance", action="store_true", default=False,
                   help="opt-in: bank a SigLIP2 view of each confirmed arrival alongside the "
                        "existing text+point record, and on a later recall, appearance-check the "
                        "first live DINO re-detection against it -- a sanity signal folded into the "
                        "Qwen supervisor's context (does NOT by itself block/override arrival; DINO "
                        "+ the supervisor still decide). Requires --goal-memory-approach.")
    p.add_argument("--siglip-model-id", type=str, default="google/siglip2-base-patch16-224")
    p.add_argument("--goal-memory-appearance-min-sim", type=float, default=0.5,
                   help="cosine similarity floor below which a re-detection is flagged as possibly "
                        "a different object than the one remembered (advisory, not a hard veto)")
    p.add_argument("--goal-memory-geo-mismatch-m", type=float, default=2.5,
                   help="geo-consistency check (go_to_object, first live re-detection after a "
                        "memory recall): if NavDP's own depth-derived goal_point, projected into "
                        "world coordinates, lands more than this far from the remembered world_xy, "
                        "the live detection is probably a different physical object (same category, "
                        "wrong instance) than the one recalled -- steer by the remembered "
                        "coordinate instead of the detection for that tick. Per user request "
                        "2026-09-10: catches what the appearance/Qwen check above can't, since a "
                        "same-category object still produces a plausible-looking crop.")
    p.add_argument("--goal-memory-paraphrase-min-sim", type=float, default=0.90,
                   help="tier-1 memory lookup: SigLIP2 text-embedding cosine-similarity floor for "
                        "matching a NEW phrasing (e.g. 'the fridge') against a past query_text "
                        "(e.g. 'white air cooler') when no exact normalized match exists -- same "
                        "default as resolve_semantic_goal_verified.py's --memory-min-similarity")
    p.add_argument("--goal-memory-min-vlm-confidence", type=float, default=0.6,
                   help="Qwen recall-verification: a 'no' at/above this confidence drops the "
                        "current DINO lock and forces a fresh search instead of trusting the "
                        "memory-recalled match. Lower than the offline pipeline's 0.75 default "
                        "(--min-vlm-confidence) since a live single-view crop is weaker evidence "
                        "than that pipeline's multi-view render.")

    # fact3r-style LIVE ENTITY MEMORY (nav_pipeline/fact3r_live_memory.py) --
    # SAM2 class-agnostic proposals + SigLIP2 embeddings, each tagged with a
    # metric odom-frame position, accumulated over EVERY frame while driving.
    # A later object leg queries it BEFORE the DINO search, so "go to X" can
    # recall the position of an X only ever seen in passing (never navigated
    # to, so no --goal-memory JSONL record). Heavier than --goal-memory
    # (loads SAM2 -- needs SAM2_ROOT); opt-in.
    p.add_argument("--fact3r-live-memory", dest="fact3r_live_memory", action="store_true", default=False,
                   help="build a live SAM2+SigLIP2 entity map while driving and consult it at the "
                        "start of each object leg (nav_pipeline/fact3r_live_memory.py).")
    p.add_argument("--no-fact3r-live-memory", dest="fact3r_live_memory", action="store_false")
    p.add_argument("--fact3r-observe-period-s", type=float, default=4.0,
                   help="min seconds between SAM2 mapping ticks. SAM2 runs on a background worker "
                        "(the control loop never blocks on it), but it still shares the GPU with "
                        "NavDP/DINO/the supervisor, so a longer period = smoother detection.")
    p.add_argument("--fact3r-points-per-side", type=int, default=12,
                   help="SAM2 automatic-mask-generator grid density (N*N point prompts). Lower = "
                        "faster masking / less GPU pressure (smoother streaming), fewer/coarser "
                        "proposals. Library default is 16.")
    p.add_argument("--fact3r-observe-during-metric", dest="fact3r_observe_during_metric",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="also map during move: legs.")
    p.add_argument("--fact3r-observe-during-turn", dest="fact3r_observe_during_turn",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="also map during turn: legs, so an object the rover rotates past (e.g. on "
                        "'turn right 90') still gets into the entity map. Only sensible now that "
                        "--turn-max-angular is slow (~0.35 rad/s); _observe_entities still skips any "
                        "tick where the instantaneous yaw rate is high.")
    p.add_argument("--fact3r-sam-model", type=str, default="facebook/sam2.1-hiera-small")
    p.add_argument("--fact3r-depth-max", type=float, default=8.0,
                   help="ignore SAM2 proposals whose median depth exceeds this (m). Per user "
                        "request 2026-09-10 kept at 8 m so objects across a larger room are still "
                        "mapped -- note that unprojection error grows with depth, so an object "
                        "first seen at 7-8 m has a noisier banked position (LiveEntity.position "
                        "already down-weights far/partial views, and --fact3r-recall-max-spread-m "
                        "still drops fragmented entities at recall).")
    p.add_argument("--fact3r-recall", dest="fact3r_recall", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="query the entity map for the phrase at the start of each object leg and, on "
                        "a hit, drive to the stored position (blind NavDP GOTO) before the "
                        "DINO+supervisor loop -- which still makes the confirmed approach.")
    p.add_argument("--fact3r-recall-verify", dest="fact3r_recall_verify",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="two-stage recall: SigLIP2 shortlist -> the Qwen supervisor verifies each "
                        "candidate crop against the phrase -> stored position (fact3r's own "
                        "query_verified). Falls back to SigLIP-only text match with no supervisor.")
    p.add_argument("--fact3r-recall-merge", dest="fact3r_recall_merge",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="run merge_nearby_duplicates() before each recall query (fold fragments of "
                        "one object seen from many angles back together).")
    p.add_argument("--fact3r-recall-min-obs", type=int, default=3,
                   help="ignore entities with fewer than this many observations at recall.")
    p.add_argument("--fact3r-recall-min-score", type=float, default=0.10,
                   help="SigLIP-only recall (--no-fact3r-recall-verify): text-embedding cosine floor "
                        "(raw SigLIP2 scores here sit around 0.1).")
    p.add_argument("--fact3r-recall-min-confidence", type=float, default=0.5,
                   help="verified recall: Qwen decision=yes confidence floor.")
    p.add_argument("--fact3r-recall-max-spread-m", type=float, default=2.0,
                   help="verified recall: drop entities whose observed positions disagree by more "
                        "than this before the VLM sees them (no trustworthy position to drive to).")
    p.add_argument("--fact3r-recall-max-depth-m", type=float, default=8.0,
                   help="recall: drop entities whose MEDIAN observation depth exceeds this (m) "
                        "from the shortlist. Kept equal to --fact3r-depth-max (8 m) per user "
                        "request so nothing mapped is excluded from recall; lower it (e.g. 4-5) "
                        "if far, noisily-localised entities start winning the shortlist. "
                        "0 disables the filter entirely.")
    p.add_argument("--fact3r-memory-persist", dest="fact3r_memory_persist",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="save/restore the entity map to --fact3r-memory-path across process restarts "
                        "(unlike --goal-memory, which is wiped on every --serve start).")
    p.add_argument("--fact3r-memory-path", type=str, default="logs/live_fact3r_memory.json")

    # step tracing
    p.add_argument("--debug-steps", dest="debug_steps", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="print every decision an object leg makes: raw clause -> interpreted "
                        "target phrase (+ who set it), fact3r recall (SigLIP shortlist + per-"
                        "candidate Qwen verdict), JSONL goal-memory lookup, DINO detections + "
                        "which box was locked, supervisor retargets. Verbose -- for debugging.")

    # self-check
    p.add_argument("--odom-selfcheck", action="store_true", default=True,
                   help="drive 1 m out and back and verify against odometry before the run (default ON)")
    p.add_argument("--no-odom-selfcheck", dest="odom_selfcheck", action="store_false")
    p.add_argument("--selfcheck-dist-m", type=float, default=1.0)
    p.add_argument("--selfcheck-tol-m", type=float, default=0.15)

    # live BEV occupancy map (real depth + odometry, carved the same way
    # fact3r-map's own offline return-path does -- see live_occupancy.py)
    p.add_argument("--occupancy", dest="occupancy", action="store_true", default=True,
                   help="carve a live occupancy grid from depth+odometry each tick and publish "
                        "a cropped PNG on 'multileg/occupancy' for multileg_gui.py's BEV panel")
    p.add_argument("--no-occupancy", dest="occupancy", action="store_false")
    p.add_argument("--occupancy-extent-m", type=float, default=30.0,
                   help="total grid size (square); the session origin (0,0) is the centre")
    p.add_argument("--occupancy-res", type=float, default=0.10, help="grid cell size, metres")
    p.add_argument("--occupancy-range-m", type=float, default=6.0,
                   help="half-width of the cropped window published for the GUI -- keep in sync "
                        "with multileg_gui.py's TRAIL_RANGE_M")
    p.add_argument("--occupancy-publish-period-s", type=float, default=0.5)
    return p


def _config_summary(args) -> dict:
    dino = args.object_grounder == "dino"
    return dict(
        object_grounder=args.object_grounder,
        use_dino=dino, use_qwen_instruction=not dino,
        supervisor=(args.supervisor and dino),
        supervisor_model_id=args.supervisor_model_id if (args.supervisor and dino) else None,
        supervisor_period_s=args.supervisor_period_s,
        qwen_model_id=args.qwen_model_id, qwen_fp16=args.qwen_fp16,
        qwen_appearance_lock=args.qwen_appearance_lock, qwen_max_candidates=args.qwen_max_candidates,
        qwen_goal_consistency_m=args.qwen_goal_consistency_m,
        stop_distance=args.stop_distance,
        max_linear=args.max_linear, max_angular=args.max_angular,
        metric_max_angular=args.metric_max_angular, turn_max_angular=args.turn_max_angular,
        angular_slew_max=args.angular_slew_max, fov=args.fov,
        predict_hz=args.predict_hz, metric_hz=args.metric_hz,
        wheel_deadband_correction=args.wheel_deadband_correction,
        odom_selfcheck=args.odom_selfcheck, imu_min_mag_calib=args.imu_min_mag_calib,
        serve=args.serve, occupancy=args.occupancy,
        goal_memory=args.goal_memory, goal_memory_path=args.goal_memory_path if args.goal_memory else None,
        goal_memory_approach=args.goal_memory_approach,
        goal_memory_appearance=(args.goal_memory_appearance and args.goal_memory_approach),
        siglip_model_id=args.siglip_model_id if args.goal_memory_appearance else None,
        goal_memory_paraphrase_min_sim=(args.goal_memory_paraphrase_min_sim
                                        if args.goal_memory_appearance else None),
        goal_memory_min_vlm_confidence=(args.goal_memory_min_vlm_confidence
                                        if args.goal_memory_appearance else None),
        fact3r_live_memory=args.fact3r_live_memory,
        fact3r_recall=(args.fact3r_recall if args.fact3r_live_memory else None),
        fact3r_recall_verify=(args.fact3r_recall_verify if args.fact3r_live_memory else None),
        fact3r_observe_period_s=(args.fact3r_observe_period_s if args.fact3r_live_memory else None),
        fact3r_memory_persist=(args.fact3r_memory_persist if args.fact3r_live_memory else None),
    )


def main():
    args = build_arg_parser().parse_args()
    plan = None
    if args.serve:
        print("\n[serve mode] models load once below; instructions arrive later over Zenoh "
              "('multileg/instruction') -- no plan to resolve yet.")
    else:
        plan = resolve_plan(args)
        print("\nResolved plan:")
        for i, st in enumerate(plan):
            print(f"  {i + 1}. {st!r}    <- {st.raw!r}")
    print(f"\nRest between subtasks: {args.rest_s} s")
    print("Config:")
    for k, v in _config_summary(args).items():
        print(f"  {k:28} {v}")

    if args.dry_run:
        print("\n[dry-run] not connecting to the rover.")
        return 0

    if not args.yes:
        try:
            input("\nPress Enter to start on the REAL ROVER (Ctrl-C to cancel) ... ")
        except (EOFError, KeyboardInterrupt):
            print("\ncancelled.")
            return 1

    try:
        import zenoh
    except ImportError:
        print("ERROR: zenoh not importable -- run this under the internnav conda env "
              "(LAUNCH/launch_rover_multileg.sh does that for you).")
        return 1

    _load_nav_pipeline()
    dino_mode = args.object_grounder == "dino"
    cfg_kw = dict(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        stop_distance=args.stop_distance,
        angular_slew_max=args.angular_slew_max,
        navdp_angular_gain=args.navdp_angular_gain,
        use_belief_goal=not args.no_belief_goal,
        guard=GuardConfig(),
        wheel_deadband_correction=args.wheel_deadband_correction,
        use_appearance_reid=True,           # DINOv2 identity lock (both grounders use it)
    )
    if dino_mode:
        cfg_kw.update(use_dino=True, use_qwen_instruction=False, use_qwen_search=False,
                      appearance_min_similarity=args.dino_appearance_min_sim,
                      appearance_iou_override=args.dino_appearance_iou_override,
                      clip_min_similarity=args.dino_clip_min_sim)
        if args.dino_target_threshold is not None:
            cfg_kw["detect_score_min"] = args.dino_target_threshold
    else:
        cfg_kw.update(
            use_dino=False, use_qwen_instruction=True,
            qwen_model_id=args.qwen_model_id,
            qwen_load_in_4bit=not args.qwen_fp16,
            qwen_instruction_period_s=args.qwen_instruction_period_s,
            qwen_goal_consistency_m=args.qwen_goal_consistency_m,
            qwen_max_candidates=args.qwen_max_candidates,
            qwen_appearance_lock=args.qwen_appearance_lock,
        )
    cfg = PipelineConfig(**cfg_kw)
    print(f"\n[load] building pipeline (grounder={args.object_grounder}: "
          f"{'Grounding-DINO' if dino_mode else 'Qwen-pixel'} + NavDP + DepthAnything + DINOv2 re-id) ...")
    pipeline = DinoNavDPPipeline(cfg)

    supervisor = None
    if dino_mode and args.supervisor:
        supervisor = TaskSupervisor(args.supervisor_model_id, load_in_4bit=args.supervisor_4bit,
                                    device=args.device)
        try:
            supervisor._ensure_loaded()   # absorb the load now, not mid-drive
        except Exception as e:
            print(f"[supervisor] eager load failed ({e}) -- continuing DINO-only")
            supervisor = None

    zcfg = zenoh.Config()
    if args.pi_ip:
        zcfg.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[zenoh] connecting tcp/{args.pi_ip}:7447")
    else:
        print("[zenoh] multicast scouting")
    session = zenoh.open(zcfg)

    node = DinoNavDPZenohNode(session, pipeline, "multileg", args.predict_hz,
                              stop_confirm_count=args.stop_confirm_count,
                              odometry_log_dir=args.odometry_log_dir,
                              imu_min_mag_calib=args.imu_min_mag_calib)
    if args.encoder_only:
        node.odom.encoder_only = True
        node.odom.auto_detect_frozen_imu = False   # nothing left to detect -- already forced off
        print("[odometry] encoder-only mode: IMU heading fusion disabled, theta is pure "
              "wheel-differential dead reckoning (turn's IMU-calibration wait is skipped)")
    else:
        node.odom.auto_detect_frozen_imu = args.auto_encoder_only
        if args.auto_encoder_only:
            print("[odometry] auto-encoder-only detection armed: will self-switch to encoder-only "
                  "dead reckoning if the IMU heading is ever caught frozen while calibrated "
                  "(see nav_pipeline/odometry_logger.py's _check_imu_alive)")

    occupancy = None
    if args.occupancy:
        occupancy = LiveOccupancy(center_xy=(0.0, 0.0), extent_m=args.occupancy_extent_m,
                                  res=args.occupancy_res, guard=GuardConfig(), max_range=4.0)
        print(f"[occupancy] live grid enabled: {args.occupancy_extent_m:.0f}m extent, "
              f"{args.occupancy_res:.2f}m cells, publishing a ±{args.occupancy_range_m:.0f}m "
              f"crop on 'multileg/occupancy'")

    # Start every fresh process (i.e. every GUI (re)start, which spawns a new
    # --serve process) with a clean slate: the occupancy grid above is
    # already fresh-constructed, and per user request the goal-memory file
    # (past confirmed arrivals) should not carry over either -- old entries
    # from a previous session shouldn't silently exist for a later
    # --goal-memory-approach run to find.
    if args.goal_memory and args.goal_memory_path:
        try:
            p = Path(args.goal_memory_path)
            if p.exists():
                p.unlink()
                print(f"[memory] cleared stale goal-memory file: {p}")
        except Exception as e:
            print(f"[memory] could not clear {args.goal_memory_path}: {e}")

    fact3r_mem = None
    if args.fact3r_live_memory:
        if Fact3rLiveMemory is None:
            print("[fact3r] --fact3r-live-memory requested but nav_pipeline.fact3r_live_memory "
                  "could not be imported (SAM2_ROOT?) -- continuing without it")
        else:
            try:
                fact3r_mem = Fact3rLiveMemory(
                    device=args.device, siglip_model=args.siglip_model_id,
                    discovery_model=args.fact3r_sam_model, depth_max=args.fact3r_depth_max,
                    points_per_side=args.fact3r_points_per_side)
                print("[fact3r] loading SAM2 + SigLIP2 for the live entity map ...")
                fact3r_mem._ensure_loaded()   # absorb the load now, not mid-drive
                if args.fact3r_memory_persist:
                    k = fact3r_mem.load(args.fact3r_memory_path)
                    if k:
                        print(f"[fact3r] restored {k} entities from {args.fact3r_memory_path}")
                else:
                    # per-session map -- make sure a leftover file from an
                    # earlier --fact3r-memory-persist run isn't silently
                    # around; the entity map is rebuilt from scratch each run
                    try:
                        _fp = Path(args.fact3r_memory_path)
                        if _fp.exists():
                            _fp.unlink()
                            print(f"[fact3r] cleared stale entity-map file: {_fp} "
                                  f"(persistence off -- map is per-session)")
                    except Exception as _e:
                        print(f"[fact3r] could not clear {args.fact3r_memory_path}: {_e}")
            except Exception as e:
                print(f"[fact3r] disabled -- SAM2/SigLIP2 load failed ({e})")
                fact3r_mem = None

    runner = PlanRunner(node, args, supervisor=supervisor, occupancy=occupancy, fact3r_mem=fact3r_mem)
    signal.signal(signal.SIGINT, runner.request_abort)
    signal.signal(signal.SIGTERM, runner.request_abort)
    try:
        if args.serve:
            _serve_loop(runner, session)
        else:
            runner.run(plan)
    finally:
        node._running = False       # stop the heartbeat thread...
        time.sleep(2 * node.HEARTBEAT_PERIOD_S)   # ...and let it exit its loop
        try:
            session.close()
        except Exception:
            pass
    return 0


def _serve_loop(runner: "PlanRunner", session) -> None:
    """--serve mode: everything above (models, Zenoh, node, occupancy) is
    already built ONCE; loop accepting new free-text instructions over
    'multileg/instruction' and running each to completion without rebuilding
    or reloading anything. '__stop__' aborts just the in-progress plan (the
    server stays up); '__shutdown__' (or Ctrl-C while idle) ends the loop
    cleanly. Sentinels are handled the instant they arrive, even mid-plan --
    not queued behind whatever is currently running."""
    import queue as _queue
    next_instr: "_queue.Queue[str]" = _queue.Queue()
    running = threading.Event()

    def _on_instruction(sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
        except Exception:
            return
        if not text:
            return
        if text == "__stop__":
            if running.is_set():
                runner.soft_abort()
            else:
                print("[serve] '__stop__' received but nothing is running -- ignoring")
            return
        if text == "__shutdown__":
            print("[serve] '__shutdown__' received -- exiting once idle")
            runner._exit_requested = True
            if running.is_set():
                runner.soft_abort()
            else:
                next_instr.put("__shutdown__")   # wake the idle wait below
            return
        if text == "__clear_map__":
            runner.clear_map()
            return
        next_instr.put(text)

    session.declare_subscriber("multileg/instruction", _on_instruction)
    session.declare_subscriber("rt/multileg/instruction", _on_instruction)
    print("\n[serve] ready -- waiting for an instruction on 'multileg/instruction' "
          "('__stop__' aborts the current plan, '__shutdown__' exits, Ctrl-C while idle exits)")

    try:
        while True:
            runner._abort = False
            text = None
            while text is None:
                try:
                    text = next_instr.get(timeout=0.5)
                except _queue.Empty:
                    if runner._exit_requested:
                        print("[serve] exiting (idle)")
                        return
                    continue
            if text == "__shutdown__":
                print("[serve] exiting (shutdown while idle)")
                return
            try:
                plan = runner.plan_from_instruction(text)
            except Exception as e:
                print(f"[serve] could not parse instruction {text!r}: {e}")
                continue
            print(f"\n[serve] new instruction: {text!r}")
            for i, st in enumerate(plan):
                print(f"  {i + 1}. {st!r}")
            running.set()
            try:
                runner.run(plan)
            finally:
                running.clear()
            if runner._exit_requested:
                print("[serve] exiting (interrupted during a run)")
                return
            print("[serve] plan finished -- ready for the next instruction")
    finally:
        # The one place --serve's odometry logger actually gets closed (see
        # PlanRunner._cleanup(close_odom=...)) -- every return above leaves
        # it open across instructions, so it must be closed here once the
        # server is genuinely going down (any of those returns, or Ctrl-C).
        runner._cleanup(close_odom=True)


if __name__ == "__main__":
    sys.exit(main())
