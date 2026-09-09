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
    r"^(?:go|walk|drive|move|head|navigate|proceed)\s+(?:up\s*to|to|towards?|toward|near)\s+"
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
Images: [1] the robot's current forward camera view. {crop_note}

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

    def check(self, rgb, clause: str, target: str, dino_state: str, crop=None) -> Optional[dict]:
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image
            img = Image.fromarray(rgb.astype("uint8"))
            images = [img]
            crop_note = ""
            if crop is not None and getattr(crop, "size", 0) and min(crop.shape[:2]) >= 8:
                images.append(Image.fromarray(crop.astype("uint8")))
                crop_note = "[2] a close crop of what the detector is currently locked on -- judge if it is the right object."
            prompt = SUPERVISOR_PROMPT.format(clause=clause, target=target, dino_state=dino_state,
                                              crop_note=crop_note)
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
                out = self._model.generate(**inputs, max_new_tokens=120, do_sample=False)
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
                 occupancy=None):
        self.node = node
        self.args = args
        self.supervisor = supervisor
        self.results: List[dict] = []
        self._abort = False
        self._exit_requested = False   # set by request_abort; --serve checks this to fully stop
        self._last_w = 0.0
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
        if not a.goal_memory or _goal_memory_load is None or _goal_memory_normalize is None:
            return False
        try:
            records = _goal_memory_load(Path(a.goal_memory_path))
        except Exception:
            return False
        records = [r for r in records if "world_xy" in r]
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
        if hit is None:
            return False
        gx, gy = hit["world_xy"]
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
                    route = thin_waypoints(path, spacing=a.goal_memory_waypoint_spacing_m)
                    print(f"  [{label}] [memory] A* route: {len(route)} waypoints, "
                          f"{info.get('length_m', 0):.1f}m over {info.get('cells', 0)} cells")
                else:
                    reason = info.get("reason", "no path") if isinstance(info, dict) else "no path"
                    print(f"  [{label}] [memory] A* found no route ({reason}) -- "
                          f"falling back to a straight line at the remembered point")
            except Exception as e:
                print(f"  [{label}] [memory] A* routing failed ({e}) -- falling back to a straight line")
        t_start = time.time()
        turn_sign: Optional[float] = None   # latched while |bearing| > 70deg -- see below
        while not self._abort and time.time() - t_start < a.goal_memory_approach_timeout_s:
            x, y, th = node.odom.x, node.odom.y, node.odom.theta
            if math.hypot(gx - x, gy - y) < a.goal_memory_arrive_m:
                print(f"  [{label}] [memory] within {a.goal_memory_arrive_m:.1f}m of the "
                      f"remembered point -- handing off to DINO")
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
            with node.lock:
                rgb, depth, intr = node.latest_rgb, node.latest_depth, node.latest_intrinsics
            if rgb is None:
                time.sleep(period)
                continue
            fwd, lat = body_delta(tx - x, ty - y, th)
            bearing = math.atan2(lat, fwd)
            # NavDP's diffusion trajectories are forward-biased (trained on
            # ego-forward motion) -- fed a remembered point that's actually
            # BEHIND the rover (|bearing| beyond ~70deg), none of the sampled
            # trajectories make real progress toward it, so pipe.step's GOTO
            # branch has no good option and the rover visibly stalls/creeps
            # sideways instead of turning around. Live-observed 2026-09-09:
            # a remembered air cooler at fwd=-0.92m (bearing=122deg) never
            # got approached, just timed out. Fix: turn in place to face the
            # remembered point first (same mechanism as the turn subtask,
            # targeting bearing instead of a fixed heading delta), THEN hand
            # off to NavDP's obstacle-aware GOTO once it's within a forward
            # cone NavDP can actually plan through.
            if abs(bearing) > math.radians(70.0):
                # 1.2*bearing is ALREADY saturated at +/-turn_max_angular
                # for this entire branch (1.2*radians(70) = 1.47rad, far
                # past turn_max_angular=0.6) -- so the only thing that ever
                # varies here is the SIGN of bearing, and atan2 is
                # numerically unstable exactly at +/-180deg: with the target
                # nearly directly behind, ordinary odometry noise flips
                # `lat`'s sign tick to tick, flipping bearing between just
                # under and just over +/-180deg -- and the commanded turn
                # direction flips WITH it, every tick, for as long as the
                # target sits back there. Live-observed 2026-09-09: reads as
                # the rover twitching/reversing direction mid-turn instead
                # of committing to one way around -- no DINO or Qwen
                # involved at all (this whole branch is pure odometry, see
                # the external_goal branch of pipeline.py's step() which
                # skips the detector entirely). Fix: decide the turn
                # direction ONCE on entry to this state and hold it until
                # the state is exited (bearing back under 70deg) -- a fresh
                # waypoint/re-entry gets to decide again, which is fine,
                # that's a genuinely new decision point.
                if turn_sign is None:
                    turn_sign = math.copysign(1.0, bearing)
                w = turn_sign * a.turn_max_angular
                node.publish_cmd(0.0, self._slew(w))
                self._integrate_occupancy()
                self._publish_status("object", f"{phrase} (memory approach: turning to face, "
                                               f"{math.degrees(bearing):+.0f}deg)")
                time.sleep(period)
                continue
            turn_sign = None   # exited the behind-us state -- next entry decides fresh
            try:
                gres = node.pipe.step(rgb, "", depth=depth, pose=(x, y, th), intrinsics=intr,
                                      external_goal=np.array([fwd, lat, 0.0], dtype=np.float32))
                node.publish_cmd(gres.linear, gres.angular)
            except Exception as e:
                print(f"  [{label}] [memory] approach step failed: {e} -- handing off to DINO")
                break
            self._integrate_occupancy()
            self._publish_status("object", f"{phrase} (memory approach)")
            time.sleep(period)
        else:
            print(f"  [{label}] [memory] approach timed out -- handing off to DINO")
        node.publish_cmd(0.0, 0.0)
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
            self._integrate_occupancy()
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
                   error_m=round(abs(travelled - signed_dist), 3),
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
                print(f"  [{label}] IMU never reached mag-calib >= {odom.imu_min_mag_calib} in "
                      f"{a.imu_calib_wait_s:.0f}s -- ABORTING (calib=[{odom.decode_calib(odom.last_imu_calib)}])")
                self._safe_stop()
                self.write_results(aborted_at=label, reason="imu_uncalibrated")
                self._cleanup()
                sys.exit(2)
            if time.time() - last_msg > 5.0:
                print(f"  [{label}] waiting for IMU mag calibration "
                      f"(calib=[{odom.decode_calib(odom.last_imu_calib)}]) ... "
                      f"drive/rotate the rover by hand or wait")
                last_msg = time.time()
            time.sleep(0.5)

        odom.start_new_goal(label, reset_pose=False)
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
        # Qwen reviews the RAW instruction clause (not the regex-cleaned
        # `phrase`) and decides the actual noun phrase Grounding-DINO should
        # search for -- the regex stripper (_strip_goto_prefix) only exists
        # as the seed/fallback below when no supervisor is loaded at all
        # (--no-supervisor); whenever Qwen IS available, it -- not a hand-
        # written pattern -- is what the raw instruction is handed to.
        if self.supervisor is not None:
            with node.lock:
                rgb0 = node.latest_rgb
            if rgb0 is not None:
                v0 = self.supervisor.check(rgb0, raw_clause, phrase,
                                           "not started yet -- no detection attempted")
                if v0 and v0.get("target"):
                    cand = str(v0["target"]).strip()
                    if cand and cand.lower() not in ("...", "none", "n/a", ""):
                        print(f'  [{label}] Qwen interprets "{raw_clause}" -> target \'{cand}\'')
                        phrase = cand
        node.target = phrase
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

        used_memory = self._recall_and_approach(phrase, label) if a.goal_memory_approach else False
        # Consume once -- _recall_and_approach() re-sets this fresh at the
        # start of the NEXT object leg's own call, but clear it here too so
        # nothing leaks if this leg's supervisor retargets DINO onto a
        # DIFFERENT phrase mid-leg (see the retarget branch below): the
        # remembered embedding was for the ORIGINAL recalled phrase only.
        recalled_embedding = self._recalled_appearance_embedding
        self._recalled_appearance_embedding = None
        last_det_box = None       # last DINO box seen, for _remember_arrival's appearance embed
        appearance_sim: Optional[float] = None      # set once, first live re-detection after recall
        appearance_checked = False

        while not self._abort and not node._goal_reached:
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
            if det is not None:
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
            if (self.supervisor is not None and st in ("SEARCH", "?")
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
                    v = self.supervisor.check(rgb, phrase, node.target, dstate, crop=crop)
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
                   outcome=outcome, used_memory=used_memory,
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

        for i, st in enumerate(plan):
            if self._abort:
                break
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
    p.add_argument("--metric-max-angular", type=float, default=0.6, help="heading-hold cap during a straight move, rad/s")
    p.add_argument("--turn-max-angular", type=float, default=0.6, help="in-place turn cap, rad/s")
    p.add_argument("--angular-slew-max", type=float, default=0.10, help="rad/s change per tick (0 disables)")
    p.add_argument("--fov", type=float, default=55.5, help="camera horizontal FOV deg (D435i bootstrap; camera_info overrides)")

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
    p.add_argument("--imu-calib-wait-s", type=float, default=30.0, help="max wait for IMU mag calibration before a turn (then abort)")
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
    p.add_argument("--search-creep-v", type=float, default=0.06,
                   help="forward speed while DINO has no lock -- replaces the pipeline's blind "
                        "in-place SEARCH spin so the view keeps changing (0 = spin in place)")
    p.add_argument("--search-turn", type=float, default=0.25,
                   help="turn rate (rad/s) for a supervisor left/right steering hint / scan while lost")
    p.add_argument("--search-creep-max-m", type=float, default=1.2,
                   help="stop the leg if it creeps this far with no DINO re-acquire -- stops the "
                        "rover driving off across the room past a target it just lost")
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
                   default=False,
                   help="opt-in: actually DRIVE toward a remembered point before DINO search "
                        "starts. Default OFF per user request -- recording a confirmed arrival "
                        "still happens (see --goal-memory), it just isn't auto-used to steer "
                        "anywhere until this is explicitly passed.")
    p.add_argument("--no-goal-memory-approach", dest="goal_memory_approach", action="store_false")
    p.add_argument("--goal-memory-path", type=str, default="logs/live_goal_memory.jsonl")
    p.add_argument("--goal-memory-approach-timeout-s", type=float, default=30.0)
    p.add_argument("--goal-memory-arrive-m", type=float, default=1.5,
                   help="stop the memory-guided approach once within this distance and hand off "
                        "to DINO -- never trust memory alone for the final approach")
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

    runner = PlanRunner(node, args, supervisor=supervisor, occupancy=occupancy)
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
                plan = plan_from_instruction(text)
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
