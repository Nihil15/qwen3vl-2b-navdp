#!/usr/bin/env python3
"""Memory-first + multi-candidate-verify variant of run_rover_multileg.py.

Two behaviour changes over the stock multi-subtask runner, both ON by default
here (the stock runner and LAUNCH/launch_rover_multileg_gui.sh are untouched):

 1. MEMORY FIRST, AUTHORITATIVE.  On every ``object:<phrase>`` leg the runner
    looks the phrase up in the live goal-memory journal FIRST
    (logs/live_goal_memory.jsonl -- tier-0 normalised text match, tier-1
    SigLIP2 paraphrase match).  If a confirmed past arrival exists, the rover
    drives to that remembered point (A* over the live occupancy grid, same
    machinery as run_rover_multileg's ``--goal-memory-approach``) and, once it
    is there, that counts as ``reached_memory`` -- the fresh Grounding-DINO
    search is NOT run.  Detection only happens when memory has nothing for the
    phrase (or the blind approach could not close the distance).

 2. MANY DINO BOXES, QWEN PICKS.  Stock behaviour locks the FIRST object leg
    onto Grounding-DINO's single top-scoring box.  Here, before the lock, the
    runner takes ALL of DINO's candidate boxes for the phrase and -- when two
    or more are comparably scored -- shows their crops to the Qwen supervisor
    and asks which one the instruction actually means.  Qwen's answer both
    (a) refines the phrase to something distinctive and (b) seeds the
    pipeline's tracker onto the chosen box for the first lock.

Everything else (Zenoh/odom plumbing, metric legs, occupancy carving, the
DINO+supervisor object loop, goal-memory recording) is inherited unchanged
from run_rover_multileg.PlanRunner.

Run via LAUNCH/launch_rover_memfirst_dino_gui.sh (GUI) or
LAUNCH/launch_rover_memfirst_dino.sh (headless).  Direct:

  python run_rover_memfirst_dino.py --pi-ip 10.146.234.125 --yes \\
      --subtask "object:red bucket" --subtask turn:-90 --subtask "object:red bucket"
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import run_rover_multileg as rrm

# bound now, before main() patches rrm.build_arg_parser -> our wrapper below,
# so the wrapper can call the original without recursing
_rrm_build_arg_parser = rrm.build_arg_parser


# ======================================================================
#  CLI  --  stock parser + one extra group (new flags default ON)
# ======================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = _rrm_build_arg_parser()
    g = p.add_argument_group("memory-first + multi-candidate verify")
    g.add_argument("--memory-first-authoritative", dest="memory_first_authoritative",
                   action="store_true", default=True,
                   help="on an object leg, check goal memory FIRST; on a hit, drive to the "
                        "remembered point and accept arrival there as success WITHOUT a fresh "
                        "DINO search. Only miss -> detection. Default ON.")
    g.add_argument("--no-memory-first-authoritative", dest="memory_first_authoritative",
                   action="store_false",
                   help="fall back to run_rover_multileg behaviour: memory is only a head start, "
                        "DINO still has to confirm the arrival.")
    g.add_argument("--memory-authoritative-arrive-m", type=float, default=1.5,
                   help="how close to the remembered point counts as 'arrived by memory' "
                        "(max of this and --goal-memory-arrive-m is used).")
    g.add_argument("--memory-verify-frame", dest="memory_verify_frame",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="after a memory approach reaches the remembered point, show the current "
                        "camera frame to Qwen and ask if the target is actually there. If NOT, "
                        "don't accept the memory arrival -- hand to a detection close-in with "
                        "--no-object-search (DINO drives to the target if it can see it; if it "
                        "can't, the leg ends -- no search-spin).")
    g.add_argument("--memory-verify-settle-s", type=float, default=0.8,
                   help="hold still this long (let motion blur clear) before the frame check.")
    g.add_argument("--dino-candidate-verify", dest="dino_candidate_verify",
                   action="store_true", default=True,
                   help="before the first lock of an object leg, if Grounding-DINO returns >=2 "
                        "comparably-scored boxes, ask the Qwen supervisor which one the "
                        "instruction means and lock onto that. Default ON.")
    g.add_argument("--no-dino-candidate-verify", dest="dino_candidate_verify",
                   action="store_false",
                   help="fall back to run_rover_multileg behaviour: lock onto DINO's top box.")
    g.add_argument("--dino-candidate-max", type=int, default=6,
                   help="cap on how many DINO candidate crops are shown to Qwen.")
    g.add_argument("--dino-candidate-margin", type=float, default=0.15,
                   help="only ask Qwen to disambiguate when the 2nd-best box score is within this "
                        "of the best (i.e. genuinely ambiguous). <=0 = always verify when >=2 boxes.")
    # This variant leans on memory, so the fact3r live entity map (SAM2 +
    # SigLIP2, metric positions -- see nav_pipeline/fact3r_live_memory.py) is
    # ON by default here, whereas run_rover_multileg.py keeps it opt-in.
    # PERSISTENCE stays OFF (the run_rover_multileg default): per user request
    # the entity map is per-session -- it lives in this --serve process and is
    # erased when the process ends / the GUI is closed, like the occupancy
    # grid. Pass --fact3r-memory-persist explicitly to keep it on disk.
    # Everything else about it (the --fact3r-* flags, _observe_entities /
    # _fact3r_recall / _drive_to_point) is inherited unchanged from
    # run_rover_multileg.PlanRunner.
    # RealSense depth fails on monitor screens / glass / black objects -> the
    # obstacle guard drives through them and fact3r never maps them. Fill
    # those holes from monocular depth by default in this variant.
    # Also soften NavDP's steering contribution (user 2026-09-10: "lower the
    # gain when navdp gives the waypoints") -- 0.6 = 40% less aggressive yaw
    # while following NavDP waypoints. Override with --navdp-angular-gain.
    #
    # goal_memory_paraphrase_min_sim 0.90 -> 0.94: SigLIP2 text cosine of
    # 'air cooler' vs a stored 'fan' arrival is 0.91, which cleared the 0.90
    # default and made an authoritative "reached_memory" at the fan's spot
    # (live 2026-09-10). Tighter here; and a tier-1 paraphrase hit is never
    # authoritative anyway now (see _memfirst_hit / go_to_object step 1b) --
    # it's a prior that DINO+supervisor must still confirm.
    # qwen_orchestrator ON by default here (user 2026-09-10: "the whole
    # instruction goes to qwen -> subtasks; qwen handles memory recalling,
    # else object search; qwen supervises the whole workflow").
    # goal_memory_arrive_m 1.5 -> 0.5: a "come back to X" recall drives blind on
    # memory as far as it can (to ~0.5 m of the remembered point, or until the
    # stall watchdog fires) and only THEN activates object search (user
    # 2026-09-10: "operate from memory to its maximum extent, then search").
    p.set_defaults(fact3r_live_memory=True, debug_steps=True, depth_fill_holes=True,
                   navdp_angular_gain=0.6, goal_memory_paraphrase_min_sim=0.94,
                   qwen_orchestrator=True, goal_memory_arrive_m=0.5)
    return p


# ======================================================================
#  Qwen supervisor  --  + a candidate-picker call
# ======================================================================
CANDIDATE_PICK_PROMPT = """A ground robot must go to the object named in this instruction:
INSTRUCTION: "{clause}"

Its open-vocabulary detector found {n} candidate region(s) for "{target}". You are shown:
  [1] the robot's full forward camera view.
  [2..{last}] one close crop per candidate, IN ORDER
      (candidate 1 = image [2], candidate 2 = image [3], ...).
Detector scores, in the same order: {scores}

Pick the ONE candidate crop that is the object the instruction refers to.
- If several crops show the SAME physical object, pick the clearest one.
- If NONE of the crops is the right object, answer index 0.

Also give a short DISTINCTIVE noun phrase (2-6 words, lowercase) that would help
the detector re-find that specific one next frame, e.g. "red bucket by the wall",
"left white air cooler".

Reply with ONLY a JSON object:
{{"index": <0..{n}>, "phrase": "..."}}"""


DECOMPOSE_PROMPT = """You are the PLANNER for a ground robot. Break the instruction below into an
ORDERED list of primitive subtasks. The only primitives are:

  {{"kind": "move",   "meters":  <number>}}   drive straight; NEGATIVE = backward
  {{"kind": "turn",   "degrees": <number>}}   turn in place; NEGATIVE = right/clockwise, POSITIVE = left
  {{"kind": "object", "phrase":  "<noun phrase>"}}   go to a named object (bare noun phrase, no "go to")

Rules:
- Keep the order of the instruction.
- "turn right" with no angle -> -90 ; "turn left" with no angle -> 90 ; "turn around" -> 180.
- "go to X" / "go back to X" / "come to X" / "come back to X" / "return to X" / "find X" /
  "head to X" -> {{"kind":"object","phrase":"X"}}
  (strip the leading verb and any article; keep distinguishing words, e.g. "white air cooler near the window").
- The SAME object named twice in one instruction is still two "object" subtasks -- do not drop the
  repeat. (A later visit to an object seen earlier just gets recalled from memory automatically.)
- A distance like "2.5 m" / "2 meters" / "50 cm" -> a move (cm -> convert to metres).
- If the instruction is just a single object with no motion verbs, still emit one object subtask.

INSTRUCTION: {instruction}

Reply with ONLY a JSON array of subtask objects, nothing else."""


MEMORY_DECISION_PROMPT = """You are supervising a ground robot about to go to an object.

TARGET: "{phrase}"

The robot has two memories it could recall a location from instead of searching:
  A) Confirmed past arrivals (text label + stored position):
{jsonl_block}
  B) A live map of {fact3r_count} object(s) it has seen while driving this session
     (no text labels; each will be visually re-checked against "{phrase}" if you choose to try it).

Decide how to proceed:
  "recall_jsonl" - one of the past-arrival labels in (A) clearly IS "{phrase}" (or an obvious synonym);
                   give which label in "label".
  "recall_map"   - no good label match, but it is worth letting the live map (B) try to find "{phrase}".
  "search"       - go straight to a fresh camera search; memory is unlikely to help.

Reply ONLY: {{"action": "recall_jsonl"|"recall_map"|"search", "label": "<exact label from A, or empty>", "reason": "<one short sentence>"}}"""


WORKFLOW_PROMPT = """You are supervising a ground robot executing a multi-step instruction.

ORIGINAL INSTRUCTION: "{instruction}"

Subtasks done so far (in order), with outcome:
{history}

The subtask that just finished: {last}
Remaining subtasks: {remaining}

Decide what the robot should do now:
  "continue" - the last subtask is good enough; move on.
  "retry"    - the last subtask clearly failed (e.g. an object leg ended 'timeout' / 'spin_stall' /
               'lost' / 'abort', or a turn/move errored badly) AND retrying it is worthwhile.
  "abort"    - the plan can't sensibly continue.

Reply ONLY: {{"action": "continue"|"retry"|"abort", "reason": "<one short sentence>"}}"""


VERIFY_REACHED_PROMPT = """The robot just drove to a location it REMEMBERED for an object -- BLIND, with no
live detection. Now check whether it actually worked.

TARGET: "{phrase}"

Look at the camera view [1]. Is the robot NOW at the target -- is "{phrase}"
clearly visible, close (roughly within 1.5 m), and near the centre / directly
ahead?
  reached = true  ONLY if "{phrase}" is right there in front of the robot.
  reached = false if it is far away, off to one side, only partly visible, not
            visible at all, or what's ahead is a different object.

Reply with ONLY: {{"reached": true|false, "reason": "<one short sentence>"}}"""


def _parse_json_value(reply: str):
    """First JSON value in a VLM reply -- a top-level ARRAY when one starts
    before any object (so a decompose reply of `[{...},{...}]` isn't
    truncated to its first element by a brace-only scanner), else the first
    balanced object. Strips a ```json fence. Returns dict / list / None."""
    import json as _json

    text = reply
    if "```" in text:
        for part in text.split("```"):
            p = part[4:] if part.lower().lstrip().startswith("json") else part
            if "[" in p or "{" in p:
                text = p
                break

    lb, cb = text.find("["), text.find("{")
    if lb != -1 and (cb == -1 or lb < cb):
        rb = text.rfind("]")
        if rb > lb:
            frag = text[lb:rb + 1]
            try:
                return _json.loads(frag)
            except Exception:
                # trailing junk after the array -> shrink to the last ']'
                # that parses
                for k in range(rb, lb, -1):
                    if text[k] == "]":
                        try:
                            return _json.loads(text[lb:k + 1])
                        except Exception:
                            continue
    v = rrm._extract_json(text)
    if v is None:
        rb = text.rfind("]")
        lb2 = text.find("[")
        if 0 <= lb2 < rb:
            try:
                v = _json.loads(text[lb2:rb + 1])
            except Exception:
                v = None
    if v is None:
        print(f"[orchestrator] unparsed VLM reply: {reply[:200]!r}")
    return v


class MemFirstSupervisor(rrm.TaskSupervisor):
    """rrm.TaskSupervisor + pick_candidate(). Same constructor / check() /
    verify_recall(); adds one more throttled VLM call used once per object
    leg, before the lock, to choose among DINO's candidate boxes."""

    def pick_candidate(self, rgb: np.ndarray, clause: str, target: str,
                       crops: List[np.ndarray], scores: List[float]) -> Optional[dict]:
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image

            images = [Image.fromarray(rgb.astype("uint8"))]
            for c in crops:
                images.append(Image.fromarray(c.astype("uint8")))
            prompt = CANDIDATE_PICK_PROMPT.format(
                clause=clause, target=target, n=len(crops), last=len(crops) + 1,
                scores=", ".join(f"#{i + 1}={s:.2f}" for i, s in enumerate(scores)))
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            text = self._processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
            inputs = self._processor(text=[text], images=images, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            reply = self._processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            v = rrm._extract_json(reply)
            if not v or "index" not in v:
                print(f"[supervisor] pick_candidate: unparsed reply {reply[:100]!r}")
                return None
            try:
                idx = int(v.get("index", 0))
            except (TypeError, ValueError):
                idx = 0
            phrase = str(v.get("phrase", "")).strip().lower() or None
            return {"index": idx, "phrase": phrase}
        except Exception as e:
            print(f"[supervisor] pick_candidate failed ({e}) -- keeping DINO's top box")
            return None

    # -- generic text/vision -> JSON --------------------------------
    def _ask_json(self, prompt: str, images: Optional[List[np.ndarray]] = None,
                  max_new_tokens: int = 320):
        """One deterministic VLM call, returns the first parsed JSON value
        (dict or list) or None. images optional -- most orchestrator calls
        are text-only."""
        self._ensure_loaded()
        import torch
        from PIL import Image
        pil = [Image.fromarray(np.asarray(im).astype("uint8")) for im in (images or [])]
        content = [{"type": "image", "image": im} for im in pil]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kw = dict(text=[text], return_tensors="pt")
        if pil:
            kw["images"] = pil
        inputs = self._processor(**kw).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return _parse_json_value(reply)

    def decompose(self, instruction: str) -> Optional[list]:
        v = self._ask_json(DECOMPOSE_PROMPT.format(instruction=repr(instruction)))
        if isinstance(v, dict):
            v = v.get("subtasks") or v.get("plan") or [v]
        return v if isinstance(v, list) else None

    def memory_decision(self, phrase: str, jsonl_labels: list, fact3r_count: int) -> Optional[dict]:
        block = "\n".join(f"      - {lbl}  @ ({x:+.2f},{y:+.2f})" for lbl, x, y in jsonl_labels) \
            or "      (none)"
        v = self._ask_json(MEMORY_DECISION_PROMPT.format(
            phrase=phrase, jsonl_block=block, fact3r_count=fact3r_count))
        return v if isinstance(v, dict) and v.get("action") in ("recall_jsonl", "recall_map", "search") else None

    def verify_reached(self, rgb: np.ndarray, phrase: str) -> Optional[dict]:
        v = self._ask_json(VERIFY_REACHED_PROMPT.format(phrase=phrase), images=[rgb], max_new_tokens=120)
        if isinstance(v, dict) and "reached" in v:
            return {"reached": bool(v.get("reached")), "reason": str(v.get("reason", ""))}
        return None

    def workflow_step(self, instruction: str, history: str, last: str, remaining: str) -> Optional[dict]:
        v = self._ask_json(WORKFLOW_PROMPT.format(
            instruction=instruction, history=history or "  (none)", last=last, remaining=remaining or "(none)"))
        return v if isinstance(v, dict) and v.get("action") in ("continue", "retry", "abort") else None


# ======================================================================
#  Runner  --  overrides only go_to_object
# ======================================================================
class MemFirstDinoRunner(rrm.PlanRunner):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._orchestrator_instruction = ""
        self._orchestrator_history: List[str] = []

    # ==================================================================
    #  Qwen orchestrator: plan + workflow supervision + memory decision
    # ==================================================================
    def plan_from_instruction(self, text: str) -> "List[rrm.Subtask]":
        """The WHOLE instruction goes to Qwen, which decomposes it into
        move/turn/object subtasks. Regex decomposer is the fallback."""
        self._orchestrator_instruction = text
        self._orchestrator_history = []
        if not getattr(self.args, "qwen_orchestrator", False) or self.supervisor is None:
            return rrm.plan_from_instruction(text)
        try:
            items = self.supervisor.decompose(text)
            plan: List[rrm.Subtask] = []
            for it in (items or []):
                if not isinstance(it, dict):
                    continue
                k = str(it.get("kind", "")).lower()
                if k == "move" and it.get("meters") is not None:
                    plan.append(rrm.Subtask("move", float(it["meters"]), text))
                elif k == "turn" and it.get("degrees") is not None:
                    plan.append(rrm.Subtask("turn", float(it["degrees"]), text))
                elif k == "object" and it.get("phrase"):
                    plan.append(rrm.Subtask("object", rrm._strip_goto_prefix(str(it["phrase"])), text))
            regex_plan = rrm.plan_from_instruction(text)
            if plan:
                # Safety net: if the instruction is clearly multi-clause
                # (has 'and'/'then'/',' joining verbs) but Qwen collapsed it
                # to fewer subtasks than the regex splitter found, trust the
                # longer one -- a dropped clause ("turn left AND go to the
                # bucket" -> just the turn) is the worse failure.
                if len(plan) < len(regex_plan) and rrm._CLAUSE_SPLIT_RE.search(text):
                    self._dbg(f"[orchestrator] Qwen plan has {len(plan)} subtask(s) but the "
                              f"instruction splits into {len(regex_plan)} -- using the regex plan "
                              f"so no clause is dropped: " + "  ".join(repr(s) for s in regex_plan))
                    return regex_plan
                self._dbg("[orchestrator] Qwen decomposed the instruction -> "
                          + "  ".join(repr(s) for s in plan))
                return plan
            self._dbg("[orchestrator] Qwen returned no usable subtasks -- regex fallback")
        except Exception as e:
            self._dbg(f"[orchestrator] Qwen decompose failed ({e}) -- regex fallback")
        return rrm.plan_from_instruction(text)

    def _on_subtask_done(self, st, res, remaining, retries) -> str:
        a = self.args
        outc = res.get("outcome", "?")
        self._orchestrator_history.append(f"  {st!r} -> {outc}")
        if not getattr(a, "qwen_orchestrator", False) or self.supervisor is None:
            return "continue"
        # cheap local gate: only bother the VLM when something looks wrong or
        # a retry budget remains for a bad object outcome
        bad = outc in ("timeout", "spin_stall", "lost", "abort", "imu_uncalibrated") or \
            (st.kind == "turn" and abs(res.get("error_deg", 0.0)) > 15.0) or \
            (st.kind == "move" and abs(res.get("error_m", 0.0)) > 0.5)
        if not bad:
            self._dbg(f"[orchestrator] subtask ok ({outc}) -- continue (no VLM call)")
            return "continue"
        try:
            v = self.supervisor.workflow_step(
                self._orchestrator_instruction or str(st),
                "\n".join(self._orchestrator_history[:-1]),
                self._orchestrator_history[-1].strip(),
                "  ".join(repr(s) for s in remaining))
        except Exception as e:
            self._dbg(f"[orchestrator] workflow_step failed ({e}) -- continue")
            return "continue"
        if not v:
            return "continue"
        act, why = v.get("action", "continue"), v.get("reason", "")
        if act == "retry" and retries >= getattr(a, "orchestrator_max_retries", 0):
            self._dbg(f"[orchestrator] Qwen wanted retry but budget spent -- continue ({why})")
            return "continue"
        self._dbg(f"[orchestrator] Qwen workflow verdict: {act.upper()} -- {why}")
        return act

    def _orchestrator_memory_choice(self, phrase: str, label: str) -> Tuple[Optional[str], Optional[str]]:
        """Ask Qwen whether to recall a remembered location or search fresh.
        Returns (action, recall_label): action is
        'recall_jsonl'/'recall_map'/'search' or None (keep the default
        cascade); recall_label is the CLEAN stored label Qwen matched (e.g.
        'red bucket' when the leg phrase was 'come to the red bucket') or
        None."""
        if not getattr(self.args, "qwen_orchestrator", False) or self.supervisor is None:
            return None, None
        try:
            recs = rrm._goal_memory_load(Path(self.args.goal_memory_path)) if rrm._goal_memory_load else []
        except Exception:
            recs = []
        stored_labels = [str(r.get("query_text", "?")) for r in recs if "world_xy" in r]
        labels = [(str(r.get("query_text", "?")), float(r["world_xy"][0]), float(r["world_xy"][1]))
                  for r in recs if "world_xy" in r][-8:]
        nf = len(self.fact3r_mem.entities) if self.fact3r_mem is not None else 0
        try:
            v = self.supervisor.memory_decision(phrase, labels, nf)
        except Exception as e:
            self._dbg(f"[{label}] [orchestrator] memory_decision failed ({e}) -- default cascade")
            return None, None
        if not v:
            return None, None
        act = v["action"]
        rl = str(v.get("label", "") or "").strip()
        # only honour a label that actually normalizes to a stored one
        if rl and rrm._goal_memory_normalize is not None:
            rn = rrm._goal_memory_normalize(rl)
            if not any(rrm._goal_memory_normalize(s) == rn for s in stored_labels):
                rl = ""
        self._dbg(f"[{label}] [orchestrator] Qwen memory decision: {act.upper()} "
                  f"(match label={rl or v.get('label','')!r}) -- {v.get('reason','')}")
        return act, (rl or None)

    def _frame_says_reached(self, phrase: str, label: str) -> Optional[bool]:
        """After a memory approach: settle, then ask Qwen if `phrase` is
        actually in the current frame. True/False, or None when the check is
        off / unavailable (caller then accepts the memory arrival as before)."""
        a = self.args
        if not getattr(a, "memory_verify_frame", False) or self.supervisor is None:
            return None
        node = self.node
        node.publish_cmd(0.0, 0.0)
        if not self._abort and getattr(a, "memory_verify_settle_s", 0.0) > 0:
            time.sleep(a.memory_verify_settle_s)
        with node.lock:
            rgb = None if node.latest_rgb is None else node.latest_rgb.copy()
        if rgb is None:
            return None
        try:
            v = self.supervisor.verify_reached(rgb, phrase)
        except Exception as e:
            self._dbg(f"[{label}] [mem-first] verify_reached failed ({e}) -- accepting memory arrival")
            return None
        if not v:
            return None
        self._dbg(f"[{label}] [mem-first] Qwen frame check for \"{phrase}\": "
                  f"reached={v['reached']} -- {v.get('reason','')}")
        return v["reached"]

    # -- memory lookup (no driving) -----------------------------------
    def _memfirst_hit(self, phrase: str) -> Optional[Tuple[dict, bool]]:
        """Return (record, is_exact) for `phrase` in the goal-memory JSONL, or
        None. is_exact=True means the stored query_text NORMALIZES to the same
        string (trustworthy); is_exact=False means it only matched via tier-1
        SigLIP2 text-paraphrase similarity ('air cooler' ~= stored 'fan' at
        0.91) -- a weak signal that must NOT be treated as authoritative."""
        a = self.args
        if not a.goal_memory or rrm._goal_memory_load is None or rrm._goal_memory_normalize is None:
            return None
        try:
            records = rrm._goal_memory_load(Path(a.goal_memory_path))
        except Exception:
            return None
        records = [r for r in records if "world_xy" in r]
        if not records:
            return None
        tnorm = rrm._goal_memory_normalize(phrase)
        # tier 0 -- exact normalized text match (newest wins)
        for r in reversed(records):
            if rrm._goal_memory_normalize(str(r.get("query_text", ""))) == tnorm:
                return r, True
        # tier 1 -- SigLIP2 paraphrase (only if the exact scan found nothing)
        embedder = self._ensure_appearance_embedder()
        if embedder is not None and rrm._goal_memory_find_hit is not None:
            try:
                hit = rrm._goal_memory_find_hit(phrase, records, embedder,
                                                min_similarity=a.goal_memory_paraphrase_min_sim)
                if hit is not None:
                    return hit, False
            except Exception:
                pass
        return None

    # -- many DINO boxes -> Qwen picks -------------------------------
    def _verify_candidates(self, phrase: str, raw_clause: str,
                           label: str) -> Tuple[str, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Returns (refined_phrase, seed) where seed is (box, rgb) to lock
        the pipeline's first pick onto, or None to leave stock behaviour."""
        a = self.args
        node = self.node
        det = getattr(node.pipe, "detector", None)
        if det is None:
            return phrase, None
        with node.lock:
            rgb = None if node.latest_rgb is None else node.latest_rgb.copy()
        if rgb is None:
            return phrase, None
        self._dbg(f"[candidate-verify] calling Grounding-DINO detect(\"{phrase}\") to list ALL boxes")
        try:
            dets = det.detect(rgb, phrase)
        except Exception as e:
            print(f"  [{label}] [verify] DINO detect failed ({e})")
            return phrase, None
        self._dbg(f"[candidate-verify] DINO returned {len(dets)} box(es): "
                  + "; ".join(f"'{getattr(d,'label','?')}'@{d.score:.2f}" for d in dets[:6]))
        if len(dets) < 2:
            self._dbg("[candidate-verify] <2 boxes -> nothing to disambiguate, using DINO top box")
            return phrase, None
        if a.dino_candidate_margin > 0 and (dets[0].score - dets[1].score) > a.dino_candidate_margin:
            self._dbg(f"[candidate-verify] top box wins by >{a.dino_candidate_margin} "
                      f"({dets[0].score:.2f} vs {dets[1].score:.2f}) -> not ambiguous, using it")
            return phrase, None  # top box is a clear winner, nothing to disambiguate

        H, W = rgb.shape[:2]
        crops: List[np.ndarray] = []
        kept = []
        for d in dets[:max(2, a.dino_candidate_max)]:
            x0, y0, x1, y1 = (int(v) for v in d.box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops.append(rgb[y0:y1, x0:x1].copy())
            kept.append(d)
        if len(kept) < 2:
            return phrase, None

        print(f"  [{label}] [verify] {len(kept)} candidate boxes for '{phrase}' "
              f"(scores {', '.join(f'{d.score:.2f}' for d in kept)}) -- asking Qwen which one")
        self._dbg(f"[candidate-verify] supervisor.pick_candidate(clause=\"{raw_clause}\", "
                  f"target=\"{phrase}\", {len(kept)} crops) -- Qwen returns which crop + a distinctive phrase")
        v = self.supervisor.pick_candidate(rgb, raw_clause, phrase, crops, [d.score for d in kept])
        self._dbg(f"[candidate-verify] Qwen pick_candidate -> {v}")
        if not v:
            return phrase, None
        idx = v.get("index", 0)
        refined = v.get("phrase") or phrase
        if not isinstance(idx, int) or idx < 1 or idx > len(kept):
            print(f"  [{label}] [verify] Qwen says none of the candidates match -- "
                  f"searching for '{refined}' with no seed")
            return refined, None
        chosen = kept[idx - 1]
        print(f"  [{label}] [verify] Qwen picked candidate #{idx} "
              f"(score {chosen.score:.2f}) -> target '{refined}', seeding the first lock")
        self._dbg(f"[candidate-verify] NOTE: the phrase became \"{refined}\" -- this is Qwen's "
                  f"'distinctive noun phrase' for the box it chose. If you asked for "
                  f"\"{raw_clause}\" and got e.g. \"chair with black jacket\", THIS is where "
                  f"that happened: Qwen described the chosen chair by its most salient feature.")
        return refined, (np.asarray(chosen.box, dtype=np.float32), rgb)

    # -- object leg -------------------------------------------------
    def go_to_object(self, phrase: str, timeout: float, label: str,
                     raw_clause: Optional[str] = None) -> dict:
        a = self.args
        node = self.node
        raw_clause = raw_clause or phrase

        def _reached_memory_res(t0, infer0, grounder, gx, gy):
            node.publish_cmd(0.0, 0.0)
            self._publish_status("object", phrase, force=True,
                                 memory={"event": "reached_memory", "phrase": phrase,
                                         "world_xy": [round(gx, 3), round(gy, 3)]})
            return dict(
                label=label, kind="object", instruction=phrase, final_target=phrase,
                grounder=grounder, outcome="reached_memory", used_memory=True, used_fact3r=(grounder == "fact3r-map"),
                end=[round(node.odom.x, 3), round(node.odom.y, 3), round(node.odom.theta, 4)],
                steps=node._infer_count - infer0, dur_s=round(time.time() - t0, 1),
                theta_src_final=node.odom.theta_source, supervisor=[])

        attempted_memory = False
        arrive_m = max(a.memory_authoritative_arrive_m, a.goal_memory_arrive_m)
        self._dbg(f"=== memfirst.go_to_object({label}) instruction=\"{raw_clause}\" ===")
        self._dbg(f"order: [1a] fact3r entity map -> [1b] JSONL goal-memory -> "
                  f"[2] DINO-candidate Qwen-verify -> [3] stock DINO+supervisor detection. "
                  f"memory_first_authoritative={a.memory_first_authoritative} (a hit in 1a/1b ends the leg, no detection)")

        # ---- 0. Qwen decides: recall from memory, or search fresh? ----
        # (only when --qwen-orchestrator; otherwise the default 1a->1b->3
        #  cascade below runs unchanged)
        _search_off = False   # set True if a memory approach's Qwen frame-check says "not reached"
        _mem_choice, _recall_label = self._orchestrator_memory_choice(phrase, label)
        _allow_fact3r = _mem_choice in (None, "recall_map")
        _allow_jsonl = _mem_choice in (None, "recall_jsonl", "recall_map")
        if _mem_choice == "search":
            self._dbg(f"[{label}] [orchestrator] Qwen: SEARCH -- skipping fact3r + JSONL recall, "
                      f"going straight to detection")
        if _recall_label and _recall_label.lower() != phrase.lower():
            # the leg phrase was e.g. "come to the red bucket" -- Qwen matched
            # it to the stored label "red bucket"; run the whole leg under the
            # clean label so _memfirst_hit / _recall_and_approach / DINO all
            # key on the same string the arrival was recorded under.
            self._dbg(f"[{label}] [orchestrator] leg phrase \"{phrase}\" -> recalling as "
                      f"\"{_recall_label}\" (the stored arrival label)")
            phrase = _recall_label

        # ---- 1a. FACT3R LIVE ENTITY MAP (authoritative) --------------
        # An object only ever seen in passing (never navigated to -> no
        # goal-memory JSONL record) can still be recalled here from the
        # SAM2+SigLIP2 entity map as a stored position.
        _f3_on = (a.memory_first_authoritative and a.fact3r_recall and _allow_fact3r
                  and self.fact3r_mem is not None and not self._abort)
        self._dbg(f"[1a] fact3r recall: {'RUNNING' if _f3_on else 'skipped'} "
                  f"(authoritative={a.memory_first_authoritative}, fact3r_recall={a.fact3r_recall}, "
                  f"map={'loaded' if self.fact3r_mem is not None else 'none'}, "
                  f"entities={len(self.fact3r_mem.entities) if self.fact3r_mem is not None else 0})")
        if _f3_on:
            attempted_memory = True   # ran the fact3r query -> super() must not re-run it
            fxy = self._fact3r_recall(phrase, label)
            if fxy is not None:
                t0, infer0 = time.time(), node._infer_count
                left = self._drive_to_point(fxy[0], fxy[1], phrase, label,
                                            a.goal_memory_approach_timeout_s)
                if left <= arrive_m and not self._abort:
                    if self._frame_says_reached(phrase, label) is not False:
                        print(f"  [{label}] [mem-first] arrived by fact3r entity map: {left:.2f} m "
                              f"from the stored position -- accepting")
                        return _reached_memory_res(t0, infer0, "fact3r-map", fxy[0], fxy[1])
                    print(f"  [{label}] [mem-first] fact3r point reached but Qwen says '{phrase}' "
                          f"is NOT in the frame -- detection close-in (search OFF)")
                    _search_off = True
                else:
                    print(f"  [{label}] [mem-first] fact3r approach left {left:.2f} m -- "
                          f"trying JSONL recall / detection")

        # ---- 1b. GOAL-MEMORY JSONL (authoritative) ------------------
        _js_on = a.memory_first_authoritative and a.goal_memory and a.goal_memory_approach and _allow_jsonl
        self._dbg(f"[1b] JSONL goal-memory lookup: {'RUNNING _memfirst_hit' if _js_on else 'skipped'}")
        if _js_on:
            _res = self._memfirst_hit(phrase)
            hit, is_exact = (_res if _res is not None else (None, False))
            attempted_memory = True   # ran the JSONL lookup -> super() must not re-run its recall
            self._dbg(f"[1b] _memfirst_hit(\"{phrase}\") -> "
                      f"{('HIT ' + repr(hit.get('query_text')) + (' [EXACT]' if is_exact else ' [paraphrase -- NOT authoritative]')) if hit else 'no record'}")
            if hit is not None:
                t0, infer0 = time.time(), node._infer_count
                gx, gy = hit["world_xy"]
                print(f"  [{label}] [mem-first] '{phrase}' {'is in goal memory' if is_exact else 'loosely matches stored ' + repr(hit.get('query_text'))} "
                      f"(remembered ({gx:+.2f},{gy:+.2f})) -- driving there as a {'trusted target' if is_exact else 'prior, then confirming with detection'}")
                used = self._recall_and_approach(phrase, label)
                # consume recall side-channel state so the stock leg below (if
                # we fall through) starts clean
                self._recalled_appearance_embedding = None
                self._recalled_world_xy = None
                self._recalled_hit_age_s = None
                dist = math.hypot(node.odom.x - gx, node.odom.y - gy)
                if used and dist <= arrive_m and not self._abort and is_exact:
                    if self._frame_says_reached(phrase, label) is not False:
                        print(f"  [{label}] [mem-first] arrived by memory (EXACT text match): "
                              f"{dist:.2f} m from the remembered point -- accepting")
                        return _reached_memory_res(t0, infer0, "memory", gx, gy)
                    print(f"  [{label}] [mem-first] remembered point reached but Qwen says "
                          f"'{phrase}' is NOT in the frame -- detection close-in (search OFF)")
                    _search_off = True
                elif used and dist <= arrive_m and not is_exact:
                    self._dbg(f"[1b] reached the remembered point for a PARAPHRASE match "
                              f"('{phrase}' ~= '{hit.get('query_text')}') -- NOT accepting blind; "
                              f"handing to DINO+supervisor to confirm what is actually here")
                elif not _search_off:
                    print(f"  [{label}] [mem-first] blind approach left {dist:.2f} m to go -- "
                          f"falling back to detection")

        # ---- 2. MANY DINO BOXES -> QWEN PICKS -----------------------
        refined, seed = phrase, None
        _cv_on = a.dino_candidate_verify and self.supervisor is not None and not self._abort
        self._dbg(f"[2] DINO-candidate verify: {'RUNNING' if _cv_on else 'skipped'} "
                  f"(dino_candidate_verify={a.dino_candidate_verify}, supervisor={'yes' if self.supervisor else 'no'})")
        if _cv_on:
            refined, seed = self._verify_candidates(phrase, raw_clause, label)
        self._dbg(f"[3] handing to stock DINO+supervisor leg with phrase=\"{refined}\" "
                  f"(seed box {'set' if seed is not None else 'none'})")

        # ---- 3. stock DINO + supervisor object leg -----------------
        saved_gma, saved_fr = a.goal_memory_approach, a.fact3r_recall
        saved_os = getattr(a, "object_search", True)
        if attempted_memory:
            # already tried memory (fact3r and/or JSONL) this leg -- don't let
            # super().go_to_object() blind-drive to either one again
            a.goal_memory_approach = False
            a.fact3r_recall = False
        if _search_off:
            # memory approach reached the point but Qwen said the target
            # isn't in view -> detection-only: DINO drives to it if it can
            # see it from here; if it can't, the leg ends (no search spin).
            a.object_search = False
            self._dbg(f"[3] search OFF for this leg -- DINO close-in to \"{refined}\" only")
        orig_reset = node.pipe.reset
        if seed is not None:
            box, srgb = seed

            def _reset_and_seed():
                orig_reset()
                try:
                    node.pipe._last_box = np.asarray(box, dtype=np.float32)
                    node.pipe._locked_embed = node.pipe._embed_or_none(srgb, np.asarray(box, dtype=np.float32))
                    node.pipe._box_miss_count = 0
                except Exception as e:
                    print(f"  [{label}] [verify] seed failed (ignoring): {e}")

            node.pipe.reset = _reset_and_seed
        try:
            res = super().go_to_object(refined, timeout, label, raw_clause=raw_clause)
        finally:
            node.pipe.reset = orig_reset
            a.goal_memory_approach, a.fact3r_recall = saved_gma, saved_fr
            a.object_search = saved_os
        if attempted_memory:
            res["used_memory"] = True
        return res


# ======================================================================
#  main  --  reuse rrm.main() with the subclasses patched in
# ======================================================================
def main() -> int:
    rrm.build_arg_parser = build_arg_parser
    rrm.PlanRunner = MemFirstDinoRunner
    rrm.TaskSupervisor = MemFirstSupervisor
    print("[memfirst] memory-first-authoritative + dino-candidate-verify + fact3r live "
          "entity map ON by default (all three opt-in in run_rover_multileg.py)")
    return rrm.main()


if __name__ == "__main__":
    import sys
    sys.exit(main())
