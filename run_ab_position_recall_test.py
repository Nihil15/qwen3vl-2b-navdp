#!/usr/bin/env python3
"""The full flow, end to end:

  1. The rover sees object A and drives to it with Qwen + NavDP (live vision).
  2. WHILE driving, object B is visible along the way, and fact3r's live entity
     memory stores it -- appearance AND a metric position.
  3. On arriving at A, the instruction becomes "go to object B".
  4. B is NOT visible from A. Nothing to re-identify, nothing to ground.
  5. So the memory is QUERIED by text, returns B's stored position, and NavDP
     drives there blind -- `pipeline.step(external_goal=...)`, state "GOTO",
     obstacle guard live, target never entering the camera.

Step 5 is why this uses `fact3r_live_memory.Fact3rLiveMemory` (positions) and
not `fact3r_entity_recall.Fact3rEntityRecallGuide` (appearance): the latter
re-identifies B when it comes back into view, which by construction never
happens here.

Frames come from habitat-sim rendering a real scanned HM3D house; ground truth
positions are used only to score the result, never given to the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from nav_pipeline.fact3r_goal_source import Fact3rGoalSource, MapToOdom
from nav_pipeline.fact3r_live_memory import (
    Fact3rLiveMemory, _enforce_reason_decision_consistency, parse_verification,
)
from nav_pipeline.live_occupancy import LiveOccupancy, thin_waypoints
from nav_pipeline.goal_utils import intrinsics_from_fov
from nav_pipeline.obstacle_guard import GuardConfig, apply_avoid_cooldown, depth_to_obstacle_points, forward_guard
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig, StepResult, bearing_to_angular
from run_ab_recall_hm3d_test import (
    EYE_HEIGHT_M, FOV_DEG, H, W,
    build_sim, category_instances, clearance_m, find_dataset_config, find_scene_glb,
    geodesic, largest_island, pick_pair, pick_start, render,
)


def nearest_instance(instances, xy, ref_height: float | None = None, max_height_diff: float = 1.5):
    """GOAT scores an `object` goal against the NEAREST instance of the
    category, because such a goal names a category with no instance and any
    instance satisfies it (fact3r/experiments/goat.py says the same). Scoring
    "go to the chair" against one hand-picked chair would fail a rover that
    correctly drove to a different, equally valid chair.

    `ref_height`, if given, filters out instances more than `max_height_diff`
    m away in Y (habitat's up-axis) before ranking by XY distance alone --
    this whole harness tracks pose as a flat (x, z, yaw) and renders from a
    FIXED cam_y for the entire run (never climbs stairs), so a "nearby"
    instance from a pure-XY distance can silently be on a different floor of
    a multi-story house. Measured real case (TbHJrupSAjP, sofa->bed): the
    XY-nearest sofa to a leg1 final pose was 0.55m away by (x,z) but 1.28m
    UP in Y (start floor was -2.59m) -- a real geodesic check confirmed it
    needs a 21.3m walk (via stairs) to actually reach, while the robot had
    only travelled 2.34m total. Without this filter that XY-adjacent-but-
    different-floor instance silently credited a false "success". Default
    1.5m is generous for one floor's real furniture height variation
    (chair/table/sofa AABB centers) while excluding a genuine story change."""
    if not instances:
        return None, float("inf")
    if ref_height is not None:
        on_floor = [p for p in instances if abs(float(p[1]) - ref_height) <= max_height_diff]
        if on_floor:
            instances = on_floor
    ds = [(dist((float(p[0]), float(p[2])), xy), p) for p in instances]
    ds.sort(key=lambda t: t[0])
    return ds[0][1], ds[0][0]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


_LIVE_WINDOW = "run_ab_position_recall_test -- live view (q to close)"
_live_view_open = True


def show_live(rgb: np.ndarray, lines: list[str]) -> None:
    """Live cv2 window: current camera frame + a text overlay (leg/step/
    state/etc, caller-supplied). No-op once the window's been closed with
    'q' -- the run keeps going headless after that rather than erroring."""
    global _live_view_open
    if not _live_view_open:
        return
    import cv2

    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 60), 1, cv2.LINE_AA)
    cv2.imshow(_LIVE_WINDOW, frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        _live_view_open = False
        cv2.destroyWindow(_LIVE_WINDOW)


def state_counts(log, leg):
    out: dict[str, int] = {}
    for r in log:
        if r["leg"] == leg:
            out[r["state"]] = out.get(r["state"], 0) + 1
    return out


# ----------------------------- live viewer feed ---------------------------- #
# When --live-dir is set, every phase drops an atomic snapshot (current camera
# frame + a top-down of the carved occupancy/trajectory + a state.json) into
# that directory. ab_recall_viewer_gui.py polls it to show the run in real
# time. Kept entirely optional and side-effect-free when the flag is unset.

def _atomic_write(path: "Path", data: bytes) -> None:
    import os
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _render_topdown(occ, log, markers, pose):
    """Top-down of the carved occupancy grid, rendered with planar +y UP
    (image row 0 = max y). All overlays are drawn AFTER the raster flip so
    text stays upright."""
    import cv2
    g = occ.grid
    n_rows = g.shape[0]

    def px(x, y):                       # planar (x, y) -> pixel (col, row), +y up
        row, col = occ.to_cell(x, y)
        return (int(col), int(n_rows - 1 - row))

    img = np.full((n_rows, g.shape[1], 3), 70, np.uint8)   # UNKNOWN grey
    img[g == 0] = (240, 240, 240)                          # FREE
    img[g == 1] = (35, 35, 35)                             # OCCUPIED wall
    img = cv2.flip(img, 0)                                 # raster only

    if log:
        pts = [px(r["pose"][0], r["pose"][1]) for r in log if r.get("pose")]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (255, 170, 30), 1, cv2.LINE_AA)

    cols = {"A": (0, 170, 0), "B": (0, 0, 220), "start": (180, 180, 0),
            "recalled": (220, 0, 220)}
    for name, xy in (markers or {}).items():
        if xy is None:
            continue
        col, row = px(xy[0], xy[1])
        c = cols.get(name, (255, 255, 255))
        cv2.circle(img, (col, row), 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, name, (col + 6, row + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, c, 1, cv2.LINE_AA)

    if pose is not None:
        col, row = px(pose[0], pose[1])
        tip = (int(col + 9 * math.cos(pose[2])), int(row - 9 * math.sin(pose[2])))
        cv2.arrowedLine(img, (col, row), tip, (0, 200, 255), 2, tipLength=0.45)

    nz = np.argwhere(cv2.flip(g, 0) != -1)
    if len(nz):
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
        m = 10
        img = img[max(0, r0 - m):r1 + m + 1, max(0, c0 - m):c1 + m + 1]
    h, w = img.shape[:2]
    scale = 460.0 / max(1, w)
    img = cv2.resize(img, (460, max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    return img


def emit_live(live_dir, rgb, meta: dict, occ=None, log=None, markers=None) -> None:
    if not live_dir:
        return
    import cv2, json
    d = Path(live_dir)
    try:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            _atomic_write(d / "frame.jpg", buf.tobytes())
        if occ is not None:
            td = _render_topdown(occ, log, markers, meta.get("pose"))
            ok, buf = cv2.imencode(".jpg", td, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                _atomic_write(d / "topdown.jpg", buf.tobytes())
        meta = dict(meta)
        meta["ts"] = time.time()
        _atomic_write(d / "state.json", json.dumps(meta).encode())
    except Exception as exc:   # a viewer feed must never break the run
        print(f"  [live] emit failed: {exc}")


def build_pipeline(args, guard: GuardConfig) -> DinoNavDPPipeline:
    cfg = PipelineConfig(
        device=args.device, horizontal_fov_deg=FOV_DEG, policy_type=args.policy,
        max_linear=args.max_linear, max_angular=args.max_angular,
        stop_distance=args.stop_distance, avoid_enabled=not args.no_avoid,
        use_dino=False, use_sam=False, use_clip=False, use_qwen_search=False,
        use_qwen_instruction=True, qwen_model_id=args.qwen_model_id,
        qwen_load_in_4bit=args.qwen_4bit,
        qwen_instruction_period_s=args.instruction_period_s,
        # use_appearance_reid loads the DINOv2 embedder (independent of DINO
        # detection); qwen_appearance_lock needs it to gate Qwen candidates on
        # appearance before the belief commits -- see PipelineConfig.
        use_appearance_reid=args.qwen_appearance_lock,
        qwen_appearance_lock=args.qwen_appearance_lock,
        qwen_appearance_min_similarity=args.qwen_appearance_sim,
        qwen_grounding_crops=args.qwen_grounding_crops,
        use_scene_tagger=False, use_belief_goal=True,
        search_angular=args.search_angular, guard=guard,
    )
    if args.goal_consistency_m is not None:
        # Default 1.5 m rejects EVERY fresh Qwen grounding that lands >1.5 m
        # from the current lock as "a different door/target" and coasts the
        # stale belief until it goes to SEARCH -- fatal for a plain "go to
        # the <object>" instruction where the object legitimately re-grounds
        # metres away as the camera closes in. Widen it (davinci uses 8 m).
        cfg.qwen_goal_consistency_m = args.goal_consistency_m
    return DinoNavDPPipeline(cfg, use_depth_estimator=False)


def step_pose(pose, v, w, dt, pf, island, cam_y):
    v, w = float(v), float(w)
    nx = pose[0] + v * math.cos(pose[2]) * dt
    ny = pose[1] + v * math.sin(pose[2]) * dt
    snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny]), island))
    pose[0], pose[1] = float(snapped[0]), float(snapped[2])
    pose[2] = wrap(pose[2] + w * dt)


class PurePursuitDriver:
    """A*-only local execution for BLIND driving legs: pure goal-bearing
    servo, with NavDP left out of the loop entirely -- reuses pipeline.py's
    own reactive obstacle-guard functions and formulas verbatim (same
    GuardConfig, same bearing_to_angular/apply_avoid_cooldown), just without
    its NavDP-sampled-trajectory contribution.

    Built after measuring exactly why NavDP-driven blind GOTO stalls: on a
    real leg 2 run (bed->cabinet -> chair->table swap, 2026-09-05),
    dist_to_recalled stayed FLAT at 2.72-2.79m for 75 of 78 steps -- NavDP's
    clearance-vetoed trajectory sampling has no visual grounding to work
    with on a blind leg (the goal isn't in frame), so in a cluttered room
    its candidates can all read as costly and "best of a bad set" nets near-
    zero progress. pipeline.py's own final command is ALREADY a blend of
    this exact open-space bearing-servo (weight 0 at slow_dist) with NavDP's
    trajectory (weight 1 at hard_stop_dist) -- see PipelineConfig.guard's
    slow_dist docstring. This class is that blend with the weight pinned to
    0: pure bearing-servo above hard_stop_dist, and pipeline.py's UNCHANGED
    reactive AVOID escape (same hysteresis, stall/wedge detection,
    escalating turn+reverse, cooldown bias) below it -- so close-range
    safety behavior is identical to every other test run today; only the
    normal-range driving law changes.

    Honest tradeoff: between hard_stop_dist and slow_dist, pipeline.py would
    proactively curve away from a nearing obstacle via NavDP's clearance
    scoring. This drives straight at the goal bearing through that whole
    band and relies on the hard AVOID trigger as the safety backstop instead
    of curving early -- expect more frequent AVOID engagements, not fewer,
    in a cluttered room; the point is escaping the stall, not smoothness.
    """

    def __init__(self, guard_cfg: GuardConfig, max_linear: float, max_angular: float,
                 ang_min_cmd: float = 0.12, servo_deadband: float = 0.05,
                 servo_ramp_deg: float = 35.0, avoid_confirm_ticks: int = 2,
                 avoid_stall_ticks: int = 10, avoid_stall_distance: float = 0.12,
                 avoid_cooldown_ticks: int = 8, avoid_bias_gain: float = 0.15):
        self.guard_cfg = guard_cfg
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.ang_min_cmd = ang_min_cmd
        self.servo_deadband = servo_deadband
        self.servo_ramp_deg = servo_ramp_deg
        self.avoid_confirm_ticks = avoid_confirm_ticks
        self.avoid_stall_ticks_cfg = avoid_stall_ticks
        self.avoid_stall_distance = avoid_stall_distance
        self.avoid_cooldown_ticks = avoid_cooldown_ticks
        self.avoid_bias_gain = avoid_bias_gain
        self._avoid_streak = 0
        self._avoid_stall_ticks = 0
        self._avoid_stall_anchor: Optional[tuple[float, float]] = None
        self._avoid_side = 0.0
        self._avoid_cooldown = 0

    def step(self, goal_point, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
             pose) -> StepResult:
        obstacle_pts = depth_to_obstacle_points(depth, fx, fy, cx, cy, self.guard_cfg)
        min_fwd, escape = forward_guard(obstacle_pts, self.guard_cfg)

        if min_fwd < self.guard_cfg.hard_stop_dist:
            self._avoid_streak += 1
            if self._avoid_streak >= self.avoid_confirm_ticks:
                if self._avoid_stall_anchor is None:
                    self._avoid_stall_anchor = (float(pose[0]), float(pose[1]))
                    self._avoid_stall_ticks = 0
                else:
                    moved = math.hypot(float(pose[0]) - self._avoid_stall_anchor[0],
                                       float(pose[1]) - self._avoid_stall_anchor[1])
                    if moved >= self.avoid_stall_distance:
                        self._avoid_stall_anchor = (float(pose[0]), float(pose[1]))
                        self._avoid_stall_ticks = 0
                    else:
                        self._avoid_stall_ticks += 1
                wedged = self._avoid_stall_ticks >= self.avoid_stall_ticks_cfg
                linear = -0.5 * self.max_linear if (min_fwd < self.guard_cfg.reverse_dist or wedged) else 0.0
                urgency = 1.0 if wedged else float(np.clip(
                    (self.guard_cfg.hard_stop_dist - min_fwd)
                    / max(self.guard_cfg.hard_stop_dist - self.guard_cfg.reverse_dist, 1e-6),
                    0.0, 1.0))
                angular = escape * (self.ang_min_cmd + urgency * (self.max_angular - self.ang_min_cmd))
                self._avoid_side = escape
                self._avoid_cooldown = self.avoid_cooldown_ticks
                return StepResult(linear=linear, angular=angular, state="AVOID",
                                  obstacle_points=obstacle_pts, min_forward=min_fwd)
        else:
            self._avoid_streak = 0
            self._avoid_stall_ticks = 0
            self._avoid_stall_anchor = None

        bearing = float(np.arctan2(goal_point[1], goal_point[0]))
        angular = bearing_to_angular(bearing, self.max_angular, self.ang_min_cmd,
                                     self.servo_deadband, math.radians(self.servo_ramp_deg))
        linear = self.max_linear * max(0.2, 1.0 - 0.8 * abs(angular) / self.max_angular)
        angular, self._avoid_cooldown = apply_avoid_cooldown(
            angular, "GOTO", self._avoid_side, self._avoid_cooldown, self.avoid_bias_gain, self.max_angular)
        return StepResult(linear=linear, angular=angular, state="GOTO",
                          obstacle_points=obstacle_pts, min_forward=min_fwd)


def verify_arrival_frame(rgb: np.ndarray, category: str, qwen, max_new_tokens: int = 200) -> dict:
    """Ask Qwen3-VL whether a just-declared STOP is really at `category`,
    from the single live frame at the moment of arrival (no stored crops --
    this checks a fresh self-declared STOP, not a memory entity, so mirrors
    Fact3rLiveMemory.verify()'s prompt/parsing but for one raw frame).

    Built directly in response to visual evidence, not a hunch: two
    different runs (category 'chair' and category 'bed') both self-declared
    arrival while the saved leg1_arrived_at_A.png showed a bookshelf / a
    fireplace -- no chair or bed anywhere in frame. Traced to pipeline.py's
    own AVOID-triggered STOP heuristic ("a close obstacle near our own
    tracked goal is probably the goal", guard.slow_dist~3.2m) declaring
    arrival on proximity alone, with no category check. This is the caller-
    level guard for that: pipeline.py itself is untouched (shared by every
    launcher), this only gates whether THIS script's loop trusts a STOP."""
    import json as _json

    import torch
    from PIL import Image

    qwen._ensure_loaded()
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((224, 224))
    prompt = (
        "A mobile robot just declared it has ARRIVED at its navigation target because "
        "something is very close in front of it. Look at this image (the robot's current "
        "camera view) and judge whether the close object filling/near the center of frame "
        f"is really the target.\n\nTarget: {_json.dumps(category.strip())}\n\n"
        "Answer no if the close object is a DIFFERENT piece of furniture, a wall, or "
        "clutter, even if it's plausible-looking -- a wrong arrival is worse than a missed "
        "one. Answer uncertain only if nothing is clearly close/identifiable at all. Return "
        'ONLY one JSON object: {"decision":"yes|no|uncertain","confidence":0.0,'
        '"predicted_object":"short noun phrase","reason":"one short sentence"}'
    )
    content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    chat = qwen._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen._processor(text=[chat], images=[image], return_tensors="pt").to(qwen._model.device)
    with torch.no_grad():
        out = qwen._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = qwen._processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _enforce_reason_decision_consistency(category, parse_verification(answer))


class ArrivalVerifier:
    """Caller-level guard on a self-declared STOP -- see verify_arrival_frame
    for why this exists. Rejecting a false STOP alone isn't enough: the
    pipeline's own belief is still fixated on the same nearby wrong object
    and would just re-declare the same false STOP again next tick. So a
    rejection also forces an escape (same pattern as InstanceLock's mismatch
    handling), and the caller is expected to call pipe.reset() alongside a
    rejection (clears the stale belief) -- this class only owns the escape,
    not the reset.

    TURN-THEN-PUSH, not turn-only -- same lesson StuckRecovery already
    learned earlier the same day and this class initially failed to reuse.
    Measured directly (chair category, real run 2026-09-05): a turn-only
    escape left the rover's POSITION essentially frozen (pose moved <0.15m
    from step 20 to step 100 while yaw swept almost a full circle) -- with
    no net relocation, spinning in a cluttered spot just faces a NEW nearby
    object every few degrees, and the pipeline's own close-object STOP
    heuristic re-fires on each new thing in turn (a wall, a couch, a shelf)
    instead of ever actually leaving. Rotating a fixed sweep THEN forcing
    several ticks of forward push actually relocates the rover into new
    territory before the next STOP attempt, exactly like StuckRecovery.

    turn_ticks/push_ticks trimmed from StuckRecovery's own 12/15 (real cost
    found 2026-09-05, same exact episode replayed with the identical A/B
    pair + start pose as an earlier documented 0.54m clean success): this
    scene's real geometry near a bed triggers the pipeline's own close-
    object STOP heuristic VERY often (measured: 99 raw STOP declarations
    out of 220 ticks -- a coat rack, a wall, a staircase railing all
    triggered it once each). With the original 12/15 (27 ticks/rejection),
    just 3 real rejections consumed 81 ticks -- 37% of the whole leg1
    budget spent escaping instead of making progress, a real, measurable
    regression risk this verifier itself introduces on top of the false-
    accept problem it fixes. Shortened to reduce that tax; still long
    enough to actually relocate per the turn-only failure above, not yet
    independently re-verified as the right number for every scene."""

    def __init__(self, qwen, turn_ticks: int = 8, push_ticks: int = 8,
                 turn_w: float = 0.8, push_v: float = 0.15):
        self.qwen = qwen
        self.turn_ticks = turn_ticks
        self.push_ticks = push_ticks
        self.turn_w = turn_w
        self.push_v = push_v
        self._phase = None       # None | "turn" | "push"
        self._phase_left = 0
        self._turn_dir = 1.0
        self.reject_count = 0
        self.accept_count = 0
        self.just_rejected = False

    def check(self, rgb, res, category: str, v: float, w: float) -> tuple[float, float, bool, bool]:
        """Call every tick, after any other override. Returns
        (v, w, overridden, confirmed_stop) -- confirmed_stop is True only on
        the tick a verified arrival is accepted."""
        self.just_rejected = False

        if self._phase == "turn":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase, self._phase_left = "push", self.push_ticks
            return 0.0, self.turn_w * self._turn_dir, True, False
        if self._phase == "push":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase = None
            return self.push_v, 0.0, True, False

        if res.state != "STOP":
            return v, w, False, False

        verdict = verify_arrival_frame(rgb, category, self.qwen)
        if verdict.get("decision") == "yes":
            self.accept_count += 1
            return v, w, False, True

        self.reject_count += 1
        self.just_rejected = True
        print(f"  [arrival-verify] REJECTED STOP -- Qwen says this is "
              f"{verdict.get('predicted_object', '?')!r}, not {category!r} "
              f"({verdict.get('reason', '')[:100]})")
        self._turn_dir *= -1.0
        self._phase, self._phase_left = "turn", self.turn_ticks
        return 0.0, self.turn_w * self._turn_dir, True, False


class StuckRecovery:
    """Detects a stalled local minimum and overrides the commanded (v, w).

    Root cause this exists for: `step_pose`'s `pf.snap_point` clips any
    proposed move that leaves the navmesh (e.g. NavDP servoing a heading
    that points into a wall/obstacle) back onto the nearest navigable point.
    When that keeps happening tick after tick, the *commanded* v looks
    normal in the log but the *realized* displacement is ~0 -- the pipeline
    has no way to see this itself (it only sees rendered frames, not the
    navmesh clipping), so a local minimum silently eats the whole step
    budget instead of surfacing as an AVOID/replan event.

    A first version of this rotated in place and handed straight back to
    the policy. Measured result (bed->cabinet, HM3D): it fired, but net
    displacement toward the actual goal over the WHOLE stuck episode was
    ~0 anyway -- rotate-then-resume just re-locks onto the same blocked
    heading a tick later. So this is turn-THEN-COMMIT: after rotating by a
    fixed sweep, force several ticks of forward motion at a reduced,
    guard-respecting speed before returning control to the policy, so the
    escape actually relocates the robot into new territory rather than just
    repointing it. Escalates (bigger sweep, alternating direction) each time
    it re-triggers without the robot having moved on from where the
    previous escape left it.
    """

    def __init__(self, window: int = 30, min_progress_m: float = 0.15,
                 turn_ticks: int = 12, push_ticks: int = 15,
                 escape_w: float = 0.8, push_v: float = 0.15):
        self.window = window
        self.min_progress_m = min_progress_m
        self.turn_ticks = turn_ticks
        self.push_ticks = push_ticks
        self.escape_w = escape_w
        self.push_v = push_v
        self._history: list[tuple[float, float]] = []
        self._phase = None       # None | "turn" | "push"
        self._phase_left = 0
        self._escape_dir = 1.0
        self._escape_start_xy: Optional[tuple[float, float]] = None
        self.trigger_count = 0
        self.escalation = 0
        self.just_triggered = False

    def override(self, pose, v: float, w: float) -> tuple[float, float, bool]:
        self.just_triggered = False
        self._history.append((pose[0], pose[1]))
        if len(self._history) > self.window:
            self._history.pop(0)

        if self._phase == "turn":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase, self._phase_left = "push", self.push_ticks
            return 0.0, self.escape_w * self._escape_dir, True

        if self._phase == "push":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase = None
                # did the whole turn+push episode actually relocate us?
                if self._escape_start_xy is not None:
                    moved = float(np.hypot(pose[0] - self._escape_start_xy[0],
                                            pose[1] - self._escape_start_xy[1]))
                    self.escalation = 0 if moved > self.min_progress_m else self.escalation + 1
                self._history.clear()
            return self.push_v, 0.0, True

        if len(self._history) == self.window:
            p0 = np.array(self._history[0])
            p1 = np.array(self._history[-1])
            net = float(np.linalg.norm(p1 - p0))
            if net < self.min_progress_m:
                self._escape_dir *= -1.0  # alternate so it doesn't re-settle on the same heading
                self._escape_start_xy = (pose[0], pose[1])
                # escalate the turn size each consecutive failed escape --
                # a shallow sweep wasn't enough to clear whatever's blocking
                turn_ticks = self.turn_ticks * (1 + min(self.escalation, 3))
                self._phase, self._phase_left = "turn", turn_ticks
                self.trigger_count += 1
                self.just_triggered = True
                return 0.0, self.escape_w * self._escape_dir, True

        return v, w, False


class InstanceLock:
    """Appearance-locks leg 1's category-only grounding onto the FIRST
    instance it tracks, so a long obstacle detour can't silently swap the
    target to a different instance of the same category.

    `pipe.step()`'s own belief-coasting already rejects a fresh grounding
    candidate too far in WORLD position from the current lock -- but that
    protection depends on the dead-reckoned belief still being confident.
    After a long detour (many lost ticks -> sigma exceeds max -> forced
    SEARCH), the belief resets and the NEXT fresh grounding is accepted
    with no memory of what the original target looked like. Measured: with
    4 "bed" instances in one house, a 100/220-step obstacle-heavy detour
    ended 4.47m from the instance actually paired with B -- reached a
    DIFFERENT bed, confidently.

    This is a caller-level safety net, not a pipeline patch: it can't undo
    the pipeline's own internal belief state, but it CAN refuse to let the
    robot drive toward a re-acquisition that doesn't visually match the
    first one, forcing a brief search turn instead so a later re-grounding
    (from a different angle) gets another chance at the real target instead
    of confidently arriving at the wrong one.

    Reuses `memory`'s already-loaded SigLIP2 encoder (same primitive
    `fact3r_entity_recall.py` uses for leg 2 re-identification, applied here
    to leg 1's own targeting instead) -- no second model load."""

    def __init__(self, memory, similarity_threshold: float = 0.75,
                 crop_radius_px: int = 60, turn_ticks: int = 10, turn_w: float = 0.6):
        self.memory = memory
        self.similarity_threshold = similarity_threshold
        self.crop_radius_px = crop_radius_px
        self.turn_ticks = turn_ticks
        self.turn_w = turn_w
        self.reference_embedding = None
        self._turn_left = 0
        self._turn_dir = 1.0
        self.mismatch_count = 0
        self.just_triggered = False

    def _crop_embed(self, rgb, u: float, v: float):
        h, w = rgb.shape[:2]
        r = self.crop_radius_px
        x0, y0 = max(0, int(u - r)), max(0, int(v - r))
        x1, y1 = min(w, int(u + r)), min(h, int(v + r))
        if x1 <= x0 or y1 <= y0:
            return None
        emb = self.memory._embed_images([rgb[y0:y1, x0:x1]])
        return None if emb is None else emb[0]

    def check(self, rgb, res, v: float, w: float) -> tuple[float, float, bool]:
        """Call every tick, after any other command overrides. Returns
        (v, w, overridden)."""
        self.just_triggered = False
        if self._turn_left > 0:
            self._turn_left -= 1
            return 0.0, self.turn_w * self._turn_dir, True

        if res.qwen_pixel_goal is None:
            return v, w, False  # coasting tick -- nothing fresh to check

        self.memory._ensure_loaded()
        emb = self._crop_embed(rgb, *res.qwen_pixel_goal)
        if emb is None:
            return v, w, False

        if self.reference_embedding is None:
            self.reference_embedding = emb
            return v, w, False

        sim = float(np.dot(emb, self.reference_embedding))
        if sim >= self.similarity_threshold:
            # slow moving-average so lighting/angle drift over the walk is
            # tolerated without needing an exact match every time
            self.reference_embedding = 0.8 * self.reference_embedding + 0.2 * emb
            self.reference_embedding /= np.linalg.norm(self.reference_embedding) + 1e-12
            return v, w, False

        self.mismatch_count += 1
        self._turn_left = self.turn_ticks
        self._turn_dir *= -1.0
        self.just_triggered = True
        return 0.0, self.turn_w * self._turn_dir, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dataset", choices=("hm3d", "mp3d"), default="hm3d",
                    help="which real-scene dataset to load --scene from. Only one HM3D scene "
                         "(wcojb4TFT35) and one MP3D scene (17DRP5sb8fy, habitat-sim's public "
                         "example scene) are present on this machine -- 'mp3d' exists to test on "
                         "a genuinely DIFFERENT real house (11 chair / 4 table / 2 sofa / 2 bed "
                         "instances, verified 2026-09-05) instead of the one HM3D scene every "
                         "earlier run today used, since the exact same start position/room kept "
                         "reproducing the same local-minimum regardless of which fix was tried.")
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--mp3d-root", default="/home/gpu/Desktop/pineapple/mp3d_data")
    ap.add_argument("--scene-glb", default=None,
                    help="explicit mesh path override (e.g. a raw MP3D v1 house rotated to Y-up, "
                         "see fact3r-navdp-bridge memory) -- bypasses --scene-dataset's canonical "
                         "lookup; build_sim() bakes a navmesh at runtime if none is shipped")
    ap.add_argument("--mp3d-house", default=None,
                    help="path to a raw MP3D .house file -- loads real semantic category_instances() "
                         "on a raw scan via habitat_sim.scene.SemanticScene.load_mp3d_house(), so "
                         "--category-a/--category-b can be real categories (chair, table, cushion, "
                         "cabinet, plant, counter, ...) instead of needing the toy example scene's "
                         "pre-baked semantics. Only meaningful together with --scene-glb.")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--category-a", default="chair")
    ap.add_argument("--instruction-a", default=None,
                    help="override leg1's instruction (default: bare 'go to the {category_a}', "
                         "a category-only goal with no instance disambiguation -- GOAT scores this "
                         "against the NEAREST instance for exactly that reason). Pass a genuine "
                         "language-description goal instead, e.g. 'go to the chair near the piano', "
                         "to test whether a landmark-qualified description lets Qwen ground on the "
                         "SPECIFIC picked-A instance instead of whichever instance is closest/most "
                         "salient -- this is GOAT-Bench's actual 'description' goal modality, never "
                         "exercised by this harness before since leg1 always used a bare category.")
    ap.add_argument("--category-b", default="table")
    ap.add_argument("--query-b", default="a table", help="the text used to recall B from the map")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true")
    ap.add_argument("--instruction-period-s", type=float, default=0.0,
                    help="0 = ground on every tick. The gate is a WALL-CLOCK throttle while the "
                         "sim advances on a fixed dt, so any nonzero value makes the run depend on "
                         "model-latency jitter: which sim ticks get a fresh grounding shifts between "
                         "runs and the trajectory diverges (measured: the same config scored 0.30 m "
                         "and 2.51 m on consecutive runs). On a real robot the wall clock is the "
                         "right master; in an offline sim it just injects noise.")
    ap.add_argument("--map-every", type=int, default=12,
                    help="run the fact3r mapper every N SIM STEPS (not seconds) -- same "
                         "reproducibility argument as --instruction-period-s")
    ap.add_argument("--lookaround-turns", type=float, default=1.0,
                    help="full 2*pi in-place rotations to do AT A, after leg 1 arrives and before "
                         "recall queries the map, mapping every tick while turning. 0 disables it. "
                         "Exists because leg 1's beeline path to A only maps whatever happened to be "
                         "along that one route -- B, by definition, usually wasn't on it. Fixes "
                         "nothing about IDENTIFYING B (that's context crops / consistency-checking / "
                         "dedup, already in fact3r_live_memory.py); this only widens the chance B was "
                         "ever OBSERVED at all before recall is asked to find it.")
    ap.add_argument("--lookaround-angular-rate", type=float, default=0.5,
                    help="rad/s for the --lookaround-turns sweep")
    ap.add_argument("--explore-frontiers", type=int, default=2,
                    help="after the look-around, walk to this many frontier points (boundary "
                         "between seen and unseen space, see LiveOccupancy.frontier_points) before "
                         "giving up and querying recall, mapping en route with the same A*/waypoint "
                         "machinery leg 2 uses. 0 disables it. Exists because a spin in place at A "
                         "only sees what's visible FROM A -- if B is in a different room with no "
                         "sightline from A, no rotation ever sees it; walking toward a few "
                         "unexplored frontiers is what actually extends the seen area beyond A's "
                         "own room. Still bounded and NOT full house coverage -- it visits a few "
                         "nearby frontiers, not every room.")
    ap.add_argument("--explore-radius", type=float, default=6.0,
                    help="max distance from A a frontier point may be to be considered (keeps "
                         "this a bounded 'walk a bit further' step, not open-ended exploration)")
    ap.add_argument("--explore-max-steps-per-frontier", type=int, default=100)
    ap.add_argument("--min-observations", type=int, default=3)
    ap.add_argument("--recall-top", type=int, default=6,
                    help="how many SigLIP2-shortlisted candidates get sent to Qwen verification "
                         "-- the true object can be observed but rank below the cutoff (e.g. a "
                         "cluttered scene with many visually-similar distractors scoring higher "
                         "on raw SigLIP similarity), in which case it never gets a chance at "
                         "verification at all")
    ap.add_argument("--min-vlm-confidence", type=float, default=0.5)
    ap.add_argument("--max-position-spread", type=float, default=2.0)
    ap.add_argument("--cam-height", type=float, default=0.9,
                    help="camera height above the floor. The guard is told the SAME value, so "
                         "screening and driving agree; they silently disagreed before.")
    ap.add_argument("--pitch-deg", type=float, default=0.0,
                    help="downward camera tilt. Kept at 0 by default: the guard's depth->height "
                         "math (z_up = cam_height - y_cam) assumes a LEVEL camera, so a tilt it "
                         "isn't told about reads the floor as rising terrain. The near-field "
                         "blindness a tilt would fix is already gone now that the camera sits at "
                         "cam_height rather than 2.4 m -- floor is visible from 1.2 m out.")
    ap.add_argument("--min-ab", type=float, default=3.0, help="min A<->B geodesic (m)")
    ap.add_argument("--max-ab", type=float, default=14.0)
    ap.add_argument("--start-dist", type=float, default=4.0, help="desired start->A distance (m)")
    ap.add_argument("--start-xy", type=float, nargs=2, default=None, metavar=("X", "Z"),
                    help="override pick_start: snap this (x, z) to the navmesh and start there "
                         "instead. Use a pose with real line-of-sight to A (e.g. one verified by "
                         "find_ab_episode.py) when pick_start's 'past A, away from B' point lands "
                         "in a walled-off pocket.")
    ap.add_argument("--goal-consistency-m", type=float, default=None,
                    help="override PipelineConfig.qwen_goal_consistency_m (default 1.5). Fresh "
                         "Qwen groundings farther than this from the current goal lock are "
                         "rejected as noise; 1.5 is too tight for 'go to the <object>' -- try "
                         "5-8 so the goal actually tracks the object as you approach.")
    ap.add_argument("--min-start-geodesic", type=float, default=2.5)
    ap.add_argument("--min-clearance", type=float, default=1.5,
                    help="required forward clearance at the start pose (m)")
    ap.add_argument("--qwen-appearance-lock", action=argparse.BooleanOptionalAction, default=True,
                    help="gate leg1's Qwen pixel-goal candidates on DINOv2 appearance similarity "
                         "to the first-locked object, INSIDE pipeline.py, before the belief "
                         "commits (see PipelineConfig.qwen_appearance_lock). This is the fix for "
                         "the root cause found 2026-09-05: 3 of 4 identical replays failed at "
                         "3.08m because Qwen grounded on doors/a staircase railing right in front "
                         "of the rover, and no downstream fix (STOP radius, arrival verifier, "
                         "escape tuning, budget) could recover from a wrong lock the belief had "
                         "already accepted.")
    ap.add_argument("--qwen-appearance-sim", type=float, default=0.55,
                    help="cosine similarity floor for --qwen-appearance-lock")
    ap.add_argument("--qwen-grounding-crops", action=argparse.BooleanOptionalAction, default=True,
                    help="generate Qwen proposals from the full frame plus focused overlapping crops, "
                         "then let pipeline.py's appearance/continuity/clearance scorer select one; "
                         "use --no-qwen-grounding-crops for the pre-fix A/B baseline")
    ap.add_argument("--start-instance-filter", action=argparse.BooleanOptionalAction, default=True,
                    help="reject start candidates geodesically closer to a DIFFERENT instance of "
                         "A's category than to the paired one (added 2026-09-04 pt.4 to stop leg1 "
                         "walking to the wrong bed). --no-start-instance-filter reverts to the "
                         "older, looser selection: the filter rejects most candidates (46/60 "
                         "measured), so survivors are often cramped low-clearance poses, and leg1 "
                         "degraded right after it landed -- this flag exists to A/B that.")
    ap.add_argument("--min-a-clearance", type=float, default=0.8,
                    help="reject candidate A instances with no approach angle clearing this bar "
                         "before picking by geodesic distance (see pick_pair's own docstring) -- "
                         "0 or negative disables this and picks purely by distance, the original "
                         "behavior. Exists because leg1 can track the CORRECT instance the whole "
                         "time (0 instance-lock mismatches) and still fail to close the last 1-3m "
                         "if that instance itself has no clear approach -- measured 2026-09-05, "
                         "StuckRecovery firing 8 times in one 500-step run with near-zero net "
                         "progress toward a genuinely wedged-in target.")
    ap.add_argument("--success-distance", type=float, default=1.0,
                    help="GOAT-Bench counts a subtask successful within 1 m of the goal")
    ap.add_argument("--map-extent", type=float, default=30.0, help="occupancy grid side (m)")
    ap.add_argument("--map-res", type=float, default=0.10, help="occupancy cell size (m)")
    ap.add_argument("--robot-radius", type=float, default=0.25,
                    help="netted off the free-space distance transform when planning")
    ap.add_argument("--astar-nav-search-m", type=float, default=3.5,
                    help="how far occ.astar() may search around start/goal for a cell to "
                         "anchor A* on before giving up with 'no navigable cell nearby' -- "
                         "widened from LiveOccupancy.astar's own 2.0m default after a real "
                         "case (2026-09-05) where the rover's own final pose sat inside a "
                         "genuinely dense pocket of OCCUPIED cells (close furniture), so "
                         "neither start nor goal found an anchor within 2.0m even with "
                         "unknown cells passable")
    ap.add_argument("--waypoint-spacing", type=float, default=0.6,
                    help="A* returns cell-by-cell points 0.1 m apart; the bearing to the next "
                         "one is noise, so thin the route before steering to it")
    ap.add_argument("--waypoint-radius", type=float, default=0.45,
                    help="advance to the next waypoint within this distance (the LAST waypoint "
                         "uses --reached-radius instead)")
    ap.add_argument("--visual-reacquire", action=argparse.BooleanOptionalAction, default=True,
                    help="while blind-driving leg 2 toward the recalled position, periodically "
                         "probe with Qwen instruction-grounding (NavDP's pixel-goal path, same "
                         "as leg 1) and switch to it for the final approach if the real object "
                         "comes into view -- corrects for the recalled position's dead-reckoned "
                         "error instead of trusting it all the way in")
    ap.add_argument("--visual-recheck-every", type=int, default=15,
                    help="ticks between visual-acquisition probes while in blind mode")
    ap.add_argument("--visual-lock-patience", type=int, default=5,
                    help="consecutive non-TRACK ticks before giving up visual mode and "
                         "reverting to blind GOTO on the recalled position")
    ap.add_argument("--visual-max-deviation-m", type=float, default=3.0,
                    help="reject a visual TRACK lock whose implied world position is farther "
                         "than this from the recalled position -- a lookalike object elsewhere "
                         "grounds and TRACKs just as confidently as the real one, so a lock is "
                         "only trusted if it's plausibly the SAME object the recall step found")
    ap.add_argument("--blind-driver", choices=("purepursuit", "navdp"), default="navdp",
                    help="local execution for leg 2's BLIND driving ticks (visual reacquisition "
                         "ticks are unaffected either way). 'navdp' (default): the original "
                         "pipe.step(external_goal=...) path, letting NavDP's clearance-vetoed "
                         "trajectory sampling drive. 'purepursuit': pure goal-bearing servo + the "
                         "same reactive AVOID guard, NavDP not called -- built to test whether "
                         "removing NavDP from blind driving would fix an observed stall; an 8-trial "
                         "A/B comparison (2026-09-05, logs/ab_recall_session2/compare_*) found NO "
                         "improvement (navdp: mean leg2 error 2.48m, 31.5% AVOID, 0.97m net "
                         "progress; purepursuit: 3.26m, 43.7% AVOID, 0.55m progress) -- kept "
                         "available for further experimentation, not because it measured better.")
    ap.add_argument("--visual-reid", action=argparse.BooleanOptionalAction, default=True,
                    help="at each visual probe, try SigLIP re-identification against the "
                         "recalled entity's OWN stored appearance (memory.reidentify) before "
                         "falling back to Qwen text-grounding -- appearance-matches the actual "
                         "recalled object rather than re-deriving identity from the query text, "
                         "so it can't be fooled by a same-category lookalike the way grounding "
                         "'the cabinet' on a different cabinet-like object can")
    ap.add_argument("--visual-reid-threshold", type=float, default=None,
                    help="SigLIP similarity floor for a re-ID match (default: the memory's own "
                         "associate_threshold -- the same bar used to decide two observations "
                         "are one entity)")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=1.0)
    ap.add_argument("--reached-radius", type=float, default=1.0)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--max-steps-leg1", type=int, default=220)
    ap.add_argument("--max-steps-leg2", type=int, default=260)
    ap.add_argument("--live-view", action="store_true",
                    help="open a live cv2 window showing the current camera frame + "
                         "state/step overlay, updated every tick -- for watching a run "
                         "as it happens instead of only reading logs/saved frames after "
                         "the fact. Requires a reachable X11 display (DISPLAY set); run "
                         "in the FOREGROUND (not backgrounded/nohup) so the window can "
                         "actually open. Press 'q' in the window to close it early "
                         "(the run keeps going headless after that).")
    ap.add_argument("--live-dir", default=None,
                    help="directory to drop an atomic per-tick snapshot into "
                         "(frame.jpg + topdown.jpg + state.json) for "
                         "ab_recall_viewer_gui.py to display the run in real time. "
                         "Unlike --live-view this needs no X11 in the run process "
                         "and works fine backgrounded/nohup.")
    ap.add_argument("--out", default="logs/ab_position_recall")
    args = ap.parse_args()
    if args.live_dir:
        Path(args.live_dir).mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    import cv2

    guard = GuardConfig(cam_height=args.cam_height)
    if args.scene_glb:
        sim = build_sim(args.scene_glb, None, pitch_deg=args.pitch_deg, mp3d_house=args.mp3d_house)
    elif args.scene_dataset == "mp3d":
        mp3d_root = Path(args.mp3d_root)
        scene_glb = str(mp3d_root / "versioned_data" / "mp3d_example_scene_1.1" / args.scene / f"{args.scene}.glb")
        dataset_config = str(mp3d_root / "versioned_data" / "mp3d_example_scene_1.1" / "mp3d.scene_dataset_config.json")
        if not Path(dataset_config).is_file():
            dataset_config = None
        sim = build_sim(scene_glb, dataset_config, pitch_deg=args.pitch_deg)
    else:
        hm3d_root = Path(args.hm3d_root)
        sim = build_sim(find_scene_glb(hm3d_root, args.scene), find_dataset_config(hm3d_root),
                        pitch_deg=args.pitch_deg)
    pf = sim.pathfinder
    island = largest_island(pf)
    a_point, b_point, ab_geo = pick_pair(
        sim, island, args.category_a, args.category_b, args.min_ab, args.max_ab,
        min_a_clearance=(args.min_a_clearance if args.min_a_clearance > 0 else None),
        cfg=guard, cam_height=args.cam_height)
    # Every OTHER instance of A's category (there are often several -- see
    # pick_start's docstring): the start point must not sit closer to one of
    # THOSE than to the specific instance actually paired with B, or "go to
    # the bed" reliably walks to the wrong bed.
    # --no-start-instance-filter reverts to the pre-2026-09-04-pt.4 behavior:
    # that fix is correct in intent (it stops leg 1 walking to a DIFFERENT
    # instance of A's category) but its own notes flagged an unresolved side
    # effect -- it rejects the large majority of start candidates (measured
    # 46/60), so what survives is often a cramped, low-clearance pose, and
    # leg 1 degraded right after it landed. Exposed as a flag so the two can
    # actually be A/B compared instead of assumed.
    other_a_instances = ([inst for inst in category_instances(sim, args.category_a, island)
                          if not np.allclose(inst, a_point)]
                         if args.start_instance_filter else None)
    start_point, start_geo, clearance = pick_start(
        sim, pf, island, a_point, b_point, args.start_dist,
        min_clearance=args.min_clearance, cam_height=args.cam_height, cfg=guard,
        min_geodesic=args.min_start_geodesic, other_a_instances=other_a_instances)
    if args.start_xy is not None:
        snapped = np.array(pf.snap_point(
            np.array([args.start_xy[0], a_point[1], args.start_xy[1]]), island))
        if not np.all(np.isfinite(snapped)):
            raise SystemExit(f"--start-xy {args.start_xy} does not snap to island {island}")
        start_point = snapped
        start_geo = geodesic(pf, start_point, a_point) or float("nan")
        clearance = clearance_m(sim, np.array([start_point[0], start_point[2],
                                               math.atan2(a_point[2] - start_point[2],
                                                          a_point[0] - start_point[0])]),
                                float(start_point[1]) + args.cam_height, guard)
        print(f"[--start-xy] overriding pick_start -> {np.round(start_point, 2)} "
              f"(start->A {start_geo:.2f} m, clearance {clearance:.2f} m)")
    cam_y = float(start_point[1]) + args.cam_height

    a_xy = (float(a_point[0]), float(a_point[2]))
    b_xy = (float(b_point[0]), float(b_point[2]))
    start_xy = (float(start_point[0]), float(start_point[2]))
    yaw0 = math.atan2(a_xy[1] - start_xy[1], a_xy[0] - start_xy[0])
    print(f"A ({args.category_a}) @ {a_xy}   B ({args.category_b}) @ {b_xy}   A<->B geodesic={ab_geo:.2f} m")
    print(f"start @ {start_xy} (start->A {start_geo:.2f} m, clearance {clearance:.2f} m)")

    pipe = build_pipeline(args, guard)
    memory = Fact3rLiveMemory(device=args.device)
    print("loading SAM2 + SigLIP2 for the live map...")
    memory._ensure_loaded()

    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pose = np.array([start_xy[0], start_xy[1], yaw0])
    log: list[dict] = []
    occ = LiveOccupancy(start_xy, extent_m=args.map_extent, res=args.map_res, guard=guard)
    occ.set_fov(FOV_DEG)

    live_markers = {"A": a_xy, "B": b_xy, "start": start_xy, "recalled": None}

    # ---------------- leg 1: drive to A, mapping B on the way ---------------- #
    instruction_a = args.instruction_a or f"go to the {args.category_a}"
    print(f"\n=== Leg 1: '{instruction_a}' -- Qwen+NavDP, fact3r mapping in parallel ===")
    outcome1 = "max_steps"
    stuck1 = StuckRecovery()
    lock1 = InstanceLock(memory)
    arrival1 = ArrivalVerifier(pipe._qwen_pixel_guide)
    for i in range(args.max_steps_leg1):
        tick = time.time()
        rgb, depth = render(sim, pose, cam_y)
        res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                        instruction=instruction_a, intrinsics=intr)
        v_cmd, w_cmd, escaping = stuck1.override(pose, res.linear, res.angular)
        if not escaping:
            v_cmd, w_cmd, mismatched = lock1.check(rgb, res, v_cmd, w_cmd)
        else:
            mismatched = False
        confirmed_stop = False
        if not escaping and not mismatched:
            v_cmd, w_cmd, verify_overridden, confirmed_stop = arrival1.check(
                rgb, res, args.category_a, v_cmd, w_cmd)
        log.append({"leg": "leg1", "step": i, "pose": [float(p) for p in pose],
                    "state": res.state, "v": float(v_cmd), "w": float(w_cmd),
                    "escaping": escaping, "instance_mismatch": mismatched,
                    "qwen_pixel_goal": list(res.qwen_pixel_goal) if res.qwen_pixel_goal else None})
        if stuck1.just_triggered:
            print(f"  [stuck] step {i}: <{stuck1.min_progress_m}m over last {stuck1.window} steps "
                  f"-> escape #{stuck1.trigger_count} (turn {stuck1._escape_dir:+.0f}, escalation={stuck1.escalation})")
            # The escape (turn+push) only relocates the ROBOT -- pipe.step()
            # keeps running every tick throughout it (escaping=True only
            # overrides v/w, not whether pipe.step() is called), so its
            # belief just keeps coasting toward the SAME lock that caused
            # the stall in the first place, and often re-hits the same wall
            # the moment control returns. Measured real case (QUCTc6BB5sX,
            # sink->toilet): 4 separate stuck-recovery escapes, each
            # re-trapped within ~30 steps, net leg1 displacement ~0.5m over
            # the ENTIRE 220-step budget -- the physical escape was working
            # exactly as designed, but the STALE BELIEF it kept escaping
            # back into never was. pipe.reset() here discards that lock so
            # the ~27 ticks of the escape's own turn+push (which changes the
            # camera view a lot) get spent on a genuinely fresh re-ground
            # instead of coasting straight back to the blocked target.
            pipe.reset()
        if lock1.just_triggered:
            print(f"  [instance-lock] step {i}: re-acquisition doesn't match the locked "
                  f"{args.category_a} (mismatch #{lock1.mismatch_count}) -- turning away instead of driving there")
        if i == 0:
            cv2.imwrite(str(frames_dir / "leg1_start.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        occ.integrate(depth, pose, intr)   # the route home is carved on the way out

        if i % args.map_every == 0:
            n = memory.observe(rgb, depth, pose, intr, step=i)
            print(f"  [map] step {i}: +{n} proposals -> {len(memory.entities)} entities")

        if args.live_view:
            show_live(rgb, [
                f"leg1  step {i}/{args.max_steps_leg1}  state={res.state}",
                f"target: go to the {args.category_a}",
                f"v={v_cmd:+.2f} w={w_cmd:+.2f}  escaping={escaping} mismatched={mismatched}",
                f"arrival-verify: accepts={arrival1.accept_count} rejects={arrival1.reject_count}",
            ])
        if args.live_dir:
            emit_live(args.live_dir, rgb, {
                "phase": "leg 1 -- drive to A", "instruction": instruction_a,
                "step": i, "max_steps": args.max_steps_leg1, "state": res.state,
                "pose": [float(p) for p in pose],
                "dist_to_A_m": dist((pose[0], pose[1]), a_xy),
                "entities": len(memory.entities), "escaping": bool(escaping),
                "scene": (args.scene_glb or args.scene),
            }, occ=(occ if i % 3 == 0 else None), log=log, markers=live_markers)

        if confirmed_stop:
            outcome1 = "reached"
            cv2.imwrite(str(frames_dir / "leg1_arrived_at_A.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            break
        if arrival1.just_rejected:
            pipe.reset()   # break the belief's fixation on the wrong nearby object
        step_pose(pose, v_cmd, w_cmd, args.dt, pf, island, cam_y)
        rem = args.dt - (time.time() - tick)
        if rem > 0:
            time.sleep(rem)

    leg1_final = [float(p) for p in pose]
    err_a_picked = dist(leg1_final, a_xy)
    a_instances = category_instances(sim, args.category_a, island)
    _, err_a = nearest_instance(a_instances, (leg1_final[0], leg1_final[1]),
                                ref_height=cam_y - args.cam_height)
    print(f"outcome={outcome1}  final=({leg1_final[0]:.2f},{leg1_final[1]:.2f})  "
          f"error_to_nearest_{args.category_a}={err_a:.2f} m (picked instance {err_a_picked:.2f} m, "
          f"{len(a_instances)} instances)  entities={len(memory.entities)}  "
          f"states={state_counts(log,'leg1')}")

    # ---------------- look-around: widen COVERAGE before recall ------------- #
    # Leg 1's beeline path to A only mapped whatever happened to be along
    # THAT ONE route -- B, by definition, usually wasn't on it (measured:
    # a run where every one of 6 shortlisted candidates was >3m from the
    # true cabinet, because it was simply never in any leg-1 frame). No
    # verification or dedup fix can recall an object that was never
    # observed. This is the cheapest thing that helps: a scripted in-place
    # rotation at A, mapping every tick, before recall is asked to find
    # anything -- if B is visible from anywhere near A, this is what gives
    # memory a chance to have actually seen it.
    n_lookaround_entities_before = len(memory.entities)
    n_lookaround_new = 0
    if args.lookaround_turns > 0:
        n_steps = max(1, int(2 * math.pi * args.lookaround_turns / (args.lookaround_angular_rate * args.dt)))
        w_sweep = args.lookaround_angular_rate
        print(f"\n=== Look-around: {args.lookaround_turns:.1f} turn(s) at A, "
              f"mapping every tick ({n_steps} steps) ===")
        for j in range(n_steps):
            rgb, depth = render(sim, pose, cam_y)
            occ.integrate(depth, pose, intr)
            n = memory.observe(rgb, depth, pose, intr, step=args.max_steps_leg1 + j)
            log.append({"leg": "lookaround", "step": j, "pose": [float(p) for p in pose],
                       "state": "SPIN", "v": 0.0, "w": float(w_sweep)})
            if args.live_dir:
                emit_live(args.live_dir, rgb, {
                    "phase": "look-around at A -- widen the map before recall",
                    "step": j, "max_steps": n_steps, "state": "SPIN",
                    "pose": [float(p) for p in pose],
                    "entities": len(memory.entities),
                    "scene": (args.scene_glb or args.scene),
                }, occ=(occ if j % 3 == 0 else None), log=log, markers=live_markers)
            step_pose(pose, 0.0, w_sweep, args.dt, pf, island, cam_y)
        n_lookaround_new = len(memory.entities) - n_lookaround_entities_before
        print(f"  look-around added {n_lookaround_new} new entities -> {len(memory.entities)} total")

    # ---------------- explore: walk toward a few frontiers past A ----------- #
    # A spin in place only sees what's visible FROM A -- if B is in a
    # different room with no sightline from A (a real, common case: the
    # look-around above measured every candidate still >2m from the true
    # object because it was never in ANY leg-1-or-spin frame), no rotation
    # ever sees it. This walks toward a few nearby frontier points --
    # boundary cells between what's been seen and what hasn't -- using the
    # exact same A*-plan + waypoint-follow + StuckRecovery machinery leg 2
    # already relies on, mapping continuously en route. Bounded by
    # --explore-frontiers (how many) and --explore-radius (how far), so
    # this is "walk a bit further", not full house coverage.
    n_explore_new = 0
    frontiers_visited = 0
    if args.explore_frontiers > 0:
        n_before_explore = len(memory.entities)
        targets = occ.frontier_points((pose[0], pose[1]), robot_radius=args.robot_radius,
                                      max_radius=args.explore_radius, k=args.explore_frontiers)
        print(f"\n=== Explore: {len(targets)} frontier point(s) beyond A, mapping en route ===")
        for fi, target in enumerate(targets):
            # allow_unknown=True: a frontier point is, by definition, at the
            # boundary of what's been seen -- astar's own strict clearance
            # test (grid == FREE only, same issue frontier_points had) will
            # often refuse to route onto it at all. Frontiers exist
            # specifically to be walked toward past the known edge, so this
            # is the one caller that should reach for allow_unknown first,
            # not as a fallback.
            path, info = occ.astar((pose[0], pose[1]), target, robot_radius=args.robot_radius,
                                   allow_unknown=True, nav_search_m=args.astar_nav_search_m)
            if path is None:
                # Even with allow_unknown, A* can still refuse a route that
                # was, in fact, just walked -- e.g. one noisy depth reading
                # during leg 1 marked a cell OCCUPIED forever (`_carve_free`
                # never clears an occupied cell once set) somewhere on the
                # real corridor. Rather than skip the frontier outright,
                # fall back to the same blind-dead-reckon-with-live-
                # obstacle-guard mechanism leg 2's own no-route case uses --
                # it doesn't need a pre-planned route, just a direction.
                print(f"  frontier {fi}: no route even with allow_unknown "
                      f"({info.get('reason', info)}) -- blind dead-reckon fallback")
                waypoints = [(pose[0], pose[1]), tuple(target)]
            else:
                waypoints = thin_waypoints(path, spacing=args.waypoint_spacing)
            if len(waypoints) < 2:
                continue
            wp_i = 1
            src = Fact3rGoalSource(waypoints[wp_i], MapToOdom.identity(),
                                   reached_radius=args.waypoint_radius)
            stuck_explore = StuckRecovery()
            j, arrived = 0, False
            while j < args.explore_max_steps_per_frontier:
                rgb, depth = render(sim, pose, cam_y)
                res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                                external_goal=src.tick(pose), intrinsics=intr)
                v_cmd, w_cmd, escaping = stuck_explore.override(pose, res.linear, res.angular)
                occ.integrate(depth, pose, intr)
                if j % args.map_every == 0:
                    memory.observe(rgb, depth, pose, intr, step=20000 + fi * 1000 + j)
                log.append({"leg": "explore", "step": j, "frontier": fi,
                           "pose": [float(p) for p in pose], "state": res.state,
                           "v": float(v_cmd), "w": float(w_cmd), "escaping": escaping})
                if args.live_dir:
                    emit_live(args.live_dir, rgb, {
                        "phase": f"explore -- frontier {fi + 1}/{len(targets)} past A",
                        "step": j, "max_steps": args.explore_max_steps_per_frontier,
                        "state": res.state, "pose": [float(p) for p in pose],
                        "entities": len(memory.entities), "escaping": bool(escaping),
                        "scene": (args.scene_glb or args.scene),
                    }, occ=(occ if j % 3 == 0 else None), log=log, markers=live_markers)
                if src.reached(pose) and not escaping:
                    if wp_i >= len(waypoints) - 1:
                        arrived = True
                        break
                    wp_i += 1
                    src = Fact3rGoalSource(waypoints[wp_i], MapToOdom.identity(),
                                           reached_radius=args.waypoint_radius)
                step_pose(pose, v_cmd, w_cmd, args.dt, pf, island, cam_y)
                j += 1
            frontiers_visited += int(arrived)
            print(f"  frontier {fi} @ ({target[0]:.2f},{target[1]:.2f}): "
                  f"{'arrived' if arrived else 'max steps'} after {j} steps, "
                  f"{len(memory.entities)} entities so far")
        n_explore_new = len(memory.entities) - n_before_explore
        print(f"  explore added {n_explore_new} new entities -> {len(memory.entities)} total")

    # One physical object seen from different angles across the walk often
    # fails observe()'s conservative live per-tick threshold and fragments
    # into several entities -- merge those before recall sees the shortlist
    # (see Fact3rLiveMemory.merge_nearby_duplicates's docstring).
    n_merged = memory.merge_nearby_duplicates()
    if n_merged:
        print(f"  merged {n_merged} fragmented duplicate(s) -> {len(memory.entities)} entities remain")

    # ---------------- recall: B is NOT visible; query the map ---------------- #
    print(f"\n=== Recall: query the map for {args.query_b!r} (B is not in view) ===")
    rgb_at_a, _ = render(sim, pose, cam_y)
    cv2.imwrite(str(frames_dir / "recall_view_from_A.png"), cv2.cvtColor(rgb_at_a, cv2.COLOR_RGB2BGR))
    if args.live_dir:
        emit_live(args.live_dir, rgb_at_a, {
            "phase": f"recall -- querying the map for {args.query_b!r}",
            "step": 0, "max_steps": 0, "state": "RECALL",
            "pose": [float(p) for p in pose], "entities": len(memory.entities),
            "scene": (args.scene_glb or args.scene),
        }, occ=occ, log=log, markers=live_markers)

    ent, position, chosen, records = memory.query_verified(
        args.query_b, pipe._qwen_pixel_guide, min_observations=args.min_observations,
        min_confidence=args.min_vlm_confidence, max_position_spread=args.max_position_spread,
        top=args.recall_top)
    # Same call `query_verified` made internally to build `records` -- redone
    # here (cheap: SigLIP text-embed only, no VLM calls) purely to get the
    # actual LiveEntity objects back so every shortlisted candidate's crops
    # can be saved, not just the one that won. Debugging "did it pick the
    # right object" needs to see what it was choosing BETWEEN.
    shortlist_entities = {e.entity_id: e for _, e in memory.ranked(
        args.query_b, min_observations=args.min_observations, top=args.recall_top)}
    for rank, rec in enumerate(records, 1):
        px, py = rec["position_xy"]
        v = rec["vlm"]
        print(f"  rank {rank}: {rec['entity_id']} siglip={rec['siglip_score']:.4f} "
              f"obs={rec['observations']} pos=({px:.2f},{py:.2f}) "
              f"spread={rec['position_spread_m']:.2f} m dist_to_true_B={dist((px,py), b_xy):.2f} m")
        print(f"           qwen: {v['decision']} conf={v['confidence']:.2f} "
              f"{v.get('predicted_object','')!r} -- {v.get('reason','')}")
        rec["dist_to_true_B_m"] = dist((px, py), b_xy)
        cand = shortlist_entities.get(rec["entity_id"])
        if cand is not None:
            # Save what Qwen actually judged (context_crops, padded with
            # surrounding scene), not the tight SigLIP crop -- debugging
            # "why did it say yes" needs the same evidence Qwen had.
            for i, crop in enumerate((cand.context_crops or cand.crops)[:4]):
                cv2.imwrite(str(frames_dir / f"shortlist_rank{rank}_{rec['entity_id']}_view{i}.png"),
                            cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    result = {
        "scene": (args.scene_glb or args.scene), "category_a": args.category_a,
        "instruction_a": instruction_a, "category_b": args.category_b,
        "query_b": args.query_b, "blind_driver": args.blind_driver,
        "mechanism": "fact3r live entity memory (SigLIP2 shortlist -> Qwen3-VL verification) "
                     "-> stored POSITION -> NavDP blind GOTO (B never in view)",
        "protocol": {"name": "GOAT-Bench criterion", "success_distance_m": args.success_distance,
                     "note": "GOAT-Bench's official episode files are gated on HuggingFace (401), "
                             "so this runs GOAT's scoring criterion on an HM3D scene from the same "
                             "family, with A/B taken from the scene's own semantic annotations."},
        "camera": {"height_m": args.cam_height, "pitch_deg": args.pitch_deg,
                   "guard_cam_height_m": guard.cam_height},
        "ground_truth": {"A_xy": a_xy, "B_xy": b_xy, "A_B_geodesic_m": ab_geo, "start_xy": start_xy},
        "leg1": {"outcome": outcome1, "final_pose": leg1_final,
                 "error_to_nearest_instance_m": err_a, "error_to_picked_A_m": err_a_picked,
                 "instances_of_category": len(a_instances),
                 "success": bool(err_a < args.success_distance),
                 "entities_mapped": len(memory.entities), "fragments_merged": n_merged,
                 "lookaround_new_entities": n_lookaround_new,
                 "instance_lock_mismatches": lock1.mismatch_count,
                 "arrival_verify_rejections": arrival1.reject_count,
                 "arrival_verify_accepts": arrival1.accept_count,
                 "explore_new_entities": n_explore_new, "explore_frontiers_visited": frontiers_visited,
                 "state_counts": state_counts(log, "leg1"),
                 "steps": sum(1 for r in log if r["leg"] == "leg1")},
        "recall_shortlist": records, "recall": None, "leg2": None,
    }

    if ent is None:
        print("  no entity passed Qwen verification -- nothing to drive to")
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
        # leg1/lookaround/explore already carved this map -- save it here too,
        # not just on the leg2-reached path below, so report_ab_recall.py
        # (and anything else reading occupancy.npy) still works on a run
        # that fails at recall.
        np.save(out_dir / "occupancy.npy", occ.grid)
        if args.live_dir:
            emit_live(args.live_dir, rgb_at_a, {
                "phase": "DONE -- recall found no verified match for B",
                "step": 0, "max_steps": 0, "state": "DONE", "done": True,
                "pose": [float(p) for p in pose], "scene": (args.scene_glb or args.scene),
                "leg1_outcome": outcome1, "leg1_err_m": round(err_a, 2),
                "leg1_success": bool(result["leg1"]["success"]),
                "recall_err_m": None, "leg2_outcome": "not_run",
                "success_rate": 0.0,
            }, occ=occ, log=log, markers=live_markers)
        if args.live_view:
            cv2.destroyAllWindows()
        sim.close()
        return

    err_pos = dist(position, b_xy)
    print(f"  recalled {ent.entity_id}: position=({position[0]:.2f},{position[1]:.2f}) "
          f"qwen_conf={chosen['vlm']['confidence']:.2f} obs={ent.observations} "
          f"-> {err_pos:.2f} m from the true {args.category_b}")
    result["recall"] = {"entity_id": ent.entity_id, "siglip_score": chosen["siglip_score"],
                        "vlm": chosen["vlm"], "observations": ent.observations,
                        "position_xy": list(position), "position_spread_m": ent.position_spread,
                        "position_error_vs_true_B_m": err_pos}
    for i, crop in enumerate(ent.crops[:4]):
        cv2.imwrite(str(frames_dir / f"recall_{ent.entity_id}_view{i}.png"),
                    cv2.cvtColor(np.asarray(crop, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    live_markers["recalled"] = (float(position[0]), float(position[1]))

    # ---------------- leg 2: drive to the stored position, blind ------------- #
    print(f"\n=== Leg 2: 'go to B' -- A* over the carved map, NavDP follows, B never in view ===")
    print(f"  occupancy carved on leg 1: {occ.stats()}")
    path, info = occ.astar((leg1_final[0], leg1_final[1]), position,
                           robot_radius=args.robot_radius, nav_search_m=args.astar_nav_search_m)
    result["plan"] = {"occupancy": occ.stats(), "info": {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}}

    if path is None:
        # Tier 2: no route through CONFIRMED-free cells -- almost always the
        # case, since B by definition usually isn't on the leg-1 path. Retry
        # allowing UNKNOWN cells (not OCCUPIED -- a real wall still blocks)
        # at a penalty, so A* can propose a route through unexplored space
        # toward the goal instead of refusing outright. The live obstacle
        # guard still runs every tick during execution and reacts to
        # whatever is actually there; this only changes what route gets
        # proposed.
        print(f"  A* (free-only) found no route ({info.get('reason', info)}) -- "
              f"retrying with unknown cells passable at a penalty")
        path, info2 = occ.astar((leg1_final[0], leg1_final[1]), position,
                                robot_radius=args.robot_radius, allow_unknown=True,
                                nav_search_m=args.astar_nav_search_m)
        result["plan"]["info_free_only"] = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()}

        if path is not None:
            waypoints = thin_waypoints(path, spacing=args.waypoint_spacing)
            print(f"  A* (frontier) route: {info2['cells']} cells, {info2['length_m']:.2f} m "
                  f"-> {len(waypoints)} waypoints")
            result["plan"]["waypoints"] = [list(w) for w in waypoints]
            result["plan"]["fallback"] = "astar_through_unknown"
            result["plan"]["info"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in info2.items()}
        else:
            # Tier 3: still nothing (goal cell itself has no nearby navigable
            # cell even counting UNKNOWN, e.g. surrounded by OCCUPIED) --
            # last resort, the mechanism run_ab_recall_hm3d_test.py's leg 2
            # uses: steer straight at the recalled point and let the LIVE
            # obstacle guard handle avoidance reactively with no pre-planned
            # route at all.
            print(f"  A* (frontier) also found no route ({info2.get('reason', info2)}) -- "
                  f"falling back to blind dead-reckon GOTO")
            result["plan"]["fallback"] = "blind_dead_reckon_no_route"
            waypoints = [(leg1_final[0], leg1_final[1]), tuple(position)]
    else:
        waypoints = thin_waypoints(path, spacing=args.waypoint_spacing)
        print(f"  A* route: {info['cells']} cells, {info['length_m']:.2f} m "
              f"-> {len(waypoints)} waypoints")
        result["plan"]["waypoints"] = [list(w) for w in waypoints]
        result["plan"]["fallback"] = None

    pipe.reset()
    outcome2 = "max_steps"
    if len(waypoints) < 2:
        # A* snapped start and goal into the same navigable cell: the recalled
        # position is already within a cell of where the rover stands, so there
        # is nothing to drive. Not an error -- but it is only "success" if the
        # recalled position is actually B, which the metrics below decide.
        print("  route is a single cell -- already standing at the recalled position")
        waypoints = [ (leg1_final[0], leg1_final[1]), tuple(position) ]
    wp_i, i = 1, 0   # waypoints[0] is where the rover already stands
    src = Fact3rGoalSource(waypoints[wp_i], MapToOdom.identity(),
                           reached_radius=args.waypoint_radius, query=args.query_b)
    stuck2 = StuckRecovery()
    pursuit = PurePursuitDriver(guard, args.max_linear, args.max_angular)

    # Visual reacquisition: A* + external_goal gets the rover to the
    # RECALLED position, which is only ever as good as the live memory's
    # position estimate (measured 1-3 m off true B from position-spread
    # noise) -- blind dead-reckoning can never do better than that number.
    # NavDP's OTHER goal source -- Qwen instruction-grounded pixel-goal,
    # the exact mechanism leg 1 drives on -- can: if the real B comes into
    # camera view anywhere along the route, ground on it directly and let
    # NavDP's own pixel+depth 3D-goal geometry take over for the final
    # approach instead of trusting the recalled point. `instruction` is
    # silently ignored whenever `external_goal` is passed (pipeline.py's
    # own precedence), so this alternates single-call-per-tick between the
    # two goal sources rather than ever calling step() twice in one tick:
    # probe with `instruction` every `visual_recheck_every` blind ticks: a
    # real acquisition (state "TRACK") switches into visual mode outright
    # and USES that tick's command; while in visual mode, `patience`
    # consecutive non-"TRACK" ticks (lost it again) reverts to blind GOTO.
    visual_mode = False
    visual_fail_streak = 0
    visual_lock_count = 0
    reid_lock_count = 0
    visual_ticks = 0
    visual_reject_count = 0
    best_known_xy = position   # improves in-place if SigLIP re-ID fires; the
                                # Qwen-grounding fallback's plausibility gate
                                # checks against this, not the frozen recall
    # Two-tier visual reacquisition, checked in order of trustworthiness:
    #
    # 1. SigLIP RE-IDENTIFICATION (memory.reidentify): "does anything in
    #    this frame match `ent`'s OWN stored appearance?" -- appearance
    #    matching against the actual recalled entity's embedding bank, not
    #    a text label. Can't be fooled by a differently-shaped lookalike
    #    that merely fits the query text (a bed headboard can satisfy "the
    #    cabinet" query text; it can't satisfy "looks like THIS specific
    #    previously-observed cabinet"). When it fires, its geometry is a
    #    FRESH position fix -- strictly better than the stale recalled one,
    #    dead-reckoned from however long ago the entity was last actually
    #    seen -- so it replaces `best_known_xy`/`src`, not just gates a lock.
    #
    # 2. Qwen instruction-grounding (pipeline's own pixel-goal path, same
    #    mechanism leg 1 drives on): fallback when SigLIP finds nothing
    #    (re-ID needs the object at a recognisable angle/distance; text
    #    grounding can succeed from views re-ID misses, or vice versa).
    #    Grounding "the cabinet" with no prior fix on WHICH cabinet has no
    #    guarantee it's the one recalled -- a lookalike elsewhere in the
    #    house grounds and TRACKs just as confidently (measured: a run held
    #    a false lock for 200+ ticks and ended further from true B than
    #    blind dead-reckoning would have) -- so its lock is sanity-checked
    #    against `best_known_xy` (inverting the same local-frame transform
    #    Fact3rGoalSource.tick() uses) and rejected if implausibly far.
    #
    # `instruction` is silently ignored whenever `external_goal` is passed
    # (pipeline.py's own precedence), so only one of {reid, qwen, blind}
    # ever calls step() per tick.

    def goal_point_to_world(goal_point, pose) -> Optional[np.ndarray]:
        if goal_point is None:
            return None
        fwd, left = float(goal_point[0]), float(goal_point[1])
        x, y, th = pose
        return np.array([x + fwd * math.cos(th) - left * math.sin(th),
                         y + fwd * math.sin(th) + left * math.cos(th)])

    while i < args.max_steps_leg2:
        tick = time.time()
        rgb, depth = render(sim, pose, cam_y)

        probing = args.visual_reacquire and (not visual_mode) and args.visual_recheck_every > 0 \
                  and i > 0 and i % args.visual_recheck_every == 0

        reid_hit = None
        if ent is not None and args.visual_reid and (visual_mode or probing):
            reid_hit = memory.reidentify(rgb, depth, tuple(pose), intr, ent,
                                         threshold=args.visual_reid_threshold)

        if reid_hit is not None:
            (rx, ry), sim_score = reid_hit
            if not visual_mode:
                reid_lock_count += 1
                print(f"  [visual] step {i}: SigLIP re-identified the recalled entity "
                      f"in-frame (sim={sim_score:.3f}) -> fresh position fix, driving there")
            visual_mode, visual_fail_streak = True, 0
            visual_ticks += 1
            best_known_xy = (rx, ry)
            wp_i = len(waypoints) - 1
            src = Fact3rGoalSource(best_known_xy, MapToOdom.identity(), query=args.query_b,
                                   reached_radius=args.reached_radius)
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            external_goal=src.tick(pose), intrinsics=intr)
            mode = "visual_reid"
        elif visual_mode or probing:
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            instruction=args.query_b, intrinsics=intr)
            if res.state == "TRACK":
                world_xy = goal_point_to_world(res.goal_point, pose)
                plausible = world_xy is None or dist(world_xy, best_known_xy) <= args.visual_max_deviation_m
                if not plausible:
                    visual_reject_count += 1
                    print(f"  [visual] step {i}: REJECTED lock -- implied position "
                          f"{dist(world_xy, best_known_xy):.2f} m from the recalled point "
                          f"(> {args.visual_max_deviation_m} m) -- likely a lookalike, not B")
                    res.state, res.linear, res.angular = "SEARCH", 0.0, 0.0
                    if visual_mode:
                        visual_fail_streak += 1
                        if visual_fail_streak >= args.visual_lock_patience:
                            visual_mode = False
                else:
                    if not visual_mode:
                        visual_lock_count += 1
                        print(f"  [visual] step {i}: Qwen grounded '{args.query_b}' in-frame "
                              f"({dist(world_xy, best_known_xy):.2f} m from recalled -> plausible) "
                              f"-> switching to NavDP pixel-goal for final approach")
                    visual_mode, visual_fail_streak = True, 0
                    visual_ticks += 1
            elif visual_mode:
                visual_fail_streak += 1
                if visual_fail_streak >= args.visual_lock_patience:
                    print(f"  [visual] step {i}: lost '{args.query_b}' for "
                          f"{visual_fail_streak} ticks -> back to blind GOTO on the recalled point")
                    visual_mode = False
            mode = "visual_qwen"
        elif args.blind_driver == "purepursuit":
            # A*-only execution (see PurePursuitDriver): pure goal-bearing
            # servo toward this waypoint + the same reactive AVOID guard
            # pipeline.py itself uses, NavDP not called at all this tick.
            res = pursuit.step(src.tick(pose), depth, intr[0], intr[1], intr[2], intr[3], pose)
            mode = "blind"
        else:
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            external_goal=src.tick(pose), intrinsics=intr)
            mode = "blind"

        v_cmd, w_cmd, escaping = stuck2.override(pose, res.linear, res.angular)
        d_goal = dist((pose[0], pose[1]), position)
        log.append({"leg": "leg2", "step": i, "pose": [float(p) for p in pose],
                    "state": res.state, "v": float(v_cmd), "w": float(w_cmd),
                    "waypoint": wp_i, "dist_to_recalled": d_goal, "escaping": escaping,
                    "mode": mode,
                    "qwen_pixel_goal": list(res.qwen_pixel_goal) if res.qwen_pixel_goal else None})
        if stuck2.just_triggered:
            print(f"  [stuck] step {i}: <{stuck2.min_progress_m}m over last {stuck2.window} steps "
                  f"-> escape #{stuck2.trigger_count} (turn {stuck2._escape_dir:+.0f}, escalation={stuck2.escalation})")
        if i % 20 == 0:
            print(f"  [step {i:3d}] state={res.state:6s} mode={mode:6s} wp {wp_i}/{len(waypoints)-1} "
                  f"dist_to_recalled={d_goal:5.2f} m")
        if args.live_view:
            show_live(rgb, [
                f"leg2  step {i}/{args.max_steps_leg2}  state={res.state}  mode={mode}",
                f"target: {args.query_b}  (waypoint {wp_i}/{len(waypoints) - 1})",
                f"v={v_cmd:+.2f} w={w_cmd:+.2f}  dist_to_recalled={d_goal:.2f} m  escaping={escaping}",
            ])
        if args.live_dir:
            emit_live(args.live_dir, rgb, {
                "phase": "leg 2 -- blind drive to the recalled position of B",
                "instruction": f"go to {args.query_b}", "mode": mode,
                "step": i, "max_steps": args.max_steps_leg2, "state": res.state,
                "pose": [float(p) for p in pose], "dist_to_recalled_m": d_goal,
                "waypoint": f"{wp_i}/{len(waypoints) - 1}", "escaping": bool(escaping),
                "scene": (args.scene_glb or args.scene),
            }, occ=(occ if i % 3 == 0 else None), log=log, markers=live_markers)

        if mode == "visual_qwen" and res.state == "STOP" and not escaping:
            # A genuine visual arrival at the real object -- strictly better
            # ground truth than the recalled position's dead-reckoned
            # distance, since NavDP declared it from actually seeing B close
            # up, the same criterion leg 1 arrives on.
            outcome2 = "reached_visual"
            cv2.imwrite(str(frames_dir / "leg2_arrived_at_B.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            break
        # visual_reid drives via external_goal same as blind (no self-declared
        # STOP -- see pipeline.py's own note on why GOTO never does), just
        # toward a fresher position; both use src.reached() for arrival.
        if mode in ("blind", "visual_reid") and src.reached(pose) and not escaping:
            if wp_i >= len(waypoints) - 1:
                outcome2 = "reached"
                cv2.imwrite(str(frames_dir / "leg2_arrived_at_B.png"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                break
            wp_i += 1
            last = wp_i >= len(waypoints) - 1
            src = Fact3rGoalSource(
                waypoints[wp_i], MapToOdom.identity(), query=args.query_b,
                reached_radius=args.reached_radius if last else args.waypoint_radius)
        step_pose(pose, v_cmd, w_cmd, args.dt, pf, island, cam_y)
        i += 1
        rem = args.dt - (time.time() - tick)
        if rem > 0:
            time.sleep(rem)
    np.save(out_dir / "occupancy.npy", occ.grid)
    result["leg1"]["stuck_recovery_triggers"] = stuck1.trigger_count
    result["leg2_stuck_recovery_triggers"] = stuck2.trigger_count
    result["leg2_visual_reacquisition"] = {
        "enabled": args.visual_reacquire,
        "siglip_reid_enabled": args.visual_reid, "siglip_reid_lock_count": reid_lock_count,
        "qwen_grounding_lock_count": visual_lock_count, "qwen_grounding_reject_count": visual_reject_count,
        "ticks_in_visual_mode": visual_ticks}

    leg2_final = [float(p) for p in pose]
    # Same GOAT category-goal scoring as leg1 (see nearest_instance's own
    # docstring): the recall query (args.query_b, e.g. "a table") names a
    # CATEGORY with no distinguishing description, so any real instance of
    # it satisfies the goal -- b_xy is only the ONE instance pick_pair
    # happened to select for building a scoreable episode, not a specific-
    # instance target the pipeline was ever asked to find. Measured real
    # case this fixes: a run scored "fail" at 3.11m from b_xy was actually
    # 0.78m from the nearest real table -- a genuine category-goal success
    # mischaracterized as a failure by scoring against the wrong reference.
    b_instances = category_instances(sim, args.category_b, island)
    _, err_b_nearest = nearest_instance(b_instances, (leg2_final[0], leg2_final[1]),
                                        ref_height=cam_y - args.cam_height)
    err_b_picked = dist(leg2_final, b_xy)
    print(f"outcome={outcome2}  final=({leg2_final[0]:.2f},{leg2_final[1]:.2f})  "
          f"error_to_recalled={dist(leg2_final, position):.2f} m  "
          f"error_to_nearest_{args.category_b}={err_b_nearest:.2f} m "
          f"(picked instance {err_b_picked:.2f} m, {len(b_instances)} instances)  "
          f"states={state_counts(log,'leg2')}")
    result["leg2"] = {"outcome": outcome2, "final_pose": leg2_final,
                      "error_to_recalled_m": dist(leg2_final, position),
                      "error_to_nearest_instance_m": err_b_nearest,
                      "error_to_picked_B_m": err_b_picked,
                      "instances_of_category": len(b_instances),
                      "success": bool(err_b_nearest < args.success_distance),
                      "state_counts": state_counts(log, "leg2"),
                      "steps": sum(1 for r in log if r["leg"] == "leg2"),
                      "used_planning_fallback": result["plan"]["fallback"] is not None}
    subtasks = [{"index": 0, "goal": args.category_a, "distance_to_goal": err_a,
                 "success": result["leg1"]["success"]},
                {"index": 1, "goal": args.category_b, "distance_to_goal": err_b_nearest,
                 "success": result["leg2"]["success"]}]
    result["goat_metrics"] = {
        "subtasks": subtasks,
        "success_rate": float(np.mean([float(s["success"]) for s in subtasks])),
        "mean_distance_to_goal_m": float(np.mean([s["distance_to_goal"] for s in subtasks])),
    }
    print(f"\nGOAT criterion (<{args.success_distance} m): "
          f"leg1 {'SUCCESS' if subtasks[0]['success'] else 'fail'} ({err_a:.2f} m), "
          f"leg2 {'SUCCESS' if subtasks[1]['success'] else 'fail'} ({subtasks[1]['distance_to_goal']:.2f} m) "
          f"-> SR={result['goat_metrics']['success_rate']:.2f}")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'}")
    if args.live_dir:
        emit_live(args.live_dir, rgb, {
            "phase": "DONE", "step": 0, "max_steps": 0, "state": "DONE",
            "pose": [float(p) for p in pose], "scene": (args.scene_glb or args.scene),
            "done": True,
            "leg1_outcome": outcome1, "leg1_err_m": round(err_a, 2),
            "leg1_success": bool(subtasks[0]["success"]),
            "recall_err_m": round(err_pos, 2),
            "leg2_outcome": outcome2, "leg2_err_m": round(err_b_nearest, 2),
            "leg2_success": bool(subtasks[1]["success"]),
            "success_rate": result["goat_metrics"]["success_rate"],
        }, occ=occ, log=log, markers=live_markers)
    if args.live_view:
        cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()
