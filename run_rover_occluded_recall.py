"""Occluded go-behind / recall test on the REAL rover.

Outbound: a directional instruction -- "go behind the <occluder>" -- drives the
rover around an opaque obstacle to a real object B parked in the obstacle's
shadow. While driving it:

  * records a VISUAL SUBGOAL track (nav_pipeline/visual_subgoals.py): a waypoint
    spine with a forced keyframe at every sharp turn (the corner it makes to get
    around the obstacle), each node carrying a SigLIP frame embedding;
  * SigLIP-encodes the OPAQUE OBSTACLE itself into an obstacle memory
    (nav_pipeline/obstacle_memory.py, is_obstacle=True) whenever
    LiveOccupancy.occlusion_shadow reports the goal ray is blocked by an
    occupied cell with UNKNOWN directly behind it;
  * routes every goal-set through an encode-once SigLIP cache
    (nav_pipeline/siglip_encoding_cache.py): a text goal already encoded on a
    previous leg / run is reused, not re-embedded.

Recall: "go to <B>" (or "come back"). Loads the subgoal track + obstacle memory
+ encode cache, follows VisualSubgoalTrack.reverse_spine() as the return spine
(A* over the live grid only for local repair of a blocked segment), re-identifies
the obstacle from its stored SigLIP signature as it passes, then hands the final
approach to the normal DINO+supervisor object leg and remembers the arrival.

Everything not new here is inherited unchanged from run_rover_multileg.PlanRunner
(Zenoh/odom plumbing, occupancy carving, drive_straight/turn_in_place,
_recall_and_approach, goal_memory).  Run via LAUNCH/launch_rover_multileg.sh's
bring-up, then this script instead of run_rover_multileg.py.  Example:

  python run_rover_occluded_recall.py --pi-ip 10.86.180.125 --yes \\
      --outbound "go behind the cabinet" --recall "the blue storage box"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import run_rover_multileg as rrm
from run_rover_multileg import Subtask, body_delta, clamp, wrap_angle

# heavy nav_pipeline classes, bound in main() after rrm._load_nav_pipeline()
LiveOccupancy = thin_waypoints = Siglip2Embedder = None
SiglipEncodingCache = VisualSubgoalTrack = ObstacleMemory = bearing_crop_box = None
occlusion_shadow_beyond_m = 0.4


# ======================================================================
#  directional-aware clause parsing
# ======================================================================
_DIR_RE = re.compile(
    r"^(?:go|move|drive|head|navigate|proceed|get)?\s*(?:to\s+)?"
    r"(behind|beyond|past|around|in front of|left of|right of|"
    r"to the left of|to the right of)\s+(?:the\s+|a\s+|an\s+)?(.+)$",
    flags=re.I,
)
_DIR_CANON = {
    "behind": "behind", "beyond": "behind", "past": "behind", "around": "around",
    "in front of": "front", "left of": "left", "to the left of": "left",
    "right of": "right", "to the right of": "right",
}


def classify_occ_clause(clause: str) -> Subtask:
    """Directional clause -> Subtask(kind='directional', arg='<dir>::<anchor>').
    Anything else falls through to run_rover_multileg's own move/turn/object
    classifier unchanged."""
    c = clause.strip().strip(" .,")
    m = _DIR_RE.match(c)
    if m and "turn" not in c.lower():
        direction = _DIR_CANON[m.group(1).lower()]
        anchor = re.sub(r"^(the|a|an)\s+", "", m.group(2).strip(), flags=re.I).strip()
        if anchor:
            return Subtask("directional", f"{direction}::{anchor}", clause)
    return rrm._classify_clause(clause)


_RECALL_RE = re.compile(
    r"^(?:recall|remember|go\s+back\s+to|come\s+back\s+to|return\s+to|"
    r"find\s+(?:again\s+)?(?:the\s+)?)\s*", flags=re.I)


def plan_from_occ_instruction(instruction: str) -> List[Subtask]:
    """Free-text -> ordered subtasks, directional- and recall-aware.  Used by
    the GUI/serve path.  A clause led by recall/'go back to'/'return to' is a
    recall leg (follow the subgoal spine); a behind/left/right/around clause is
    a directional leg; everything else falls to run_rover_multileg's own
    move/turn/object classifier."""
    clauses: List[str] = []
    for c in rrm._CLAUSE_SPLIT_RE.split(instruction):
        c = c.strip()
        if c:
            clauses.extend(p for p in rrm._split_compound(c) if p)
    plan: List[Subtask] = []
    for c in clauses:
        m = _RECALL_RE.match(c)
        if m:
            phrase = rrm._strip_goto_prefix(c[m.end():].strip()) or c
            plan.append(Subtask("recall", phrase, c))
        else:
            plan.append(classify_occ_clause(c))
    return plan or [classify_occ_clause(instruction)]


def _occ_serve_loop(runner: "OccludedRecallRunner", session) -> None:
    """--serve loop: models load once; instructions arrive on
    'multileg/instruction' and are parsed with plan_from_occ_instruction (so
    'go behind the X, then recall the B' works from the GUI box).  Mirrors
    run_rover_multileg._serve_loop's sentinel handling exactly."""
    import queue as _queue
    import threading as _threading
    next_instr: "_queue.Queue[str]" = _queue.Queue()
    running = _threading.Event()

    def _on_instruction(sample):
        try:
            text = rrm.parse_string(bytes(sample.payload)).strip()
        except Exception:
            return
        if not text:
            return
        if text == "__stop__":
            if running.is_set():
                runner.soft_abort()
            return
        if text == "__shutdown__":
            runner._exit_requested = True
            if running.is_set():
                runner.soft_abort()
            else:
                next_instr.put("__shutdown__")
            return
        if text == "__clear_map__":
            runner.clear_map()
            return
        next_instr.put(text)

    session.declare_subscriber("multileg/instruction", _on_instruction)
    session.declare_subscriber("rt/multileg/instruction", _on_instruction)
    print("\n[serve] ready -- waiting for an instruction on 'multileg/instruction' "
          "(directional + recall aware: 'go behind the X, then recall the B')")
    try:
        while True:
            runner._abort = False
            text = None
            while text is None:
                try:
                    text = next_instr.get(timeout=0.5)
                except _queue.Empty:
                    if runner._exit_requested:
                        return
                    continue
            if text == "__shutdown__":
                return
            try:
                plan = plan_from_occ_instruction(text)
            except Exception as e:
                print(f"[serve] could not parse {text!r}: {e}")
                continue
            print(f"\n[serve] new instruction: {text!r}")
            for i, st in enumerate(plan):
                print(f"  {i + 1}. {st.kind}:{st.arg}")
            running.set()
            try:
                runner.run(plan)
            finally:
                running.clear()
            if runner._exit_requested:
                return
            print("[serve] plan finished -- ready for the next instruction")
    finally:
        runner._cleanup(close_odom=True)


def resolve_occ_plan(args) -> List[Subtask]:
    plan: List[Subtask] = []
    if args.subtask:
        for spec in args.subtask:
            kind, _, val = spec.partition(":")
            k = kind.strip().lower()
            if k in ("directional", "dir"):
                plan.append(Subtask("directional", val.strip(), spec))
            elif k == "recall":
                plan.append(Subtask("recall", rrm._strip_goto_prefix(val.strip()), spec))
            else:
                plan.append(rrm._parse_explicit(spec))
    if args.outbound:
        clauses: List[str] = []
        for c in rrm._CLAUSE_SPLIT_RE.split(args.outbound):
            c = c.strip()
            if c:
                clauses.extend(p for p in rrm._split_compound(c) if p)
        plan.extend(classify_occ_clause(c) for c in clauses)
    if args.recall:
        plan.append(Subtask("recall", rrm._strip_goto_prefix(args.recall.strip()), args.recall))
    if not plan:
        raise SystemExit("give a plan: --outbound \"go behind the X\" [--recall \"the B\"] "
                         "or --subtask directional:behind::X / recall:the B")
    return plan


# ======================================================================
#  runner
# ======================================================================
class OccludedRecallRunner(rrm.PlanRunner):
    def __init__(self, node, args, supervisor=None, occupancy=None):
        super().__init__(node, args, supervisor=supervisor, occupancy=occupancy)
        self.persist_dir = Path(args.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._enc_embedder = None
        self.enc_cache = SiglipEncodingCache(
            embedder=None, path=args.encoding_cache_path or (self.persist_dir / "encoding_cache.npz"))
        self.subgoals = VisualSubgoalTrack(spacing_m=args.subgoal_spacing_m,
                                           sharp_turn_deg=args.sharp_turn_deg)
        self.obstacles = ObstacleMemory(associate_sim=args.obstacle_associate_sim,
                                        associate_pos_gate_m=args.obstacle_pos_gate_m)
        self._outbound_active = False
        self._cur_goal_xy: Optional[Tuple[float, float]] = None
        self._cur_goal_key: str = ""
        self._cur_anchor: Optional[str] = None
        self._last_obstacle_obs_t = 0.0
        self._reid_events: List[dict] = []
        self._goal_set_log: List[dict] = []

    # -- SigLIP embedder (always available here, unlike the opt-in parent) --
    def _embedder(self):
        if self._enc_embedder is None:
            print(f"[enc] loading SigLIP2 ({self.args.siglip_model_id}) for the encode-once cache "
                  "+ subgoal/obstacle embeddings ...")
            self._enc_embedder = Siglip2Embedder(model_id=self.args.siglip_model_id,
                                                 device=self.args.device)
            self.enc_cache.embedder = self._enc_embedder
        return self._enc_embedder

    def _embed_frame(self, rgb) -> Optional[np.ndarray]:
        try:
            h, w = rgb.shape[:2]
            return self._embedder().embed(rgb, np.array([0, 0, w, h], dtype=float))
        except Exception:
            return None

    def _set_goal(self, key: str) -> np.ndarray:
        """Enquire the encode-once cache for `key`; encode fresh on a miss,
        reuse the stored vector on a hit. Called at the start of every leg."""
        self._embedder()                       # ensure the cache has an encoder
        emb, cached = self.enc_cache.ensure_text(key)
        tag = "HIT->reuse" if cached else "MISS->encode"
        print(f"  [goal-set] key={key!r}  siglip-cache {tag}")
        self._goal_set_log.append({"key": key, "cached": bool(cached), "ts": time.time()})
        return emb

    # -- per-tick hook: subgoal recording + opaque-obstacle encoding -------
    def _integrate_occupancy(self):
        super()._integrate_occupancy()
        if not self._outbound_active or self.occupancy is None:
            return
        node = self.node
        x, y, th = node.odom.x, node.odom.y, node.odom.theta
        with node.lock:
            rgb = node.latest_rgb
        self.subgoals.maybe_add((x, y, th), time.time(), rgb=rgb, embed_fn=self._embed_frame)

        goal = self._cur_goal_xy
        if goal is None:                       # plain move leg: probe straight ahead
            goal = (x + 3.0 * math.cos(th), y + 3.0 * math.sin(th))
        shadow = self.occupancy.occlusion_shadow((x, y), goal, beyond_m=occlusion_shadow_beyond_m)
        if shadow is None:
            return
        (hx, hy), opaque = shadow
        now = time.time()
        if not opaque or rgb is None or now - self._last_obstacle_obs_t < self.args.obstacle_obs_period_s:
            return
        # don't re-log the same hit from a stationary/wedged rover -- only
        # observe when the viewpoint has actually changed.
        prev = getattr(self, "_last_obstacle_obs_xy", None)
        if prev is not None and math.hypot(x - prev[0], y - prev[1]) < 0.10 \
                and abs(wrap_angle(th - getattr(self, "_last_obstacle_obs_th", th))) < math.radians(8):
            return
        self._last_obstacle_obs_t = now
        self._last_obstacle_obs_xy = (x, y)
        self._last_obstacle_obs_th = th
        fwd, lat = body_delta(hx - x, hy - y, th)
        bearing = math.atan2(lat, fwd)
        box = self._anchor_box_this_tick()
        if box is None:
            h, w = rgb.shape[:2]
            box = bearing_crop_box(w, h, bearing, self.args.fov)
        try:
            emb = self._embedder().embed(rgb, np.asarray(box, dtype=float))
        except Exception:
            emb = None
        if emb is None:
            return
        ent = self.obstacles.observe(emb, (hx, hy), opaque=True,
                                     goal_key=self._cur_goal_key, ts=now)
        print(f"  [obstacle] opaque occluder for {self._cur_goal_key!r} at "
              f"({hx:+.2f},{hy:+.2f}) -> {ent.obstacle_id} "
              f"({ent.observations} obs, blocks {ent.blocked_goals})")
        self._publish_status("directional", self._cur_goal_key, force=True,
                             memory={"event": "obstacle_encoded", "obstacle_id": ent.obstacle_id,
                                     "world_xy": [round(hx, 3), round(hy, 3)], "ts": now})

    def _anchor_box_this_tick(self):
        """A live Grounding-DINO box for the named anchor this tick, if the
        pipeline has one -- the preferred obstacle crop when the occluder is
        the instruction's own anchor."""
        if self._cur_anchor is None:
            return None
        res = getattr(self.node, "last_result", None)
        det = getattr(res, "detection", None) if res is not None else None
        box = getattr(det, "box", None) if det is not None else None
        return np.asarray(box, dtype=float) if box is not None else None

    # -- resolve the anchor object's world position -----------------------
    def _resolve_anchor_world(self, anchor: str, timeout_s: float) -> Optional[Tuple[float, float, float]]:
        """Point Grounding-DINO at `anchor`, tick until it has a box, and
        unproject that box (median depth) to a world (x, y) + apparent radius.
        Returns None if nothing is detected inside `timeout_s`."""
        from nav_pipeline.goal_utils import goal_point_from_detection
        node = self.node
        node.target = anchor
        node._goal_reached = False
        node.pipe.reset()
        node.odom._history.clear()
        t0 = time.time()
        while not self._abort and time.time() - t0 < timeout_s:
            node._tick()
            res = node.last_result
            det = getattr(res, "detection", None) if res is not None else None
            box = getattr(det, "box", None) if det is not None else None
            with node.lock:
                depth, intr = node.latest_depth, node.latest_intrinsics
            if box is not None and depth is not None and intr is not None:
                pt = goal_point_from_detection(np.asarray(box, dtype=float), depth, *intr)
                if pt is not None:
                    x, y, th = node.odom.x, node.odom.y, node.odom.theta
                    fwd, lat = float(pt[0]), float(pt[1])
                    wx = x + fwd * math.cos(th) - lat * math.sin(th)
                    wy = y + fwd * math.sin(th) + lat * math.cos(th)
                    rad = 0.5
                    print(f"  [anchor] '{anchor}' at world ({wx:+.2f},{wy:+.2f}), "
                          f"{math.hypot(fwd, lat):.2f} m ahead")
                    return wx, wy, rad
            self._integrate_occupancy()
            time.sleep(1.0 / self.args.predict_hz)
        return None

    # -- shared point-approach loop (A* spine + NavDP external_goal) ------
    def _plan_route(self, gx: float, gy: float, label: str):
        """A* over the live grid to (gx, gy). Strict pass first, then
        allow_unknown, both with a wide start/goal anchor search (the rover is
        often nose-to-occluder when this is called for a re-plan). Returns a
        thinned waypoint list or None."""
        if self.occupancy is None or thin_waypoints is None:
            return None
        x0, y0 = self.node.odom.x, self.node.odom.y
        try:
            path, info = self.occupancy.astar((x0, y0), (gx, gy), nav_search_m=3.0)
            if path is None:
                path, info = self.occupancy.astar((x0, y0), (gx, gy), allow_unknown=True,
                                                  nav_search_m=3.0)
            if path and len(path) >= 2:
                route = thin_waypoints(path, spacing=self.args.goal_memory_waypoint_spacing_m)
                print(f"  [{label}] A* route: {len(route)} wp, {info.get('length_m', 0):.1f} m")
                return route
            print(f"  [{label}] A* found no route ({info.get('reason', '?') if isinstance(info, dict) else '?'})")
        except Exception as e:
            print(f"  [{label}] A* failed ({e})")
        return None

    def _approach_point(self, gx: float, gy: float, label: str, *, arrive_m: float,
                        timeout_s: float, reid: bool = False, replan: bool = False,
                        plan: bool = True) -> dict:
        """Drive to (gx, gy) via an A* waypoint spine + NavDP external_goal.
        When `replan`, a stall (no real motion for `--directional-replan-after-s`
        while still far from the goal -- the rover wedged against the occluder
        it can now see) triggers a fresh A* over the by-now-carved grid, so it
        routes AROUND the occluder instead of grinding into it. After
        `--directional-max-replans` failed re-plans it gives up with
        outcome='no_path'."""
        node, a = self.node, self.args
        period = 1.0 / a.metric_hz
        route = self._plan_route(gx, gy, label) if plan else None
        route_idx = 0
        t0 = time.time()
        face_latch_sign: Optional[float] = None   # latched turn dir while facing a behind-us point
        last_reid = 0.0
        stall_xy = (node.odom.x, node.odom.y)
        stall_t = time.time()
        replans = 0
        while not self._abort and time.time() - t0 < timeout_s:
            x, y, th = node.odom.x, node.odom.y, node.odom.theta
            far = math.hypot(gx - x, gy - y)
            if far < arrive_m:
                break
            if replan and time.time() - stall_t > a.directional_replan_after_s:
                moved = math.hypot(x - stall_xy[0], y - stall_xy[1])
                if moved < a.directional_replan_min_move_m and far > arrive_m + 0.3:
                    replans += 1
                    if replans > a.directional_max_replans:
                        print(f"  [{label}] wedged {replans - 1}x, no route around the occluder "
                              f"-- giving up (no navigable path)")
                        node.publish_cmd(0.0, 0.0)
                        return self._approach_result(gx, gy, arrive_m, outcome="no_path")
                    print(f"  [{label}] stalled ({moved * 100:.0f} cm in "
                          f"{a.directional_replan_after_s:.0f} s), {far:.2f} m from goal -- "
                          f"re-planning A* with the occluder now mapped ({replans})")
                    new_route = self._plan_route(gx, gy, f"{label}-replan{replans}")
                    route, route_idx = (new_route or route), 0
                stall_xy, stall_t = (x, y), time.time()
            tx, ty = gx, gy
            if route:
                while route_idx < len(route) - 1 and \
                        math.hypot(route[route_idx][0] - x, route[route_idx][1] - y) < a.goal_memory_waypoint_advance_m:
                    route_idx += 1
                tx, ty = route[route_idx]
            with node.lock:
                rgb, depth, intr = node.latest_rgb, node.latest_depth, node.latest_intrinsics
            if rgb is None:
                time.sleep(period)
                continue
            fwd, lat = body_delta(tx - x, ty - y, th)
            # Turn in place to face a point outside NavDP's ~70deg forward
            # cone before handing to GOTO. P-controller on the WRAPPED
            # heading error (not bang-bang on atan2(lat, fwd), which
            # sign-flips on odometry noise when the point is near dead
            # astern and reads as the rover twitching left/right without
            # turning round -- see run_rover_multileg._recall_and_approach
            # for the full write-up). Latch the direction on entry; use
            # hysteresis (enter >70deg, leave <25deg) so boundary noise
            # can't chatter it in and out.
            desired_th = math.atan2(ty - y, tx - x)
            face_err = wrap_angle(desired_th - th)
            if face_latch_sign is not None and abs(face_err) < math.radians(25.0):
                face_latch_sign = None
            if face_latch_sign is not None or abs(face_err) > math.radians(70.0):
                if face_latch_sign is None:
                    face_latch_sign = math.copysign(1.0, face_err if face_err != 0.0 else 1.0)
                w = face_latch_sign * min(abs(a.turn_kp * face_err), a.goal_memory_face_max_angular)
                node.publish_cmd(0.0, self._slew(w))
                self._integrate_occupancy()
                time.sleep(period)
                continue
            try:
                gres = node.pipe.step(rgb, "", depth=depth, pose=(x, y, th), intrinsics=intr,
                                      external_goal=np.array([fwd, lat, 0.0], dtype=np.float32))
                node.publish_cmd(gres.linear, gres.angular)
            except Exception as e:
                print(f"  [{label}] approach step failed: {e}")
                break
            self._integrate_occupancy()
            if reid and self.obstacles.entities and time.time() - last_reid > a.reid_period_s:
                last_reid = time.time()
                self._reid_check(rgb, x, y, th, label)
            self._publish_status(label, f"approach ({far:.2f} m)")
            time.sleep(period)
        node.publish_cmd(0.0, 0.0)
        self._last_w = 0.0
        return self._approach_result(gx, gy, arrive_m)

    def _approach_result(self, gx: float, gy: float, arrive_m: float,
                         outcome: Optional[str] = None) -> dict:
        x, y = self.node.odom.x, self.node.odom.y
        err = math.hypot(gx - x, gy - y)
        return {"target_xy": [round(gx, 3), round(gy, 3)],
                "final_xy": [round(x, 3), round(y, 3)], "error_m": round(err, 3),
                "outcome": outcome or ("abort" if self._abort else
                                       ("ok" if err < arrive_m else "timeout"))}

    def _scan_in_place(self, half_deg: float, label: str) -> None:
        """Sweep +half_deg, -half_deg, back to start, carving occupancy the
        whole time -- maps the occluder's extent and the open floor to its
        sides so the subsequent A* can route around it. Turn rate stays under
        _integrate_occupancy's turning-fast gate so carving isn't skipped."""
        node, a = self.node, self.args
        period = 1.0 / a.metric_hz
        th0 = node.odom.theta
        rate = min(a.turn_max_angular, 0.28)
        for off in (half_deg, -half_deg, 0.0):
            tgt = wrap_angle(th0 + math.radians(off))
            t0 = time.time()
            while not self._abort and time.time() - t0 < 10.0:
                err = wrap_angle(tgt - node.odom.theta)
                if abs(err) < math.radians(5.0):
                    break
                w = clamp(1.5 * err, -rate, rate)
                if 0.0 < abs(w) < a.turn_w_floor:
                    w = math.copysign(a.turn_w_floor, w)
                node.publish_cmd(0.0, self._slew(w))
                self._integrate_occupancy()
                time.sleep(period)
            node.publish_cmd(0.0, 0.0)
        self._last_w = 0.0
        if self.occupancy is not None:
            print(f"  [{label}] scan done -- occupancy {self.occupancy.stats()}")

    def _pick_open_side(self, ax: float, ay: float, ux: float, uy: float, label: str) -> str:
        """After the scan, count OCCUPIED cells in a fan on each side of the
        rover->anchor axis (alongside and just past the occluder). Fewer
        occupied hits = more room to pass -> go that way."""
        if self.occupancy is None:
            return "right"
        occ = self.occupancy

        def occ_hits(perp_x: float, perp_y: float) -> int:
            hits = 0
            for d_along in (-0.2, 0.3, 0.8, 1.3, 1.8):
                for d_side in (0.4, 0.7, 1.0, 1.3, 1.6):
                    r, c = occ.to_cell(ax + ux * d_along + perp_x * d_side,
                                       ay + uy * d_along + perp_y * d_side)
                    if occ._in_bounds(r, c) and occ.grid[r, c] == 1:
                        hits += 1
            return hits

        left = occ_hits(-uy, ux)
        right = occ_hits(uy, -ux)
        side = "left" if left < right else "right"
        print(f"  [{label}] auto side-pick: occupied cells  left={left}  right={right}  -> {side}")
        return side

    def _go_around(self, ax: float, ay: float, ux: float, uy: float, arad: float,
                   side: str, label: str) -> dict:
        """Scripted three-hop maneuver to get BEHIND the occluder centred at
        (ax, ay): step out to the open `side`, advance forward past the
        occluder, step back in. Deterministic -- doesn't need A* to find a
        route through a mostly-unknown map. Each hop's turn-to-face is a real
        corner, so the visual-subgoal track picks it up as a sharp turn."""
        node, a = self.node, self.args
        clear = a.go_around_clearance_m + arad
        depth = a.directional_standoff_m + arad + a.go_around_depth_m
        px, py = (-uy, ux) if side == "left" else (uy, -ux)
        x0, y0 = node.odom.x, node.odom.y
        s_out = (x0 + px * clear, y0 + py * clear)
        s_past = (s_out[0] + ux * depth, s_out[1] + uy * depth)
        s_in = (s_past[0] - px * clear, s_past[1] - py * clear)
        last = self._approach_result(x0, y0, 0.4, outcome="ok")
        for name, (wx, wy) in (("out", s_out), ("past", s_past), ("in", s_in)):
            if self._abort:
                break
            print(f"  [{label}] go-around/{side} {name}: -> ({wx:+.2f},{wy:+.2f})")
            last = self._approach_point(wx, wy, f"{label}-{name}", arrive_m=0.4,
                                        timeout_s=a.go_around_leg_timeout_s, plan=False)
            if last["outcome"] != "ok":
                print(f"  [{label}] go-around hop '{name}' -> {last['outcome']} "
                      f"({last['error_m']:.2f} m) -- that side looks blocked")
                return dict(last, outcome="no_path")
        return last

    def _reid_check(self, rgb, x, y, th, label):
        best = min(self.obstacles.entities,
                   key=lambda e: math.hypot(e.position[0] - x, e.position[1] - y))
        fwd, lat = body_delta(best.position[0] - x, best.position[1] - y, th)
        if fwd <= 0.1:
            return
        h, w = rgb.shape[:2]
        box = bearing_crop_box(w, h, math.atan2(lat, fwd), self.args.fov)
        try:
            emb = self._embedder().embed(rgb, box)
        except Exception:
            return
        if emb is None:
            return
        hit = self.obstacles.reidentify(emb, threshold=self.args.obstacle_reid_sim)
        if hit is not None:
            ent, sim = hit
            print(f"  [{label}] re-identified obstacle {ent.obstacle_id} (sim {sim:.2f}) "
                  f"while passing ({ent.position[0]:+.2f},{ent.position[1]:+.2f})")
            self._reid_events.append({"obstacle_id": ent.obstacle_id, "sim": round(sim, 3),
                                      "at_xy": [round(x, 3), round(y, 3)], "ts": time.time()})

    # -- legs ------------------------------------------------------------
    def directional_leg(self, spec: str, label: str) -> dict:
        direction, _, anchor = spec.partition("::")
        # "behind the left/right side of X" / "behind X on the right" -> pull
        # the pass side out of the anchor phrase; else fall back to the flag.
        around_side: Optional[str] = None
        ms = re.match(r"^(left|right)\s+side\s+of\s+(.+)$|^(left|right)\s+of\s+(.+)$", anchor, re.I)
        if ms:
            around_side = (ms.group(1) or ms.group(3)).lower()
            anchor = (ms.group(2) or ms.group(4)).strip()
        else:
            ms = re.search(r"\bon\s+the\s+(left|right)\b|\b(left|right)\s+side\b", anchor, re.I)
            if ms:
                around_side = (ms.group(1) or ms.group(2)).lower()
                anchor = re.sub(r"\s*(?:on\s+the\s+(?:left|right)|(?:left|right)\s+side)\s*$", "",
                                anchor, flags=re.I).strip()
        if around_side is None and self.args.go_around_side != "auto":
            around_side = self.args.go_around_side
        key = f"{direction}::{anchor}"
        self._set_goal(key)
        node = self.node
        aw = self._resolve_anchor_world(anchor, self.args.anchor_search_timeout_s)
        x, y, th = node.odom.x, node.odom.y, node.odom.theta
        if aw is None:
            ax = x + self.args.directional_fallback_m * math.cos(th)
            ay = y + self.args.directional_fallback_m * math.sin(th)
            arad = 0.5
            print(f"  [{label}] anchor '{anchor}' not detected -- assuming it "
                  f"{self.args.directional_fallback_m:.1f} m ahead")
        else:
            ax, ay, arad = aw
        ux, uy = ax - x, ay - y
        n = math.hypot(ux, uy) or 1.0
        ux, uy = ux / n, uy / n
        m = arad + self.args.shadow_margin_m
        if direction in ("behind", "around"):
            gx, gy = ax + ux * m, ay + uy * m
        elif direction == "front":
            gx, gy = ax - ux * m, ay - uy * m
        elif direction == "left":
            gx, gy = ax - uy * m, ay + ux * m
        else:  # right
            gx, gy = ax + uy * m, ay - ux * m
        print(f"  [{label}] {direction} '{anchor}': driving to world ({gx:+.2f},{gy:+.2f})")

        self._cur_anchor = anchor
        self._cur_goal_xy = (gx, gy)
        self._cur_goal_key = key
        self._outbound_active = True
        # The ONLY reason to scan-and-map is to auto-pick the open side. If the
        # instruction (or --go-around-side) already names the side, or the user
        # passed --no-directional-scan, skip straight to the go-around.
        do_scan = (self.args.directional_scan and around_side is None
                   and direction in ("behind", "around"))
        try:
            if direction in ("behind", "around", "left", "right"):
                # drive up to ~standoff_m in front of the occluder (a normal
                # hop, not a lingering stop), so the go-around box starts from
                # a known distance
                sx = ax - ux * self.args.directional_standoff_m
                sy = ay - uy * self.args.directional_standoff_m
                print(f"  [{label}] approach to standoff ({sx:+.2f},{sy:+.2f}), "
                      f"{self.args.directional_standoff_m:.1f} m in front of '{anchor}'")
                self._approach_point(sx, sy, f"{label}-standoff", arrive_m=0.6,
                                     timeout_s=max(25.0, self.args.directional_timeout_s * 0.4))
                if do_scan and not self._abort:
                    print(f"  [{label}] scan +/-{self.args.directional_scan_deg:.0f} deg to pick the open side")
                    self._scan_in_place(self.args.directional_scan_deg, f"{label}-scan")
            if direction in ("behind", "around") and not self._abort:
                side = around_side or (self._pick_open_side(ax, ay, ux, uy, label) if do_scan
                                       else "right")
                print(f"  [{label}] go-around via the {side} side "
                      f"({self.args.shadow_margin_m:.1f} m past the occluder)")
                res = self._go_around(ax, ay, ux, uy, arad, side, label)
                if res["outcome"] == "no_path":
                    print(f"  [{label}] scripted go-around blocked -- one A* attempt as a fallback")
                    res = self._approach_point(gx, gy, label, arrive_m=self.args.directional_arrive_m,
                                               timeout_s=self.args.directional_timeout_s, replan=True)
            else:
                print(f"  [{label}] phase 3: routing to the target point")
                res = self._approach_point(gx, gy, label, arrive_m=self.args.directional_arrive_m,
                                           timeout_s=self.args.directional_timeout_s, replan=True)
            self.subgoals.force_node((node.odom.x, node.odom.y, node.odom.theta), time.time())
        finally:
            self._outbound_active = False
            self._cur_goal_xy = None
            self._cur_anchor = None
        self._persist_outbound()
        res.update(label=label, kind="directional", instruction=spec,
                   direction=direction, anchor=anchor,
                   anchor_xy=[round(ax, 3), round(ay, 3)],
                   subgoal_nodes=len(self.subgoals.nodes),
                   sharp_turns=sum(1 for nd in self.subgoals.nodes if nd.sharp_turn),
                   obstacles=[e.obstacle_id for e in self.obstacles.entities])
        return res

    def recall_leg(self, phrase: str, label: str) -> dict:
        self._set_goal(f"recall::{phrase}")
        self._load_persisted()
        spine = self.subgoals.reverse_spine()
        print(f"  [{label}] return spine: {len(spine)} subgoal waypoints "
              f"({sum(1 for n in self.subgoals.nodes if n.sharp_turn)} sharp turns), "
              f"{len(self.obstacles.entities)} known obstacle(s)")
        seg_results = []
        for k, (wx, wy) in enumerate(spine[1:], start=1):
            if self._abort:
                break
            seg = self._approach_point(wx, wy, f"{label}-wp{k}",
                                       arrive_m=self.args.goal_memory_waypoint_advance_m,
                                       timeout_s=self.args.recall_segment_timeout_s, reid=True)
            seg_results.append(seg)
        # final confirmed approach to B via the normal DINO+supervisor object leg
        obj = self.go_to_object(phrase, self.args.object_timeout_s, f"{label}-final", raw_clause=phrase)
        node = self.node
        try:
            with node.lock:
                rgb = node.latest_rgb
            box = self._anchor_box_this_tick()
            self._remember_arrival(phrase, node.odom.x, node.odom.y,
                                   det_box=box, rgb=rgb)
        except Exception:
            pass
        return {"label": label, "kind": "recall", "instruction": phrase,
                "spine_waypoints": len(spine), "segments": seg_results,
                "reid_events": self._reid_events, "final_object_leg": obj,
                "outcome": obj.get("outcome", "unknown")}

    # -- persistence ---------------------------------------------------
    def _persist_outbound(self):
        try:
            self.subgoals.save(self.persist_dir / "subgoals.json")
            self.obstacles.to_jsonl(self.persist_dir / "obstacles.jsonl")
            self.enc_cache.save()
            print(f"  [persist] {self.persist_dir}/ : subgoals.json, obstacles.jsonl, "
                  f"encoding_cache.npz  ({self.enc_cache.summary()})")
        except Exception as e:
            print(f"  [persist] failed: {e}")

    def _load_persisted(self):
        sg = self.persist_dir / "subgoals.json"
        ob = self.persist_dir / "obstacles.jsonl"
        if not self.subgoals.nodes and sg.exists():
            self.subgoals = VisualSubgoalTrack.load(sg)
            print(f"  [persist] loaded {len(self.subgoals.nodes)} subgoal nodes from {sg}")
        if not self.obstacles.entities and ob.exists():
            self.obstacles = ObstacleMemory.from_jsonl(
                ob, associate_sim=self.args.obstacle_associate_sim,
                associate_pos_gate_m=self.args.obstacle_pos_gate_m)
            print(f"  [persist] loaded {len(self.obstacles.entities)} obstacle(s) from {ob}")

    def _write_occ_result(self, results: List[dict]):
        out = {
            "format": "occluded-recall-result",
            "outbound": self.args.outbound, "recall": self.args.recall,
            "results": results,
            "goal_set_log": self._goal_set_log,
            "reid_events": self._reid_events,
            "encoding_cache": self.enc_cache.stats,
            "subgoals": self.subgoals.to_dict(),
            "obstacles": [json.loads(x) for x in
                          (self.persist_dir / "obstacles.jsonl").read_text().splitlines()
                          if x.strip()] if (self.persist_dir / "obstacles.jsonl").exists() else [],
        }
        p = self.persist_dir / "result.json"
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[result] {p}")
        self._topdown_png()

    def _topdown_png(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        fig, ax = plt.subplots(figsize=(7, 7))
        poses = np.asarray([(p[0], p[1]) for p in self.subgoals.poses], dtype=float)
        if len(poses):
            ax.plot(poses[:, 0], poses[:, 1], "-", color="0.6", lw=1, label="outbound path")
        sp = np.asarray(self.subgoals.reverse_spine(), dtype=float)
        if len(sp):
            ax.plot(sp[:, 0], sp[:, 1], "--", color="tab:blue", lw=1.5, label="return spine")
        for nd in self.subgoals.nodes:
            if nd.sharp_turn:
                ax.plot(*nd.xy, "*", color="tab:red", ms=15)
        for e in self.obstacles.entities:
            ax.plot(*e.position, "s", color="k" if e.opaque else "0.5", ms=12,
                    label=f"{e.obstacle_id}{' (opaque)' if e.opaque else ''}")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("occluded recall: outbound path, sharp turns (*), obstacles (blk=opaque)")
        fig.savefig(self.persist_dir / "topdown.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[result] {self.persist_dir}/topdown.png")

    # -- lifecycle (trimmed run(): dispatch our extra subtask kinds) -----
    def run(self, plan: List[Subtask]):
        self._plan = plan
        self.results = []
        a = self.args
        if not self.wait_for_inputs(a.input_wait_s):
            print("[fatal] no camera and/or /rover/rpm -- is the Pi up? Aborting.")
            self._cleanup()
            sys.exit(4)
        if not self._pose_reset_done:
            self.node.odom.reset_pose()
            self._occ_origin = (0.0, 0.0)
            self._pose_reset_done = True
            print("[frame] odometry origin set at current pose (theta = 0)")
            if a.odom_selfcheck and not self.odom_selfcheck():
                self._safe_stop()
                self.write_results(aborted_at="odom_selfcheck", reason="odometry_error")
                self._cleanup()
                sys.exit(3)
            self.settle(a.rest_s)

        for i, st in enumerate(plan):
            if self._abort:
                break
            self._current_index = i
            self._publish_status(st.kind, f"{st.kind}:{st.arg}", force=True)
            print(f"\n=== subtask {i + 1}/{len(plan)} : {st.kind}:{st.arg} ===")
            if st.kind == "move":
                self._outbound_active = True
                try:
                    res = self.drive_straight(st.arg, a.move_tol_m, a.move_timeout_s, f"{i}_move")
                finally:
                    self._outbound_active = False
            elif st.kind == "turn":
                res = self.turn_in_place(st.arg, a.turn_tol_deg, a.turn_timeout_s, f"{i}_turn")
            elif st.kind == "directional":
                res = self.directional_leg(st.arg, f"{i}_dir")
            elif st.kind == "recall":
                res = self.recall_leg(st.arg, f"{i}_recall")
            else:
                res = self.go_to_object(st.arg, a.object_timeout_s, f"{i}_object", raw_clause=st.raw)
            self.results.append(res)
            self.settle(a.rest_s)

        self._current_index = len(plan)
        self._publish_status("done", "finished" if not self._abort else "aborted", force=True)
        self._safe_stop()
        try:
            self._write_occ_result(self.results)
        except Exception as e:
            print(f"[result] write failed: {e}")
        self.write_results(aborted_at="interrupt" if self._abort else None)
        self._cleanup(close_odom=not self.args.serve)


# ======================================================================
#  CLI
# ======================================================================
def build_parser():
    p = rrm.build_arg_parser()
    p.description = "Occluded go-behind + recall test on the real rover"
    g = p.add_argument_group("occluded-recall")
    g.add_argument("--outbound", type=str, default=None,
                   help='directional outbound instruction, e.g. "go behind the cabinet"')
    g.add_argument("--recall", type=str, default=None,
                   help='recall target after the outbound leg, e.g. "the blue storage box"')
    g.add_argument("--persist-dir", type=str, default="logs/occluded_recall",
                   help="where subgoals.json / obstacles.jsonl / encoding_cache.npz / result.json go")
    g.add_argument("--encoding-cache-path", type=str, default=None,
                   help="explicit encode-once cache file (default: <persist-dir>/encoding_cache.npz)")
    g.add_argument("--subgoal-spacing-m", type=float, default=0.6)
    g.add_argument("--sharp-turn-deg", type=float, default=35.0)
    g.add_argument("--directional-arrive-m", type=float, default=0.6)
    g.add_argument("--directional-timeout-s", type=float, default=90.0)
    g.add_argument("--directional-fallback-m", type=float, default=1.5,
                   help="assumed anchor distance ahead when the anchor is never detected")
    g.add_argument("--anchor-search-timeout-s", type=float, default=20.0)
    g.add_argument("--shadow-margin-m", type=float, default=0.8,
                   help="how far past the anchor (plus its radius) the behind-point sits")
    g.add_argument("--directional-standoff-m", type=float, default=1.1,
                   help="phase-1 stop distance in FRONT of the occluder before scanning + routing around")
    g.add_argument("--directional-scan", action=argparse.BooleanOptionalAction, default=True,
                   help="in-place sweep to auto-pick the open side; automatically skipped when the "
                        "instruction or --go-around-side already names the side. --no-directional-scan "
                        "forces it off (rover just goes -- no mapping stop)")
    g.add_argument("--directional-scan-deg", type=float, default=55.0,
                   help="in-place sweep half-angle when --directional-scan is used")
    g.add_argument("--directional-replan-after-s", type=float, default=7.0,
                   help="phase-3: re-plan A* if the rover moves < replan-min-move in this long (wedged)")
    g.add_argument("--directional-replan-min-move-m", type=float, default=0.20)
    g.add_argument("--directional-max-replans", type=int, default=4,
                   help="give up the directional leg (outcome=no_path) after this many failed re-plans")
    g.add_argument("--go-around-side", choices=("auto", "left", "right"), default="auto",
                   help="which side to pass the occluder on for behind/around legs; 'auto' picks the "
                        "less-occupied side from the post-scan map (an instruction like "
                        "'behind the right side of the curtain' overrides this)")
    g.add_argument("--go-around-clearance-m", type=float, default=0.9,
                   help="how far out to the side the scripted go-around steps to clear the occluder edge")
    g.add_argument("--go-around-depth-m", type=float, default=1.0,
                   help="how far PAST the occluder the go-around drives before stepping back in behind it")
    g.add_argument("--go-around-leg-timeout-s", type=float, default=40.0)
    g.add_argument("--obstacle-obs-period-s", type=float, default=1.5)
    g.add_argument("--obstacle-associate-sim", type=float, default=0.82)
    g.add_argument("--obstacle-pos-gate-m", type=float, default=1.2)
    g.add_argument("--obstacle-reid-sim", type=float, default=0.85)
    g.add_argument("--reid-period-s", type=float, default=2.0)
    g.add_argument("--recall-segment-timeout-s", type=float, default=40.0)
    return p


def main():
    args = build_parser().parse_args()
    plan = None if args.serve else resolve_occ_plan(args)
    if plan is not None:
        print("\nResolved plan:")
        for i, st in enumerate(plan):
            print(f"  {i + 1}. {st.kind}:{st.arg}   <- {st.raw!r}")
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
        print("ERROR: zenoh not importable -- run under the rover conda env.")
        return 1

    rrm._load_nav_pipeline()
    global LiveOccupancy, thin_waypoints, Siglip2Embedder
    global SiglipEncodingCache, VisualSubgoalTrack, ObstacleMemory, bearing_crop_box
    from nav_pipeline.live_occupancy import LiveOccupancy as _LO, thin_waypoints as _tw
    from nav_pipeline.siglip2_embedder import Siglip2Embedder as _SE
    from nav_pipeline.siglip_encoding_cache import SiglipEncodingCache as _SEC
    from nav_pipeline.visual_subgoals import VisualSubgoalTrack as _VST
    from nav_pipeline.obstacle_memory import ObstacleMemory as _OM, bearing_crop_box as _bcb
    LiveOccupancy, thin_waypoints, Siglip2Embedder = _LO, _tw, _SE
    SiglipEncodingCache, VisualSubgoalTrack, ObstacleMemory, bearing_crop_box = _SEC, _VST, _OM, _bcb

    args.object_grounder = "dino"              # DINO grounding is what the anchor + final leg rely on
    args.goal_memory = True
    pipeline, supervisor = _build_pipeline(args)

    zcfg = zenoh.Config()
    if args.pi_ip:
        zcfg.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(zcfg)
    node = rrm.DinoNavDPZenohNode(session, pipeline, "multileg", args.predict_hz,
                                  stop_confirm_count=args.stop_confirm_count,
                                  odometry_log_dir=args.odometry_log_dir,
                                  imu_min_mag_calib=args.imu_min_mag_calib)
    occupancy = LiveOccupancy(center_xy=(0.0, 0.0), extent_m=args.occupancy_extent_m,
                              res=args.occupancy_res, guard=rrm.GuardConfig(), max_range=4.0)
    runner = OccludedRecallRunner(node, args, supervisor=supervisor, occupancy=occupancy)
    signal.signal(signal.SIGINT, runner.request_abort)
    signal.signal(signal.SIGTERM, runner.request_abort)
    try:
        if args.serve:
            _occ_serve_loop(runner, session)
        else:
            runner.run(plan)
    finally:
        node._running = False
        time.sleep(2 * node.HEARTBEAT_PERIOD_S)
        try:
            session.close()
        except Exception:
            pass
    return 0


def _build_pipeline(args):
    cfg_kw = dict(
        device=args.device, horizontal_fov_deg=args.fov, max_linear=args.max_linear,
        max_angular=args.max_angular, stop_distance=args.stop_distance,
        angular_slew_max=args.angular_slew_max, use_belief_goal=not args.no_belief_goal,
        guard=rrm.GuardConfig(), wheel_deadband_correction=args.wheel_deadband_correction,
        use_appearance_reid=True, use_dino=True, use_qwen_instruction=False,
        use_qwen_search=False, appearance_min_similarity=args.dino_appearance_min_sim,
        appearance_iou_override=args.dino_appearance_iou_override,
        clip_min_similarity=args.dino_clip_min_sim,
    )
    if args.dino_target_threshold is not None:
        cfg_kw["detect_score_min"] = args.dino_target_threshold
    pipeline = rrm.DinoNavDPPipeline(rrm.PipelineConfig(**cfg_kw))
    supervisor = None
    if args.supervisor:
        supervisor = rrm.TaskSupervisor(args.supervisor_model_id, load_in_4bit=args.supervisor_4bit,
                                        device=args.device)
        try:
            supervisor._ensure_loaded()
        except Exception as e:
            print(f"[supervisor] eager load failed ({e}) -- continuing DINO-only")
            supervisor = None
    return pipeline, supervisor


if __name__ == "__main__":
    raise SystemExit(main())
