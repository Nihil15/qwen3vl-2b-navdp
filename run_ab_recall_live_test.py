#!/usr/bin/env python3
"""Real Qwen + real NavDP, two-instruction "go to A, then recall B" test.

No discrete grid actions anywhere -- every command is DinoNavDPPipeline's own
continuous (v, w) NavDP output, dead-reckoned through a unicycle model. No
VLN-CE benchmark scaffolding either. This exercises the actual live pipeline:

  Leg 1  instruction="go to the chair" (target A)
         -> every tick: pipe.step(..., instruction=...) -- the REAL
            Qwen3-VL pixel-goal grounder (nav_pipeline/qwen_pixel_goal.py)
            picks the waypoint pixel, depth lifts it to 3D, NavDP samples/
            follows a trajectory toward it (state "TRACK"), continuously.
         -> in parallel, a second real Qwen probe ("is anything else in
            view?") runs periodically. The first time it lands on a real
            object that ISN'T the current chair goal, that object's 3D
            position is computed (same pixel+depth+intrinsics geometry the
            pipeline itself uses) and converted to the odom frame -- this
            is the "notices B and remembers it" moment. No ground truth is
            consulted to decide whether a memory entry is written; only
            Qwen's own answer + the depth at the pixel it pointed to.
  Leg 2  instruction="go to B" -- B is NOT re-detected. Its remembered
         odom-frame position is handed to pipe.step(external_goal=...) every
         tick (nav_pipeline/fact3r_goal_source.Fact3rGoalSource.tick) --
         state "GOTO", same continuous obstacle-guard + NavDP machinery,
         driving on memory alone.

Since there is no camera on this machine, RGB-D frames are rendered from a
tiny synthetic world (one chair, one lamp, known ground-truth positions) via
exact pinhole projection -- geometry is exact, Qwen's answers on the pixels
are real inference. Ground truth is used only for the accuracy report, never
fed to the pipeline or the recall step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from nav_pipeline.fact3r_goal_source import Fact3rGoalSource, MapToOdom
from nav_pipeline.goal_utils import intrinsics_from_fov, pixel_depth_to_point
from nav_pipeline.obstacle_guard import GuardConfig
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

W, H, FOV_DEG = 640, 480, 90.0
BG_DEPTH_M = 9.0

WORLD = {
    "chair": {"xy": (5.5, 0.0), "w": 0.60, "h": 0.90, "color": (139, 90, 43)},   # straight ahead
    "lamp":  {"xy": (3.0, -1.8), "w": 0.35, "h": 1.10, "color": (235, 205, 70)},  # off to the side, en route
}

PROBE_PROMPT = (
    "Look at this image from a mobile robot's camera. Is there any distinct "
    "object visible OTHER than the chair the robot is driving toward? If "
    "yes, point to it. If no other object is visible, point to (0, 0). "
    "Reply with only one pixel coordinate as (x, y)."
)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def local_to_odom(pose, fwd: float, left: float):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return x + (fwd * c - left * s), y + (fwd * s + left * c)


def render(pose, intr):
    """Exact pinhole projection of WORLD onto a synthetic RGB-D frame."""
    fx, fy, cx, cy = intr
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    rgb[: H // 2] = (150, 148, 145)   # wall
    rgb[H // 2:] = (95, 90, 85)       # floor
    depth = np.full((H, W), BG_DEPTH_M, dtype=np.float32)

    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    visible = {}
    for name, obj in WORLD.items():
        dx, dy = obj["xy"][0] - x, obj["xy"][1] - y
        fwd = dx * c + dy * s
        left = -dx * s + dy * c
        if fwd < 0.3:
            continue
        bearing = math.atan2(left, fwd)
        if abs(bearing) > math.radians(FOV_DEG / 2.0) * 1.15:
            continue
        u = cx - fx * left / fwd
        half_w = fx * (obj["w"] / 2.0) / fwd
        half_h = fy * (obj["h"] / 2.0) / fwd
        v = cy + min(70.0, fy * 0.10 / fwd)
        u0, u1 = int(round(u - half_w)), int(round(u + half_w))
        v0, v1 = int(round(v - half_h)), int(round(v + half_h))
        cu0, cu1 = max(0, u0), min(W, u1)
        cv0, cv1 = max(0, v0), min(H, v1)
        if cu1 <= cu0 or cv1 <= cv0:
            continue
        if name == "chair":
            # seat + backrest + four legs, not a featureless block -- gives a
            # real VLM something chair-shaped to actually recognize.
            seat_v0 = cv0 + int(0.55 * (cv1 - cv0))
            cv2.rectangle(rgb, (cu0, seat_v0), (cu1, cv1), obj["color"], -1)          # seat
            back_w = max(1, int(0.12 * (cu1 - cu0)))
            cv2.rectangle(rgb, (cu0, cv0), (cu0 + back_w, seat_v0), obj["color"], -1)  # backrest (left post)
            cv2.rectangle(rgb, (cu0, cv0), (cu1, cv0 + max(1, back_w // 2)), obj["color"], -1)  # backrest top rail
            leg_h = min(H - 1, cv1 + int(0.35 * (cv1 - cv0)))
            for lu in (cu0 + 1, cu1 - 1):
                cv2.line(rgb, (lu, cv1), (lu, leg_h), (40, 25, 12), 2)                 # legs
        else:
            cv2.rectangle(rgb, (cu0, cv0), (cu1, cv1), obj["color"], -1)
            pole_u = int(round(u))  # a little shape variety: a pole under the shade
            cv2.line(rgb, (pole_u, cv1), (pole_u, min(H - 1, cv1 + int(half_h))), (60, 60, 60), 2)
        depth[cv0:cv1, cu0:cu1] = float(fwd)
        visible[name] = (cu0, cv0, cu1, cv1, float(fwd))
    return rgb, depth, visible


def within(bbox, u: int, v: int) -> bool:
    if bbox is None:
        return False
    u0, v0, u1, v1, _ = bbox
    return u0 <= u < u1 and v0 <= v < v1


def build_pipeline(args) -> DinoNavDPPipeline:
    cfg = PipelineConfig(
        device=args.device,
        horizontal_fov_deg=FOV_DEG,
        policy_type=args.policy,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        stop_distance=args.stop_distance,
        avoid_enabled=not args.no_avoid,
        use_dino=False,
        use_sam=False,
        use_clip=False,
        use_qwen_search=False,
        use_qwen_instruction=True,
        qwen_model_id=args.qwen_model_id,
        qwen_load_in_4bit=args.qwen_4bit,
        use_appearance_reid=False,
        use_scene_tagger=False,
        use_belief_goal=True,
        search_angular=args.search_angular,
        guard=GuardConfig(),
    )
    return DinoNavDPPipeline(cfg, use_depth_estimator=False)


def run_leg(pipe, intr, pose, dt, max_steps, kind, extra, log):
    """kind='track' drives on instruction='go to the chair' (live Qwen);
    kind='goto' drives on extra=Fact3rGoalSource (memory recall). Returns
    (outcome, notice) -- notice is only ever set during a 'track' leg."""
    notice = None
    last_probe = 0.0
    first_search_step = None
    for i in range(max_steps):
        rgb, depth, visible = render(pose, intr)

        if kind == "track":
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            instruction="go to the chair", intrinsics=intr)
        else:
            ext = extra.tick(pose)
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            external_goal=ext, intrinsics=intr)

        log.append({
            "leg": kind, "step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            "state": res.state, "v": float(res.linear), "w": float(res.angular),
            "qwen_pixel_goal": res.qwen_pixel_goal,
        })

        if kind == "track" and notice is None:
            now = time.time()
            if now - last_probe >= 2.0:
                last_probe = now
                pg = pipe._qwen_pixel_guide.ground(rgb, PROBE_PROMPT)
                if pg is not None and pg.confidence >= 0.3:
                    u, v = int(round(pg.u)), int(round(pg.v))
                    if 0 <= u < W and 0 <= v < H:
                        d = float(depth[v, u])
                        if d < BG_DEPTH_M - 0.5 and not within(visible.get("chair"), u, v):
                            local = pixel_depth_to_point(u, v, d, *intr)
                            ox, oy = local_to_odom(pose, float(local[0]), float(local[1]))
                            notice = {
                                "step": i, "pixel": [u, v], "depth_m": d,
                                "odom_xy": [ox, oy], "qwen_confidence": pg.confidence,
                            }
                            print(f"  [notice] step {i}: Qwen flagged something at pixel ({u},{v}) "
                                  f"depth={d:.2f}m conf={pg.confidence:.2f} -> odom ({ox:.2f}, {oy:.2f})")

        if kind == "track" and res.state == "SEARCH" and first_search_step is None:
            first_search_step = i
            print(f"  [lost] step {i}: chair goal lost, entering SEARCH at pose "
                  f"({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}deg)")
        if kind == "track" and res.state == "TRACK" and first_search_step is not None:
            print(f"  [reacquired] step {i}: back to TRACK after {i - first_search_step} SEARCH steps")
            first_search_step = None

        if kind == "track" and res.state == "STOP":
            return "reached", notice
        if kind == "goto" and extra.reached(pose):
            return "reached", notice
        if kind == "goto" and res.state == "STOP":
            return "pipeline_stop", notice

        v_, w_ = float(res.linear), float(res.angular)
        pose[0] += v_ * math.cos(pose[2]) * dt
        pose[1] += v_ * math.sin(pose[2]) * dt
        pose[2] = wrap(pose[2] + w_ * dt)

    return "max_steps", notice


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def state_counts(log, kind):
    counts: dict[str, int] = {}
    for row in log:
        if row["leg"] == kind:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=1.2)
    ap.add_argument("--reached-radius", type=float, default=1.0)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4,
                    help="SEARCH spin rate (rad/s); default (0.15) is real-rover-slow and can time out "
                         "a synthetic run's step budget before completing even one full turn")
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--max-steps-leg1", type=int, default=350)
    ap.add_argument("--max-steps-leg2", type=int, default=250)
    ap.add_argument("--out", default="logs/ab_recall_test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"World: chair(A) at {WORLD['chair']['xy']}, lamp(B) at {WORLD['lamp']['xy']} (ground truth, "
          f"never given to the pipeline or the recall step)")

    pipe = build_pipeline(args)
    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pose = np.array([0.0, 0.0, 0.0])
    log: list[dict] = []

    print("\n=== Leg 1: instruction = 'go to the chair' (live Qwen grounding + NavDP) ===")
    outcome1, notice = run_leg(pipe, intr, pose, args.dt, args.max_steps_leg1, "track", None, log)
    leg1_final = [float(pose[0]), float(pose[1]), float(pose[2])]
    leg1_error = dist(leg1_final, WORLD["chair"]["xy"])
    print(f"outcome={outcome1}  final_pose=({leg1_final[0]:.2f},{leg1_final[1]:.2f})  "
          f"error_to_chair={leg1_error:.2f} m  states={state_counts(log,'track')}")

    result = {
        "world": {k: v["xy"] for k, v in WORLD.items()},
        "leg1": {"instruction": "go to the chair", "outcome": outcome1,
                 "final_pose": leg1_final, "error_to_A_m": leg1_error,
                 "state_counts": state_counts(log, "track"), "steps": sum(1 for r in log if r["leg"] == "track")},
        "notice": notice,
    }

    if notice is None:
        print("\n=== Leg 2 skipped: Qwen never flagged a second object during leg 1 -- nothing to recall ===")
        result["leg2"] = None
    else:
        b_error_geometry = dist(notice["odom_xy"], WORLD["lamp"]["xy"])
        print(f"\nrecall target B: recorded odom ({notice['odom_xy'][0]:.2f},{notice['odom_xy'][1]:.2f}) "
              f"vs true lamp ({WORLD['lamp']['xy'][0]:.2f},{WORLD['lamp']['xy'][1]:.2f}) "
              f"-> detection geometry error {b_error_geometry:.2f} m")

        print("\n=== Leg 2: instruction = 'go to B' (memory recall, NO re-detection) ===")
        src = Fact3rGoalSource(notice["odom_xy"], MapToOdom.identity(),
                               reached_radius=args.reached_radius, query="B")
        outcome2, _ = run_leg(pipe, intr, pose, args.dt, args.max_steps_leg2, "goto", src, log)
        leg2_final = [float(pose[0]), float(pose[1]), float(pose[2])]
        leg2_err_recorded = dist(leg2_final, notice["odom_xy"])
        leg2_err_true = dist(leg2_final, WORLD["lamp"]["xy"])
        print(f"outcome={outcome2}  final_pose=({leg2_final[0]:.2f},{leg2_final[1]:.2f})  "
              f"error_to_recorded_B={leg2_err_recorded:.2f} m  error_to_true_B={leg2_err_true:.2f} m  "
              f"states={state_counts(log,'goto')}")

        result["leg2"] = {
            "instruction": "go to B", "outcome": outcome2, "final_pose": leg2_final,
            "error_to_recorded_B_m": leg2_err_recorded, "error_to_true_B_m": leg2_err_true,
            "state_counts": state_counts(log, "goto"), "steps": sum(1 for r in log if r["leg"] == "goto"),
        }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'} and {out_dir/'trajectory.json'}")


if __name__ == "__main__":
    main()
