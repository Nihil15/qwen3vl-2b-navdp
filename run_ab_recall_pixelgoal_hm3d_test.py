#!/usr/bin/env python3
"""Real Qwen + real NavDP, "go to A, then recall B" -- recall via a PIXEL
GOAL from fact3r-map's own building blocks (SAM2 + SigLIP2), not a 3D point.

Supersedes run_ab_recall_hm3d_test.py's recall mechanism. That version
stored B's location as an odom-frame xy (computed from one Qwen-probed
pixel's depth) and drove leg 2 blind on it via `external_goal` / state
"GOTO" -- correct NavDP usage, but it never touched fact3r-map's own code,
and it needed a metric depth reading to work at all.

Here, leg 2 drives exactly like leg 1 does: a `ground_candidates()` call
every throttled tick returns a `PixelGoal` in the CURRENT frame, depth lifts
it to 3D, NavDP samples/follows a trajectory toward it -- pipeline.py's
existing TRACK / belief-coasting / SEARCH-and-reacquire path, unmodified.
The only difference from leg 1 is which object supplies that pixel:

  Leg 1  pipe._qwen_pixel_guide = QwenVLPixelGoal      instruction="go to the chair"
  Leg 2  pipe._qwen_pixel_guide = Fact3rEntityRecallGuide   instruction="go to B"

Fact3rEntityRecallGuide (nav_pipeline/fact3r_entity_recall.py) is SAM2 mask
proposals + a SigLIP2 crop embedding, matched by cosine similarity against
one embedding learned during leg 1 (nav_pipeline.fact3r_entity_recall's
docstring). No 3D point is ever computed or stored for B -- if it isn't
recognized in the current frame, ground_candidates() returns [], and
pipeline.py's own SEARCH state takes over exactly as it would for a lost
Qwen grounding.

Frames, scene setup and world/agent geometry are shared with
run_ab_recall_hm3d_test.py (imported directly, not duplicated).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from nav_pipeline.fact3r_entity_recall import Fact3rEntityRecallGuide
from nav_pipeline.goal_utils import intrinsics_from_fov
from nav_pipeline.obstacle_guard import GuardConfig
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig
from run_ab_recall_hm3d_test import (
    EYE_HEIGHT_M,
    FOV_DEG,
    H,
    W,
    build_sim,
    find_dataset_config,
    find_scene_glb,
    largest_island,
    pick_pair,
    pick_start,
    render,
)

PROBE_PROMPT_TMPL = (
    "Look at this image from a mobile robot's camera. Is there any distinct "
    "object visible OTHER than the {target} the robot is driving toward? If "
    "yes, point to it. If no other object is visible, point to (0, 0). "
    "Reply with only one pixel coordinate as (x, y)."
)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def build_pipeline(args) -> DinoNavDPPipeline:
    cfg = PipelineConfig(
        device=args.device,
        horizontal_fov_deg=FOV_DEG,
        policy_type=args.policy,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        stop_distance=args.stop_distance,
        avoid_enabled=not args.no_avoid,
        use_dino=False, use_sam=False, use_clip=False, use_qwen_search=False,
        use_qwen_instruction=True,
        qwen_model_id=args.qwen_model_id,
        qwen_load_in_4bit=args.qwen_4bit,
        qwen_instruction_period_s=args.instruction_period_s,
        use_appearance_reid=False, use_scene_tagger=False, use_belief_goal=True,
        search_angular=args.search_angular,
        guard=GuardConfig(),
    )
    return DinoNavDPPipeline(cfg, use_depth_estimator=False)


def state_counts(log, kind):
    counts: dict[str, int] = {}
    for row in log:
        if row["leg"] == kind:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
    return counts


def run_leg1(pipe, sim, pf, island, cam_y, intr, pose, dt, max_steps, target_label,
            recall, log, frames_dir, realtime_pacing, probe_period_s) -> tuple[str, bool]:
    """Drives on Qwen's own pixel goal, exactly as pipe._qwen_pixel_guide was
    constructed. In parallel, probes for a second object every probe_period_s
    of wall time for the WHOLE leg (not just until the first hit) and hands
    each flagged pixel to `recall.remember()` -- banking one more diverse
    SigLIP2 view of B into memory each time the object is seen again from a
    different distance/angle while the robot continues toward A, the same
    "build the memory while you go" shape as fact3r-map's own live mapper."""
    remembered = False
    last_probe = 0.0
    for i in range(max_steps):
        tick_start = time.time()
        rgb, depth = render(sim, pose, cam_y)
        res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                        instruction=f"go to the {target_label}", intrinsics=intr)
        log.append({"leg": "leg1", "step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "state": res.state, "v": float(res.linear), "w": float(res.angular)})

        if i == 0 and frames_dir is not None:
            _save_frame(frames_dir, "leg1_start.png", rgb)

        now = time.time()
        if now - last_probe >= probe_period_s:
            last_probe = now
            pg = pipe._qwen_pixel_guide.ground(rgb, PROBE_PROMPT_TMPL.format(target=target_label))
            if pg is None:
                print(f"  [probe] step {i}: Qwen returned no candidate")
            elif pg.confidence < 0.3:
                print(f"  [probe] step {i}: Qwen confidence too low ({pg.confidence:.2f})")
            else:
                u, vv = int(round(pg.u)), int(round(pg.v))
                margin_u, margin_v = 0.04 * W, 0.04 * H
                on_border = u < margin_u or u > W - margin_u or vv < margin_v or vv > H - margin_v
                if not (0 <= u < W and 0 <= vv < H) or on_border:
                    print(f"  [probe] step {i}: pixel ({u},{vv}) rejected as a border/no-answer cop-out")
                else:
                    before = recall.views_stored
                    ok = recall.remember(rgb, (u, vv), label="B", step=i)
                    if ok and recall.views_stored > before:
                        remembered = True
                        print(f"  [remember] step {i}: banked memory view #{recall.views_stored} at "
                              f"pixel ({u},{vv}) (Qwen conf={pg.confidence:.2f})")
                        if frames_dir is not None:
                            _save_frame(frames_dir, f"leg1_remember_{recall.views_stored}.png", rgb,
                                       marker=(u, vv))
                    elif ok:
                        print(f"  [probe] step {i}: pixel ({u},{vv}) matched an already-banked view "
                              f"(not a new one)")
                    else:
                        print(f"  [probe] step {i}: pixel ({u},{vv}) had no nearby SAM2 proposal to embed")

        if res.state == "STOP":
            if frames_dir is not None:
                _save_frame(frames_dir, "leg1_final.png", rgb)
            return "reached", remembered

        v_, w_ = float(res.linear), float(res.angular)
        nx = pose[0] + v_ * math.cos(pose[2]) * dt
        ny = pose[1] + v_ * math.sin(pose[2]) * dt
        snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny]), island))
        pose[0], pose[1] = float(snapped[0]), float(snapped[2])
        pose[2] = wrap(pose[2] + w_ * dt)

        if realtime_pacing:
            remaining = dt - (time.time() - tick_start)
            if remaining > 0:
                time.sleep(remaining)

    if frames_dir is not None:
        rgb, _ = render(sim, pose, cam_y)
        _save_frame(frames_dir, "leg1_final.png", rgb)
    return "max_steps", remembered


def run_leg2(pipe, sim, pf, island, cam_y, intr, pose, dt, max_steps, log, frames_dir, realtime_pacing) -> str:
    """Drives on Fact3rEntityRecallGuide's pixel goal -- installed as
    pipe._qwen_pixel_guide, so this is the identical call shape as leg 1."""
    for i in range(max_steps):
        tick_start = time.time()
        rgb, depth = render(sim, pose, cam_y)
        res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                        instruction="go to B", intrinsics=intr)
        log.append({"leg": "leg2", "step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "state": res.state, "v": float(res.linear), "w": float(res.angular),
                    "match_score": pipe._qwen_pixel_guide.last_match_score,
                    "match_view": pipe._qwen_pixel_guide.last_match_view_index})

        if i == 0 and frames_dir is not None:
            _save_frame(frames_dir, "leg2_start.png", rgb)
        if i % 50 == 0 and i > 0 and frames_dir is not None:
            _save_frame(frames_dir, f"leg2_step{i}.png", rgb)

        if i % 10 == 0:
            score = pipe._qwen_pixel_guide.last_match_score
            score_txt = f"{score:.3f}" if score is not None else "n/a"
            print(f"  [step {i:3d}] state={res.state:6s} best_match={score_txt}")

        if res.state == "STOP":
            if frames_dir is not None:
                _save_frame(frames_dir, "leg2_final.png", rgb)
            return "reached"

        v_, w_ = float(res.linear), float(res.angular)
        nx = pose[0] + v_ * math.cos(pose[2]) * dt
        ny = pose[1] + v_ * math.sin(pose[2]) * dt
        snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny]), island))
        pose[0], pose[1] = float(snapped[0]), float(snapped[2])
        pose[2] = wrap(pose[2] + w_ * dt)

        if realtime_pacing:
            remaining = dt - (time.time() - tick_start)
            if remaining > 0:
                time.sleep(remaining)

    if frames_dir is not None:
        rgb, _ = render(sim, pose, cam_y)
        _save_frame(frames_dir, "leg2_final.png", rgb)
    return "max_steps"


def _save_frame(frames_dir: Path, name: str, rgb: np.ndarray, marker: tuple[int, int] | None = None) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if marker is not None:
        cv2.drawMarker(bgr, marker, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.circle(bgr, marker, 18, (0, 0, 255), 2)
    cv2.imwrite(str(frames_dir / name), bgr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--category-a", default="chair")
    ap.add_argument("--category-b", default="table")
    ap.add_argument("--min-geodesic", type=float, default=2.0)
    ap.add_argument("--max-geodesic", type=float, default=14.0)
    ap.add_argument("--start-offset", type=float, default=3.5)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true")
    ap.add_argument("--instruction-period-s", type=float, default=1.5)
    ap.add_argument("--recall-discovery-model", default="facebook/sam2.1-hiera-small")
    ap.add_argument("--recall-siglip-model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--recall-match-threshold", type=float, default=0.72)
    ap.add_argument("--recall-max-views", type=int, default=5)
    ap.add_argument("--probe-period-s", type=float, default=2.0)
    ap.add_argument("--no-realtime-pacing", dest="realtime_pacing", action="store_false",
                    help="don't sleep to real wall-clock dt each tick -- breaks "
                         "qwen_instruction_period_s's real-time throttle")
    ap.add_argument("--save-frames", action="store_true", help="save key RGB frames under <out>/frames/")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=1.0)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--max-steps-leg1", type=int, default=200)
    ap.add_argument("--max-steps-leg2", type=int, default=250)
    ap.add_argument("--out", default="logs/ab_recall_pixelgoal_hm3d_test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hm3d_root = Path(args.hm3d_root)
    dataset_config = find_dataset_config(hm3d_root)
    scene_glb = find_scene_glb(hm3d_root, args.scene)
    print(f"Scene: {args.scene}  glb={scene_glb}")

    sim = build_sim(scene_glb, dataset_config)
    pf = sim.pathfinder
    island = largest_island(pf)
    print(f"navmesh: {pf.num_islands} islands, using largest (#{island}, area={pf.island_area(island):.1f} m^2)")

    a_point, b_point, ab_geodesic = pick_pair(sim, island, args.category_a, args.category_b,
                                              args.min_geodesic, args.max_geodesic)
    start_point, start_geodesic, start_clearance = pick_start(sim, pf, island, a_point, b_point, args.start_offset)
    cam_y = float(start_point[1]) + EYE_HEIGHT_M

    a_xy = (float(a_point[0]), float(a_point[2]))
    b_xy = (float(b_point[0]), float(b_point[2]))
    start_xy = (float(start_point[0]), float(start_point[2]))
    yaw0 = math.atan2(a_xy[1] - start_xy[1], a_xy[0] - start_xy[0])

    print(f"A ({args.category_a}) @ {a_xy}   B ({args.category_b}) @ {b_xy}   A<->B geodesic={ab_geodesic:.2f}m")
    print(f"start @ {start_xy}  (start->A geodesic={start_geodesic:.2f}m, clearance={start_clearance:.2f}m)")

    pipe = build_pipeline(args)
    qwen_guide = pipe._qwen_pixel_guide  # leg 1's grounder, saved so leg 2 can swap it back out
    recall = Fact3rEntityRecallGuide(
        device=args.device, discovery_model=args.recall_discovery_model,
        siglip_model=args.recall_siglip_model, match_threshold=args.recall_match_threshold,
        max_views=args.recall_max_views,
    )
    print("Loading SAM2 + SigLIP2 for the recall guide...")
    recall._ensure_loaded()

    frames_dir = None
    if args.save_frames:
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pose = np.array([start_xy[0], start_xy[1], yaw0])
    log: list[dict] = []

    print(f"\n=== Leg 1: instruction = 'go to the {args.category_a}' (Qwen pixel-goal + NavDP) ===")
    outcome1, remembered = run_leg1(pipe, sim, pf, island, cam_y, intr, pose, args.dt, args.max_steps_leg1,
                                    args.category_a, recall, log, frames_dir, args.realtime_pacing,
                                    args.probe_period_s)
    leg1_final = [float(pose[0]), float(pose[1]), float(pose[2])]
    leg1_error = dist(leg1_final, a_xy)
    print(f"outcome={outcome1}  final=({leg1_final[0]:.2f},{leg1_final[1]:.2f})  "
          f"error_to_A={leg1_error:.2f} m  memory_views={recall.views_stored}  states={state_counts(log,'leg1')}")

    result = {
        "scene": args.scene, "category_a": args.category_a, "category_b": args.category_b,
        "mechanism": "pixel-goal recall (SAM2 + SigLIP2 multi-view memory, no 3D point)",
        "ground_truth": {"A_xy": a_xy, "B_xy": b_xy, "A_B_geodesic_m": ab_geodesic,
                         "start_xy": start_xy, "start_A_geodesic_m": start_geodesic},
        "leg1": {"instruction": f"go to the {args.category_a}", "outcome": outcome1,
                 "final_pose": leg1_final, "error_to_A_m": leg1_error, "memory_views": recall.views_stored,
                 "state_counts": state_counts(log, "leg1"), "steps": sum(1 for r in log if r["leg"] == "leg1")},
    }

    if not remembered:
        print("\n=== Leg 2 skipped: no recall target was learned during leg 1 ===")
        result["leg2"] = None
    else:
        print(f"\n=== Leg 2: instruction = 'go to B' (pipe._qwen_pixel_guide swapped to "
              f"Fact3rEntityRecallGuide, {recall.views_stored} memory views, "
              f"match_threshold={args.recall_match_threshold}) ===")
        pipe._qwen_pixel_guide = recall
        # A fresh instruction means a fresh belief: leg 1's locked-on chair
        # belief (already inside stop_distance -- that's why it stopped) would
        # otherwise coast straight into a false STOP on leg 2's first tick,
        # before the recall guide is ever consulted.
        pipe.reset()
        outcome2 = run_leg2(pipe, sim, pf, island, cam_y, intr, pose, args.dt, args.max_steps_leg2, log,
                            frames_dir, args.realtime_pacing)
        pipe._qwen_pixel_guide = qwen_guide  # restore, in case the pipeline is reused
        leg2_final = [float(pose[0]), float(pose[1]), float(pose[2])]
        leg2_error = dist(leg2_final, b_xy)
        print(f"outcome={outcome2}  final=({leg2_final[0]:.2f},{leg2_final[1]:.2f})  "
              f"error_to_true_B={leg2_error:.2f} m  states={state_counts(log,'leg2')}")
        result["leg2"] = {
            "instruction": "go to B", "outcome": outcome2, "final_pose": leg2_final,
            "error_to_true_B_m": leg2_error, "state_counts": state_counts(log, "leg2"),
            "steps": sum(1 for r in log if r["leg"] == "leg2"),
        }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'} and {out_dir/'trajectory.json'}")
    sim.close()


if __name__ == "__main__":
    main()
