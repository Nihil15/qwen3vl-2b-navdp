#!/usr/bin/env python3
"""Find an A/B episode that can actually test the thing we mean to test.

The scenario is: drive to A, see B along the way, then recall B from memory
and go there without B in view. That has a precondition nothing was checking
-- B must genuinely be visible during the drive to A. When it is not, the
mapper never stores B, and the recall query can only ever return the nearest
plausible distractor. Three full simulator runs failed that way before this
existed, and the failure looked like a retrieval bug rather than an episode
that was impossible from the start.

So verify the precondition up front, with the semantic sensor (exact and
cheap) rather than by running the whole stack for 25 minutes:

  * A is visible from the start pose        -- else Qwen has nothing to ground
  * B is visible from >= min_b_frames poses along the start->A route
  * the start pose has real obstacle clearance

Writes an episode JSON that run_ab_position_recall_test.py can load, so the
expensive run always begins from a feasible setup.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import habitat_sim
from nav_pipeline.obstacle_guard import GuardConfig
from run_ab_recall_hm3d_test import (FOV_DEG, H, W, clearance_m,
                                     find_dataset_config, find_scene_glb,
                                     geodesic, largest_island)


def build_sim_semantic(scene_glb, dataset_config):
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH),
                        ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, stype
        spec.resolution = [H, W]
        spec.position = np.array([0.0, 0.0, 0.0])
        spec.hfov = FOV_DEG
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def look(sim, xy, yaw, cam_y):
    from habitat_sim.utils.common import quat_from_angle_axis
    state = habitat_sim.AgentState()
    state.position = np.array([xy[0], cam_y, xy[1]], dtype=np.float32)
    state.rotation = quat_from_angle_axis(-(yaw + math.pi / 2.0), np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)
    return sim.get_sensor_observations()


def visible_pixels(obs, semantic_id: int) -> int:
    return int((np.asarray(obs["semantic"]) == semantic_id).sum())


def instances(sim, category, island):
    """Semantic instances of `category` with the id the sensor reports."""
    out = []
    for obj in sim.semantic_scene.objects or []:
        if obj is None or obj.category is None or obj.category.name() != category:
            continue
        snapped = np.array(sim.pathfinder.snap_point(obj.aabb.center(), island))
        if np.all(np.isfinite(snapped)):
            out.append((int(obj.semantic_id), snapped, np.array(obj.aabb.center())))
    return out


def route_points(pf, start, goal):
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = start, goal
    if not pf.find_path(path):
        return None
    return [np.array(p) for p in path.points]


def resample_route(points, spacing=0.5):
    """Dense poses along the polyline, each facing the direction of travel --
    which is what the rover's camera will actually see on the way."""
    out = []
    for a, b in zip(points, points[1:]):
        seg = math.hypot(b[0] - a[0], b[2] - a[2])
        n = max(int(seg / spacing), 1)
        yaw = math.atan2(b[2] - a[2], b[0] - a[0])
        for k in range(n):
            t = k / n
            out.append(((a[0] + (b[0] - a[0]) * t, a[2] + (b[2] - a[2]) * t), yaw))
    if points:
        last_yaw = out[-1][1] if out else 0.0
        out.append(((points[-1][0], points[-1][2]), last_yaw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--category-a", default="chair")
    ap.add_argument("--category-b", default="table")
    ap.add_argument("--cam-height", type=float, default=0.9)
    ap.add_argument("--min-ab", type=float, default=2.0)
    ap.add_argument("--max-ab", type=float, default=10.0)
    ap.add_argument("--min-start-a", type=float, default=2.5)
    ap.add_argument("--max-start-a", type=float, default=8.0)
    ap.add_argument("--min-pixels-a", type=int, default=800,
                    help="A must occupy at least this many pixels from the start pose")
    ap.add_argument("--min-pixels-b", type=int, default=600)
    ap.add_argument("--min-b-frames", type=int, default=4,
                    help="B must be this visible along the route, or the mapper never stores it")
    ap.add_argument("--min-clearance", type=float, default=1.0)
    ap.add_argument("--starts", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/episode.json")
    args = ap.parse_args()

    hm3d_root = Path(args.hm3d_root)
    sim = build_sim_semantic(find_scene_glb(hm3d_root, args.scene), find_dataset_config(hm3d_root))
    pf = sim.pathfinder
    island = largest_island(pf)
    guard = GuardConfig(cam_height=args.cam_height)

    a_list = instances(sim, args.category_a, island)
    b_list = instances(sim, args.category_b, island)
    print(f"{len(a_list)} '{args.category_a}', {len(b_list)} '{args.category_b}' instances")
    if not a_list or not b_list:
        raise SystemExit("scene lacks one of the categories")

    rng = np.random.default_rng(args.seed)
    starts = [np.array(pf.get_random_navigable_point(island_index=island))
              for _ in range(args.starts)]

    best = None
    for a_id, a_pt, _ in a_list:
        for b_id, b_pt, _ in b_list:
            ab = geodesic(pf, a_pt, b_pt)
            if ab is None or not (args.min_ab <= ab <= args.max_ab):
                continue
            for s in starts:
                d = geodesic(pf, s, a_pt)
                if d is None or not (args.min_start_a <= d <= args.max_start_a):
                    continue
                cam_y = float(s[1]) + args.cam_height
                yaw = math.atan2(a_pt[2] - s[2], a_pt[0] - s[0])
                obs = look(sim, (s[0], s[2]), yaw, cam_y)
                a_px = visible_pixels(obs, a_id)
                if a_px < args.min_pixels_a:
                    continue
                pts = route_points(pf, s, a_pt)
                if not pts:
                    continue
                b_frames, b_px_total = 0, 0
                for xy, ryaw in resample_route(pts):
                    o = look(sim, xy, ryaw, cam_y)
                    px = visible_pixels(o, b_id)
                    if px >= args.min_pixels_b:
                        b_frames += 1
                        b_px_total += px
                if b_frames < args.min_b_frames:
                    continue
                clear = clearance_m(sim, np.array([s[0], s[2], yaw]), cam_y, guard)
                if clear < args.min_clearance:
                    continue
                score = (b_frames, a_px, clear)
                if best is None or score > best["score"]:
                    best = {"score": score, "start_xy": [float(s[0]), float(s[2])],
                            "start_yaw": float(yaw), "floor_y": float(s[1]),
                            "A_xy": [float(a_pt[0]), float(a_pt[2])], "A_semantic_id": a_id,
                            "B_xy": [float(b_pt[0]), float(b_pt[2])], "B_semantic_id": b_id,
                            "A_B_geodesic_m": float(ab), "start_A_geodesic_m": float(d),
                            "A_pixels_at_start": int(a_px), "B_visible_frames": int(b_frames),
                            "B_pixels_total": int(b_px_total), "start_clearance_m": float(clear)}
                    print(f"  candidate: B visible in {b_frames} route frames, A {a_px} px at start, "
                          f"clearance {clear:.2f} m, start->A {d:.2f} m, A<->B {ab:.2f} m")

    if best is None:
        raise SystemExit("no feasible episode -- loosen --min-pixels-* / --min-b-frames")
    best.pop("score")
    best["scene"] = args.scene
    best["category_a"], best["category_b"] = args.category_a, args.category_b
    best["cam_height_m"] = args.cam_height
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    print(f"\nchosen episode -> {out}\n{json.dumps(best, indent=2)}")
    sim.close()


if __name__ == "__main__":
    main()
