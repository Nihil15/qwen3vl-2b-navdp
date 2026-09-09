#!/usr/bin/env python3
"""Render an HM3D walkthrough in the exact layout fact3r-map's depth+odometry
BEV path consumes -- no MASt3R needed.

`build_depth_semantic_bev.py` fuses *metric depth + rover odometry* into the
semantic BEV that `resolve_semantic_goal.py` queries for an entity's real
metric position. Its `--depth-dir` loader is written for simulator depth
("Simulator depth is rendered at the original square resolution"), and
`vlnce_poses_to_odom.py` turns habitat ground-truth poses into the
`odom_*.csv` it wants. So a habitat walkthrough can feed that whole path
directly, skipping MASt3R-SLAM entirely -- which is also how the real rover
runs it (Depth-Anything metric depth + wheel odometry, no MASt3R).

Writes, into --out:
    000000.png ...        RGB frames (for SAM2/SigLIP observation indexing)
    depth_000000.npy ...  metric depth, same resolution as the RGB
    groundtruth.txt       TUM poses in the habitat world frame
    walk.mp4              the same frames as a video, for run_fact3r_live.py

Then:
    conda run -n sam2 python fact3r-map/scripts/vlnce_poses_to_odom.py --poses <out>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from run_ab_recall_hm3d_test import (
    EYE_HEIGHT_M,
    FOV_DEG,
    H,
    W,
    find_dataset_config,
    find_scene_glb,
    largest_island,
    pick_pair,
    render,
)


def build_sim_tilted(scene_glb: str, dataset_config, pitch_deg: float, cam_height: float):
    """Same sim as run_ab_recall_hm3d_test.build_sim, but the camera can be
    pitched DOWN. A perfectly level camera at eye height barely sees the floor,
    so the BEV gets ~0% free space -- every depth return lands on a wall or on
    furniture inside the [0.1, 1.5] m obstacle slice and nothing is ever carved
    free, leaving the planner with nowhere legal to stand. The real rover's own
    config tilts down for exactly this reason (build_depth_semantic_bev.py's
    --pitch/--cam-height defaults are 2.75 deg at 0.5 m)."""
    import habitat_sim

    orientation = np.array([math.radians(-abs(pitch_deg)), 0.0, 0.0], dtype=np.float32)
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [H, W]
        spec.position = np.array([0.0, cam_height, 0.0])
        spec.orientation = orientation
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


def habitat_pose(pose, cam_y: float):
    """My planar (X, Yplanar, yaw) -> habitat (position xyz, quaternion xyzw),
    the exact same mapping render() applies internally."""
    X, Yp, yaw = pose
    habitat_yaw = -(yaw + math.pi / 2.0)
    half = habitat_yaw / 2.0
    # quat_from_angle_axis(habitat_yaw, [0,1,0]) == (0, sin(half), 0, cos(half))
    return (float(X), float(cam_y), float(Yp)), (0.0, math.sin(half), 0.0, math.cos(half))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--category-a", default="chair")
    ap.add_argument("--category-b", default="table")
    ap.add_argument("--fps", type=float, default=2.0, help="timestamps written into groundtruth.txt")
    ap.add_argument("--tour-points", type=int, default=0,
                    help="if >0, also tour this many extra navigable points across the island "
                         "(chained geodesically, with a pan at each) -- a straight A->B walk "
                         "observes too little of the room for the BEV to be planner-navigable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pitch-deg", type=float, default=20.0,
                    help="camera downward tilt for mapping -- a level camera sees no floor, "
                         "so the BEV carves no free space and the planner has nowhere to stand")
    ap.add_argument("--cam-height", type=float, default=EYE_HEIGHT_M)
    ap.add_argument("--room-radius", type=float, default=0.0,
                    help="if >0, keep tour points within this many metres of the A-B midpoint. "
                         "Touring the whole house spreads a fixed frame budget over a big grid and "
                         "leaves ~73%% of it unknown, which the planner refuses; concentrating the "
                         "same budget on the room that actually holds A and B is what makes the BEV "
                         "dense enough to plan over.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hm3d_root = Path(args.hm3d_root)
    sim = build_sim_tilted(find_scene_glb(hm3d_root, args.scene), find_dataset_config(hm3d_root),
                           args.pitch_deg, args.cam_height)
    pf = sim.pathfinder
    island = largest_island(pf)
    a_point, b_point, ab_geo = pick_pair(sim, island, args.category_a, args.category_b, 2.0, 14.0)
    print(f"A ({args.category_a}) @ ({a_point[0]:.2f},{a_point[2]:.2f})  "
          f"B ({args.category_b}) @ ({b_point[0]:.2f},{b_point[2]:.2f})  geodesic={ab_geo:.2f}m")

    # start a little back from A, on the far side from B, so the walk passes A then B
    start = np.array(pf.snap_point(
        np.array([a_point[0] - (b_point[0] - a_point[0]) * 0.6,
                  a_point[1],
                  a_point[2] - (b_point[2] - a_point[2]) * 0.6]), island))
    cam_y = float(a_point[1]) + args.cam_height

    poses: list[np.ndarray] = []

    def add_walk(p0, p1, n):
        yaw = math.atan2(p1[2] - p0[2], p1[0] - p0[0])
        for k in range(n):
            t = k / max(n - 1, 1)
            poses.append(np.array([p0[0] + (p1[0] - p0[0]) * t,
                                   p0[2] + (p1[2] - p0[2]) * t, yaw]))

    def add_pan(p, yaw_center, n, sweep=0.6):
        for k in range(n):
            poses.append(np.array([p[0], p[2], yaw_center + sweep * math.sin(2 * math.pi * k / n)]))

    yaw_a = math.atan2(a_point[2] - start[2], a_point[0] - start[0])
    add_walk(start, a_point, 20)
    add_pan(a_point, yaw_a, 12)
    yaw_b = math.atan2(b_point[2] - a_point[2], b_point[0] - a_point[0])
    add_walk(a_point, b_point, 16)
    add_pan(b_point, yaw_b, 12)

    if args.tour_points > 0:
        # A straight A->B walk leaves ~78% of the BEV unobserved, so
        # project_semantic_goal.py finds no navigable component to plan over.
        # Chain geodesic legs between navigable points to actually see the room,
        # panning at each stop so the depth fan sweeps rather than staring.
        import habitat_sim

        rng = np.random.default_rng(args.seed)
        midpoint = (np.asarray(a_point) + np.asarray(b_point)) / 2.0
        prev = b_point
        for _ in range(args.tour_points):
            target = None
            for _try in range(200):
                cand = np.array(pf.get_random_navigable_point(island_index=island))
                if args.room_radius > 0.0:
                    if math.hypot(cand[0] - midpoint[0], cand[2] - midpoint[2]) > args.room_radius:
                        continue
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = prev, cand
                lo = 0.4 if args.room_radius > 0.0 else 1.0   # shorter hops inside one room
                hi = args.room_radius if args.room_radius > 0.0 else 8.0
                if pf.find_path(path) and lo <= path.geodesic_distance <= max(hi, lo + 0.1):
                    target = cand
                    break
            if target is None:
                break
            yaw_t = math.atan2(target[2] - prev[2], target[0] - prev[0])
            add_walk(prev, target, 14)
            add_pan(target, yaw_t, 10, sweep=1.4)  # wider sweep = more coverage per stop
            prev = target

    import cv2

    writer = cv2.VideoWriter(str(out / "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    lines = ["# habitat camera poses: t x y z qx qy qz qw"]
    for i, p in enumerate(poses):
        rgb, depth = render(sim, p, cam_y)
        cv2.imwrite(str(out / f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(out / f"depth_{i:06d}.npy", depth.astype(np.float32))
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        (px, py, pz), (qx, qy, qz, qw) = habitat_pose(p, cam_y)
        lines.append(f"{i / args.fps:.6f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    writer.release()
    (out / "groundtruth.txt").write_text("\n".join(lines) + "\n")

    meta = {
        "scene": args.scene, "frames": len(poses), "fps": args.fps,
        "width": W, "height": H, "hfov_deg": FOV_DEG, "cam_height_m": args.cam_height, "pitch_deg": args.pitch_deg,
        "A_xy": [float(a_point[0]), float(a_point[2])],
        "B_xy": [float(b_point[0]), float(b_point[2])],
        "A_B_geodesic_m": float(ab_geo),
        "start_xy": [float(start[0]), float(start[2])],
    }
    import json
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(poses)} frames + depth + groundtruth.txt + walk.mp4 -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
