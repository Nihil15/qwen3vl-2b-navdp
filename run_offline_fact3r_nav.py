#!/usr/bin/env python3
"""Offline driver: Fact3R-Map remembered goal -> NavDP "GOTO" rollout.

Headless, no Zenoh, no rover. Feeds a recorded RGB (optionally RGB-D) walk
through ``DinoNavDPPipeline`` with the target supplied every tick by
``nav_pipeline.fact3r_goal_source.Fact3rGoalSource`` as ``external_goal`` --
the blind navigate-back path the pipeline already has (state ``"GOTO"``:
obstacle-guard + NavDP live, no self-declared STOP from proximity). This is
the "REMEMBER -> GO" leg: Fact3R-Map was built earlier from a walk through
the scene, a text query located an entity in it, and the rover now drives
back to that point.

Pose comes either from a per-frame file (``--poses``) or, with
``--integrate-cmd``, from dead-reckoning the pipeline's own velocity output
through a unicycle model -- the closed-loop "did NavDP actually get there"
check when no external odometry is available.

Two ways to give the goal:

    # (a) a goal request already produced by fact3r-map/resolve_semantic_goal.py
    python run_offline_fact3r_nav.py --frames walk/ --goal-request goal_request.json \
        --align identity --integrate-cmd --out logs/fact3r_nav/run1

    # (b) resolve it here, shelling out to that script in its SigLIP env
    python run_offline_fact3r_nav.py --frames walk.mp4 \
        --fact3r-map-root ../MASt3R-SLAM/fact3r-map \
        --semantic-map logs/fact3r_video/room-walk/room-walk_semantic.json \
        --query "the black chair" --siglip-env SAM2 \
        --align anchor --integrate-cmd --out logs/fact3r_nav/run1

    # (c) no map at all -- just exercise the GOTO loop toward a fixed odom point
    python run_offline_fact3r_nav.py --frames walk/ --goal-odom-xy "4.0 -1.5" \
        --integrate-cmd --out logs/fact3r_nav/smoke
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from nav_pipeline.fact3r_goal_source import Fact3rGoalSource, MapToOdom
from nav_pipeline.obstacle_guard import GuardConfig
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def iter_frames(source: str):
    """Yield RGB uint8 HxWx3 frames from a video file or a directory of images."""
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTS)}) in {path}")
        for f in files:
            bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not read {f}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {source}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_depth_dir(depth_dir: str | None):
    if depth_dir is None:
        return None
    files = sorted(p for p in Path(depth_dir).iterdir() if p.suffix.lower() in (".npy", ".png"))
    if not files:
        raise FileNotFoundError(f"no .npy/.png depth frames in {depth_dir}")

    def read(p: Path) -> np.ndarray:
        if p.suffix.lower() == ".npy":
            return np.load(p).astype(np.float32)
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return raw.astype(np.float32) / 1000.0  # 16-bit millimetre PNG convention

    return [read(f) for f in files]


def load_pose_file(path: str) -> np.ndarray:
    """One pose per line: either ``x y theta`` (3 cols, theta in rad) or a
    TUM row ``t x y z qx qy qz qw`` (8 cols) reduced to (x, y, yaw)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(v) for v in line.replace(",", " ").split()]
        if len(parts) == 3:
            rows.append(parts)
        elif len(parts) >= 8:
            _t, x, y, _z, qx, qy, qz, qw = parts[:8]
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            rows.append([x, y, yaw])
        else:
            raise ValueError(f"{path}: need 3 or >=8 columns per line, got {len(parts)}")
    if not rows:
        raise ValueError(f"{path}: no pose rows")
    return np.asarray(rows, dtype=np.float64)


def load_xy_track(path: str) -> np.ndarray:
    pts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(v) for v in line.replace(",", " ").split()]
        pts.append(parts[:2])
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------- #
# goal source assembly
# --------------------------------------------------------------------------- #
def build_goal_source(args) -> Fact3rGoalSource:
    if args.goal_odom_xy is not None:
        x, y = (float(v) for v in args.goal_odom_xy.split())
        return Fact3rGoalSource([x, y], MapToOdom.identity(), reached_radius=args.reached_radius)

    explicit = None
    if args.align == "explicit":
        if not args.map_to_odom:
            raise SystemExit("--align explicit needs --map-to-odom 'tx ty phi_deg scale'")
        tx, ty, phi_deg, scale = (float(v) for v in args.map_to_odom.split())
        explicit = MapToOdom.explicit(tx, ty, phi_deg, scale)

    odom_track = load_xy_track(args.odom_track) if args.odom_track else None
    common = dict(
        map_to_odom=explicit,
        align=args.align,
        odom_track_xy=odom_track,
        odom_heading_rad=math.radians(args.odom_heading_deg),
        reached_radius=args.reached_radius,
    )

    if args.goal_request:
        return Fact3rGoalSource.from_goal_request(args.goal_request, **common)

    if not (args.fact3r_map_root and args.semantic_map and args.query and args.siglip_env):
        raise SystemExit(
            "give one of: --goal-odom-xy, --goal-request, or "
            "(--fact3r-map-root --semantic-map --query --siglip-env)"
        )
    observed = tuple(args.observed_between) if args.observed_between else None
    return Fact3rGoalSource.from_map_query(
        fact3r_map_root=args.fact3r_map_root,
        semantic_map=args.semantic_map,
        query=args.query,
        siglip_env=args.siglip_env,
        device=args.siglip_device,
        work_dir=Path(args.out) / "fact3r",
        observed_between=observed,
        min_score=args.min_score,
        **common,
    )


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def hud(frame_rgb: np.ndarray, text_lines: list[str]) -> np.ndarray:
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    y = 22
    for line in text_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 24
    return out


def topdown_plot(path: Path, poses: np.ndarray, goal_xy: np.ndarray, reached_radius: float):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(poses[:, 0], poses[:, 1], "-", color="tab:blue", label="rover odom path")
    ax.plot(poses[0, 0], poses[0, 1], "o", color="tab:green", label="start")
    ax.plot(poses[-1, 0], poses[-1, 1], "s", color="tab:blue", label="end")
    ax.plot(goal_xy[0], goal_xy[1], "*", color="tab:red", markersize=16, label="fact3r goal")
    ax.add_patch(plt.Circle(tuple(goal_xy), reached_radius, color="tab:red", alpha=0.15))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="video file or directory of images")
    ap.add_argument("--depth", help="directory of per-frame .npy/.png metric depth; omit to use Depth Anything V2")
    ap.add_argument("--intrinsics", help="'fx fy cx cy' from calibration; else derived from --fov")
    ap.add_argument("--fov", type=float, default=90.0, help="horizontal FOV deg when --intrinsics is absent")

    # goal
    ap.add_argument("--goal-request", help="fact3r-map/resolve_semantic_goal.py output JSON")
    ap.add_argument("--goal-odom-xy", help="'x y' fixed odom-frame goal, bypasses fact3r entirely (smoke test)")
    ap.add_argument("--fact3r-map-root", help="path to MASt3R-SLAM/fact3r-map (for --query)")
    ap.add_argument("--semantic-map", help="*_semantic.json stem passed to resolve_semantic_goal.py")
    ap.add_argument("--query", help="text goal to resolve against the fact3r map")
    ap.add_argument("--siglip-env", help="conda env with SigLIP for resolve_semantic_goal.py")
    ap.add_argument("--siglip-device", default="0")
    ap.add_argument("--observed-between", type=int, nargs=2, metavar=("FIRST", "LAST"))
    ap.add_argument("--min-score", type=float)

    # alignment
    ap.add_argument("--align", choices=("identity", "anchor", "umeyama", "explicit"), default="identity")
    ap.add_argument("--map-to-odom", help="'tx ty phi_deg scale' for --align explicit")
    ap.add_argument("--odom-track", help="xy-per-line odom trajectory of the same walk, for --align umeyama")
    ap.add_argument("--odom-heading-deg", type=float, default=0.0, help="initial rover heading for --align anchor")
    ap.add_argument("--reached-radius", type=float, default=1.0, help="stop when this close to the goal (m)")

    # pose / dynamics
    ap.add_argument("--poses", help="per-frame pose file ('x y theta' or TUM rows); else use --integrate-cmd")
    ap.add_argument("--integrate-cmd", action="store_true", help="dead-reckon the pipeline's (v, w) output")
    ap.add_argument("--dt", type=float, default=0.1, help="control period for --integrate-cmd (s)")
    ap.add_argument("--max-steps", type=int, default=600)

    # pipeline
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--depth-encoder", choices=("vits", "vitb"), default="vits")
    ap.add_argument("--max-linear", type=float, default=0.5)
    ap.add_argument("--max-angular", type=float, default=0.4)
    ap.add_argument("--no-avoid", action="store_true", help="disable the depth obstacle guard")

    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--seed", type=int, default=-1,
                    help="seed torch + numpy before the pipeline (NavDP samples 32 trajectories "
                         "from torch.randn each tick and is otherwise unseeded). -1 = legacy "
                         "unseeded; >=0 = reproducible GOTO rollout.")
    args = ap.parse_args()

    if args.seed >= 0:
        import os as _os
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import random as _random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            torch.use_deterministic_algorithms(True)
        print(f"[seed] torch+numpy+random seeded {args.seed}, cudnn deterministic")

    if not args.poses and not args.integrate_cmd and args.goal_odom_xy is None:
        raise SystemExit("need --poses or --integrate-cmd for the pose stream")
    if not args.poses and not args.integrate_cmd:
        args.integrate_cmd = True  # smoke test defaults to closed loop

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_src = build_goal_source(args)
    print(f"[offline-fact3r-nav] {goal_src.describe()}")

    intr = tuple(float(v) for v in args.intrinsics.split()) if args.intrinsics else None
    depth_frames = load_depth_dir(args.depth)
    poses = load_pose_file(args.poses) if args.poses else None

    cfg = PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        policy_type=args.policy,
        depth_encoder=args.depth_encoder,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        avoid_enabled=not args.no_avoid,
        use_dino=False,
        use_sam=False,
        use_clip=False,
        use_qwen_search=False,
        use_qwen_instruction=False,
        use_appearance_reid=False,
        use_scene_tagger=False,
        use_belief_goal=False,  # GOTO supplies the goal every tick; belief never engages
        guard=GuardConfig(),
    )
    pipe = DinoNavDPPipeline(cfg, use_depth_estimator=depth_frames is None)

    pose = np.array([0.0, 0.0, 0.0]) if poses is None else poses[0].copy()
    writer = None
    log: list[dict] = []
    pose_hist: list[list[float]] = []
    outcome = "max_steps"

    for i, rgb in enumerate(iter_frames(args.frames)):
        if i >= args.max_steps:
            break
        if poses is not None:
            if i >= len(poses):
                outcome = "poses_exhausted"
                break
            pose = poses[i].copy()

        depth = depth_frames[i] if depth_frames is not None else None
        if depth_frames is not None and i >= len(depth_frames):
            outcome = "depth_exhausted"
            break

        ext = goal_src.tick(pose)
        res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                        external_goal=ext, intrinsics=intr)

        dist = goal_src.distance(pose)
        pose_hist.append([float(pose[0]), float(pose[1]), float(pose[2])])
        log.append({
            "step": i,
            "pose": pose_hist[-1],
            "goal_local_fwd_left": [float(ext[0]), float(ext[1])],
            "distance_to_goal": dist,
            "state": res.state,
            "linear": float(res.linear),
            "angular": float(res.angular),
            "min_forward": None if res.min_forward is None else float(res.min_forward),
        })

        if not args.no_video:
            if writer is None:
                h, w = rgb.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "rollout.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            writer.write(hud(rgb, [
                f"step {i}  state {res.state}",
                f"dist {dist:.2f} m   goal fwd/left {ext[0]:+.2f}/{ext[1]:+.2f}",
                f"cmd v={res.linear:+.2f}  w={res.angular:+.2f}",
            ]))

        if i % 20 == 0:
            print(f"[step {i:4d}] state={res.state:6s} dist={dist:5.2f}m "
                  f"v={res.linear:+.2f} w={res.angular:+.2f} pose=({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f})")

        if goal_src.reached(pose):
            outcome = "reached"
            break
        if res.state == "STOP":
            outcome = "pipeline_stop"
            break

        if poses is None:  # unicycle dead-reckon of the issued command
            v, w = float(res.linear), float(res.angular)
            pose[0] += v * math.cos(pose[2]) * args.dt
            pose[1] += v * math.sin(pose[2]) * args.dt
            pose[2] = (pose[2] + w * args.dt + math.pi) % (2 * math.pi) - math.pi

    if writer is not None:
        writer.release()

    pose_arr = np.asarray(pose_hist, dtype=np.float64) if pose_hist else np.zeros((1, 3))
    summary = {
        "outcome": outcome,
        "steps": len(log),
        "final_distance_to_goal": log[-1]["distance_to_goal"] if log else None,
        "goal_odom_xy": [float(goal_src.goal_odom_xy[0]), float(goal_src.goal_odom_xy[1])],
        "goal_map_xy": [float(goal_src.goal_map_xy[0]), float(goal_src.goal_map_xy[1])],
        "map_to_odom": goal_src.map_to_odom.__dict__,
        "query": goal_src.query,
        "pose_source": "file" if poses is not None else "integrate_cmd",
    }
    (out_dir / "trajectory.json").write_text(json.dumps({"summary": summary, "steps": log}, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    topdown_plot(out_dir / "topdown.png", pose_arr[:, :2], goal_src.goal_odom_xy, args.reached_radius)

    print(f"\n[offline-fact3r-nav] outcome={outcome} steps={len(log)} "
          f"final_dist={summary['final_distance_to_goal']} -> {out_dir}")


if __name__ == "__main__":
    main()
