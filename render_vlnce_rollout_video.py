#!/usr/bin/env python3
"""Replay a saved trajectory.json (from run_vlnce_episode_test.py) through
the same scene and encode the re-rendered frames into an mp4.

The episode test only logs poses, not images (Qwen+NavDP inference is the
expensive part, rendering is cheap and fully deterministic given the same
scene mesh), so this reconstructs the visual rollout after the fact instead
of needing to re-run the model.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_vlnce_episode_test import build_sim, render, EYE_HEIGHT_M, W, H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory with trajectory.json + result.json")
    ap.add_argument("--scene-glb", required=True)
    ap.add_argument("--fps", type=float, default=None,
                    help="default: 1/dt from the trajectory's own step spacing (real-time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj = json.loads((run_dir / "trajectory.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    out_path = args.out or str(run_dir / "rollout.mp4")

    sim = build_sim(args.scene_glb, None)

    fps = args.fps or (1.0 / 0.15)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i, t in enumerate(traj):
        x, z, yaw = t["pose"]
        rgb, _ = render(sim, (x, z, yaw), EYE_HEIGHT_M)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        label = f"step {t['step']:4d}  state={t['state']:<6s}  dist_to_goal={t['dist_to_goal']:.2f}m"
        cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"\"{result['instruction']}\"", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"\"{result['instruction']}\"", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"v={t['v']:+.2f} w={t['w']:+.2f}", (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"v={t['v']:+.2f} w={t['w']:+.2f}", (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        if i % 50 == 0:
            print(f"  rendered {i}/{len(traj)}", flush=True)

    writer.release()
    sim.close()
    print(f"wrote {out_path} ({len(traj)} frames @ {fps:.2f} fps)")


if __name__ == "__main__":
    main()
