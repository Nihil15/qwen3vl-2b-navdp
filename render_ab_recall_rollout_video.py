#!/usr/bin/env python3
"""Replay a saved trajectory.json from run_ab_position_recall_test.py (the
custom "go to A, then recall B" test) through the same scene and encode the
re-rendered frames into an mp4 -- same idea as render_vlnce_rollout_video.py,
adapted for this harness's 4 phases (leg1 / lookaround / explore / leg2) and
per-step "pose" = (x, z, yaw) with floor height resolved via the navmesh
(cam_height isn't logged per step, but the pipeline always renders from
floor + cam_height, and snap_point recovers the real floor y at each x,z).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_ab_recall_hm3d_test import H, W, build_sim, render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory with trajectory.json + result.json")
    ap.add_argument("--scene-glb", required=True)
    ap.add_argument("--mp3d-house", default=None)
    ap.add_argument("--cam-height", type=float, default=0.9)
    ap.add_argument("--fps", type=float, default=None, help="default: 1/dt, real-time (dt=0.15s)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics-banner", default=None,
                    help="optional text overlaid on every leg2 frame, e.g. a hand-computed "
                         "SR/SPL/NE this harness doesn't natively report")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj = json.loads((run_dir / "trajectory.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    out_path = args.out or str(run_dir / "rollout.mp4")

    sim = build_sim(args.scene_glb, None, mp3d_house=args.mp3d_house)
    pf = sim.pathfinder

    label_by_leg = {
        "leg1": f"leg 1: go to the {result['category_a']}",
        "lookaround": "look-around at A (mapping)",
        "explore": "explore frontiers (mapping)",
        "leg2": f"leg 2: recall + go to '{result['query_b']}'",
    }

    recall = result.get("recall")
    if recall:
        vlm = recall.get("vlm", {})
        object_line = (
            f"recalled object: {vlm.get('predicted_object') or recall['entity_id']}  "
            f"(decision={vlm.get('decision', '?')} conf={vlm.get('confidence', 0):.2f}, "
            f"true target={result['category_b']}, "
            f"error to true B={recall.get('position_error_vs_true_B_m', float('nan')):.2f}m)"
        )
    else:
        object_line = "recall: no entity accepted"

    fps = args.fps or (1.0 / 0.15)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i, t in enumerate(traj):
        x, z, yaw = t["pose"]
        snapped = pf.snap_point(np.array([x, 0.0, z], dtype=np.float32))
        cam_y = float(snapped[1]) + args.cam_height
        rgb, _ = render(sim, (x, z, yaw), cam_y)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

        top = f"{label_by_leg.get(t['leg'], t['leg'])}   step {t['step']:4d}  state={t['state']:<6s}"
        cv2.putText(frame, top, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, top, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        extra = ""
        if "dist_to_recalled" in t:
            extra = f"dist_to_recalled={t['dist_to_recalled']:.2f}m  mode={t.get('mode', '-')}"
        elif "instance_mismatch" in t:
            extra = f"instance_mismatch={t['instance_mismatch']}"
        if extra:
            cv2.putText(frame, extra, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, extra, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        pg = t.get("qwen_pixel_goal")
        if pg:
            u, v = int(round(pg[0])), int(round(pg[1]))
            cv2.drawMarker(frame, (u, v), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS,
                          markerSize=22, thickness=2)
            cv2.circle(frame, (u, v), 14, (0, 0, 255), 2)
            cv2.putText(frame, "Qwen pixel-goal", (max(0, u - 60), max(20, v - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Qwen pixel-goal", (max(0, u - 60), max(20, v - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        bottom = f"v={t.get('v', 0):+.2f} w={t.get('w', 0):+.2f}"
        cv2.putText(frame, bottom, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, bottom, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        has_leg2 = result.get("leg2") is not None
        if t["leg"] == "leg2" or (not has_leg2 and i == len(traj) - 1):
            cv2.putText(frame, object_line, (8, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, object_line, (8, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
        if t["leg"] == "leg2" or not has_leg2:
            if args.metrics_banner:
                cv2.putText(frame, args.metrics_banner, (8, H - 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, args.metrics_banner, (8, H - 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        writer.write(frame)
        if i % 100 == 0:
            print(f"  rendered {i}/{len(traj)}", flush=True)

    writer.release()
    sim.close()
    print(f"wrote {out_path} ({len(traj)} frames @ {fps:.2f} fps)")


if __name__ == "__main__":
    main()
