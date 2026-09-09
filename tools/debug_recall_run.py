#!/usr/bin/env python3
"""Debug a run_ab_position_recall_test.py run: draw the top-down trajectory
(occupancy grid + leg1/leg2 paths + A/B ground truth + recalled position),
show the actual object crops used for recall, and compute NE/SR/SPL correctly
from the raw trajectory (not just the single final-pose numbers result.json
prints).

This exists to answer one question directly: is the coordinate FRAME correct?
If leg2's path visibly walks toward the recalled marker and the recalled
marker is near-ish the true-B marker, the frame math (Fact3rGoalSource /
odom / occupancy grid_to_world) is consistent. If leg2's path goes some other
direction than the markers imply, that is a frame bug, not a semantics bug.

Usage:
    python tools/debug_recall_run.py --run-dir logs/ab_position_recall_bed_cabinet_fallback2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def path_length(poses: list[list[float]]) -> float:
    if len(poses) < 2:
        return 0.0
    p = np.array(poses)[:, :2]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def compute_metrics(result: dict, traj: list[dict]) -> dict:
    """Recompute NE/SR/SPL per-leg directly from the trajectory, independent
    of whatever result.json's own outcome bookkeeping says, as a
    cross-check."""
    gt = result["ground_truth"]
    success_distance = result["protocol"]["success_distance_m"]
    A_xy = np.array(gt["A_xy"])
    B_xy = np.array(gt["B_xy"])
    start_xy = np.array(gt["start_xy"])

    leg1_poses = [r["pose"] for r in traj if r["leg"] == "leg1"]
    leg2_poses = [r["pose"] for r in traj if r["leg"] == "leg2"]

    out = {}
    for leg_name, poses, goal_xy, optimal_from in [
        ("leg1", leg1_poses, A_xy, start_xy),
        ("leg2", leg2_poses, B_xy, (leg1_poses[-1][:2] if leg1_poses else start_xy)),
    ]:
        if not poses:
            out[leg_name] = {"NE_m": None, "SR": 0, "SPL": 0.0, "path_length_m": 0.0, "steps": 0}
            continue
        final_xy = np.array(poses[-1][:2])
        # A_xy is whichever instance was PICKED to set up an A<->B pair at
        # the right geodesic distance -- not necessarily the one closest to
        # where leg 1 ends up when there are several instances of the
        # category (there almost always are). GOAT's own criterion (and
        # the run's live printout) scores against the NEAREST instance;
        # reuse result.json's own number for leg 1 when present instead of
        # silently recomputing a different, stricter one against A_xy here.
        if leg_name == "leg1" and "error_to_nearest_instance_m" in result.get("leg1", {}):
            ne = float(result["leg1"]["error_to_nearest_instance_m"])
        else:
            ne = float(np.linalg.norm(final_xy - goal_xy))
        sr = int(ne < success_distance)
        actual_len = path_length(poses)
        optimal_len = float(np.linalg.norm(np.array(optimal_from) - goal_xy))
        spl = sr * (optimal_len / max(optimal_len, actual_len, 1e-6))
        out[leg_name] = {
            "NE_m": round(ne, 3),
            "SR": sr,
            "SPL": round(spl, 3),
            "path_length_m": round(actual_len, 3),
            "optimal_length_m": round(optimal_len, 3),
            "steps": len(poses),
        }

    out["overall"] = {
        "SR": (out["leg1"]["SR"] + out["leg2"]["SR"]) / 2.0,
        "SPL": (out["leg1"]["SPL"] + out["leg2"]["SPL"]) / 2.0,
        "mean_NE_m": None if out["leg1"]["NE_m"] is None or out["leg2"]["NE_m"] is None
                     else round((out["leg1"]["NE_m"] + out["leg2"]["NE_m"]) / 2.0, 3),
    }
    return out


def draw_trajectory(run_dir: Path, result: dict, traj: list[dict], metrics: dict) -> Path:
    gt = result["ground_truth"]
    A_xy = np.array(gt["A_xy"])
    B_xy = np.array(gt["B_xy"])
    start_xy = np.array(gt["start_xy"])

    occ_path = run_dir / "occupancy.npy"
    fig, ax = plt.subplots(figsize=(9, 9))

    if occ_path.exists():
        # LiveOccupancy convention (nav_pipeline/live_occupancy.py): UNKNOWN=-1,
        # FREE=0, OCCUPIED=1. grid[row, col] with row<-y, col<-x (world_to_grid),
        # so imshow(grid, origin="lower", extent=[x0,x1,y0,y1]) needs no transpose.
        grid = np.load(occ_path)
        res = 0.10  # default --map-res
        extent_m = 30.0  # default --map-extent
        origin_xy = (start_xy[0] - extent_m / 2.0, start_xy[1] - extent_m / 2.0)
        extent = [origin_xy[0], origin_xy[0] + extent_m, origin_xy[1], origin_xy[1] + extent_m]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#dcdcdc", "#ffffff", "#333333"])  # unknown, free, occupied
        ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")

    leg1_xy = np.array([r["pose"][:2] for r in traj if r["leg"] == "leg1"])
    leg2_xy = np.array([r["pose"][:2] for r in traj if r["leg"] == "leg2"])

    if len(leg1_xy):
        ax.plot(leg1_xy[:, 0], leg1_xy[:, 1], "-", color="tab:blue", lw=2, label="leg1 path (→A)")
        ax.scatter(*leg1_xy[0], marker="s", s=80, color="tab:blue", zorder=5)
    if len(leg2_xy):
        ax.plot(leg2_xy[:, 0], leg2_xy[:, 1], "-", color="tab:red", lw=2, label="leg2 path (→B, blind)")

    ax.scatter(*start_xy, marker="*", s=250, color="black", zorder=6, label="start")
    ax.scatter(*A_xy, marker="X", s=200, color="tab:blue", edgecolor="black", zorder=6,
              label=f"A instance picked for pairing ({result['category_a']}) -- NE scored vs NEAREST instead")
    ax.scatter(*B_xy, marker="X", s=200, color="tab:red", edgecolor="black", zorder=6, label="B ground truth (cabinet)")

    if result.get("recall") and result["recall"].get("position_xy"):
        rec_xy = np.array(result["recall"]["position_xy"])
        ax.scatter(*rec_xy, marker="o", s=180, facecolor="none", edgecolor="tab:orange", lw=3,
                   zorder=6, label="B recalled position")
        # draw a line from recalled to true B to make the frame error visually obvious
        ax.plot([rec_xy[0], B_xy[0]], [rec_xy[1], B_xy[1]], ":", color="tab:orange", lw=1.5,
               label=f"recall error ({np.linalg.norm(rec_xy - B_xy):.2f} m)")

    ax.set_xlabel("x (m, odom frame)")
    ax.set_ylabel("y (m, odom frame)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    title = (f"{result['scene']}  A={result['category_a']}  B={result['category_b']}\n"
             f"leg1 NE={metrics['leg1'].get('NE_m')}m SR={metrics['leg1']['SR']} | "
             f"leg2 NE={metrics['leg2'].get('NE_m')}m SR={metrics['leg2']['SR']} SPL={metrics['leg2']['SPL']}")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)

    out_path = run_dir / "trajectory_debug.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def build_object_frame_sheet(run_dir: Path, result: dict) -> Path | None:
    """Tile the saved recall-candidate crop frames into one sheet, labeled
    with the object name Qwen assigned and its confidence, for a human to
    eyeball -- "is this actually a cabinet?" """
    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        return None

    from PIL import Image, ImageDraw, ImageFont

    shortlist = result.get("recall_shortlist", [])
    recalled_id = (result.get("recall") or {}).get("entity_id")
    TILE = 220  # each crop upscaled to this size -- SAM2 crops are often
                # tiny (a partial/far object can be <50px), unreadable at
                # native resolution once several are tiled in one sheet
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    tiles = []
    for rank, cand in enumerate(shortlist, 1):
        eid = cand["entity_id"]
        # New runs save every shortlisted candidate's crops (not just the
        # winner) as shortlist_rank{N}_{id}_view{i}.png; fall back to the
        # older recall_{id}_view*.png (winner only) for runs from before
        # this existed, so old result dirs still render something.
        views = sorted(frames_dir.glob(f"shortlist_rank{rank}_{eid}_view*.png")) \
                or sorted(frames_dir.glob(f"recall_{eid}_view*.png"))
        if not views:
            continue
        imgs = [Image.open(v).convert("RGB").resize((TILE, TILE)) for v in views[:4]]
        row = Image.new("RGB", (TILE * len(imgs), TILE + 60), "white")
        for i, im in enumerate(imgs):
            row.paste(im, (i * TILE, 60))
        draw = ImageDraw.Draw(row)
        vlm = cand.get("vlm", {})
        tag = "   <== RECALLED (drove here)" if eid == recalled_id else ""
        line1 = f"rank {rank}: {eid}   siglip={cand['siglip_score']:.3f}   spread={cand['position_spread_m']:.2f}m   dist_to_true_B={cand.get('dist_to_true_B_m','?'):.2f}m{tag}"
        line2 = f"  qwen: {vlm.get('decision','-')} ('{vlm.get('predicted_object','-')}', conf={vlm.get('confidence',0):.2f})  {vlm.get('reason','')[:110]}"
        fill = "darkgreen" if eid == recalled_id else "black"
        draw.text((6, 4), line1, fill=fill, font=font)
        draw.text((6, 26), line2, fill="dimgray", font=font)
        tiles.append(row)

    if not tiles:
        return None

    total_h = sum(t.height for t in tiles)
    max_w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height

    out_path = run_dir / "recall_candidates_sheet.png"
    sheet.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    result = json.loads((args.run_dir / "result.json").read_text())
    traj = json.loads((args.run_dir / "trajectory.json").read_text())

    metrics = compute_metrics(result, traj)
    print("=" * 70)
    print(f"RUN: {args.run_dir}")
    print(f"Objects: A = '{result['category_a']}'  ->  B = '{result['category_b']}'  "
          f"(query: \"{result['query_b']}\")")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))

    traj_png = draw_trajectory(args.run_dir, result, traj, metrics)
    print(f"\nTrajectory plot: {traj_png}")

    sheet_png = build_object_frame_sheet(args.run_dir, result)
    if sheet_png:
        print(f"Object candidate crops: {sheet_png}")
    else:
        print("No recall candidate frames found to sheet.")

    (args.run_dir / "metrics_recomputed.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
