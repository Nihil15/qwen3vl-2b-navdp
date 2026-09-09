#!/usr/bin/env python3
"""Build a topdown trajectory + occupancy-map report for a
run_ab_position_recall_test.py output directory: the carved occupancy grid
as background, all 4 phases of the run drawn as colored paths, and the real
ground-truth / recalled positions marked -- answers "did it actually go
where the map/plan said" and "how much of the house got mapped" in one image.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEG_COLOR = {"leg1": "#1f77b4", "lookaround": "#9467bd", "explore": "#ff7f0e", "leg2": "#2ca02c"}
LEG_LABEL = {"leg1": "leg 1: go to A", "lookaround": "look-around at A",
            "explore": "explore frontiers", "leg2": "leg 2: recall -> go to B"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--map-extent", type=float, default=30.0)
    ap.add_argument("--map-res", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    traj = json.loads((run_dir / "trajectory.json").read_text())
    occupancy_path = run_dir / "occupancy.npy"
    grid = np.load(occupancy_path) if occupancy_path.is_file() else None
    out_path = args.out or str(run_dir / "topdown_report.png")

    gt = result["ground_truth"]
    start_xy = gt["start_xy"]
    origin_x = start_xy[0] - args.map_extent / 2.0
    origin_y = start_xy[1] - args.map_extent / 2.0
    extent = [origin_x, origin_x + args.map_extent, origin_y, origin_y + args.map_extent]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if grid is not None:
        # nav_pipeline.live_occupancy: UNKNOWN, FREE, OCCUPIED = -1, 0, 1
        cmap = matplotlib.colors.ListedColormap(["#e8e8e8", "#ffffff", "#333333"])
        norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        # Older run predating the fix that saves occupancy.npy on the
        # no-entity-recalled early-exit path -- still plot trajectory/
        # candidates on a blank canvas rather than failing outright.
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor("#f5f5f5")

    for leg in ("explore", "lookaround", "leg1", "leg2"):
        pts = [t["pose"] for t in traj if t["leg"] == leg]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=LEG_COLOR[leg], linewidth=1.6, alpha=0.85, label=LEG_LABEL[leg],
                zorder=3)

    ax.plot(*gt["start_xy"], marker="o", color="black", markersize=10, zorder=5, label="start")
    ax.plot(*gt["A_xy"], marker="*", color="blue", markersize=18, zorder=5,
            label=f"true A ({result['category_a']})")
    ax.plot(*gt["B_xy"], marker="*", color="red", markersize=18, zorder=5,
            label=f"true B ({result['category_b']})")
    decision_style = {"yes": ("green", "o"), "no": ("gray", "x"),
                      "uncertain": ("orange", "^"), "skipped": ("lightgray", ".")}
    for rec in result.get("recall_shortlist", []):
        vlm = rec.get("vlm", {})
        color, marker = decision_style.get(vlm.get("decision"), ("black", "?"))
        px, pz = rec["position_xy"]
        ax.plot(px, pz, marker=marker, color=color, markersize=9, zorder=4, linestyle="none")
        label = vlm.get("predicted_object") or rec["entity_id"]
        ax.annotate(f"{label}\n({vlm.get('decision')}, {rec['dist_to_true_B_m']:.2f}m)",
                    (px, pz), fontsize=6, color=color, xytext=(4, 4), textcoords="offset points")

    if result.get("recall"):
        rp = result["recall"]["position_xy"]
        ax.plot(*rp, marker="X", color="green", markersize=14, zorder=5,
                label=f"recalled B (err {result['recall']['position_error_vs_true_B_m']:.2f}m)")
    if result.get("leg2"):
        fp = result["leg2"]["final_pose"][:2]
        ax.plot(*fp, marker="s", color="darkgreen", markersize=10, zorder=5, label="leg2 final pose")
    if result.get("leg1"):
        fp1 = result["leg1"]["final_pose"][:2]
        ax.plot(*fp1, marker="s", color="navy", markersize=10, zorder=5, label="leg1 final pose")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plan = result.get("plan")
    if plan:
        map_line = (f"map: {plan['occupancy']['free_cells']}/{plan['occupancy']['cells']} cells free "
                   f"({plan['occupancy']['free']*100:.1f}%), ")
    elif grid is not None:
        free_frac = float((grid == 0).mean())
        map_line = (f"map: {int((grid == 0).sum())}/{grid.size} cells free ({free_frac*100:.1f}%) "
                   f"-- recall found no acceptable entity, leg2 never ran, ")
    else:
        map_line = "map: occupancy.npy not saved (older pre-fix run), "
    ax.set_title(f"{Path(result['scene']).stem}  --  {result['category_a']} -> {result['category_b']}\n"
                f"{map_line}"
                f"{result['leg1'].get('entities_mapped', '?')} entities mapped, "
                f"{result['leg1'].get('fragments_merged', '?')} fragments merged")
    handles, labels = ax.get_legend_handles_labels()
    for decision, (color, marker) in decision_style.items():
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle="none", markersize=8))
        labels.append(f"candidate: {decision}")
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
