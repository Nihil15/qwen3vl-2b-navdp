#!/usr/bin/env python3
"""Scout an HM3D scene for the A/B recall test: list ground-floor semantic
categories and render a look-at frame for each candidate instance so a
human can eyeball which two objects to use as A and B.

    conda run -n habitat python scout_ab_scene.py --out /tmp/scout
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from run_ab_recall_hm3d_test import (
    EYE_HEIGHT_M, build_sim, category_instances, find_dataset_config,
    find_scene_glb, largest_island, render,
)

DEFAULT_HM3D_ROOT = "/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d"
STRUCTURAL = {
    "wall", "ceiling", "floor", "window", "door", "door frame", "beam", "pipe",
    "air duct", "smoke detector", "light switch", "air vent", "fire sprinkler",
    "window shutters", "curtain rod", "unknown", "misc", "objects", "stairs",
    "handrail", "railing", "step", "baseboard", "trim", "molding", "column",
}


def look_at_frame(sim, pf, island, target_xyz, cam_y, standoff=1.6):
    """Best of 8 standing points `standoff` m from target, facing it."""
    best = None
    for k in range(8):
        th = 2 * math.pi * k / 8
        cand = np.array([target_xyz[0] + standoff * math.cos(th), target_xyz[1],
                         target_xyz[2] + standoff * math.sin(th)], dtype=np.float32)
        snap = np.array(pf.snap_point(cand, island))
        if not np.all(np.isfinite(snap)):
            continue
        d = float(np.linalg.norm(snap[[0, 2]] - np.asarray(target_xyz)[[0, 2]]))
        if best is None or abs(d - standoff) < best[0]:
            best = (abs(d - standoff), snap)
    if best is None:
        return None, None
    snap = best[1]
    yaw = math.atan2(target_xyz[2] - snap[2], target_xyz[0] - snap[0])
    pose = (float(snap[0]), float(snap[2]), yaw)
    rgb, _ = render(sim, pose, cam_y)
    return rgb, pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT)
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--cam-height", type=float, default=0.9)
    ap.add_argument("--out", default="/tmp/scout")
    args = ap.parse_args()

    root = Path(args.hm3d_root)
    glb = find_scene_glb(root, args.scene)
    cfg = find_dataset_config(root)
    sim = build_sim(glb, cfg)
    pf = sim.pathfinder
    island = largest_island(pf)
    print(f"scene={args.scene} glb={glb}")
    print(f"largest navmesh island={island} area={pf.island_area(island):.1f} m^2")

    # floor height of that island
    samp = np.array([pf.get_random_navigable_point(island_index=island) for _ in range(200)])
    floor_y = float(np.median(samp[:, 1]))
    cam_y = floor_y + args.cam_height
    print(f"island floor y~{floor_y:.2f}  cam_y={cam_y:.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # inventory: categories with >=1 instance snapped onto this island
    by_cat = {}
    for obj in sim.semantic_scene.objects or []:
        if obj is None or obj.category is None:
            continue
        name = obj.category.name()
        if name in STRUCTURAL:
            continue
        c = obj.aabb.center()
        snap = np.array(pf.snap_point(c, island))
        if not np.all(np.isfinite(snap)):
            continue
        # only keep instances whose snap landed near this island's floor
        if abs(snap[1] - floor_y) > 1.0:
            continue
        by_cat.setdefault(name, []).append((c, snap))

    print("\nground-floor nameable categories (instances reachable on this island):")
    for name in sorted(by_cat, key=lambda n: -len(by_cat[n])):
        print(f"  {name:22s} x{len(by_cat[name])}")

    # render a look-at for up to 3 instances of each category
    manifest = []
    for name, insts in sorted(by_cat.items()):
        for idx, (center, _snap) in enumerate(insts[:3]):
            rgb, pose = look_at_frame(sim, pf, island, np.asarray(center), cam_y)
            if rgb is None:
                continue
            slug = name.replace(" ", "_").replace("/", "_")
            fn = out / f"{slug}_{idx}.png"
            cv2.imwrite(str(fn), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            manifest.append((name, idx, tuple(round(float(x), 2) for x in center), str(fn)))
            print(f"  wrote {fn.name:32s} center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})")

    (out / "manifest.txt").write_text(
        "\n".join(f"{n}\t{i}\tcenter_xyz={c}\t{p}" for n, i, c, p in manifest))
    print(f"\n{len(manifest)} frames -> {out}")
    sim.close()


if __name__ == "__main__":
    main()
