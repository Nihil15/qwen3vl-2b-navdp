#!/usr/bin/env python3
"""Build the occupancy layer of a fact3r BEV from habitat's own navmesh.

`build_depth_semantic_bev.py` gives us two layers: WHERE entities are (the
semantic layer -- real, and what the recall query needs) and WHAT is walkable
(the occupancy layer). On a camera-only tour the second one comes out ~50%
unknown / ~0.1% free: at camera height nearly every depth return lands on
wall or furniture inside the [0.1, 1.5] m obstacle slice, so ray-carving can
never win and `project_semantic_goal.py` correctly refuses to hand A* a goal
("nowhere to send the robot").

In simulation the walkable set is known exactly -- it's the navmesh. This
writes an occupancy grid in the SAME frame/resolution/origin as the semantic
BEV, so the real fact3r semantic positions still drive the query and only the
free/occupied layer comes from ground truth. That isolates "can it recall a
position and plan A* to it" from the orthogonal question of how good a
camera-only occupancy map is.

Frame (verified against a walk whose last pose is a known object):
    grid_x = x_habitat        grid_y = -z_habitat
    x = origin_x + (col + 0.5) * res     y = origin_y + (row + 0.5) * res
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_ab_recall_hm3d_test import (
    build_sim,
    find_dataset_config,
    find_scene_glb,
    largest_island,
)

UNKNOWN = -1
FREE = 0
OCCUPIED = 97


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semantic-bev", required=True, help="<stem>_semantic_bev.npz from build_depth_semantic_bev.py")
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--floor-y", type=float, default=0.19491982, help="habitat floor height in this room")
    ap.add_argument("--out", required=True, help="output .npy occupancy grid")
    args = ap.parse_args()

    z = np.load(args.semantic_bev, allow_pickle=False)
    shape = z["occupancy"].shape
    origin_xy = z["origin_xy"]
    res = float(z["resolution"])
    print(f"grid {shape[1]}x{shape[0]} @ {res} m, origin {origin_xy}")

    hm3d_root = Path(args.hm3d_root)
    sim = build_sim(find_scene_glb(hm3d_root, args.scene), find_dataset_config(hm3d_root))
    pf = sim.pathfinder
    island = largest_island(pf)

    rows, cols = np.indices(shape)
    xs = origin_xy[0] + (cols + 0.5) * res          # grid x  == habitat x
    ys = origin_xy[1] + (rows + 0.5) * res          # grid y  == -habitat z
    grid = np.full(shape, UNKNOWN, dtype=np.int8)

    n_free = 0
    for r in range(shape[0]):
        for c in range(shape[1]):
            hx, hz = float(xs[r, c]), -float(ys[r, c])
            point = np.array([hx, args.floor_y, hz], dtype=np.float32)
            if pf.is_navigable(point):
                snapped = np.array(pf.snap_point(point, island))
                # snap_point returns NaN when the point isn't on the requested
                # island; keep those out so A* can't route through another floor.
                if np.all(np.isfinite(snapped)) and abs(float(snapped[1]) - args.floor_y) < 0.5:
                    grid[r, c] = FREE
                    n_free += 1
                    continue
            grid[r, c] = OCCUPIED

    np.save(args.out, grid)
    total = grid.size
    print(f"free {n_free / total:6.1%}   occupied {(total - n_free) / total:6.1%}   -> {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
