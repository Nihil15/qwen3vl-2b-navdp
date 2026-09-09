#!/usr/bin/env python3
"""Find a STRAIGHT-SHOT A->B pair in a converted MP3D house: an A instance
and a B instance ~min..max m apart by geodesic whose geodesic path is nearly
a straight line (geo/euclid < ratio), plus a navmesh start point placed
`back` m behind A on the A->B line. Prints ready-to-paste --start-xy /
--min-ab / --max-ab for run_ab_position_recall_test.py.

    conda run -n habitat python scout_mp3d_pair.py \
        --scene-glb .../2azQ1b91cZZ_yup.ply --mp3d-house .../2azQ1b91cZZ.house \
        --cat-a chair --cat-b table
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from run_ab_recall_hm3d_test import (
    EYE_HEIGHT_M, build_sim, category_instances, geodesic, largest_island,
)
import habitat_sim


def euclid(a, b):
    return float(math.hypot(a[0] - b[0], a[2] - b[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-glb", required=True)
    ap.add_argument("--mp3d-house", required=True)
    ap.add_argument("--cat-a", default="chair")
    ap.add_argument("--cat-b", default="table")
    ap.add_argument("--min-ab", type=float, default=2.5)
    ap.add_argument("--max-ab", type=float, default=5.0)
    ap.add_argument("--ratio", type=float, default=1.15,
                    help="max geodesic/euclidean ratio -- 1.0 is a perfect straight line")
    ap.add_argument("--back", type=float, default=3.0, help="start distance behind A, m")
    args = ap.parse_args()

    sim = build_sim(args.scene_glb, None, mp3d_house=args.mp3d_house)
    pf = sim.pathfinder
    island = largest_island(pf)
    A = category_instances(sim, args.cat_a, island)
    B = category_instances(sim, args.cat_b, island)
    print(f"island {island}  area {pf.island_area(island):.1f} m^2  "
          f"|{args.cat_a}|={len(A)}  |{args.cat_b}|={len(B)}")

    cands = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            g = geodesic(pf, a, b)
            if g is None or not (args.min_ab <= g <= args.max_ab):
                continue
            e = euclid(a, b)
            if e < 0.3:
                continue
            ratio = g / e
            if ratio > args.ratio:
                continue
            # start point: back along the B->A direction, past A
            d = np.array([a[0] - b[0], 0.0, a[2] - b[2]])
            d = d / (np.linalg.norm(d) + 1e-9)
            s = np.array([a[0] + d[0] * args.back, a[1], a[2] + d[2] * args.back], dtype=np.float32)
            s = np.array(pf.snap_point(s, island))
            if not np.all(np.isfinite(s)):
                continue
            sg = geodesic(pf, s, a)
            se = euclid(s, a)
            if sg is None:
                continue
            s_ratio = sg / max(se, 1e-6)
            cands.append((ratio + s_ratio, g, ratio, s_ratio, i, j, a, b, s, sg))

    if not cands:
        print("no straight-shot pair found -- loosen --ratio / widen --min-ab..--max-ab")
        sim.close()
        return
    cands.sort(key=lambda c: c[0])
    print(f"\ntop straight-shot {args.cat_a}->{args.cat_b} pairs:\n")
    for score, g, ratio, sr, i, j, a, b, s, sg in cands[:6]:
        print(f"  A#{i} ({a[0]:+.2f},{a[2]:+.2f})  ->  B#{j} ({b[0]:+.2f},{b[2]:+.2f})   "
              f"geo={g:.2f}m  path/straight A-B={ratio:.2f}  start->A={sg:.2f}m ratio={sr:.2f}")
        print(f"      --start-xy {s[0]:.2f} {s[2]:.2f}  --min-ab {g - 0.25:.2f}  --max-ab {g + 0.25:.2f}\n")
    sim.close()


if __name__ == "__main__":
    main()
