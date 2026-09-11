#!/usr/bin/env python3
"""Standalone diagnostic: is SAM2 (+ SigLIP2) in Fact3rLiveMemory actually working?

Loads the entity-memory models, runs SAM2 automatic-mask generation on one
image, and prints how many proposals it makes + their sizes, then a full
observe() -> query() cycle so you can see the whole path.

    SAM2_ROOT=/home/gpu/Desktop/pineapple/packages/sam2 \\
    python tools/check_fact3r_sam.py [IMAGE.png] [--points-per-side 12] [--query "a chair"]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_SAM2_ROOT = os.environ.get("SAM2_ROOT", "/home/gpu/Desktop/pineapple/packages/sam2")
if _SAM2_ROOT and _SAM2_ROOT not in sys.path:
    sys.path.insert(0, _SAM2_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?",
                    default="logs/ab_recall_hm3d_test/start_frame.png",
                    help="RGB image to run SAM2 on")
    ap.add_argument("--points-per-side", type=int, default=12)
    ap.add_argument("--query", default="a chair", help="text to try recall with")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    print(f"[check] SAM2_ROOT       = {_SAM2_ROOT}  (exists={os.path.isdir(_SAM2_ROOT)})")
    try:
        import sam2  # noqa: F401
        print(f"[check] import sam2     = OK ({sam2.__file__})")
    except Exception as e:
        print(f"[check] import sam2     = FAIL -> {e!r}")
        return 1
    import importlib.util
    if importlib.util.find_spec("iopath") is not None:
        print("[check] import iopath   = OK")
    else:
        print("[check] import iopath   = FAIL -> missing  (pip install iopath -- SAM2's hieradet needs it)")
        return 1

    from PIL import Image
    if not os.path.isfile(a.image):
        print(f"[check] image not found: {a.image}")
        return 1
    rgb = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.uint8)
    print(f"[check] image           = {a.image}  {rgb.shape}")

    from nav_pipeline.fact3r_live_memory import Fact3rLiveMemory
    mem = Fact3rLiveMemory(device=a.device, points_per_side=a.points_per_side)
    t0 = time.time()
    print(f"[check] loading SAM2 + SigLIP2 (points_per_side={a.points_per_side}) ...")
    mem._ensure_loaded()
    print(f"[check] models loaded   = OK  ({time.time() - t0:.1f}s)")

    # 1) raw SAM2 proposals
    t0 = time.time()
    anns = mem._propose(rgb)
    dt = time.time() - t0
    print(f"\n[SAM2] _propose() -> {len(anns)} masks in {dt:.2f}s")
    if not anns:
        print("[SAM2] *** ZERO masks -- SAM2 is loaded but not segmenting this image ***")
        return 2
    areas = sorted((int(m["area"]) for m in anns), reverse=True)
    print(f"[SAM2] mask areas px: max={areas[0]} med={areas[len(areas) // 2]} min={areas[-1]}")
    print(f"[SAM2] >= min_mask_region_area({mem.min_mask_region_area}): "
          f"{sum(1 for x in areas if x >= mem.min_mask_region_area)}")
    for i, m in enumerate(sorted(anns, key=lambda z: -z["area"])[:8]):
        x, y, w, h = (int(v) for v in m["bbox"])
        print(f"       mask {i}: area={int(m['area']):6d} bbox=({x},{y},{w}x{h}) "
              f"iou={m.get('predicted_iou', 0):.2f} stab={m.get('stability_score', 0):.2f}")

    # 2) full observe() cycle with synthetic depth (SAM2 crop -> SigLIP embed -> entity)
    depth = np.full(rgb.shape[:2], 2.5, np.float32)   # flat 2.5 m so every mask banks
    intr = (rgb.shape[1] * 0.9, rgb.shape[1] * 0.9, rgb.shape[1] / 2, rgb.shape[0] / 2)
    t0 = time.time()
    n = mem.observe(rgb, depth, (0.0, 0.0, 0.0), intr, step=0)
    print(f"\n[observe] absorbed {n} proposals -> {len(mem.entities)} entities  "
          f"({time.time() - t0:.2f}s)")

    # 3) SigLIP text recall
    hits = mem.ranked(a.query, min_observations=1, top=8)
    print(f"\n[query] ranked('{a.query}') top-8 (raw SigLIP2 text->crop cosine):")
    for score, ent in hits:
        x, y = ent.position
        print(f"       {ent.entity_id}  score={score:+.3f}  obs={ent.observations}  "
              f"xy=({x:+.2f},{y:+.2f})  spread={ent.position_spread:.2f}m")
    print("\n[note] SigLIP2 raw cosines here sit ~0.05-0.15 with little separation -- "
          "that is expected; recall leans on the Qwen verify stage (query_verified).")
    print("[note] SAM2 gives UNLABELLED masks. fact3r entities carry an appearance "
          "embedding + position, NOT a text label -- the label on a live bounding box "
          "comes from Grounding-DINO (Detection.label) or the Qwen supervisor's phrase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
