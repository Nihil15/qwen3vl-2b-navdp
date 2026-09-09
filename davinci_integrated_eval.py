#!/usr/bin/env python3
"""DaViNCi outdoor eval with the FULL integrated pipeline (Qwen + NavDP +
obstacle guard), not just the VLM.

Why this exists: davinci_qwen_eval.py only measures the Qwen half, because
the Discrete split is a topological graph (pick an edge from panorama crops)
with no continuous control and no depth -- NavDP has nothing to do there.
The Continuous split would exercise the whole system but ships trajectories
only, no frames, and needs CARLA (not installed here).

So this drives a CONTINUOUS pose over the discrete graph's real panoramas:

    pose (x, y, yaw) in the graph's own metric frame
      -> RGB   = horizontal FOV window of the NEAREST node's panorama at yaw
      -> depth = Depth-Anything V2 monocular metric depth on that RGB
      -> DinoNavDPPipeline.step(rgb, depth, instruction=...) -- real Qwen
         pixel-goal grounding, real NavDP (v, w), real obstacle guard
      -> integrate (v, w) -> new pose, repeat

Every component that runs in the HM3D/VLN-CE tests runs here, on real
outdoor CARLA imagery. Two honest caveats about what the frames are:

  * The panorama is sampled at the nearest NODE, so translation between
    nodes does not produce true parallax -- the view updates in steps as the
    nearest node changes, while rotation is continuous. Depth is therefore
    also node-quantised.
  * Depth is monocular-estimated, not sensor depth, so the obstacle guard is
    reacting to inferred geometry (this is also true of the real rover, but
    not of the habitat tests where depth is ground truth).

Scoring is the same VLN convention as davinci_qwen_eval.py: NE / SR / OSR /
SPL against the goal node, success radius 1.5x mean edge length.

  conda run -n habitat python davinci_integrated_eval.py --episodes 3
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from davinci_qwen_eval import DAVINCI, crop_by_heading, load_graph, mean_edge_length
from nav_pipeline.goal_utils import intrinsics_from_fov
from nav_pipeline.obstacle_guard import GuardConfig
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

W, H, FOV_DEG = 640, 480, 90.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def nearest_node(nodes, xy):
    best, best_d = None, float("inf")
    for nid, p in nodes.items():
        d = math.dist(p, xy)
        if d < best_d:
            best, best_d = nid, d
    return best, best_d


def render(nodes, panos_dir, pose, pano_cache):
    """RGB the rover would see at `pose`, from the nearest node's panorama."""
    nid, _ = nearest_node(nodes, (pose[0], pose[1]))
    if nid not in pano_cache:
        p = panos_dir / f"{nid}.jpeg"
        if not p.is_file():
            return None, nid
        pano_cache[nid] = Image.open(p).convert("RGB")
        if len(pano_cache) > 24:      # bounded cache; panos are 3840x1440
            pano_cache.pop(next(iter(pano_cache)))
    # graph headings are degrees clockwise-from-north in the dataset's frame;
    # pose yaw is radians CCW-from-+x, so convert before sampling the pano.
    heading_deg = (90.0 - math.degrees(pose[2])) % 360.0
    crop = crop_by_heading(pano_cache[nid], heading_deg, FOV_DEG).resize((W, H))
    return np.asarray(crop, dtype=np.uint8), nid


def build_pipeline(args) -> DinoNavDPPipeline:
    cfg = PipelineConfig(
        device=args.device, horizontal_fov_deg=FOV_DEG, policy_type=args.policy,
        max_linear=args.max_linear, max_angular=args.max_angular,
        stop_distance=args.stop_distance, avoid_enabled=not args.no_avoid,
        use_dino=False, use_sam=False, use_clip=False, use_qwen_search=False,
        use_qwen_instruction=True, qwen_model_id=args.qwen_model_id,
        # PipelineConfig defaults this True, but bitsandbytes isn't installed
        # in the habitat env -- the other harnesses all pass False too.
        qwen_load_in_4bit=args.qwen_4bit,
        qwen_instruction_period_s=args.instruction_period_s,
        qwen_goal_consistency_m=args.goal_consistency_m,
        # Outdoors the road ahead reads 10-20m and Qwen readily grounds
        # something 3-5m away, which the indoor-scale arrival check then
        # calls 'arrived' within 3 ticks (measured: every episode stopped
        # after 0.18m). Trust metric depth only up close; past that, drive
        # the BEARING to a far look-ahead, which also gives NavDP a goal
        # far enough away to actually accelerate toward.
        qwen_depth_trust_horizon_m=args.depth_trust_horizon,
        qwen_far_lookahead_m=args.far_lookahead,
        qwen_stop_confirm_ticks=args.stop_confirm_ticks,
        use_appearance_reid=False, use_scene_tagger=False, use_belief_goal=True,
        search_angular=args.search_angular,
        # outdoor: the guard's camera sits at car height and the useful
        # obstacle band is much deeper than an indoor rover's
        guard=GuardConfig(cam_height=args.cam_height, max_range=args.guard_max_range,
                          slow_dist=args.guard_slow_dist),
    )
    return DinoNavDPPipeline(cfg, use_depth_estimator=True)   # Depth-Anything V2


def run_episode(ep, nodes, links, panos_dir, pipe, args, success_dist):
    route = [int(p) for p in ep["route_panoids"]]
    if args.route_prefix and args.route_prefix < len(route):
        route = route[:args.route_prefix]   # short-distance variant
    start, goal = route[0], route[-1]
    if start not in nodes or goal not in nodes:
        return None
    optimal = sum(math.dist(nodes[route[i]], nodes[route[i + 1]])
                  for i in range(len(route) - 1) if route[i] in nodes and route[i + 1] in nodes)

    # start facing the first real edge of the ground-truth route
    nxt = route[1] if len(route) > 1 and route[1] in nodes else goal
    yaw0 = math.atan2(nodes[nxt][1] - nodes[start][1], nodes[nxt][0] - nodes[start][0])
    pose = np.array([nodes[start][0], nodes[start][1], yaw0], dtype=np.float64)

    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pano_cache, log, path_xy = {}, [], [(pose[0], pose[1])]
    oracle_err = math.dist((pose[0], pose[1]), nodes[goal])
    pipe.reset()
    outcome = "max_steps"

    for i in range(args.max_steps):
        tick = time.time()
        rgb, nid = render(nodes, panos_dir, pose, pano_cache)
        if rgb is None:
            outcome = "no_panorama"
            break
        res = pipe.step(rgb, target_text="", pose=tuple(pose),
                        instruction=ep["navigation_text"], intrinsics=intr)
        d = math.dist((pose[0], pose[1]), nodes[goal])
        oracle_err = min(oracle_err, d)
        log.append({"step": i, "node": nid, "pose": [float(x) for x in pose],
                    "state": res.state, "v": float(res.linear), "w": float(res.angular),
                    "dist_to_goal": d})
        if i % 20 == 0:
            print(f"    [step {i:3d}] node={nid} state={res.state:6s} dist_to_goal={d:7.1f}m")
        if res.state == "STOP":
            outcome = "stopped"
            break
        pose[0] += res.linear * math.cos(pose[2]) * args.dt
        pose[1] += res.linear * math.sin(pose[2]) * args.dt
        pose[2] = wrap(pose[2] + res.angular * args.dt)
        path_xy.append((pose[0], pose[1]))
        if args.realtime_pacing:
            rem = args.dt - (time.time() - tick)
            if rem > 0:
                time.sleep(rem)

    arr = np.asarray(path_xy)
    travelled = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum()) if len(arr) > 1 else 0.0
    ne = math.dist((pose[0], pose[1]), nodes[goal])
    success = ne < success_dist
    spl = (optimal / max(optimal, travelled)) if (success and travelled > 0) else 0.0
    return {
        "route_id": ep["route_id"], "route_nodes": len(route), "outcome": outcome,
        "optimal_length_m": optimal, "path_length_m": travelled, "steps": len(log),
        "navigation_error_m": ne, "oracle_error_m": oracle_err,
        "success": bool(success), "oracle_success": bool(oracle_err < success_dist),
        "spl": spl, "goal_node": goal,
        "state_counts": {s: sum(1 for r in log if r["state"] == s) for s in
                         {r["state"] for r in log}},
        "log": log,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", default="town03")
    ap.add_argument("--split", default="test")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true", help="needs bitsandbytes (absent in the habitat env)")
    ap.add_argument("--instruction-period-s", type=float, default=0.0)
    ap.add_argument("--goal-consistency-m", type=float, default=8.0,
                    help="outdoor scale: legitimate re-sightings of a described place are much "
                         "farther apart than the 1.5m indoor default")
    ap.add_argument("--max-linear", type=float, default=2.0, help="m/s -- outdoor driving")
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=2.5)
    ap.add_argument("--cam-height", type=float, default=1.5)
    ap.add_argument("--guard-max-range", type=float, default=20.0)
    ap.add_argument("--guard-slow-dist", type=float, default=12.0)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--no-realtime-pacing", dest="realtime_pacing", action="store_false")
    ap.add_argument("--route-prefix", type=int, default=0,
                    help="use only the first N nodes of each route -- a genuinely SHORT A->B "
                         "drive (~30-60m) instead of the full ~217m multi-clause route")
    ap.add_argument("--depth-trust-horizon", type=float, default=8.0)
    ap.add_argument("--far-lookahead", type=float, default=15.0)
    ap.add_argument("--stop-confirm-ticks", type=int, default=8)
    ap.add_argument("--out", default="logs/davinci_integrated")
    args = ap.parse_args()

    town_dir = DAVINCI / args.town
    nodes, links = load_graph(town_dir)
    success_dist = 1.5 * mean_edge_length(nodes, links)
    eps = [json.loads(l) for l in (town_dir / "data" / f"{args.split}.json").read_text().splitlines() if l.strip()]
    eps.sort(key=lambda e: len(e["route_panoids"]))
    eps = eps[:args.episodes]

    print(f"{args.town}/{args.split}: {len(nodes)} nodes | success_distance={success_dist:.2f} m")
    print("INTEGRATED: Qwen pixel-goal + NavDP + obstacle guard + Depth-Anything V2 depth\n")
    pipe = build_pipeline(args)

    results = []
    for ep in eps:
        print(f"  route {ep['route_id']} ({len(ep['route_panoids'])} nodes)")
        r = run_episode(ep, nodes, links, town_dir / "panos", pipe, args, success_dist)
        if r is None:
            print("    skipped (start/goal node missing from graph)")
            continue
        results.append(r)
        print(f"    -> outcome={r['outcome']} steps={r['steps']} NE={r['navigation_error_m']:.1f}m "
              f"oracle={r['oracle_error_m']:.1f}m success={r['success']} states={r['state_counts']}")

    if not results:
        print("no runnable episodes")
        return
    agg = {
        "episodes": len(results), "success_distance_m": success_dist,
        "mean_navigation_error_m": float(np.mean([r["navigation_error_m"] for r in results])),
        "mean_oracle_error_m": float(np.mean([r["oracle_error_m"] for r in results])),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "oracle_success_rate": float(np.mean([r["oracle_success"] for r in results])),
        "mean_spl": float(np.mean([r["spl"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_path_length_m": float(np.mean([r["path_length_m"] for r in results])),
    }
    print("\n=== DaViNCi INTEGRATED metrics ===")
    for k, v in agg.items():
        print(f"  {k:28s}: {v}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.town}_result.json").write_text(json.dumps(
        {"aggregate": agg, "model": args.qwen_model_id,
         "mechanism": "Qwen pixel-goal + NavDP + obstacle guard, panorama-rendered RGB + "
                      "Depth-Anything V2 depth, continuous pose over the discrete graph",
         "episodes": results}, indent=2))
    print(f"\nwrote {out_dir / f'{args.town}_result.json'}")


if __name__ == "__main__":
    main()
