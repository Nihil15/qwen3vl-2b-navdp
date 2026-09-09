#!/usr/bin/env python3
"""DaViNCi (outdoor CARLA VLN) discrete-graph eval with THIS repo's Qwen model.

Rebuilt 2026-09-05 (the 2026-09-03 version lived in a session scratchpad and
is gone). Two honest scope limits, stated up front because they decide what
this number means:

  * Only the DISCRETE split is runnable here. It is a TOPOLOGICAL GRAPH task:
    at each node the agent picks an outgoing edge from real panorama crops.
    There is no continuous control, no depth, no obstacle geometry -- so
    **NavDP cannot participate at all**. This measures the Qwen half of the
    integrated system (instruction following + visual grounding), not the
    integrated system end to end.
  * The CONTINUOUS split ships trajectories only (no frames) and needs CARLA
    to render; `carla` is not installed on this machine, so it cannot be run.

Scoring is standard VLN: NE / SR / OSR / SPL against the true goal node,
with success_distance = 1.5x the graph's mean edge length (the same
convention the earlier run used, ~18.5m on town03).

  conda run -n habitat python davinci_qwen_eval.py --town town03 --episodes 3
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

DAVINCI = Path("/home/gpu/Desktop/Nihil/MASt3R-SLAM/fact3r-map/datasets/davinci/Discrete")


def load_graph(town_dir: Path):
    """nodes.txt: id,?,x,y   links.txt: from_node,heading_deg,to_node"""
    nodes = {}
    for line in (town_dir / "graph" / "nodes.txt").read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        nodes[int(parts[0])] = (float(parts[2]), float(parts[3]))
    links = {}
    for line in (town_dir / "graph" / "links.txt").read_text().splitlines():
        if not line.strip():
            continue
        a, heading, b = line.split(",")
        links.setdefault(int(a), []).append((float(heading), int(b)))
        # graph is undirected in practice -- add the reverse edge with a
        # flipped heading so a route can be walked in either direction
        links.setdefault(int(b), []).append((float(heading) + 180.0, int(a)))
    return nodes, links


def mean_edge_length(nodes, links) -> float:
    ds = []
    for a, outs in links.items():
        for _, b in outs:
            if a in nodes and b in nodes:
                ds.append(math.dist(nodes[a], nodes[b]))
    return float(np.mean(ds)) if ds else 1.0


def crop_by_heading(pano: Image.Image, heading_deg: float, fov_deg: float = 90.0) -> Image.Image:
    """Horizontal window of an equirectangular panorama centred on `heading`
    -- same formulation as the dataset's own instance_generate.py."""
    W, H = pano.size
    cx = (heading_deg % 360.0) / 360.0 * W
    half = (fov_deg / 360.0) * W / 2.0
    x0, x1 = int(cx - half), int(cx + half)
    if x0 < 0:                       # wrap around the seam
        left = pano.crop((W + x0, 0, W, H))
        right = pano.crop((0, 0, x1, H))
        out = Image.new("RGB", (left.width + right.width, H))
        out.paste(left, (0, 0)); out.paste(right, (left.width, 0))
        return out
    if x1 > W:
        left = pano.crop((x0, 0, W, H))
        right = pano.crop((0, 0, x1 - W, H))
        out = Image.new("RGB", (left.width + right.width, H))
        out.paste(left, (0, 0)); out.paste(right, (left.width, 0))
        return out
    return pano.crop((x0, 0, x1, H))


class QwenChooser:
    def __init__(self, model_id: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float16, device_map=device)
        self.model.eval()

    def choose(self, instruction: str, crops, max_new_tokens: int = 24) -> str:
        prompt = (
            "You are driving a vehicle following a navigation instruction in a city.\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"You are at an intersection. {len(crops)} possible directions are shown, "
            f"image 0 through image {len(crops) - 1}, each looking down one road you could take.\n\n"
            "Reply with ONLY the number of the image whose road continues the instruction, "
            "or ONLY the word STOP if you have already arrived at the described destination."
        )
        content = [{"type": "image", "image": c} for c in crops]
        content.append({"type": "text", "text": prompt})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=crops, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def run_episode(ep, nodes, links, panos_dir, chooser, max_steps, success_dist):
    route = [int(p) for p in ep["route_panoids"]]
    start, goal = route[0], route[-1]
    optimal = sum(math.dist(nodes[route[i]], nodes[route[i + 1]])
                  for i in range(len(route) - 1) if route[i] in nodes and route[i + 1] in nodes)

    cur, visited, log = start, [start], []
    path_len, stopped, dead_end = 0.0, False, False
    oracle_err = math.dist(nodes[cur], nodes[goal]) if cur in nodes and goal in nodes else float("inf")

    for step in range(max_steps):
        outs = links.get(cur, [])
        if not outs:
            dead_end = True
            break
        pano_path = panos_dir / f"{cur}.jpeg"
        if not pano_path.is_file():
            dead_end = True
            break
        pano = Image.open(pano_path).convert("RGB")
        crops = [crop_by_heading(pano, h).resize((448, 336)) for h, _ in outs]
        raw = chooser.choose(ep["navigation_text"], crops)

        choice = None
        if "STOP" in raw.upper():
            choice = "STOP"
        else:
            digits = "".join(ch for ch in raw if ch.isdigit())
            if digits and int(digits) < len(outs):
                choice = int(digits)
        log.append({"step": step, "node": cur, "n_options": len(outs), "raw": raw, "choice": choice})

        if choice == "STOP" or choice is None:
            stopped = True
            break
        nxt = outs[choice][1]
        if cur in nodes and nxt in nodes:
            path_len += math.dist(nodes[cur], nodes[nxt])
        cur = nxt
        visited.append(cur)
        if cur in nodes and goal in nodes:
            oracle_err = min(oracle_err, math.dist(nodes[cur], nodes[goal]))

    ne = math.dist(nodes[cur], nodes[goal]) if cur in nodes and goal in nodes else float("inf")
    success = ne < success_dist
    spl = (optimal / max(optimal, path_len)) if (success and path_len > 0) else 0.0
    return {
        "route_id": ep["route_id"], "route_nodes": len(route),
        "optimal_length_m": optimal, "path_length_m": path_len, "steps": len(log),
        "stopped": stopped, "dead_end": dead_end,
        "budget_exhausted": len(log) >= max_steps,
        "navigation_error_m": ne, "oracle_error_m": oracle_err,
        "success": bool(success), "oracle_success": bool(oracle_err < success_dist),
        "spl": spl, "final_node": cur, "goal_node": goal, "log": log,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", default="town03")
    ap.add_argument("--split", default="test")
    ap.add_argument("--episodes", type=int, default=3, help="shortest-N episodes (tractability)")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct",
                    help="defaults to the SAME model the nav pipeline runs")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="logs/davinci_qwen3vl2b")
    args = ap.parse_args()

    town_dir = DAVINCI / args.town
    nodes, links = load_graph(town_dir)
    success_dist = 1.5 * mean_edge_length(nodes, links)
    eps = [json.loads(l) for l in (town_dir / "data" / f"{args.split}.json").read_text().splitlines() if l.strip()]
    eps.sort(key=lambda e: len(e["route_panoids"]))
    eps = eps[:args.episodes]

    print(f"{args.town}/{args.split}: {len(nodes)} nodes, {len(links)} with outgoing links")
    print(f"success_distance = {success_dist:.2f} m (1.5x mean edge length)")
    print(f"model: {args.model_id}\n")

    chooser = QwenChooser(args.model_id, args.device)
    results = []
    for ep in eps:
        r = run_episode(ep, nodes, links, town_dir / "panos", chooser, args.max_steps, success_dist)
        results.append(r)
        print(f"  route {r['route_id']}: steps={r['steps']} NE={r['navigation_error_m']:.1f}m "
              f"success={r['success']} spl={r['spl']:.3f}")

    agg = {
        "episodes": len(results),
        "success_distance_m": success_dist,
        "mean_navigation_error_m": float(np.mean([r["navigation_error_m"] for r in results])),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "oracle_success_rate": float(np.mean([r["oracle_success"] for r in results])),
        "mean_spl": float(np.mean([r["spl"] for r in results])),
        "mean_steps": float(np.mean([r["steps"] for r in results])),
    }
    print("\n=== DaViNCi metrics ===")
    for k, v in agg.items():
        print(f"  {k:28s}: {v}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.town}_result.json").write_text(
        json.dumps({"aggregate": agg, "model": args.model_id, "episodes": results}, indent=2))
    print(f"\nwrote {out_dir / f'{args.town}_result.json'}")


if __name__ == "__main__":
    main()
