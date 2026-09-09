#!/usr/bin/env python3
"""Real Qwen + real NavDP, two-instruction "go to A, then recall B" test --
on a REAL scanned house (HM3D), not a procedural render.

Same mechanism as run_ab_recall_live_test.py (see its docstring for the full
design) with one change: frames come from habitat-sim rendering an actual
HM3D scene (1059 annotated real objects) instead of a synthetic pinhole
renderer, so Qwen is grounding on real imagery. Still no discrete grid
actions -- every command is DinoNavDPPipeline's own continuous (v, w) NavDP
output.

  Leg 1  instruction="go to the <target-a-category>" -- live Qwen pixel-goal
         grounding + NavDP, state "TRACK", exactly as the real rover uses it.
         A parallel Qwen probe watches for a second real object; the first
         time it lands on something that isn't the current target, that
         object's 3D position is computed (pixel + depth + intrinsics, the
         same geometry pipeline.py itself uses) and stored as odom-frame xy.
  Leg 2  instruction="go to B" -- B is NOT re-detected; its remembered
         odom-frame position drives pipe.step(external_goal=...) every tick
         (state "GOTO") via nav_pipeline.fact3r_goal_source.Fact3rGoalSource.

Target categories, the scene, and the start pose are chosen automatically:
the two HM3D semantic categories (default "chair" / "table") with a real,
navmesh-connected instance pair at a walkable geodesic distance, using the
scene's own pathfinder -- no hand-placed objects. Ground truth (both
instances' standable positions) is used only for the accuracy report, never
given to the pipeline or the recall step.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError as exc:  # pragma: no cover
    raise SystemExit("habitat_sim is required -- run this under the `habitat` conda env") from exc

from nav_pipeline.fact3r_goal_source import Fact3rGoalSource, MapToOdom
from nav_pipeline.goal_utils import intrinsics_from_fov, pixel_depth_to_point
from nav_pipeline.obstacle_guard import GuardConfig, depth_to_obstacle_points, forward_guard
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

W, H, FOV_DEG = 640, 480, 90.0
EYE_HEIGHT_M = 1.2

PROBE_PROMPT_TMPL = (
    "Look at this image from a mobile robot's camera. Is there any distinct "
    "object visible OTHER than the {target} the robot is driving toward? If "
    "yes, point to it. If no other object is visible, point to (0, 0). "
    "Reply with only one pixel coordinate as (x, y)."
)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def local_to_odom(pose, fwd: float, left: float):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return x + (fwd * c - left * s), y + (fwd * s + left * c)


# --------------------------------------------------------------------------- #
# scene setup: real HM3D house, real object instances, real navmesh
# --------------------------------------------------------------------------- #
def find_dataset_config(hm3d_root: Path) -> str | None:
    for pattern in ("*annotated*basis.scene_dataset_config.json", "*basis.scene_dataset_config.json"):
        matches = sorted(hm3d_root.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def find_scene_glb(hm3d_root: Path, scene: str) -> str:
    matches = sorted(hm3d_root.rglob(f"*-{scene}/{scene}.basis.glb"))
    if not matches:
        raise FileNotFoundError(f"no {scene}.basis.glb under {hm3d_root}")
    return str(matches[0])


def load_mp3d_semantics(sim, house_path: str) -> None:
    """Load real MP3D .house semantic annotations (category_instances() below
    then finds real object categories/positions) into a scene whose mesh was
    rotated -90deg about X at export time -- raw MP3D v1 scans are natively
    Z-up, habitat/R2R-CE are Y-up (see fact3r-navdp-bridge memory's "Real
    MP3D v1 raw scan" notes for how the matching mesh rotation was derived
    and verified). This is a fixed axis-convention constant, not scene-
    specific, so the same rotation is reused verbatim for every raw house."""
    import magnum as mn
    q = mn.Quaternion.rotation(mn.Deg(-90.0), mn.Vector3.x_axis())
    rotvec = mn.Vector4(q.vector.x, q.vector.y, q.vector.z, q.scalar)
    ok = habitat_sim.scene.SemanticScene.load_mp3d_house(house_path, sim.semantic_scene, rotvec)
    print(f"[note] loaded MP3D semantics from {house_path}: ok={ok}, "
          f"{len(sim.semantic_scene.objects)} objects")


def build_sim(scene_glb: str, dataset_config: str | None, pitch_deg: float = 0.0,
             mp3d_house: str | None = None):
    # Sensor offset is ZERO on purpose: render() already places the AGENT at
    # cam_y = floor + camera height, and habitat adds the sensor's offset on
    # top of the agent's position. Carrying EYE_HEIGHT_M here as well put the
    # camera at 2 * 1.2 = 2.4 m above the floor -- ceiling height -- where a
    # level 90-degree camera sees no floor nearer than 3.3 m and reads upper
    # wall and ceiling as close-range obstacles, so the guard hard-stopped
    # almost everywhere. Height belongs in exactly one place: cam_y.
    orientation = np.array([math.radians(-abs(pitch_deg)), 0.0, 0.0], dtype=np.float32)
    specs = []
    for uuid, stype in (("rgb", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [H, W]
        spec.position = np.array([0.0, 0.0, 0.0])
        spec.orientation = orientation
        spec.hfov = FOV_DEG
        specs.append(spec)
    rgb_spec, depth_spec = specs

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:
        navmesh = str(Path(scene_glb).with_suffix(".navmesh"))
        if Path(navmesh).is_file():
            sim.pathfinder.load_nav_mesh(navmesh)
    if not sim.pathfinder.is_loaded:
        # Raw scene dumps (e.g. a MP3D v1 research-release mesh, no shipped
        # .navmesh) have nothing to load -- bake one from the mesh itself.
        settings = habitat_sim.nav.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = EYE_HEIGHT_M
        ok = sim.recompute_navmesh(sim.pathfinder, settings)
        print(f"[note] no shipped navmesh for {scene_glb} -- baked one at runtime "
              f"(recompute_navmesh ok={ok})")
    if mp3d_house:
        load_mp3d_semantics(sim, mp3d_house)
    return sim


def largest_island(pf) -> int:
    return max(range(pf.num_islands), key=pf.island_area)


def category_instances(sim, category: str, island: int):
    """Real semantic-annotation instances of `category`, snapped to the
    navmesh (habitat (x, y, z), y = navmesh floor height)."""
    out = []
    for obj in sim.semantic_scene.objects or []:
        if obj is None or obj.category is None or obj.category.name() != category:
            continue
        snapped = np.array(sim.pathfinder.snap_point(obj.aabb.center(), island))
        if np.all(np.isfinite(snapped)):
            out.append(snapped)
    return out


def geodesic(pf, a, b) -> float | None:
    path = habitat_sim.ShortestPath()
    path.requested_start = a
    path.requested_end = b
    if not pf.find_path(path):
        return None
    return float(path.geodesic_distance)


def best_approach_clearance(sim, pf, point, standoff: float = 1.0, cam_height: float = EYE_HEIGHT_M,
                            cfg: GuardConfig | None = None, n_angles: int = 8) -> float:
    """Max obstacle-guard clearance found facing `point` from any of
    `n_angles` standing points `standoff` away on the navmesh -- "is there at
    least one decent way to approach this instance", not an average or worst
    case (only one real approach angle needs to be open). Cheap relative to
    a full sim step: n_angles renders, no NavDP/Qwen inference."""
    best = -1.0
    for k in range(n_angles):
        theta = 2.0 * math.pi * k / n_angles
        cand = np.array([point[0] + standoff * math.cos(theta), point[1],
                         point[2] + standoff * math.sin(theta)], dtype=np.float32)
        snapped = np.array(pf.snap_point(cand, pf.get_island(point)))
        if not np.all(np.isfinite(snapped)):
            continue
        yaw = math.atan2(point[2] - snapped[2], point[0] - snapped[0])   # face back toward `point`
        pose = (float(snapped[0]), float(snapped[2]), yaw)
        clr = clearance_m(sim, pose, cam_height, cfg)
        if clr > best:
            best = clr
    return best


def pick_pair(sim, island, cat_a: str, cat_b: str, min_d: float, max_d: float,
             min_a_clearance: float | None = None, cfg: GuardConfig | None = None,
             cam_height: float = EYE_HEIGHT_M):
    """min_a_clearance (optional): reject candidate A instances with no
    decent approach angle (see best_approach_clearance) before picking by
    geodesic distance, instead of picking purely by distance and letting a
    wedged-in instance (e.g. a chair jammed against a wall/other furniture)
    become leg 1's target regardless of how hard it is to actually reach.
    Measured need for this (2026-09-05, MP3D scene): repeated real runs
    where leg 1 tracked the CORRECT picked instance the whole time (0
    instance_lock_mismatches) but still failed to close the last 1-3m --
    StuckRecovery fired repeatedly (up to 8 times in one 500-step run) with
    near-zero net progress, consistent with the target itself sitting
    somewhere with no clear approach, not a navigation/identity bug.
    Candidates are tried in ascending-distance order and the first to clear
    the bar wins; if NONE clear it, falls back to the closest candidate
    anyway (same behavior as before this parameter existed) rather than
    raising -- a caller can inspect the returned clearance-check outcome via
    logging in `main`, this function still always returns a pair to drive."""
    pf = sim.pathfinder
    a_insts, b_insts = category_instances(sim, cat_a, island), category_instances(sim, cat_b, island)
    candidates = []
    for a in a_insts:
        for b in b_insts:
            d = geodesic(pf, a, b)
            if d is not None and min_d <= d <= max_d:
                candidates.append((d, a, b))
    if not candidates:
        raise SystemExit(f"no navmesh-connected '{cat_a}'<->'{cat_b}' pair with {min_d}-{max_d} m geodesic "
                         f"distance in this scene (found {len(a_insts)} {cat_a}, {len(b_insts)} {cat_b})")
    candidates.sort(key=lambda c: c[0])
    if min_a_clearance is None:
        best = candidates[0]
    else:
        best = None
        for d, a, b in candidates:
            clr = best_approach_clearance(sim, pf, a, cam_height=cam_height, cfg=cfg)
            if clr >= min_a_clearance:
                best = (d, a, b)
                print(f"  [pick_pair] picked A instance with best-approach clearance {clr:.2f}m "
                     f"(>= {min_a_clearance}m bar), {d:.2f}m geodesic to B")
                break
        if best is None:
            best = candidates[0]
            print(f"  [pick_pair] no candidate A instance cleared the {min_a_clearance}m approach-"
                 f"clearance bar -- falling back to the closest-geodesic candidate anyway")
    return best[1], best[2], best[0]


def clearance_m(sim, pose, cam_y: float, cfg: GuardConfig | None = None) -> float:
    """min_forward the obstacle guard would read from `pose` -- real HM3D
    rooms are cluttered enough that a plain navmesh point can still be
    pressed right up against furniture (camera a few cm from a chair leg,
    say), which the guard would (correctly) refuse to drive from.

    `cfg` must be the SAME GuardConfig the pipeline will run with: its
    cam_height sets where the guard thinks the camera is, so screening starts
    with one value and driving with another screens for the wrong thing."""
    cfg = cfg or GuardConfig()
    _, depth = render(sim, pose, cam_y)
    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pts = depth_to_obstacle_points(depth, *intr, cfg)
    mf, _ = forward_guard(pts, cfg)
    return float(mf)


def pick_start(sim, pf, island, a_point, b_point, desired_dist: float, min_clearance: float = 1.0,
               cam_height: float = EYE_HEIGHT_M, cfg: GuardConfig | None = None,
               min_geodesic: float = 1.0, other_a_instances=None):
    """A navmesh point ~desired_dist past `a_point`, away from `b_point`,
    with real obstacle clearance -- gives leg 1 a real short walk with a
    start-to-A geodesic path that doesn't hard-fault the guard on frame one.

    other_a_instances, if given: every OTHER instance of A's category
    (`pick_pair` already picked `a_point` specifically for its geodesic
    distance to B -- among possibly several instances of the same category,
    e.g. 4 different "bed"s in one house). "Go to the bed" grounds on
    whatever bed-like region is most salient from the start pose, with no
    way to specify which instance; if the start point sits closer to a
    DIFFERENT bed than to `a_point`, leg 1 reliably walks to that other one
    instead (measured: reached a bed 3.85-5.36m from the one actually paired
    with B, while landing within 1-2m of some OTHER bed -- "success" by
    GOAT's own nearest-instance criterion, but not the location anything
    about B is actually near). Candidates where some other instance's
    geodesic distance from the start is less than `a_point`'s are excluded
    outright so this can't happen by construction, not just by scoring luck."""
    dx, dz = a_point[0] - b_point[0], a_point[2] - b_point[2]
    n = math.hypot(dx, dz) or 1.0
    ux, uz = dx / n, dz / n

    # 60 random candidates was enough headroom when every one only had to
    # clear a distance + clearance bar. `other_a_instances` (below) can
    # reject the large majority of them outright (measured: 46/60, only 14
    # left) -- with that few survivors, "best clearance among what's left"
    # readily lands on something bad (measured: 0.52m against a 1.5m bar,
    # triggering an unrelated false-arrival failure once the robot started
    # there). More candidates give the OTHER constraint room to still find
    # a clear one instead of settling for whatever survives.
    n_random = 300 if other_a_instances else 60
    candidates = []
    direct = np.array([a_point[0] + ux * desired_dist, a_point[1], a_point[2] + uz * desired_dist])
    candidates.append(np.array(pf.snap_point(direct, island)))
    for _ in range(n_random):
        candidates.append(np.array(pf.get_random_navigable_point(island_index=island)))

    def a_point_is_closest(p, d_to_a):
        if not other_a_instances:
            return True
        for other in other_a_instances:
            d_other = geodesic(pf, p, other)
            if d_other is not None and d_other < d_to_a:
                return False
        return True

    # Escalating search, instead of one pass then settling for the least-bad
    # candidate. Measured why (2026-09-05): with other_a_instances active the
    # filter rejects the large majority of candidates (46/60 in the original
    # note), so on a real scene NOTHING in a single pass clears min_clearance
    # and this silently returned a 0.52m-clearance start. Leg 1 then spent
    # 78/220 ticks in AVOID fighting its way out and ended 3.08m away, versus
    # 3/220 AVOID and clean tracking from an open start on the same episode
    # -- the start pose, not grounding, was the dominant failure. The
    # instance filter itself is NEVER relaxed (walking to the wrong instance
    # is a correctness failure, not a quality one); only the DISTANCE window
    # widens per round, since a slightly-farther open start beats a close
    # cramped one for leg 1 by a wide margin.
    def search(cands, dist_max_mult: float):
        best_local = None
        rejected = 0
        for p in cands:
            d = geodesic(pf, p, a_point)
            if d is None or not (min_geodesic <= d <= desired_dist * dist_max_mult):
                continue
            if not a_point_is_closest(p, d):
                rejected += 1
                continue
            cam_y = float(p[1]) + cam_height
            yaw = math.atan2(a_point[2] - p[2], a_point[0] - p[0])
            mf = clearance_m(sim, np.array([p[0], p[2], yaw]), cam_y, cfg)
            if mf >= min_clearance:
                return p, d, mf, rejected, None       # cleared the bar outright
            if best_local is None or mf > best_local[0]:
                best_local = (mf, d, p)
        return None, None, None, rejected, best_local

    best = None
    rejected_wrong_instance = 0
    for round_i, dist_mult in enumerate((2.5, 3.5, 5.0)):
        if round_i > 0:   # fresh candidates, widened distance window
            cands = [np.array(pf.get_random_navigable_point(island_index=island))
                     for _ in range(n_random)]
        else:
            cands = candidates
        p, d, mf, rejected, best_local = search(cands, dist_mult)
        rejected_wrong_instance += rejected
        if p is not None:
            if round_i:
                print(f"  [pick_start] found a start clearing {min_clearance}m on round "
                      f"{round_i + 1} (distance window widened to {desired_dist * dist_mult:.1f}m)")
            return p, d, mf
        if best_local is not None and (best is None or best_local[0] > best[0]):
            best = best_local

    if rejected_wrong_instance:
        print(f"  [pick_start] rejected {rejected_wrong_instance} candidate(s) closer to a "
              f"DIFFERENT instance of A's category than to the paired one")
    if best is not None:
        print(f"  [warn] no candidate start cleared {min_clearance}m across 3 rounds; using the "
              f"best available (clearance={best[0]:.2f}m) -- leg 1 will likely fight obstacles")
        return best[2], best[1], best[0]
    # Real bug this replaces (2026-09-06, HM3D wcojb4TFT35, chair->table):
    # this used to silently `return a_point.copy(), 0.0, 0.0` -- placing the
    # START at the exact same point as A when literally nothing (300
    # candidates x 3 rounds up to 5x desired_dist) was geodesically
    # connected to it. Leg 1 then trivially "succeeded" at 0m real distance
    # travelled, and that degenerate result was reported as a genuine
    # navigation success before the ground_truth.start_xy == A_xy identity
    # was noticed by inspection. Fail loudly instead, matching pick_pair's
    # own convention for "no valid candidate exists" -- a caller must pick a
    # different A/B pair, not silently receive a zero-distance test.
    raise SystemExit(
        f"pick_start: no navmesh point geodesically connects to A={tuple(a_point)} within "
        f"{desired_dist * 5.0:.1f}m across 3 escalating rounds -- this instance is likely in an "
        f"isolated navmesh pocket. Re-run with a different --category-a/--min-ab/--max-ab so "
        f"pick_pair selects a different A instance; do not treat this as a start point.")


# --------------------------------------------------------------------------- #
def build_pipeline(args) -> DinoNavDPPipeline:
    cfg = PipelineConfig(
        device=args.device,
        horizontal_fov_deg=FOV_DEG,
        policy_type=args.policy,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        stop_distance=args.stop_distance,
        avoid_enabled=not args.no_avoid,
        use_dino=False, use_sam=False, use_clip=False, use_qwen_search=False,
        use_qwen_instruction=True,
        qwen_model_id=args.qwen_model_id,
        qwen_load_in_4bit=args.qwen_4bit,
        use_appearance_reid=False, use_scene_tagger=False, use_belief_goal=True,
        search_angular=args.search_angular,
        guard=GuardConfig(),
    )
    return DinoNavDPPipeline(cfg, use_depth_estimator=False)


def render(sim, pose, cam_y: float):
    """pose = (X, Y, yaw) in our standard planar frame (X=x_habitat,
    Y=z_habitat, forward=(cos(yaw),sin(yaw))). Real habitat-sim RGB-D."""
    X, Yp, yaw = pose
    habitat_yaw = -(yaw + math.pi / 2.0)  # see module docstring's frame note
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([X, cam_y, Yp], dtype=np.float32)
    state.rotation = quat_from_angle_axis(habitat_yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[:, :, :3].copy()
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return rgb, depth


def run_leg(pipe, sim, pf, island, cam_y, intr, pose, dt, max_steps, kind, extra,
           target_label: str, log):
    notice = None
    last_probe = 0.0
    import time
    for i in range(max_steps):
        rgb, depth = render(sim, pose, cam_y)

        if kind == "track":
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            instruction=f"go to the {target_label}", intrinsics=intr)
        else:
            ext = extra.tick(pose)
            res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                            external_goal=ext, intrinsics=intr)

        log.append({"leg": kind, "step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "state": res.state, "v": float(res.linear), "w": float(res.angular)})

        if kind == "track" and notice is None:
            now = time.time()
            if now - last_probe >= 2.0:
                last_probe = now
                pg = pipe._qwen_pixel_guide.ground(rgb, PROBE_PROMPT_TMPL.format(target=target_label))
                if pg is not None and pg.confidence >= 0.3:
                    u, v = int(round(pg.u)), int(round(pg.v))
                    # Qwen's own "nothing else here" cop-out lands on (0,0) or the
                    # frame's extreme corner/edge once rescaled -- not a detection.
                    margin_u, margin_v = 0.04 * W, 0.04 * H
                    on_border = u < margin_u or u > W - margin_u or v < margin_v or v > H - margin_v
                    if 0 <= u < W and 0 <= v < H and not on_border:
                        d = float(depth[v, u])
                        if 0.3 < d < 15.0:
                            local = pixel_depth_to_point(u, v, d, *intr)
                            ox, oy = local_to_odom(pose, float(local[0]), float(local[1]))
                            notice = {"step": i, "pixel": [u, v], "depth_m": d,
                                     "odom_xy": [ox, oy], "qwen_confidence": pg.confidence}
                            print(f"  [notice] step {i}: Qwen flagged something at pixel ({u},{v}) "
                                  f"depth={d:.2f}m conf={pg.confidence:.2f} -> odom ({ox:.2f}, {oy:.2f})")

        if kind == "track" and res.state == "STOP":
            return "reached", notice
        if kind == "goto" and extra.reached(pose):
            return "reached", notice
        if kind == "goto" and res.state == "STOP":
            return "pipeline_stop", notice

        v_, w_ = float(res.linear), float(res.angular)
        nx = pose[0] + v_ * math.cos(pose[2]) * dt
        ny = pose[1] + v_ * math.sin(pose[2]) * dt
        # keep the dead-reckoned pose on the real navmesh instead of walking through walls
        snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny]), island))
        pose[0], pose[1] = float(snapped[0]), float(snapped[2])
        pose[2] = wrap(pose[2] + w_ * dt)

    return "max_steps", notice


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def state_counts(log, kind):
    counts: dict[str, int] = {}
    for row in log:
        if row["leg"] == kind:
            counts[row["state"]] = counts.get(row["state"], 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default="/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d")
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--category-a", default="chair")
    ap.add_argument("--category-b", default="table")
    ap.add_argument("--min-geodesic", type=float, default=2.0)
    ap.add_argument("--max-geodesic", type=float, default=14.0)
    ap.add_argument("--start-offset", type=float, default=3.5)

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=1.0)
    ap.add_argument("--reached-radius", type=float, default=1.0)
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--max-steps-leg1", type=int, default=200)
    ap.add_argument("--max-steps-leg2", type=int, default=200)
    ap.add_argument("--out", default="logs/ab_recall_hm3d_test")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hm3d_root = Path(args.hm3d_root)
    dataset_config = find_dataset_config(hm3d_root)
    scene_glb = find_scene_glb(hm3d_root, args.scene)
    print(f"Scene: {args.scene}  glb={scene_glb}\ndataset_config={dataset_config}")

    sim = build_sim(scene_glb, dataset_config)
    pf = sim.pathfinder
    island = largest_island(pf)
    print(f"navmesh: {pf.num_islands} islands, using largest (#{island}, area={pf.island_area(island):.1f} m^2)")

    a_point, b_point, ab_geodesic = pick_pair(sim, island, args.category_a, args.category_b,
                                              args.min_geodesic, args.max_geodesic)
    start_point, start_geodesic, start_clearance = pick_start(sim, pf, island, a_point, b_point, args.start_offset)
    cam_y = float(start_point[1]) + EYE_HEIGHT_M

    a_xy = (float(a_point[0]), float(a_point[2]))
    b_xy = (float(b_point[0]), float(b_point[2]))
    start_xy = (float(start_point[0]), float(start_point[2]))
    yaw0 = math.atan2(a_xy[1] - start_xy[1], a_xy[0] - start_xy[0])

    print(f"A ({args.category_a}) @ {a_xy}   B ({args.category_b}) @ {b_xy}   A<->B geodesic={ab_geodesic:.2f}m")
    print(f"start @ {start_xy}  (start->A geodesic={start_geodesic:.2f}m, clearance={start_clearance:.2f}m)  "
          f"yaw0={math.degrees(yaw0):.0f}deg")

    pipe = build_pipeline(args)
    intr = intrinsics_from_fov(W, H, FOV_DEG)
    pose = np.array([start_xy[0], start_xy[1], yaw0])
    log: list[dict] = []

    print(f"\n=== Leg 1: instruction = 'go to the {args.category_a}' (real HM3D imagery, live Qwen + NavDP) ===")
    outcome1, notice = run_leg(pipe, sim, pf, island, cam_y, intr, pose, args.dt, args.max_steps_leg1,
                               "track", None, args.category_a, log)
    leg1_final = [float(pose[0]), float(pose[1]), float(pose[2])]
    leg1_error = dist(leg1_final, a_xy)
    print(f"outcome={outcome1}  final=({leg1_final[0]:.2f},{leg1_final[1]:.2f})  "
          f"error_to_A={leg1_error:.2f} m  states={state_counts(log,'track')}")

    result = {
        "scene": args.scene, "category_a": args.category_a, "category_b": args.category_b,
        "ground_truth": {"A_xy": a_xy, "B_xy": b_xy, "A_B_geodesic_m": ab_geodesic,
                         "start_xy": start_xy, "start_A_geodesic_m": start_geodesic},
        "leg1": {"instruction": f"go to the {args.category_a}", "outcome": outcome1,
                 "final_pose": leg1_final, "error_to_A_m": leg1_error,
                 "state_counts": state_counts(log, "track"), "steps": sum(1 for r in log if r["leg"] == "track")},
        "notice": notice,
    }

    if notice is None:
        print(f"\n=== Leg 2 skipped: Qwen never flagged a second object during leg 1 -- nothing to recall ===")
        result["leg2"] = None
    else:
        b_geom_error = dist(notice["odom_xy"], b_xy)
        print(f"\nrecall target B: recorded odom ({notice['odom_xy'][0]:.2f},{notice['odom_xy'][1]:.2f}) "
              f"vs true {args.category_b} ({b_xy[0]:.2f},{b_xy[1]:.2f}) -> geometry error {b_geom_error:.2f} m")

        print("\n=== Leg 2: instruction = 'go to B' (memory recall, NO re-detection) ===")
        src = Fact3rGoalSource(notice["odom_xy"], MapToOdom.identity(),
                               reached_radius=args.reached_radius, query="B")
        outcome2, _ = run_leg(pipe, sim, pf, island, cam_y, intr, pose, args.dt, args.max_steps_leg2,
                              "goto", src, args.category_a, log)
        leg2_final = [float(pose[0]), float(pose[1]), float(pose[2])]
        leg2_err_recorded = dist(leg2_final, notice["odom_xy"])
        leg2_err_true = dist(leg2_final, b_xy)
        print(f"outcome={outcome2}  final=({leg2_final[0]:.2f},{leg2_final[1]:.2f})  "
              f"error_to_recorded_B={leg2_err_recorded:.2f} m  error_to_true_B={leg2_err_true:.2f} m  "
              f"states={state_counts(log,'goto')}")
        result["leg2"] = {"instruction": "go to B", "outcome": outcome2, "final_pose": leg2_final,
                          "error_to_recorded_B_m": leg2_err_recorded, "error_to_true_B_m": leg2_err_true,
                          "state_counts": state_counts(log, "goto"), "steps": sum(1 for r in log if r["leg"] == "goto")}

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'} and {out_dir/'trajectory.json'}")
    sim.close()


if __name__ == "__main__":
    main()
