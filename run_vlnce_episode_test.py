#!/usr/bin/env python3
"""One real R2R-CE (VLN-CE) episode, driven by the real Qwen+NavDP pipeline
against a real MP3D scene -- continuous NavDP control throughout, scored
with the actual VLN-CE metrics (NE / SR / OSR / SPL).

Only one MP3D scene is available on this machine: `17DRP5sb8fy`, habitat-sim's
public example scene (not one of the ToS-gated full-dataset houses, but real
R2R-CE episodes in R2R_VLNCE_v1-3/val_seen genuinely reference it -- see
MASt3R-SLAM/VLNCE_RUNBOOK.md for why the rest of the benchmark is blocked
here). This runs one of those real episodes end to end:

    instruction (real R2R-CE text) --Qwen pixel-goal--> waypoint pixel
    --depth--> 3D goal --NavDP--> continuous (v, w) --dead-reckon (navmesh-
    snapped)--> new pose, repeat until STOP or the step budget runs out.

No discrete grid actions, no oracle path-following -- the same live
instruction-grounding pipeline the HM3D/GOAT tests used, just pointed at a
real multi-clause navigation instruction and a real scene instead of an
object-goal category name.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError as exc:  # pragma: no cover
    raise SystemExit("habitat_sim is required -- run this under the `habitat` conda env") from exc

from nav_pipeline.goal_utils import intrinsics_from_fov
from nav_pipeline.obstacle_guard import GuardConfig
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig

W, H, FOV_DEG = 640, 480, 90.0
EYE_HEIGHT_M = 1.2
SUCCESS_DISTANCE_M = 3.0  # VLN-CE's own convention


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


_CLAUSE_SPLIT_RE = re.compile(r"[,.]?\s+then\s+|;\s*then\s+", flags=re.I)


def split_instruction_clauses(instruction: str) -> list[str]:
    """"Go to the chair, then continue to the table." -> ["Go to the chair",
    "continue to the table"] -- the single, well-evidenced fix for a real
    measured failure: a multi-clause instruction fed whole to Qwen every
    tick has no notion of "which clause am I on", so satisfying the FIRST
    clause (often trivially, e.g. a chair right next to the start) looks
    identical to satisfying the whole instruction, and the episode
    self-stops there instead of continuing. Splitting on "then" (the only
    connective this project's own custom test instructions actually use)
    and driving ONE clause's text at a time, advancing to the next only on
    a real STOP, keeps the rest of the instruction from ever being silently
    dropped. Falls back to a single "clause" (the whole instruction,
    unchanged behavior) when no "then" is present -- real R2R-CE instructions
    mostly aren't phrased this way, so this is opt-in territory, not a
    silent behavior change for the common case."""
    parts = [p.strip(" .") for p in _CLAUSE_SPLIT_RE.split(instruction) if p.strip(" .")]
    return parts if parts else [instruction]


def load_episode(dataset_root: Path, split: str, scene: str, episode_id: str):
    path = dataset_root / split / f"{split}.json.gz"
    data = json.load(gzip.open(path, "rt"))
    for ep in data["episodes"]:
        if scene in ep["scene_id"] and str(ep["episode_id"]) == str(episode_id):
            return ep
    raise SystemExit(f"episode {episode_id} for scene {scene} not found in {path}")


def build_sim(scene_glb: str, dataset_config: str | None):
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [H, W]
    rgb_spec.position = np.array([0.0, 0.0, 0.0])  # camera AT agent y (cam_y already = floor + EYE_HEIGHT_M); do NOT double-add
    rgb_spec.hfov = FOV_DEG

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [H, W]
    depth_spec.position = np.array([0.0, 0.0, 0.0])  # camera AT agent y (cam_y already = floor + EYE_HEIGHT_M); do NOT double-add
    depth_spec.hfov = FOV_DEG

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
        # Defaults (agent_radius=0.1, agent_height=1.5) matched the shipped
        # habitat mp3d navmeshes closely enough in testing; only override
        # what's needed to keep this reproducible across scenes.
        settings = habitat_sim.nav.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = EYE_HEIGHT_M
        ok = sim.recompute_navmesh(sim.pathfinder, settings)
        print(f"[note] no shipped navmesh for {scene_glb} -- baked one at runtime "
              f"(recompute_navmesh ok={ok})")
    return sim


def render(sim, pose, cam_y: float):
    X, Yp, yaw = pose
    habitat_yaw = -(yaw + math.pi / 2.0)  # same convention as run_ab_recall_hm3d_test.py
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


class StuckRecovery:
    """Detects a stalled local minimum and overrides the commanded (v, w).

    Ported verbatim (same defaults) from run_ab_position_recall_test.py,
    where it was built after finding this exact bug: this script's own
    `pf.snap_point` clips any proposed move that leaves the navmesh (NavDP
    servoing a heading into a wall/obstacle at a tight real R2R-CE start
    pose) back onto the nearest navigable point. The *commanded* v looks
    normal in the log but *realized* displacement is ~0 -- the pipeline
    can't see this itself (only rendered frames, not the navmesh clip), so
    a local minimum silently eats the whole step budget as AVOID instead of
    triggering a replan. Measured on episode 328 (2026-09-05): 271/300 ticks
    AVOID, mean v=-0.26 (reversing), dist_to_goal only 8.14m->6.12m over the
    whole 45s budget -- this class exists to break exactly that.

    Turn-then-COMMIT, not turn-then-resume: a first version that only
    rotated in place and handed back to the policy re-locked onto the same
    blocked heading a tick later (net displacement ~0 anyway). Rotating a
    fixed sweep THEN forcing several ticks of forward motion at reduced
    speed actually relocates the robot. Escalates the turn size on repeated
    failed escapes (alternating direction so it doesn't re-settle on the
    same heading).
    """

    def __init__(self, window: int = 30, min_progress_m: float = 0.15,
                 turn_ticks: int = 12, push_ticks: int = 15,
                 escape_w: float = 0.8, push_v: float = 0.15):
        self.window = window
        self.min_progress_m = min_progress_m
        self.turn_ticks = turn_ticks
        self.push_ticks = push_ticks
        self.escape_w = escape_w
        self.push_v = push_v
        self._history: list[tuple[float, float]] = []
        self._phase = None       # None | "turn" | "push"
        self._phase_left = 0
        self._escape_dir = 1.0
        self._escape_start_xy: Optional[tuple[float, float]] = None
        self.trigger_count = 0
        self.escalation = 0
        self.just_triggered = False

    def override(self, pose, v: float, w: float) -> tuple[float, float, bool]:
        self.just_triggered = False
        self._history.append((pose[0], pose[1]))
        if len(self._history) > self.window:
            self._history.pop(0)

        if self._phase == "turn":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase, self._phase_left = "push", self.push_ticks
            return 0.0, self.escape_w * self._escape_dir, True

        if self._phase == "push":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase = None
                if self._escape_start_xy is not None:
                    moved = float(np.hypot(pose[0] - self._escape_start_xy[0],
                                            pose[1] - self._escape_start_xy[1]))
                    self.escalation = 0 if moved > self.min_progress_m else self.escalation + 1
                self._history.clear()
            return self.push_v, 0.0, True

        if len(self._history) == self.window:
            p0 = np.array(self._history[0])
            p1 = np.array(self._history[-1])
            net = float(np.linalg.norm(p1 - p0))
            if net < self.min_progress_m:
                self._escape_dir *= -1.0
                self._escape_start_xy = (pose[0], pose[1])
                turn_ticks = self.turn_ticks * (1 + min(self.escalation, 3))
                self._phase, self._phase_left = "turn", turn_ticks
                self.trigger_count += 1
                self.just_triggered = True
                return 0.0, self.escape_w * self._escape_dir, True

        return v, w, False


def habitat_yaw_from_quat(rot_xyzw) -> float:
    x, y, z, w = rot_xyzw
    return 2.0 * math.atan2(y, w)  # pure yaw-about-Y rotations, as R2R-CE start poses are


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
        qwen_instruction_period_s=args.instruction_period_s,
        qwen_goal_consistency_m=args.goal_consistency_m,
        qwen_stop_confirm_ticks=args.stop_confirm_ticks,
        use_appearance_reid=args.qwen_appearance_lock, use_scene_tagger=False, use_belief_goal=True,
        qwen_grounding_crops=args.grounding_crops,
        qwen_grounding_crop_fraction=args.grounding_crop_fraction,
        qwen_max_candidates=args.qwen_max_candidates,
        qwen_appearance_lock=args.qwen_appearance_lock,
        qwen_appearance_min_similarity=args.qwen_appearance_sim,
        search_angular=args.search_angular,
        # GuardConfig defaults to cam_height=0.35m (the real ESP32 rover's
        # actual camera height) -- this script renders from EYE_HEIGHT_M=1.2m
        # instead (habitat's human-eye convention), so the guard's ground-
        # plane math (z_up = cam_height - y_cam) was recovering a badly wrong
        # height for every point, misreading the floor as an obstacle at
        # close range in every direction (confirmed: a standalone clearance
        # probe found min_fwd_clear pinned at 0.15m -- the valid-depth floor,
        # not real geometry -- at all 12 sampled headings from episode 328's
        # start pose). run_ab_position_recall_test.py and
        # run_ab_recall_hm3d_test.py both already pass cam_height=EYE_HEIGHT_M
        # for this exact reason; this script never did.
        guard=GuardConfig(cam_height=EYE_HEIGHT_M),
    )
    return DinoNavDPPipeline(cfg, use_depth_estimator=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mp3d-root", default="/home/gpu/Desktop/pineapple/mp3d_data",
                    help="root used to load R2R-CE episodes (mp3d-root/datasets/R2R_VLNCE_v1-3/...); "
                         "unrelated to --scene-glb, which picks the mesh independently")
    ap.add_argument("--scene", default="17DRP5sb8fy")
    ap.add_argument("--split", default="val_seen")
    ap.add_argument("--episode-id", default="95")
    ap.add_argument("--scene-glb", default=None,
                    help="explicit path to a scene mesh (.glb/.obj/.ply) to use instead of the "
                         "bundled mp3d_example_scene_1.1 habitat-ready glb -- for a raw MP3D v1 "
                         "research-release house with no shipped navmesh, build_sim() bakes one "
                         "at runtime")
    ap.add_argument("--instruction", default=None,
                    help="override the episode's own R2R-CE instruction text with a custom one "
                         "-- same start position, same goal, same navmesh/scene")
    ap.add_argument("--start-yaw-override-deg", type=float, default=None,
                    help="use this heading (world-frame, my_yaw convention) instead of the "
                         "episode's own start_rotation -- R2R-CE start headings come from raw human "
                         "walkthroughs and are occasionally degenerate (e.g. facing a wall/fixture "
                         "from 30cm away); this does not move the start position or change the goal.")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", choices=("crossmodal", "extracted"), default="crossmodal")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--qwen-4bit", action="store_true")
    ap.add_argument("--instruction-period-s", type=float, default=1.5)
    ap.add_argument("--goal-consistency-m", type=float, default=1.5,
                    help="reject a re-grounded candidate this far from the current locked belief "
                         "(PipelineConfig.qwen_goal_consistency_m) -- the real-rover default (1.5m) "
                         "assumes a mostly-static target; over a long multi-room VLN-CE instruction "
                         "the robot's own motion can legitimately put 3-4m between two real sightings "
                         "of the same described place, which the default then wrongly rejects")
    ap.add_argument("--max-linear", type=float, default=0.6)
    ap.add_argument("--max-angular", type=float, default=0.6)
    ap.add_argument("--stop-distance", type=float, default=1.2)
    ap.add_argument("--stop-confirm-ticks", type=int, default=3,
                    help="PipelineConfig.qwen_stop_confirm_ticks -- consecutive close fresh-"
                         "grounding readings required before a Qwen-instruction STOP is trusted. "
                         "Raise this to require more sustained confirmation before committing to "
                         "arrival, if the run is stopping early near a plausible-but-wrong nearby "
                         "object (a real risk on multi-clause instructions with no clause-progress "
                         "tracking -- see run notes 2026-09-05).")
    ap.add_argument("--no-avoid", action="store_true")
    ap.add_argument("--search-angular", type=float, default=0.4)
    ap.add_argument("--grounding-crops", action="store_true",
                    help="PipelineConfig.qwen_grounding_crops -- Qwen scores several image crops "
                         "per tick and the candidate scorer (collision-weighted) picks the most "
                         "reachable one, instead of trusting a single whole-frame pixel goal. "
                         "Targets 'grounded a goal behind a wall'.")
    ap.add_argument("--grounding-crop-fraction", type=float, default=0.72)
    ap.add_argument("--qwen-max-candidates", type=int, default=3,
                    help="PipelineConfig.qwen_max_candidates -- pixel-goal candidates Qwen proposes "
                         "per grounding call for the collision/continuity scorer to choose among")
    ap.add_argument("--qwen-appearance-lock", action="store_true",
                    help="PipelineConfig.qwen_appearance_lock (+ enables use_appearance_reid) -- "
                         "multi-view DINOv2 bank; rejects a candidate that doesn't look like the "
                         "locked target BEFORE belief latches. Project notes: the one change that "
                         "reproducibly cut leg1 grounding error ~20%.")
    ap.add_argument("--qwen-appearance-sim", type=float, default=0.55,
                    help="PipelineConfig.qwen_appearance_min_similarity for the lock above")
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--no-realtime-pacing", dest="realtime_pacing", action="store_false",
                    help="don't sleep to real wall-clock dt between ticks -- breaks "
                         "qwen_instruction_period_s's real-time throttle, only useful for "
                         "quick smoke tests that don't care about re-grounding behavior")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--ignore-stop", action="store_true",
                    help="don't end the drive on a self-declared STOP -- clears belief "
                         "(pipe.reset()) and keeps driving instead, to measure how much "
                         "further the episode could progress once a premature 'arrived' "
                         "declaration is discounted. oracle_navigation_error_m already "
                         "reports the true closest approach regardless of this flag")
    ap.add_argument("--sequence-clauses", action="store_true",
                    help="split --instruction on 'then' into ordered clauses and drive them one "
                         "at a time, advancing (pipe.reset(), next clause's text) on each real "
                         "STOP instead of ending the episode there. Fixes a measured real failure: "
                         "a multi-clause instruction fed whole to Qwen every tick has no notion of "
                         "'which clause am I on', so reaching just the FIRST clause's target (often "
                         "trivially -- e.g. a chair right next to the start) self-stops the whole "
                         "episode before the rest of the instruction is ever addressed. Opt-in: "
                         "real R2R-CE instructions aren't reliably phrased with 'then', so this "
                         "would be a no-op there anyway, but keeping it a flag avoids silently "
                         "changing behavior for existing single-clause callers of this script.")
    ap.add_argument("--seed", type=int, default=-1,
                    help="seed torch + numpy before building the pipeline. NavDP samples 32 "
                         "trajectories from torch.randn every tick (navdp_crossmodal.py) and is "
                         "otherwise unseeded -- so two runs of the same episode diverge from tick "
                         "one, which has confounded every A/B in this project's notes. -1 = "
                         "don't seed (legacy behaviour); >=0 = reproducible.")
    ap.add_argument("--min-steps-before-stop", type=int, default=0,
                    help="ignore a self-declared STOP for the first N steps (pipe.reset() + keep "
                         "driving instead). Qwen-2B grounds on a wrong nearby doorway/object and "
                         "declares arrival after ~15 ticks / 0.1 m on some scenes -- a real "
                         "multi-metre episode cannot legitimately be done that fast.")
    ap.add_argument("--search-creep-v", type=float, default=0.0,
                    help="while the pipeline is in SEARCH/AVOID and commanding ~0 forward speed, "
                         "force at least this m/s forward. Spinning in place never changes the "
                         "view enough for Qwen to re-ground; a slow creep does. Try 0.15-0.25.")
    ap.add_argument("--stuck-push-ticks", type=int, default=15,
                    help="StuckRecovery: forward-push duration after the escape turn. The default "
                         "(15) x push_v gives ~0.3 m/escape -- too little to leave a real local "
                         "minimum; try 25-30.")
    ap.add_argument("--stuck-push-v", type=float, default=0.15,
                    help="StuckRecovery: forward speed during the escape push (try 0.3-0.4)")
    ap.add_argument("--dwell-stop-k", type=int, default=0,
                    help="STOP once the robot has stayed within --dwell-stop-radius-m for this "
                         "many consecutive ticks (past --min-steps-before-stop). Qwen+NavDP "
                         "reaches the goal region then drifts off without ever self-declaring "
                         "STOP there -- this locks in the arrival. 0 disables.")
    ap.add_argument("--dwell-stop-radius-m", type=float, default=1.5,
                    help="radius of the dwell ball for --dwell-stop-k")
    ap.add_argument("--dwell-stop-any-state", dest="dwell_stop_require_track",
                    action="store_false",
                    help="let --dwell-stop-k fire in any pipeline state, not just TRACK "
                         "(default requires TRACK so it doesn't fire on a SEARCH/stuck stall)")
    ap.add_argument("--dwell-min-disp-m", type=float, default=0.0,
                    help="only let --dwell-stop-k fire once the robot has displaced this many "
                         "metres from its start. Distinguishes 'arrived, now circling the goal' "
                         "(measured: ~1.5 m displacement) from 'stuck near start circling a bad "
                         "Qwen goal' (measured: ~0.7 m) -- WITHOUT using the goal position. Use "
                         "with --dwell-stop-any-state. Try 1.2.")
    ap.add_argument("--grounding-prompt-style", choices=("default", "route"), default="default",
                    help="'route' swaps the Qwen grounding prompt for a route-aware + "
                         "reachability-aware one: point at the IMMEDIATE next opening/doorway "
                         "that makes progress along the route (the final target is often not "
                         "visible), and only at floor the robot can physically drive to (not "
                         "through a wall/window/furniture). Targets the multi-room + "
                         "goal-behind-a-wall failures.")
    ap.add_argument("--success-distance", type=float, default=SUCCESS_DISTANCE_M,
                    help="NE below this counts as success. Default 3.0 = VLN-CE / R2R-CE "
                         "convention. Lower it (e.g. 0.3) to require the robot to physically "
                         "reach the object -- measured, no run in this project gets closer than "
                         "~1.4 m, so <=1.0 yields 0 successes.")
    ap.add_argument("--out", default="logs/vlnce_episode_test")
    args = ap.parse_args()

    if args.seed >= 0:
        import os as _os
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        import random as _random
        _random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as _e:  # older torch signature
            torch.use_deterministic_algorithms(True)
        print(f"[seed] torch+numpy+random seeded {args.seed}, cudnn deterministic, "
              f"use_deterministic_algorithms(warn_only)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp3d_root = Path(args.mp3d_root)
    episode = load_episode(mp3d_root / "datasets" / "R2R_VLNCE_v1-3", args.split, args.scene, args.episode_id)
    instruction = episode["instruction"]["instruction_text"].strip()
    if args.instruction:
        print(f"[note] overriding episode instruction {instruction!r} with custom "
              f"instruction {args.instruction!r} -- same start/goal")
        instruction = args.instruction
    optimal_length = float(episode["info"]["geodesic_distance"])
    goal_pos = episode["goals"][0]["position"]  # [x, y, z] habitat
    goal_radius = float(args.success_distance)
    start_pos = episode["start_position"]
    if args.start_yaw_override_deg is not None:
        start_yaw = math.radians(args.start_yaw_override_deg)
        print(f"[note] overriding the episode's own start heading (degenerate close-up view) with "
              f"{args.start_yaw_override_deg:.0f} deg -- same start position, same goal")
    else:
        start_yaw = -habitat_yaw_from_quat(episode["start_rotation"]) - math.pi / 2.0

    print(f"Episode {episode['episode_id']} (scene {args.scene}, split {args.split})")
    print(f"Instruction: \"{instruction}\"")
    print(f"Optimal (geodesic) length: {optimal_length:.2f} m   success radius: {goal_radius:.1f} m")

    if args.scene_glb:
        scene_glb = args.scene_glb
        dataset_config = None
    else:
        scene_glb = str(mp3d_root / "versioned_data" / "mp3d_example_scene_1.1" / args.scene / f"{args.scene}.glb")
        dataset_config = str(mp3d_root / "versioned_data" / "mp3d_example_scene_1.1" / "mp3d.scene_dataset_config.json")
        if not Path(dataset_config).is_file():
            dataset_config = None
    sim = build_sim(scene_glb, dataset_config)
    print(f"navmesh loaded: {sim.pathfinder.is_loaded}")

    pf = sim.pathfinder

    pipe = build_pipeline(args)
    if args.grounding_prompt_style == "route" and getattr(pipe, "_qwen_pixel_guide", None) is not None:
        route_multi = (
            'You are guiding a mobile ground robot along a route. Full route: "{instruction}". '
            "The final destination is often NOT visible from here yet -- do not guess where it is. "
            "Identify up to {max_candidates} candidate pixels for the best IMMEDIATE next move that "
            "makes progress along this route: a doorway, opening, hallway mouth, or clear stretch of "
            "floor leading toward the next part of the route. Every candidate MUST be a spot on the "
            "floor the robot can physically drive to -- never through a wall, window, glass panel, "
            "or furniture. If the target for THIS step is clearly visible (e.g. you are already in "
            "the final room), point at it directly. List several if more than one opening is "
            "plausible, best first. ONE candidate per line, formatted exactly as: "
            "x, y, confidence  (confidence 0 unsure .. 1 certain)."
        )
        route_single = (
            'You are guiding a mobile ground robot along a route: "{instruction}". The final '
            "destination is often not visible yet. Point at the single best IMMEDIATE next "
            "waypoint pixel -- a doorway/opening/clear floor leading toward the next part of the "
            "route, and a spot the robot can physically drive to (not through a wall, window, or "
            "furniture). Reply with only one pixel coordinate as (x, y)."
        )
        pipe._qwen_pixel_guide.multi_prompt_template = route_multi
        pipe._qwen_pixel_guide.prompt_template = route_single
        print("[grounding] using route-aware + reachability-aware Qwen prompt")
    intr = intrinsics_from_fov(W, H, FOV_DEG)

    # wrap() into [-pi,pi] -- every SUBSEQUENT pose[2] update below already
    # goes through wrap(), but the raw start_yaw computed above (episode
    # quaternion or --start-yaw-override-deg) is not guaranteed to be in
    # that range (measured: -5.24 rad on episode 328). If left unwrapped,
    # the very first odom_delta (pipeline.py's th1-th0, comparing this
    # unwrapped value against the wrapped pose after tick 0) comes out ~2pi
    # off (measured: dtheta=6.298 instead of the real ~0.015 rad), which
    # blows belief.sigma past belief_max_sigma in one tick and forces a
    # premature SEARCH before the belief ever gets to hold its first lock
    # -- this was silently collapsing TRACK to ~1 tick for the entire
    # episode budget on every run tested 2026-09-05.
    pose = np.array([float(start_pos[0]), float(start_pos[2]), wrap(start_yaw)])
    cam_y = float(start_pos[1]) + EYE_HEIGHT_M
    goal_xy = (float(goal_pos[0]), float(goal_pos[2]))

    path_xy = [(pose[0], pose[1])]
    log: list[dict] = []
    min_dist = dist((pose[0], pose[1]), goal_xy)
    outcome = "max_steps"
    stuck = StuckRecovery(push_ticks=args.stuck_push_ticks, push_v=args.stuck_push_v)
    stop_count = 0
    ignored_stops = 0
    dwell_hist: list[tuple[float, float]] = []

    clauses = split_instruction_clauses(instruction) if args.sequence_clauses else [instruction]
    clause_idx = 0
    clause_advances = 0
    if args.sequence_clauses:
        print(f"[clause-sequencing] {len(clauses)} clause(s): {clauses}")

    print(f"\n=== Driving: \"{instruction[:70]}{'...' if len(instruction) > 70 else ''}\" ===")
    for i in range(args.max_steps):
        tick_start = time.time()
        rgb, depth = render(sim, pose, cam_y)
        res = pipe.step(rgb, target_text="", depth=depth, pose=tuple(pose),
                        instruction=clauses[clause_idx], intrinsics=intr)
        if args.realtime_pacing:
            # QwenInstructionGate throttles on real wall-clock time (correct
            # for a real robot's control loop) -- an unpaced tight loop here
            # completes far faster than dt of real time once Qwen isn't in
            # the critical path, so qwen_instruction_period_s's 1.5s never
            # actually elapses and it re-grounds exactly once, ever. Sleep
            # off whatever's left of this tick's dt budget so the gate
            # behaves as it would on the real rover.
            remaining = args.dt - (time.time() - tick_start)
            if remaining > 0:
                time.sleep(remaining)
        d = dist((pose[0], pose[1]), goal_xy)
        min_dist = min(min_dist, d)

        if res.state == "STOP":
            log.append({"step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                        "state": res.state, "v": float(res.linear), "w": float(res.angular),
                        "dist_to_goal": d, "escaping": False, "clause_idx": clause_idx})
            stop_count += 1
            if clause_idx < len(clauses) - 1:
                # Reached THIS clause's target with more clauses left -- this
                # is the fix, not a diagnostic override: advance instead of
                # ending the episode, so "go to the chair, then continue to
                # the table" doesn't self-terminate the moment any chair is
                # found. pipe.reset() clears the belief that just declared
                # "arrived" so the next tick re-grounds fresh on the NEW
                # clause's text instead of coasting on the old lock.
                clause_idx += 1
                clause_advances += 1
                print(f"  [step {i:3d}] [clause-advance] reached clause {clause_idx}/{len(clauses)} "
                      f"target -- now driving: \"{clauses[clause_idx]}\"")
                pipe.reset()
                continue
            if i < args.min_steps_before_stop:
                ignored_stops += 1
                print(f"  [step {i:3d}] [early-stop-ignored #{ignored_stops}] STOP at step {i} "
                      f"(< {args.min_steps_before_stop}), dist_to_goal={d:.2f}m -- resetting belief, "
                      f"keep driving")
                pipe.reset()
                continue
            if not args.ignore_stop:
                outcome = "stopped"
                break
            # Diagnostic mode: don't trust this STOP -- it's the documented
            # premature-arrival failure mode (see fact3r-navdp-bridge memory),
            # not proof of actually reaching the goal. pipe.reset() clears the
            # belief that just declared "arrived" so the next tick is forced
            # back into a genuine fresh SEARCH/re-ground instead of sitting
            # frozen re-confirming the same false lock every tick.
            pipe.reset()
            continue

        v_, w_ = float(res.linear), float(res.angular)
        # search-creep: a spinning-in-place SEARCH/AVOID never changes the view
        # enough for Qwen to re-ground -- force a slow forward drift.
        if args.search_creep_v > 0 and res.state in ("SEARCH", "AVOID") and v_ < args.search_creep_v:
            v_ = args.search_creep_v
        v_, w_, escaping = stuck.override(pose, v_, w_)
        if stuck.just_triggered:
            print(f"  [step {i:3d}] [stuck-recovery] no progress over last {stuck.window} ticks "
                  f"-- escape #{stuck.trigger_count} (escalation={stuck.escalation})")
        log.append({"step": i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
                    "state": res.state, "v": v_, "w": w_, "dist_to_goal": d, "escaping": escaping,
                    "clause_idx": clause_idx})
        if i % 10 == 0:
            print(f"  [step {i:3d}] state={res.state:6s} dist_to_goal={d:5.2f}m "
                  f"pose=({pose[0]:+.2f},{pose[1]:+.2f})" + ("  [escaping]" if escaping else ""))

        nx = pose[0] + v_ * math.cos(pose[2]) * args.dt
        ny = pose[1] + v_ * math.sin(pose[2]) * args.dt
        if pf.is_loaded:
            snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny])))
            pose[0], pose[1] = float(snapped[0]), float(snapped[2])
        else:
            pose[0], pose[1] = nx, ny
        pose[2] = wrap(pose[2] + w_ * args.dt)
        path_xy.append((pose[0], pose[1]))

        # dwell-stop: robot has stayed in a small ball for K ticks WHILE the
        # pipeline is TRACKing (grounded on the target) -> arrival. The TRACK
        # gate is essential: without it this fires on a SEARCH/stuck stall far
        # from the goal (measured -- it made things worse).
        # dwell-stop: the robot has stayed inside a small region for K ticks
        # (rolling window -> robust to circling the goal, which pin-anchored
        # ball-distance is not). Gates: past --min-steps-before-stop, not
        # escaping, TRACK unless --dwell-stop-any-state, and (crucially)
        # displaced >= --dwell-min-disp-frac of the geodesic from start -- so
        # it fires for "arrived and circling" (ep97: moved ~3.5m, orbits the
        # goal in SEARCH) but NOT "stuck near start circling a bad Qwen goal"
        # (ep96: moved ~0.7m, orbits a point 4m out). No goal position used.
        track_dwell = res.state == "TRACK" or not args.dwell_stop_require_track
        disp_from_start = dist((pose[0], pose[1]), path_xy[0])
        disp_ok = disp_from_start >= args.dwell_min_disp_m
        if (args.dwell_stop_k > 0 and i >= args.min_steps_before_stop and not escaping
                and track_dwell and disp_ok):
            dwell_hist.append((pose[0], pose[1]))
            if len(dwell_hist) > args.dwell_stop_k:
                dwell_hist.pop(0)
            if len(dwell_hist) == args.dwell_stop_k:
                cx = sum(p[0] for p in dwell_hist) / len(dwell_hist)
                cy = sum(p[1] for p in dwell_hist) / len(dwell_hist)
                spread = max(dist(p, (cx, cy)) for p in dwell_hist)
                if spread < args.dwell_stop_radius_m:
                    print(f"  [step {i:3d}] [dwell-stop] {args.dwell_stop_k}-tick spread "
                          f"{spread:.2f}m < {args.dwell_stop_radius_m}m, disp_from_start="
                          f"{disp_from_start:.2f}m, dist_to_goal={d:.2f}m -- declaring arrival")
                    outcome = "dwell_stop"
                    break
        else:
            dwell_hist.clear()

    final_xy = (float(pose[0]), float(pose[1]))
    navigation_error = dist(final_xy, goal_xy)
    travelled = path_length(path_xy)
    success = navigation_error < goal_radius
    oracle_success = min_dist < goal_radius
    spl = (optimal_length / max(optimal_length, travelled)) if (success and travelled > 0) else 0.0

    metrics = {
        "episode_id": episode["episode_id"], "scene": args.scene, "split": args.split,
        "instruction": instruction, "outcome": outcome, "steps": len(log),
        "start_xy": [float(path_xy[0][0]), float(path_xy[0][1])],
        "goal_xy": list(goal_xy), "final_xy": list(final_xy),
        "optimal_length_m": optimal_length, "path_length_m": travelled,
        "navigation_error_m": navigation_error, "oracle_navigation_error_m": min_dist,
        "success": success, "oracle_success": oracle_success, "spl": spl,
        "success_radius_m": goal_radius,
        "stuck_recovery_triggers": stuck.trigger_count,
        "stop_count": stop_count,
        "ignored_early_stops": ignored_stops,
        "sequence_clauses": args.sequence_clauses,
        "clauses": clauses,
        "final_clause_idx": clause_idx,
        "clause_advances": clause_advances,
    }
    print("\n=== VLN-CE metrics ===")
    for k in ("outcome", "steps", "navigation_error_m", "oracle_navigation_error_m",
             "success", "oracle_success", "spl", "path_length_m", "optimal_length_m"):
        print(f"  {k:26s}: {metrics[k]}")

    (out_dir / "result.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'} and {out_dir/'trajectory.json'}")
    sim.close()


if __name__ == "__main__":
    main()
