# Qwen3-VL-2B + NavDP instruction navigation

A minimal instruction-following navigation stack for a 6WD ESP32 rover:

**Free-text instruction → Qwen3-VL-2B grounds a pixel → depth lifts it to a 3D
point goal → NavDP samples/follows a trajectory toward it.**

Qwen finds the *next waypoint pixel* for a described maneuver ("go to the
farthest chair", "walk through the doorway and stop by the desk") — not a named
object, so no Grounding DINO. Between throttled Qwen calls a GoalBelief coasts
the goal by dead-reckoned ego-motion. A model-agnostic depth-based obstacle
guard (hard stop + reactive steer-around + swept-trajectory veto) runs the whole
time.

## Layout

```
nav_pipeline/
  instruction_gui.py     Tk GUI: camera view, top-down NavDP plot, instruction box
  pipeline.py            DinoNavDPPipeline / PipelineConfig — per-frame orchestrator
  qwen_pixel_goal.py     Qwen-VL grounder: (rgb, instruction) -> ranked pixel goals
  goal_utils.py          pixel + depth + intrinsics -> 3D point goal
  goal_belief.py         ego-motion goal coasting between Qwen calls
  navdp_net.py           NavDP diffusion trajectory sampler + critic
  navdp_backbone.py      RGBD patch backbone
  navdp_crossmodal.py    NavDP cross-modal variant
  depth_estimator.py     monocular metric depth (DepthAnythingV2-metric)
  obstacle_guard.py      depth-based AVOID state + swept clearance
  zenoh_node.py          Zenoh camera / cmd_vel contract
  isaac_gui.py           shared SharedState / inference_loop / zenoh_setup helpers
  depth_anything/        vendored DepthAnything-V2 (DINOv2 ViT + DPT head)
  dino_detector.py, sam_segmenter.py, clip_verifier.py, dinov2_embedder.py,
  scene_tagger.py, relational_target.py, qwen_search_guide.py
                         imported by pipeline.py; unused on the instruction path
                         (use_dino/use_sam/use_search all off) but must be present
LAUNCH/
  launch_instruction_gui_qwen3vl.sh   entry point (this setup)
  launch_instruction_gui.sh           base launcher it wraps
  _backend.sh                         Pi bring-up + --rover/--hiwonder selection
```

## Run

```bash
conda activate <your-env>          # needs: torch, transformers>=4.57, opencv,
                                    #        numpy, pillow, eclipse-zenoh, tkinter
export HF_HOME=/path/to/hf_cache
export PI_PASS=<rover-pi-ssh-password>     # _backend.sh has no default (scrubbed)

./LAUNCH/launch_instruction_gui_qwen3vl.sh <ROVER_PI_IP>
./LAUNCH/launch_instruction_gui_qwen3vl.sh <ROVER_PI_IP> --instruction "go to the farthest chair"
```

`launch_instruction_gui_qwen3vl.sh` = the base launcher plus
`--qwen-model-id Qwen/Qwen3-VL-2B-Instruct --qwen-fp16 --stop-distance 0.5`.

The launch scripts assume a local conda setup (edit the `source .../conda.sh`
line in `launch_instruction_gui.sh` for your machine) and a Pi running the
`rover-camera` / `rover-agent` / `rover-zenoh` services publishing over Zenoh
(`image_raw/compressed`, `rover/rpm`, optional `depth_raw`) and taking `cmd_vel`.

## Model weights (not included — download separately)

| What | Path the code expects |
|---|---|
| NavDP weights | `checkpoints/navdp_extracted.pth` |
| NavDP RGB backbone | `checkpoints/depth_anything_v2_vits.pth` |
| Metric depth estimator | `checkpoints/depth_anything_v2_metric_hypersim_vits.pth` (or `_vitb` with `--depth-encoder vitb`) |
| Grounder | `Qwen/Qwen3-VL-2B-Instruct` — auto-downloaded to `$HF_HOME` (~4.5 GB) |

`qwen_pixel_goal.py` loads the grounder via `AutoModelForImageTextToText`, so any
Qwen2.5-VL / Qwen3-VL id works via `--qwen-model-id`.

## Long-range grounding

Monocular metric depth is unreliable past ~6–8 m. When a grounded pixel's depth
is missing (a hole) or reads beyond `qwen_depth_trust_horizon_m` (6 m), the goal
is placed at `qwen_far_lookahead_m` (3.5 m) along that pixel's bearing and the
consistency gate is skipped — the rover drives the correct heading and
re-grounds as it closes in, instead of rejecting every jittery far reading and
coasting a stale lock. Set `qwen_far_lookahead_m=0` in `PipelineConfig` to
disable and revert to the drop-the-candidate behavior.

## Key `PipelineConfig` knobs

| knob | default | effect |
|---|---|---|
| `stop_distance` | 1.5 m | distance to the goal point at which arrival is declared (this setup runs 0.5) |
| `qwen_instruction_period_s` | 1.5 s | throttle between Qwen calls; GoalBelief coasts in between |
| `qwen_goal_consistency_m` | 1.5 m | reject a near re-grounding this far from the locked goal |
| `qwen_max_candidates` | 3 | ranked candidates per call, scored on confidence + continuity + clearance |
| `qwen_far_lookahead_m` | 3.5 m | bearing-only look-ahead distance for far/untrusted-depth targets |
| `qwen_depth_trust_horizon_m` | 6.0 m | depth beyond this is treated as bearing-only |
