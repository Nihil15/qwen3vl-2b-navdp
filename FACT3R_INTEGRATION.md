# Fact3R-Map × NavDP — "REMEMBER → GO"

This repo grounds a goal in the **current** frame (Qwen pixel goal) and drives
to it with NavDP. It has no memory of anywhere it has already been. **Fact3R-Map**
(`../MASt3R-SLAM/fact3r-map`) is the opposite: an offline, persistent,
open-vocabulary semantic entity map built from a monocular walk, queryable long
afterwards with arbitrary text (*"return to the black chair"*).

This integration wires Fact3R-Map in as the **memory / goal source** and keeps
NavDP as the **legs**:

```
text query ─▶ fact3r-map/resolve_semantic_goal.py ─▶ entity location (map frame, metres)
                                                          │
                        MapToOdom  (2D similarity: map frame → rover odom frame)
                                                          │
   per tick:  rover pose (odom) ──▶ Fact3rGoalSource.tick ──▶ [x_fwd, y_left, z] point
                                                          │
                       DinoNavDPPipeline.step(external_goal=…)  ── state "GOTO"
                       (obstacle-guard + NavDP live; no self-declared STOP)
```

`external_goal` / the `"GOTO"` state already existed in `pipeline.py` for exactly
this ("last remembered world location of an object that isn't in view"). Nothing
in the pipeline changed — the integration is two new files plus a launcher.

## Files

| file | role |
|---|---|
| `nav_pipeline/fact3r_goal_source.py` | `MapToOdom` (map→odom 2D similarity) + `Fact3rGoalSource` (holds one resolved entity location, emits the per-tick `external_goal`). No torch/fact3r imports — pure arithmetic; `from_map_query` shells out to `resolve_semantic_goal.py` once. |
| `run_offline_fact3r_nav.py` | headless driver: recorded RGB(-D) walk → `pipeline.step(external_goal=…)` every frame, pose from a file or dead-reckoned from the pipeline's own `(v, w)`. Writes `rollout.mp4`, `trajectory.json`, `summary.json`, `topdown.png`. |
| `LAUNCH/launch_offline_fact3r_nav.sh` | interpreter/env wrapper; every driver flag passes through. |
| `tests/test_fact3r_goal_source.py` | frame-math checks (no models). `python tests/test_fact3r_goal_source.py`. |

## The two frames, and how they're aligned

* **Fact3R map frame** — `resolve_semantic_goal.py` writes a
  `fact3r-semantic-goal-request` JSON whose `winner["centroid_yx"]` is the entity
  location `[y, x]` in metres in the semantic BEV's floor-plane frame, plus
  `rover_track_yx` — the mapping camera's own trace in that same frame.
* **Rover odom frame** — `pipeline.step(pose=(x, y, theta))`: a continuous planar
  world frame anchored where this run started.

`MapToOdom` is `p_odom = scale · R(φ) · p_map + t`. Pick how it's derived with
`--align`:

| `--align` | when | how |
|---|---|---|
| `identity` (default) | the map was built from **this** run's frames and the axes already coincide | `scale=1, φ=0, t=0` |
| `anchor` | map built from the outbound leg; you know where/heading the return leg starts | pins `rover_track_yx[0]` to `--odom-heading-deg` at the odom origin |
| `umeyama` | you have an odom trajectory (`--odom-track`, xy per line) of the same walk | least-squares similarity fit of the whole trace |
| `explicit` | you measured it | `--map-to-odom "tx ty phi_deg scale"` |

> Metric BEV scale in Fact3R-Map requires calibrated intrinsics at map-build time
> (`run_fact3r_video.sh --calib …`). Without that, `scale` is arbitrary — use
> `umeyama` (free scale) or `explicit`.

## Build a map first

```bash
cd ../MASt3R-SLAM/fact3r-map
bash scripts/run_fact3r_video.sh --video /path/room_walk.mp4 --map-name room-walk \
     --sam2-env SAM2 --device 0 --calib config/my_camera.yaml     # calib optional but needed for metric scale
# then the depth-semantic-BEV stage that resolve_semantic_goal.py consumes
# (scripts/build_depth_semantic_bev.py → *_semantic.json); see DEPTH_SEMANTIC_BEV.md
```

## Run the return leg

```bash
# (a) resolve the query inside the driver (shells out to the SigLIP env)
./LAUNCH/launch_offline_fact3r_nav.sh \
    --frames /path/return_walk.mp4 \
    --fact3r-map-root ../MASt3R-SLAM/fact3r-map \
    --semantic-map ../MASt3R-SLAM/fact3r-map/logs/…/room-walk_semantic.json \
    --query "the black chair" --siglip-env SAM2 \
    --align anchor --odom-heading-deg 0 \
    --integrate-cmd --reached-radius 1.0 \
    --out logs/fact3r_nav/chair_return

# (b) reuse a goal request you already made
conda run -n SAM2 python ../MASt3R-SLAM/fact3r-map/scripts/resolve_semantic_goal.py \
    --map …/room-walk_semantic.json --query "the black chair" --output gr.json --device 0
./LAUNCH/launch_offline_fact3r_nav.sh --frames /path/return_walk.mp4 \
    --goal-request gr.json --align identity --integrate-cmd --out logs/fact3r_nav/chair_return

# (c) no map — exercise the GOTO loop toward a fixed odom point
./LAUNCH/launch_offline_fact3r_nav.sh --frames /path/walk/ --goal-odom-xy "4 -1.5" \
    --out logs/fact3r_nav/smoke
```

Pose stream: `--poses file.txt` (per-frame `x y theta`, or TUM rows) **or**
`--integrate-cmd` (dead-reckon the pipeline's `(v, w)` through a unicycle model at
`--dt`) — the closed-loop "did NavDP actually converge on the remembered point"
check when there's no external odometry. Depth: `--depth dir/` of per-frame
`.npy`/`.png`, else Depth-Anything-V2 metric runs in-pipeline.

The driver stops on `--reached-radius`, a pipeline `STOP` (obstacle-guard
arrival), frame exhaustion, or `--max-steps`. `"GOTO"` never self-declares STOP
from proximity — a remembered point can be stale from odom drift, so arrival is
the caller's call (here, `--reached-radius`).

## Environment / weights

* `NAV_ENV` (default `internnav`) needs `torch`, `cv2`, `transformers>=4.57`, and
  the Depth-Anything-V2 deps `nav_pipeline/` imports. Override `NAV_ENV` /
  `CONDA_SH` per machine. The fact3r resolve step runs in its **own** env
  (`--siglip-env`) via `conda run`.
* NavDP + Depth-Anything checkpoints go where the code already expects them —
  `navdp-cross-modal.ckpt` (default `--policy crossmodal`) or
  `checkpoints/navdp_extracted.pth` (`--policy extracted`), and
  `checkpoints/depth_anything_v2_*.pth`. See the repo README's weights table;
  symlink them in.

## Status

Frame math is unit-tested (`tests/test_fact3r_goal_source.py`, 11/11). The driver
is smoke-tested end to end (pipeline build → `NavDPCrossModal` load →
`step(external_goal=…)` → `"GOTO"` → dead-reckon → outputs) on synthetic frames
via `--goal-odom-xy` and a synthetic `--goal-request`. A run against a **real**
Fact3R-Map + matching return video has not been done here — it needs that data.
