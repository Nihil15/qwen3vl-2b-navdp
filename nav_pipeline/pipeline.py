"""DINO -> NavDP navigation pipeline for the 6WD rover.

Per frame:
  1. Grounding DINO detects the text target -> bbox.
  2. Depth: external (Isaac Sim / sensor) or Depth Anything V2 metric (RGB-only).
  3. bbox + depth + intrinsics -> 3D point goal in the robot frame.
  4. NavDP samples N candidate trajectories conditioned on the point goal
     (checkpoint-native convention: y right-positive — empirically verified,
     see scripts/diag_goal_conditioning.py).
  5. Trajectory selection: combined score of goal progress (endpoint heading /
     distance toward goal) and the NavDP critic (collision safety). This keeps
     the rover goal-directed even though the extracted checkpoint's learned
     point conditioning is weak.
  6. Stop when the depth-estimated goal point is within stop_distance.

Returns a (linear, angular) velocity command plus rich debug info.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch

from .dino_detector import Detection, GroundingDinoDetector
from .relational_target import (
    clean_label,
    parse_positional_target,
    parse_relational_target,
    select_by_position,
    select_by_relation,
)
from .goal_belief import GoalBelief
from .goal_utils import (
    goal_point_from_detection,
    intrinsics_from_fov,
    preprocess_depth,
    preprocess_rgb,
)
from .navdp_net import NavDPStandalone
from .obstacle_guard import (
    GuardConfig,
    apply_avoid_cooldown,
    depth_to_obstacle_points,
    forward_guard,
    swept_clearance,
)
from .scene_tagger import DEFAULT_VOCAB


def _bbox_to_pixel_grid(
    box: np.ndarray, width: int, height: int, image_size: int = 224,
    stride: int = 8, max_points: int = 400,
) -> np.ndarray:
    """Strided [u,v] pixel grid over a DINO box interior -- the named-obstacle
    footprint fed into S2Diff's pixel-obstacle avoidance (see
    tube_planner/pixel_obstacles.py). A bbox, not a SAM mask: cheap (no extra
    segmentation pass every tick) at the cost of including some background
    inside the box. Depth/finite validity is filtered server-side by
    PixelObstacleConfig -- this only needs pixels that land inside the image.

    ``box`` is in the ORIGINAL rgb frame (what self.detector ran on), but
    sample_pointgoal (see s2diff_http_client.py) sends the resize+center-pad
    letterboxed image_size x image_size frame goal_utils.preprocess_rgb/
    preprocess_depth actually build for NavDP's memory -- NOT the original
    resolution. Pixels left in original-frame coordinates land outside that
    224x224 image and the server 400s every request (an obstacle pixel
    outside its depth image). Reproduce preprocess_rgb's exact transform here
    so the returned pixels land correctly in the frame that gets sent."""

    prop = image_size / max(height, width)
    pad_w = max((image_size - round(width * prop)) // 2, 0)
    pad_h = max((image_size - round(height * prop)) // 2, 0)
    x0 = max(0, int(np.floor(box[0] * prop)) + pad_w)
    y0 = max(0, int(np.floor(box[1] * prop)) + pad_h)
    x1 = min(image_size - 1, int(np.ceil(box[2] * prop)) + pad_w)
    y1 = min(image_size - 1, int(np.ceil(box[3] * prop)) + pad_h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 2), dtype=np.int64)
    us = np.arange(x0, x1 + 1, stride)
    vs = np.arange(y0, y1 + 1, stride)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    pixels = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.int64)
    if pixels.shape[0] > max_points:
        idx = np.linspace(0, pixels.shape[0] - 1, max_points).astype(np.int64)
        pixels = pixels[idx]
    return pixels


def _bbox_exclude_mask(box: np.ndarray, height: int, width: int) -> np.ndarray:
    """Coarse (H,W) boolean mask, True inside a DINO box -- fallback obstacle-
    exclusion region for depth_to_obstacle_points when no SAM mask is
    available this tick (use_sam=False, SAM not due yet, cache IoU dropped,
    or CLIP rejected the crop). Cheaper and less precise than a real SAM
    mask (includes background inside the box), but strictly better than the
    prior behavior of excluding nothing at all, which let the goal object's
    own depth surface get read as an obstacle and trigger AVOID/trajectory
    vetoes against the rover's own target -- see the exclude_mask call site's
    comment in _step_inner."""
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(width, int(np.ceil(box[2])))
    y1 = min(height, int(np.ceil(box[3])))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _disc_mask(u: float, v: float, height: int, width: int, radius: int = 14) -> np.ndarray:
    """Small boolean disc mask around a pixel. Lets a Qwen instruction-
    grounded pixel goal (qwen_pixel_goal.py) reuse mask_median_depth's
    existing robust depth sampling instead of reading one noisy pixel, and
    doubles as res.mask so the obstacle guard excludes the goal's own
    surface the same way a DINO/SAM mask already does (see _step_inner's
    exclude_mask handling) -- otherwise the goal point could veto its own
    approach as an obstacle."""
    mask = np.zeros((height, width), dtype=bool)
    cu, cv = int(round(u)), int(round(v))
    r = max(int(radius), 1)
    y0, y1 = max(cv - r, 0), min(cv + r + 1, height)
    x0, x1 = max(cu - r, 0), min(cu + r + 1, width)
    if y0 >= y1 or x0 >= x1:
        return mask
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disc = (xs - cu) ** 2 + (ys - cv) ** 2 <= r * r
    mask[y0:y1, x0:x1][disc] = True
    return mask


def score_instruction_candidates(
    candidates: list, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    belief_mu: Optional[np.ndarray], belief_locked: bool, obstacle_pts: Optional[np.ndarray],
    guard_cfg: GuardConfig, semantic_weight: float, continuity_weight: float,
    continuity_scale_m: float, collision_weight: float, collision_scale_m: float = 0.5,
    far_lookahead_m: float = 0.0, depth_trust_horizon_m: float = 0.0,
) -> Optional[tuple]:
    """Score each qwen_pixel_goal.PixelGoal candidate's 3D goal (converted
    via depth) against semantic confidence, continuity with the current
    locked belief, and obstacle cost along a straight path to it; return
    (xyz, winning_candidate, disc_mask, far_mode) for the best-scoring
    candidate, or None if none scored. See PipelineConfig.qwen_max_
    candidates for the full score formula and rationale (arxiv 2605.19420,
    adapted from a dense heatmap to a handful of discrete VLM candidates).

    Long-range handling (far_lookahead_m > 0): the monocular metric depth
    model is unreliable past ~6-8m, so a grounded pixel whose depth reads
    None (a hole) or > depth_trust_horizon_m is NOT trusted as a metric
    range -- instead the goal is placed at far_lookahead_m along that
    pixel's bearing ray and far_mode=True is returned, so the caller drives
    the correct heading and re-grounds as it closes distance (rather than
    dropping the candidate, which stranded the rover coasting a stale lock).

    obstacle_pts, if given (even empty), are this tick's depth-derived
    obstacle points in robot frame (obstacle_guard.depth_to_obstacle_points'
    output) -- None skips the collision term entirely (cost 0 for every
    candidate), matching cfg.avoid_enabled=False's existing meaning
    elsewhere in this pipeline.
    """
    from .goal_utils import pixel_depth_to_point
    from .sam_segmenter import mask_median_depth

    H, W = depth.shape[:2]
    best = None
    best_score = -np.inf
    for cand in candidates:
        if not cand.in_view:
            continue
        disc = _disc_mask(cand.u, cand.v, H, W, radius=14)
        d = mask_median_depth(depth, disc)
        far_mode = False
        if far_lookahead_m > 0.0 and (
            d is None or (depth_trust_horizon_m > 0.0 and d > depth_trust_horizon_m)
        ):
            d = far_lookahead_m
            far_mode = True
        elif d is None:
            continue
        xyz = pixel_depth_to_point(cand.u, cand.v, d, fx, fy, cx, cy)

        sem = semantic_weight * float(cand.confidence)
        if belief_locked and belief_mu is not None:
            dist = float(np.linalg.norm(xyz[:2] - belief_mu[:2]))
            cont = continuity_weight * float(np.exp(-dist / max(continuity_scale_m, 1e-6)))
        else:
            cont = continuity_weight * 1.0   # nothing locked yet -- no penalty for a fresh acquisition

        collision_cost = 0.0
        if obstacle_pts is not None and obstacle_pts.shape[0] > 0:
            yaw = float(np.arctan2(xyz[1], xyz[0]))
            T = 10
            traj = np.stack([
                np.linspace(0.0, xyz[0], T, dtype=np.float32),
                np.linspace(0.0, xyz[1], T, dtype=np.float32),
                np.full(T, yaw, dtype=np.float32),
            ], axis=1)[None]
            clearance = float(swept_clearance(traj, obstacle_pts, guard_cfg)[0])
            # Normalized against collision_scale_m (default 0.5m), NOT
            # guard_cfg.slow_dist (3.2m, the far outer edge of the AVOID
            # blend zone elsewhere in this pipeline) -- real swept-clearance
            # values live in the 0-1m range (guard_cfg.margin=0.10m is
            # already "safe enough" to select a trajectory elsewhere in
            # _select_trajectory), so normalizing against slow_dist made
            # nearly every realistic clearance register as high cost,
            # swamping the semantic/continuity terms regardless of how
            # clear a path actually was (caught by testing before this
            # shipped -- a candidate with a genuinely clear 0.33m gap was
            # scored as costly as one with a hull-overlapping obstacle).
            if np.isfinite(clearance):
                collision_cost = float(np.clip(1.0 - clearance / max(collision_scale_m, 1e-6), 0.0, 1.0))

        score = sem + cont - collision_weight * collision_cost
        if score > best_score:
            best_score = score
            best = (xyz, cand, disc, far_mode)
    return best


@dataclass
class PipelineConfig:
    device: str = "cuda:0"
    horizontal_fov_deg: float = 90.0
    sample_num: int = 32
    # Grounding DINO: loaded unconditionally by default (this class's
    # original purpose). Set False only for a pure Qwen-instruction setup
    # (use_qwen_instruction=True) with no plans to ever pass avoid_text or a
    # plain text target to step() -- doing either with this off raises
    # instead of silently doing nothing. Saves ~1GB+ VRAM and load time by
    # not loading a model that mode never calls. use_scene_tagger requires
    # this True (checked at construction, not per-tick).
    use_dino: bool = True
    # policy backend:
    #   "crossmodal" - official standalone NavDP (navdp-cross-modal.ckpt),
    #                  strong trained point-goal conditioning, y-left convention
    #   "extracted"  - NavDP weights extracted from InternVLA-N1-w-NavDP,
    #                  weak conditioning, y-right convention (sign-flipped here)
    policy_type: str = "crossmodal"
    # trajectory selection
    critic_weight: float = 1.0       # safety weight vs goal progress
    critic_floor: float = -10.0      # samples below this critic value are discarded
    #   (for "extracted" use 0.16 — its critic lives in a narrow 0.17 band)
    critic_keep_frac: float = 0.5    # among clearance-safe samples, keep top fraction by critic
    # depth-based obstacle guard (hard, model-agnostic)
    avoid_enabled: bool = True
    guard: GuardConfig = field(default_factory=GuardConfig)
    # command shaping (rover caps, matching the old OmniVLA node limits)
    invert_angular: bool = False     # flip outgoing turn direction (real-rover wiring)
    kp: float = 1.0
    kp_angular: float = 2.2          # heading-error -> angular gain (saturates at max_angular)
    urgency_gain: float = 2.0        # extra angular gain as obstacles close in (< slow_dist)
    max_linear: float = 0.15
    max_angular: float = 0.25
    waypoint_index: int = 8          # look-ahead waypoint on the chosen trajectory
    # ── proven real-rover mechanics (ported from the OmniVLA node) ──
    # Steering in open space is a DETERMINISTIC visual servo on the detection
    # bearing (stable tick to tick), not the resampled diffusion heading.
    servo_deadband: float = 0.05     # |bearing| below this -> drive straight (OmniVLA value)
    ang_min_cmd: float = 0.12        # stiction floor: minimum |angular| outside the deadband —
    #                                  the 6WD skid-steer won't yaw at all on smaller commands
    servo_ramp_deg: float = 35.0     # heading error (past the deadband) at which the smooth
    #                                  bearing->angular curve reaches ~96% of max_angular -- the
    #                                  width of the actual proportional steering range (see
    #                                  bearing_to_angular; a plain linear-gain-then-clip-and-floor
    #                                  controller saturates within a few degrees, which reads as
    #                                  the rover snapping to full-rate turns instead of easing in)
    smoothing: float = 0.5           # cross-tick EMA on (lin, ang); 0 = off
    angular_slew_max: float = 0.10   # hard cap on |Δangular| per tick (rad/s), applied to the
    #                                  FINAL command in every state (unlike smoothing above, which
    #                                  skips STOP/SEARCH) -- bounds how fast the turn rate itself can
    #                                  change tick to tick, e.g. the snap when entering/leaving SEARCH
    #                                  or AVOID's full-authority turn. 0 = off. At predict_hz≈3, 0.10
    #                                  reaches max_angular (0.25 real-rover default) in ~3 ticks.
    # ESP32 6WD rover only (see clear_wheel_deadband) -- False by default
    # since PipelineConfig is also used by sim GUIs and the Hiwonder
    # backend, where this hardware-specific correction doesn't apply.
    wheel_deadband_correction: bool = False
    track_width_m: float = 0.345          # must match esp32/rover_6wd_complete.ino's TRACK_WIDTH_M
    wheel_vel_deadband_ms: float = 0.03   # must match esp32/rover_6wd_complete.ino's VEL_DEADBAND_MS
    avoid_confirm_ticks: int = 2     # consecutive guard hits before AVOID engages
    avoid_cooldown_ticks: int = 8    # keep biasing steering away from the escape side for this many
    #                                  more ticks after AVOID releases, so the rover actually clears the
    #                                  obstacle's lateral footprint before goal-bearing servo resumes --
    #                                  otherwise it snaps straight back at the just-avoided obstacle and
    #                                  oscillates AVOID/TRACK in place (see apply_avoid_cooldown)
    avoid_bias_gain: float = 0.15    # rad/s added toward the escape side during the cooldown window
    # SAM 2.1 segmentation layer (DINO bbox -> instance mask -> goal)
    use_sam: bool = True
    sam_period_s: float = 1.0        # min seconds between SAM re-segmentations (real-rover
    #                                  latency: DINO + bbox-goal still run every tick; the
    #                                  mask is cached and reused in between as long as the
    #                                  current DINO box still overlaps the one SAM last saw)
    mask_stop_frac: float = 0.20     # mask area fraction of image -> stop (unused by this real-rover
    #                                  pipeline's STOP decision, which is depth-only; kept for the
    #                                  MARS/EARTH sim GUIs that still read it)
    # CLIP verification of the SAM-segmented crop against the target phrase,
    # paired 1:1 with each SAM re-segmentation (see sam_period_s). Catches a
    # confident DINO false positive before the rover commits to a goal: a
    # crop scoring below threshold is treated as no detection this tick.
    use_clip: bool = True
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_min_similarity: float = 0.5  # softmax prob vs generic negatives; empirical starting point
    # Periodic open-vocabulary scene inventory (independent of the goal
    # target): "what's in the frame right now", counted per label and logged
    # for later route-finding. Reuses the DINO detector already loaded above.
    use_scene_tagger: bool = False
    scene_tag_period_s: float = 1.0
    scene_vocab: List[str] = field(default_factory=lambda: list(DEFAULT_VOCAB))
    scene_log_dir: str = "scene_log"   # one JSONL file per run; set to None/"" to disable file logging
    # RGB-only depth (no depth sensor): "vits" (default, fast) or "vitb"
    # (more accurate, ~2x slower) -- see depth_estimator.py's _ENCODER_CONFIGS
    depth_encoder: str = "vits"
    # goal / stopping
    stop_distance: float = 1.5       # meters from object at which to stop (depth-only; no pixel/mask-size fallback)
    bbox_stop_frac: float = 0.55     # bbox height fraction of image -> stop; same as mask_stop_frac,
    #                                  unused here now, kept for the sim GUIs
    detect_score_min: float = 0.3
    # target loss behavior
    search_angular: float = 0.15     # spin to re-acquire when target not seen
    lost_patience: int = 5           # frames to keep last goal before searching
    # Qwen2.5-VL "ghost pixel" search guidance (see qwen_search_guide.py):
    # while SEARCH is active (DINO can't currently see target_text), a frozen
    # Qwen2.5-VL-7B-Instruct points at the pixel it thinks is the best
    # direction to look/move next (a doorway, opening, hallway...), and that
    # replaces the fixed spin-toward-last-known-side. DINO keeps detecting
    # every tick regardless (_step_inner calls it unconditionally); the
    # instant it reacquires the target this branch is no longer entered, so
    # Qwen is dropped automatically rather than explicitly switched off.
    use_qwen_search: bool = False
    qwen_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # 4-bit (bitsandbytes) measured at ~6.2GB VRAM vs fp16's ~16.6GB, on top
    # of the rest of this pipeline's ~2.5GB (DINO+SAM+depth+NavDP+CLIP+
    # DINOv2) -- ~8.7GB total, comfortable headroom on a 24GB card. Load
    # time (~21s) and per-call latency (~0.7-1.0s) are close to fp16's;
    # the pointed pixel shifts slightly (still lands on-target, e.g. still
    # inside the correct DINO box on a repeat test) -- normal quantization
    # variance, not a miss. bitsandbytes is a hard requirement for this
    # default; set False if it's unavailable in a given env.
    qwen_load_in_4bit: bool = True
    qwen_search_period_s: float = 2.0  # wall-clock throttle -- 7B inference
    #                                    takes seconds, not one pipeline tick
    # Qwen2.5-VL instruction-grounded PRIMARY goal source (see
    # qwen_pixel_goal.py) -- distinct from use_qwen_search above, which only
    # ever steers SEARCH while chasing a DINO *text target*. Pass a free-text
    # instruction to step()'s `instruction` argument to use this instead of
    # DINO target detection entirely: Qwen points at the next waypoint pixel,
    # that becomes NavDP's goal via depth, same as a DINO detection would.
    use_qwen_instruction: bool = False
    qwen_instruction_period_s: float = 1.5  # wall-clock throttle between Qwen
    #  calls; GoalBelief propagates the goal by ego-motion in between -- the
    #  same belief-coasting a lost DINO detection already falls back to.
    # Consistency gate: DINO's target has box-IoU + DINOv2 appearance-reid
    # (see _select_detection) rejecting a same-class-but-different object
    # BEFORE it ever reaches belief.observe(), which itself just snaps to
    # whatever it's given (see GoalBelief.observe's docstring). Qwen's
    # instruction grounding has no such box/embedding to check -- a fresh
    # query has zero memory of the last one, so if a SECOND door enters
    # frame (e.g. mid obstacle-avoidance turn), it can get grounded and the
    # snap-to-measurement update drags the goal onto it outright. This caps
    # how far (meters) a new grounding may sit from the current CONFIDENT
    # belief (sigma <= belief_max_sigma) before being rejected as "probably
    # a different door" and coasted through instead of accepted -- the same
    # "reject unless genuinely reacquiring" shape as DINO's lost_patience +
    # appearance gate, just using position instead of appearance as the
    # continuity test. Once sigma actually exceeds belief_max_sigma (this
    # pipeline would already give up and SEARCH), the gate stops applying --
    # that's a legitimate fresh acquisition, not a distractor.
    qwen_goal_consistency_m: float = 1.5
    # Multi-candidate, obstacle-aware grounding (arxiv 2605.19420, "Beyond
    # Waypoints: Dual-Heatmap Grounding" -- adapted here to a frozen VLM
    # that can only be prompted for discrete points, not a dense heatmap
    # network). Instead of trusting Qwen's single best guess outright, ask
    # for up to qwen_max_candidates ranked points (see qwen_pixel_goal.py's
    # MULTI_CANDIDATE_PROMPT) and score each as
    #   semantic_weight * confidence
    #   + continuity_weight * exp(-dist_from_locked_belief / continuity_scale_m)
    #   - collision_weight  * clip(1 - swept_clearance/guard.slow_dist, 0, 1)
    # picking the best-scoring candidate as this tick's grounding, which
    # THEN still passes through the qwen_goal_consistency_m gate above as a
    # final check. A distractor door doesn't need a hard reject anymore --
    # it can lose on points to a candidate that's both closer to the
    # current lock AND has a clearer path, the same joint semantic/
    # navigability tradeoff the paper's score makes, just over a handful of
    # discrete candidates instead of a continuous field. 1 candidate
    # (max_candidates=1) degrades to the old single-point behavior with no
    # scoring to do.
    qwen_max_candidates: int = 3
    qwen_candidate_semantic_weight: float = 1.0
    qwen_candidate_continuity_weight: float = 1.0
    qwen_candidate_continuity_scale_m: float = 1.0   # e^-1 at this many meters off-lock
    qwen_candidate_collision_weight: float = 2.0     # weighted heavier -- safety, not preference
    qwen_candidate_collision_scale_m: float = 0.5    # clearance=0 -> cost 1, clearance>=this -> cost 0
    #   (deliberately NOT guard.slow_dist -- see score_instruction_candidates's comment on why)
    # Long-range grounding: the monocular metric depth model is unreliable
    # past ~6-8m, so a grounded pixel whose depth is missing (a hole) or
    # reads farther than qwen_depth_trust_horizon_m is treated as a BEARING
    # only -- the goal is placed qwen_far_lookahead_m along that ray and the
    # consistency gate is skipped for it, so the rover drives the right
    # heading and re-grounds as it closes in, instead of rejecting every
    # jittery far reading and coasting a stale lock forever. 0 disables
    # (revert to the old "drop the candidate if depth is missing" behavior).
    qwen_far_lookahead_m: float = 3.5
    qwen_depth_trust_horizon_m: float = 6.0
    # Ego-motion goal belief (see goal_belief.py): while the target isn't
    # detected, propagate the goal by the rover's own dead-reckoned motion
    # instead of coasting on a frozen last-seen point. False reverts to the
    # exact pre-belief behavior (goal frozen at self._last_goal). Coasting
    # duration when belief is ON is NOT bounded by lost_patience below --
    # sigma crossing belief_max_sigma is the only cutoff (see the "lost_patience
    # is NOT applied here on purpose" comment at its use site) -- lost_patience
    # only still applies to the belief-OFF fallback path.
    use_belief_goal: bool = True
    belief_sigma_init: float = 1000.0
    belief_sigma_visible: float = 0.05
    belief_odom_noise: float = 0.0015
    # see goal_belief.py's module docstring for where this starting point
    # comes from (real spin-accuracy trials, not a physical constant)
    #
    # belief_rot_noise_gain=0.35 was tuned against belief_max_sigma=1.0 (budget
    # 0.95 above sigma_visible), crossing distrust at 0.95/0.35 ~= 155deg --
    # inside the trial-backed 135-165deg range. The 2026-08-07 Hiwonder commit
    # cut belief_max_sigma to 0.15 (budget 0.10) without rescaling this, which
    # silently dropped the crossing angle to 0.10/0.35 ~= 16deg for EVERY
    # backend (not just Hiwonder) -- belief gave up and reverted to SEARCH
    # after barely any rotation instead of the intended ~135-165deg. Rescaled
    # by the same budget ratio (0.10/0.95) to restore the original ~155deg
    # crossing point: 0.35 * (0.10/0.95) ~= 0.037.
    belief_rot_noise_gain: float = 0.037
    belief_max_sigma: float = 0.15    # above this, distrust belief and go to SEARCH
    belief_decay_factor: float = 0.95
    # Target-ambiguity detection: a bare category prompt ("chair") is
    # genuinely underspecified when several same-class objects score
    # comparably -- there's no visual evidence for which one the operator
    # meant, so a "top-score wins" pick is a guess, not a decision. Flag it
    # (StepResult.ambiguous/candidate_count) instead of picking silently.
    # Empirical starting point; detector scores here typically span
    # ~0.3-0.95, so 0.15 counts two candidates within ~a third of that range
    # of each other as "comparably confident."
    ambiguity_score_margin: float = 0.15
    # Appearance re-identification for the TRACKED detection (distinct from
    # use_clip above, which verifies goal-text match, not instance identity).
    # DINOv2, not CLIP: CLIP embeddings are global/class-level and measured
    # unusable for this -- solid red vs. solid blue crops scored 0.93 cosine
    # similarity, an almost-flat band with no headroom to threshold in.
    # DINOv2 crop embeddings measured a real gap on rover-camera crops: same
    # object ~1.0, the HARDEST case (two visually near-identical doors, the
    # closest real analog to two identical chairs) ~0.77, clearly different
    # objects ~0.40. A same-class candidate must clear
    # appearance_min_similarity against the cached locked-object embedding to
    # be accepted as a continuation of the current track, geometric IoU
    # match or not -- otherwise it's rejected (a miss), not blindly taken.
    use_appearance_reid: bool = True
    dinov2_model_id: str = "facebook/dinov2-small"
    appearance_min_similarity: float = 0.85  # sits above the measured 0.77 near-identical-twin score
    # Very high IoU against the tracked reference box is itself overwhelming
    # positional evidence of identity -- two DISTINCT physical objects
    # essentially never land within this much overlap of the previous
    # frame's box. Bypass the appearance check above this IoU: real-rover
    # test 2026-07-31 ("chair near door") measured a genuine same-object
    # re-acquisition (iou=0.88) scoring only sim=0.698 once the chair had
    # gotten smaller/farther in frame (box_h_frac 0.18) -- DINOv2 crop
    # embeddings aren't scale/distance-invariant enough to stay above
    # appearance_min_similarity for a real match once apparent size changes,
    # which was repeatedly rejecting valid re-locks and cycling into SEARCH.
    appearance_iou_override: float = 0.6


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x0, y0, x1, y1] boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _ambiguous_candidates(dets: list, margin: float) -> int:
    """Count detections that share the top-scoring label and sit within
    `margin` of its score -- genuinely comparable candidates for the same
    category, not just detector noise around a single clear winner. dets
    must be score-sorted descending (detect()'s contract); count includes
    the top detection itself, so >=2 means the target is ambiguous."""
    if not dets:
        return 0
    top = dets[0]
    return sum(1 for d in dets if d.label == top.label and (top.score - d.score) <= margin)


def _estimate_motion_affine(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
    """Camera motion between two frames as a 2x3 affine (rotation/scale/
    translation, no shear), estimated from sparse optical flow + RANSAC --
    same technique as BoT-SORT's camera motion compensation (CMC).

    Track a scatter of distinctive points (goodFeaturesToTrack) from prev
    into curr (Lucas-Kanade optical flow), then fit the single affine
    transform that explains most of those point motions (estimateAffine-
    Partial2D's built-in RANSAC). Points sitting on independently-moving
    objects (a person walking, a nudged chair) get rejected as outliers
    automatically, leaving a clean estimate of pure camera rotation/pan.

    Returns None if there aren't enough reliable points to trust (e.g. a
    blank wall) -- caller should fall back to no compensation.
    """
    pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts0 is None or len(pts0) < 10:
        return None
    pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None)
    if pts1 is None or status is None:
        return None
    keep = status.reshape(-1).astype(bool)
    pts0, pts1 = pts0[keep], pts1[keep]
    if len(pts0) < 10:
        return None
    M, inliers = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < 8:
        return None
    return M


def _warp_box(box: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to a [x0,y0,x1,y1] box: warp all 4 corners (not
    just the center) and take the new axis-aligned bounding box, so a
    rotation component reshapes the box instead of only sliding it."""
    corners = np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped = cv2.transform(corners, M).reshape(-1, 2)
    x0, y0 = warped[:, 0].min(), warped[:, 1].min()
    x1, y1 = warped[:, 0].max(), warped[:, 1].max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def bearing_to_angular(bearing: float, max_angular: float, ang_min_cmd: float,
                       deadband: float, ramp_rad: float) -> float:
    """Smooth, monotonic heading-error -> angular-velocity command.

    Below `deadband`, commands 0 (drive straight). Above it, ramps
    CONTINUOUSLY via tanh from `ang_min_cmd` (the minimum that actually
    overcomes the rover's yaw stiction) up toward `max_angular` as the
    heading error grows, reaching ~96% of max_angular by `ramp_rad` past
    the deadband.

    This replaces a linear gain (kp_angular * bearing) that gets clipped to
    max_angular and then has a stiction floor/boost added on top after the
    fact: with kp_angular=2.2 and max_angular=0.25 (real-rover defaults),
    that combination saturates within about 5 degrees of heading error, and
    the floor/boost addition makes the command jump straight from 0 to
    ~64% of max_angular the instant the deadband is crossed -- there is
    almost no actual proportional range, so steering reads as snapping
    between "straight" and "turning at full rate" instead of easing in
    gradually with how far off the target is. Baking the floor into a smooth
    saturation curve instead gives a real, wide, continuous ramp.
    """
    mag = abs(bearing)
    if mag < deadband:
        return 0.0
    span = max(max_angular - ang_min_cmd, 0.0)
    x = 2.0 * (mag - deadband) / max(ramp_rad, 1e-6)   # tanh(2) ~= 0.964
    magnitude = min(ang_min_cmd + span * float(np.tanh(x)), max_angular)
    return float(np.copysign(magnitude, bearing))


def clear_wheel_deadband(linear: float, angular: float, track_width_m: float,
                         deadband: float, margin: float = 1.15) -> tuple:
    """Boost (linear, angular) just enough that BOTH differential-drive
    wheel targets clear the ESP32 rover firmware's hard per-wheel PWM
    deadband (esp32/rover_6wd_complete.ino's VEL_DEADBAND_MS -- closed_loop_
    pwm() hard-zeros any |target| below it: "stop request -> no windup").

    drive_rover() there mixes wheels as
        left_target  = linear_x - angular_z * TRACK_WIDTH_M / 2
        right_target = linear_x + angular_z * TRACK_WIDTH_M / 2
    search_angular already has its own floor for PURE rotation (see
    launch_rover.sh: 0.174 rad/s clears VEL_DEADBAND_MS=0.03 at
    TRACK_WIDTH_M=0.345 with zero margin; 0.18 is what's actually used).
    That alone doesn't cover a COMBINED small linear + small angular:
    mixing can put BOTH wheel targets under the deadband even when neither
    linear nor angular individually looks like zero -- e.g. linear=0.02,
    angular=0.0 gives left=right=0.02 < 0.03, so the firmware silently
    commands zero PWM on both sides while the pipeline reports it's
    driving. That's the "given nonzero linear/angular, rover doesn't move"
    symptom this fixes.

    A wheel target already >= deadband is left untouched. A wheel target
    that's exactly 0.0 is left untouched too -- a deliberate "this wheel
    shouldn't move" case (a true pivot), not the stall this targets. Only a
    NONZERO-but-below-deadband wheel target gets bumped up to
    deadband*margin (same "close to the floor with a bit of headroom"
    shape as search_angular's 0.174->0.18), keeping its sign; left/right
    are then converted back to (linear, angular).

    Hardware-specific to the ESP32 6WD rover -- do not apply for
    --hiwonder (different chassis/firmware entirely, see launch_rover.sh's
    comment on why search_angular's floor doesn't carry over either) or
    any sim backend.
    """
    if linear == 0.0 and angular == 0.0:
        return 0.0, 0.0   # genuine stop/hold -- never inject motion
    k = angular * track_width_m / 2.0
    left, right = linear - k, linear + k
    floor = deadband * margin

    def _bump(v: float) -> float:
        return float(np.copysign(floor, v)) if 0.0 < abs(v) < deadband else v

    left, right = _bump(left), _bump(right)
    return (left + right) / 2.0, (right - left) / track_width_m


@dataclass
class StepResult:
    linear: float = 0.0
    angular: float = 0.0
    state: str = "SEARCH"            # SEARCH | TRACK | GOTO | AVOID | STOP
    detection: Optional[object] = None
    ambiguous: bool = False           # >=2 comparably-scored same-class candidates in view
    candidate_count: int = 0          # how many (see PipelineConfig.ambiguity_score_margin)
    mask: Optional[np.ndarray] = None
    goal_point: Optional[np.ndarray] = None
    trajectory: Optional[np.ndarray] = None
    all_trajectories: Optional[np.ndarray] = None
    critic: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    min_forward: float = float("inf")
    qwen_bearing: Optional[float] = None  # radians, +left -- set only when SEARCH used Qwen guidance
    qwen_pixel_goal: Optional[tuple] = None  # (u, v) -- set only on ticks Qwen's instruction-grounded goal fired
    avoid_detection: Optional[object] = None  # this tick's named-obstacle DINO box, if avoid_text set
    avoid_pixels: Optional[np.ndarray] = None  # [N,2] u,v pixels covering it -- see _bbox_to_pixel_grid
    timing: dict = field(default_factory=dict)


class DinoNavDPPipeline:
    def __init__(self, cfg: PipelineConfig = PipelineConfig(), use_depth_estimator: bool = True):
        self.cfg = cfg
        t0 = time.time()
        self.detector = (GroundingDinoDetector(device=cfg.device, box_threshold=cfg.detect_score_min)
                          if cfg.use_dino else None)
        if cfg.use_scene_tagger and not cfg.use_dino:
            raise ValueError("use_scene_tagger requires use_dino=True (SceneTagger wraps the "
                              "Grounding DINO detector directly)")
        if cfg.policy_type == "crossmodal":
            from .navdp_crossmodal import NavDPCrossModal

            self.policy = NavDPCrossModal.load(device=cfg.device)
            self._goal_y_sign = 1.0    # checkpoint uses ROS y-left directly
        else:
            self.policy = NavDPStandalone.load(device=cfg.device)
            self._goal_y_sign = -1.0   # embedded checkpoint uses y-right
        self._memory_size = self.policy.memory_size
        self.segmenter = None
        if cfg.use_sam:
            from .sam_segmenter import Sam2Segmenter

            self.segmenter = Sam2Segmenter(device=cfg.device)
        self.clip = None
        if cfg.use_sam and cfg.use_clip:
            from .clip_verifier import ClipVerifier

            self.clip = ClipVerifier(model_id=cfg.clip_model_id, device=cfg.device)
        self.reid = None
        if cfg.use_appearance_reid:
            from .dinov2_embedder import Dinov2Embedder

            self.reid = Dinov2Embedder(model_id=cfg.dinov2_model_id, device=cfg.device)
        self._qwen_scheduler = None
        if cfg.use_qwen_search:
            from .qwen_search_guide import QwenSearchScheduler, QwenVLSearchGuide

            qwen_guide = QwenVLSearchGuide(
                model_id=cfg.qwen_model_id, device=cfg.device,
                load_in_4bit=cfg.qwen_load_in_4bit, horizontal_fov_deg=cfg.horizontal_fov_deg,
            )
            qwen_guide._ensure_loaded()  # eager, like every other model below --
            # otherwise the ~19s weight load lands on the first real mid-drive
            # SEARCH tick instead of here at startup (measured: internnav env,
            # fp16, 3090 Ti).
            self._qwen_scheduler = QwenSearchScheduler(qwen_guide, period_s=cfg.qwen_search_period_s)
        self._qwen_pixel_guide = None
        self._qwen_instruction_gate = None
        if cfg.use_qwen_instruction:
            from .qwen_pixel_goal import QwenInstructionGate, QwenVLPixelGoal

            self._qwen_pixel_guide = QwenVLPixelGoal(
                model_id=cfg.qwen_model_id, device=cfg.device, load_in_4bit=cfg.qwen_load_in_4bit,
            )
            self._qwen_pixel_guide._ensure_loaded()  # eager, same reasoning as use_qwen_search above
            self._qwen_instruction_gate = QwenInstructionGate(period_s=cfg.qwen_instruction_period_s)
        self.scene_tagger = None
        if cfg.use_scene_tagger:
            from .scene_tagger import SceneTagger

            self.scene_tagger = SceneTagger(self.detector, vocab=cfg.scene_vocab)
        self.scene_log: list = []   # [{"t": float, "objects": {label: count}}, ...]
        self._last_tag_t = 0.0
        self._scene_log_file = None
        if self.scene_tagger is not None and cfg.scene_log_dir:
            os.makedirs(cfg.scene_log_dir, exist_ok=True)
            fname = f"scene_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            self._scene_log_file = open(os.path.join(cfg.scene_log_dir, fname), "a")
        self.depther = None
        if use_depth_estimator:
            from .depth_estimator import MetricDepthEstimator

            self.depther = MetricDepthEstimator(encoder=cfg.depth_encoder, device=cfg.device)
        print(f"[pipeline] models loaded in {time.time() - t0:.1f}s")

        self.belief = GoalBelief(
            sigma_init=cfg.belief_sigma_init, sigma_visible=cfg.belief_sigma_visible,
            odom_noise=cfg.belief_odom_noise, rot_noise_gain=cfg.belief_rot_noise_gain,
            decay_factor=cfg.belief_decay_factor,
        )
        self._prev_pose: Optional[tuple] = None   # (x, y, theta) last tick, for odom_delta

        self._memory: list = []      # last processed RGB frames (max 2)
        self._memory_d: list = []
        self._lost_count = 0
        self._last_goal: Optional[np.ndarray] = None
        self._last_box: Optional[np.ndarray] = None
        self._box_miss_count = 0
        self._avoid_streak = 0
        self._avoid_side = 0.0
        self._avoid_cooldown = 0
        self._prev_cmd = (0.0, 0.0)
        self._last_sam_t = 0.0
        self._last_mask: Optional[np.ndarray] = None
        self._last_mask_box: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._last_candidate_count = 0
        self._ambiguity_warned = False
        self._locked_embed: Optional[np.ndarray] = None
        self._avoid_no_dino_warned = False  # print once per pipeline lifetime, not per reset()

    def reset(self):
        self.belief.reset()
        if self._qwen_scheduler is not None:
            self._qwen_scheduler.reset()
        if self._qwen_instruction_gate is not None:
            self._qwen_instruction_gate.reset()
        self._prev_pose = None
        self._memory, self._memory_d = [], []
        self._lost_count = 0
        self._last_goal = None
        self._last_box = None
        self._box_miss_count = 0
        self._avoid_streak = 0
        self._avoid_side = 0.0
        self._avoid_cooldown = 0
        self._last_sam_t = 0.0
        self._last_mask = None
        self._last_mask_box = None
        self._prev_gray = None
        self._last_candidate_count = 0
        self._ambiguity_warned = False
        self._locked_embed = None

    # weight given to a newly matched box when updating the tracked-box
    # reference. With several same-class objects close together, adjacent
    # boxes can each have decent overlap with the reference, and per-frame
    # detector jitter can flip which one "wins" the overlap comparison. If
    # the reference snaps to the raw winner every frame, one flip teleports
    # it onto the neighbor and biases the *next* comparison the same way --
    # a ping-pong between the two objects' goal points that reads as
    # continuous "TRACK" while the rover oscillates toward each in turn.
    # Blending instead of snapping means a single noisy frame only nudges
    # the reference a little, so the genuinely-locked object (which wins
    # most frames, not just one) keeps dominating it.
    _BOX_TRACK_ALPHA = 0.3
    # slower than _BOX_TRACK_ALPHA on purpose: the locked-object embedding
    # should drift with gradual viewing-angle changes but stay dominated by
    # the appearance history, not the newest crop (which is exactly what a
    # wrong-object steal would try to overwrite it with).
    _EMBED_TRACK_ALPHA = 0.15

    def _select_detection(self, dets: list, motion: Optional[np.ndarray] = None,
                          rgb: Optional[np.ndarray] = None,
                          initial_pick: Optional[Detection] = None) -> Optional[Detection]:
        """Pick which detection to track this frame.

        detect() returns boxes sorted by score, but with several instances of
        the same class in view, the top score flips between different physical
        objects frame-to-frame (viewing angle, partial occlusion). Each flip
        yanks the 3D goal to a different object and the rover lurches toward
        it. Prefer whichever box overlaps the previously tracked one (a
        greedy nearest-neighbor tracker) so the rover keeps heading toward
        whatever it first locked onto -- NOT whatever currently scores/looks
        closest. A single frame with no overlap (occlusion, edge clipping)
        doesn't concede the lock: it coasts (returns None, same as "not
        detected this frame") for up to lost_patience frames before falling
        back to a fresh top-score pick, so a momentary drop-out can't hand
        the lock to a different, often physically closer, object.

        motion (optional 2x3 affine, from _estimate_motion_affine -- no-op,
        exact prior behavior, if None): the reference box lives in the
        PREVIOUS frame's pixel space. If the camera panned/rotated since
        then, the same physical object shifts in the image even though
        nothing about the detection changed, and raw IoU against a stale
        reference can drop below the match threshold for no real reason.
        Warp the reference by the estimated camera motion before matching,
        so a same-object match under rotation isn't lost.

        rgb (optional -- no-op, exact prior behavior, if missing or
        use_appearance_reid is off): geometric IoU alone isn't enough in a
        room with several of the target class close together -- a
        NEIGHBORING same-class object can overlap the reference box on an
        ordinary tick (no miss ever counted, box never even coasts) and
        silently steal the lock outright, especially as the rover approaches
        a cluster and boxes drift/grow. Gate BOTH the ordinary IoU match and
        the lost_patience fallback on appearance (DINOv2 crop embedding
        cosine similarity, see PipelineConfig.appearance_min_similarity):
        the winning candidate must also look like the object we locked onto,
        or it's rejected (counted as a miss) instead of blindly accepted.

        initial_pick (optional): which detection to lock onto FIRST, overriding
        the plain top-score default -- e.g. from select_by_relation() when the
        target phrase named an anchor ("chair near the door"). Only consulted
        on the very first lock; once tracking, the same IoU/CMC/appearance
        machinery above takes over regardless of how the initial pick was made.
        """
        self._last_candidate_count = _ambiguous_candidates(dets, self.cfg.ambiguity_score_margin)

        if self._last_box is None:
            self._box_miss_count = 0
            det = initial_pick if initial_pick is not None else (dets[0] if dets else None)
            self._last_box = det.box if det is not None else None
            self._locked_embed = self._embed_or_none(rgb, det.box) if det is not None else None
            return det

        ref = self._last_box if motion is None else _warp_box(self._last_box, motion)

        best, best_iou = None, 0.0
        for d in dets:
            iou = _box_iou(d.box, ref)
            if iou > best_iou:
                best, best_iou = d, iou
        if best is not None and best_iou > 0.1:
            emb = self._embed_or_none(rgb, best.box)
            sim = float(np.dot(emb, self._locked_embed)) if (emb is not None and self._locked_embed is not None) else None
            appearance_ok = (best_iou >= self.cfg.appearance_iou_override
                             or sim is None or sim >= self.cfg.appearance_min_similarity)
            if appearance_ok:
                self._box_miss_count = 0
                a = self._BOX_TRACK_ALPHA
                self._last_box = a * best.box + (1 - a) * ref
                if emb is not None:
                    self._locked_embed = self._blend_embed(self._locked_embed, emb)
                return best
            size_frac = (best.box[3] - best.box[1]) / rgb.shape[0] if rgb is not None else float("nan")
            print(f"[reid-debug] REJECTED sim={sim:.3f} < {self.cfg.appearance_min_similarity} "
                  f"iou={best_iou:.2f} box_h_frac={size_frac:.2f} miss_count={self._box_miss_count + 1}")
        elif dets:
            print(f"[reid-debug] NO-IOU-MATCH best_iou={best_iou:.2f} miss_count={self._box_miss_count + 1} "
                  f"(occlusion/edge-clip/lock lost, not a reid rejection)")
        self._box_miss_count += 1
        if self._box_miss_count > self.cfg.lost_patience:
            self._box_miss_count = 0
            det = self._select_by_appearance(dets, rgb) if (dets and self._locked_embed is not None) \
                else (dets[0] if dets else None)
            self._last_box = det.box if det is not None else self._last_box
            if det is not None:
                new_embed = self._embed_or_none(rgb, det.box)
                if new_embed is not None:
                    self._locked_embed = new_embed
            return det
        return None

    def _embed_or_none(self, rgb: Optional[np.ndarray], box: np.ndarray) -> Optional[np.ndarray]:
        if self.reid is None or rgb is None:
            return None
        return self.reid.embed(rgb, box)

    def _blend_embed(self, prior: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        if prior is None:
            return new
        e = self._EMBED_TRACK_ALPHA
        blended = e * new + (1 - e) * prior
        return blended / max(np.linalg.norm(blended), 1e-6)

    def _select_by_appearance(self, dets: list, rgb: Optional[np.ndarray]) -> Optional[Detection]:
        """Reacquisition tiebreak: among same-class candidates with no spatial
        continuity left to lean on, prefer whichever one's crop looks most
        like the object we had locked before losing it (DINOv2 cosine
        similarity), instead of blind top detector score.

        Must still clear appearance_min_similarity -- picking whichever
        candidate is merely LEAST dissimilar (with no floor) means once the
        original object is gone for good, this happily locks onto a
        DIFFERENT physical instance of the same class instead of continuing
        to search for the one first locked onto. Returns None (refuse to
        reacquire, caller falls through to SEARCH) if nothing clears it.
        """
        if self.reid is None or rgb is None or not dets:
            return None
        best, best_sim = None, -1.0
        for d in dets:
            emb = self.reid.embed(rgb, d.box)
            if emb is None:
                continue
            sim = float(np.dot(emb, self._locked_embed))
            if sim > best_sim:
                best, best_sim = d, sim
        if best is None or best_sim < self.cfg.appearance_min_similarity:
            return None
        return best

    # ------------------------------------------------------------------ #
    def _select_trajectory(self, trajs: np.ndarray, critic: np.ndarray, goal: np.ndarray,
                           clearances: Optional[np.ndarray] = None):
        """Score = goal progress − collision risk. goal is [x fwd, y left].

        Hard vetoes first: trajectories whose clearance to depth obstacle
        points is below guard.clearance are discarded, then the bottom
        (1 - critic_keep_frac) by critic among survivors. Goal progress only
        chooses among what survived. If nothing is safe, the max-clearance
        sample wins outright.
        """
        endpoints = trajs[:, -1, :2]                          # (N, 2) x fwd, y left
        goal_xy = goal[:2]
        goal_dist = np.linalg.norm(goal_xy) + 1e-6
        # progress: how much closer the endpoint gets to the goal, normalized
        progress = (goal_dist - np.linalg.norm(endpoints - goal_xy, axis=1)) / goal_dist
        # heading alignment of the early trajectory segment toward the goal
        early = trajs[:, min(4, trajs.shape[1] - 1), :2]
        early_norm = np.linalg.norm(early, axis=1) + 1e-6
        align = (early @ goal_xy) / (early_norm * goal_dist)
        score = progress + 0.5 * align + self.cfg.critic_weight * critic

        if clearances is not None:
            safe = clearances >= self.cfg.guard.margin
            if not safe.any():
                return int(np.argmax(clearances))            # least-bad escape
            score[~safe] = -np.inf
            # among safe samples, drop the weakest critic fraction
            safe_idx = np.where(safe)[0]
            if len(safe_idx) > 2:
                thr = np.quantile(critic[safe_idx], 1.0 - self.cfg.critic_keep_frac)
                weak = safe & (critic < thr)
                if (safe & ~weak).any():
                    score[weak] = -np.inf
        else:
            unsafe = critic < self.cfg.critic_floor
            if not unsafe.all():
                score[unsafe] = -np.inf
        return int(np.argmax(score))

    def _command_from_trajectory(self, traj: np.ndarray, min_forward: float = float("inf")):
        """Look-ahead waypoint -> (v, w), matching the rover's velocity caps.

        Angular authority scales with obstacle urgency: near obstacles the
        heading->angular ramp is steepened (saturating sooner, at a smaller
        heading error) and the look-ahead uses the widest heading over the
        horizon so a curving escape trajectory commands an immediate,
        committed turn instead of a lazy drift.
        """
        wp = traj[min(self.cfg.waypoint_index, len(traj) - 1)]
        dist = float(np.linalg.norm(wp[:2]))
        heading = float(np.arctan2(wp[1], wp[0]))

        urgent = min_forward < self.cfg.guard.slow_dist
        ramp_deg = self.cfg.servo_ramp_deg
        if urgent:
            # widest waypoint heading over the horizon = the turn the policy
            # actually intends; act on it NOW rather than easing into it
            xs, ys = traj[1:, 0], traj[1:, 1]
            headings = np.arctan2(ys, np.maximum(xs, 1e-3))
            heading = float(headings[np.argmax(np.abs(headings))])
            ramp_deg /= 1.0 + self.cfg.urgency_gain * (1.0 - min_forward / self.cfg.guard.slow_dist)

        angular = bearing_to_angular(heading, self.cfg.max_angular, self.cfg.ang_min_cmd,
                                     0.0, np.radians(ramp_deg))

        linear = np.clip(self.cfg.kp * dist, 0.0, self.cfg.max_linear)
        # slow down while turning hard (fraction of angular authority in use)
        linear *= max(0.15, 1.0 - 0.8 * abs(angular) / self.cfg.max_angular)
        return float(linear), float(angular)

    # ------------------------------------------------------------------ #
    def step(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None,
             pose: Optional[tuple] = None, external_dets: Optional[list] = None,
             external_goal: Optional[np.ndarray] = None, avoid_text: str = "",
             intrinsics: Optional[tuple] = None, instruction: str = "") -> StepResult:
        """pose, if given: (x, y, theta) odometry at capture time -- stamped onto
        this tick's scene_log entry (see PipelineConfig.use_scene_tagger), and
        required for external_goal below. As of the object-map feature,
        OdometryLogger keeps pose CONTINUOUS across goals (see its module
        docstring) rather than resetting per goal, so this is one shared
        world frame, not a series of disjoint local ones.

        external_dets, if given (even an empty list): the caller has already
        resolved which single detection (or none) is this tick's target --
        e.g. nav_pipeline/remind_gui.py, which asks REMIND for the object
        carrying a specific persistent ID instead of a bare category phrase.
        Bypasses DINO + the relation/positional/appearance re-lock machinery
        below entirely (see _step_inner); None (the default) is the exact
        prior behavior for every other caller.

        external_goal, only consulted when external_dets is None or empty
        (i.e. the target isn't currently visible): a [x fwd, y left] (or
        [x,y,z]) point in the CURRENT local frame to drive toward blindly --
        e.g. remind_gui.py's object_map.py giving the last remembered world
        location of an object that isn't in view right now. Enters state
        "GOTO" and reuses the exact same obstacle-guard/NavDP/trajectory
        machinery as a live TRACK (so depth-based collision avoidance stays
        active for the whole blind leg), but never self-declares STOP from
        proximity alone -- dead-reckoning drift over a long walk between
        rooms makes a purely odometric "arrived" unsafe to trust. The caller
        decides when to stop coasting on GOTO (e.g. switch to a search spin
        once close with no visual reacquisition) and give up.

        avoid_text, if non-empty: a free-text phrase (e.g. "trash bin") DINO-
        detected every tick same as target_text, but only to mark its box as
        a named obstacle (res.avoid_pixels) -- it never affects target
        selection. Only nav_pipeline/s2diff_http_client.py currently reads
        res.avoid_pixels (merged into its S2Diff pixel-obstacle guidance);
        the plain and in-process-S2Diff launchers detect it but don't act on
        it. "" (default) skips the extra DINO pass entirely.

        intrinsics, if given: (fx, fy, cx, cy) from the camera's real
        factory calibration (sensor_msgs/CameraInfo) -- e.g.
        nav_pipeline/zenoh_node.py's CAMERA_INFO_KEYS. Used in place of
        the coarse intrinsics_from_fov(...) approximation for this tick's
        goal-point/obstacle-guard projection; None (the default, or a
        stale/missing camera_info) falls back to the FOV approximation
        exactly as before -- never regresses to "no intrinsics".

        instruction, if non-empty (requires cfg.use_qwen_instruction):
        bypasses DINO target detection entirely -- a frozen Qwen2.5-VL
        grounds this free-text instruction to a pixel goal every
        qwen_instruction_period_s, which becomes NavDP's 3D goal via depth
        exactly like a DINO detection would (see qwen_pixel_goal.py).
        target_text is ignored when this is set. Only consulted when
        external_dets/external_goal are both absent -- those take priority,
        same precedence as target_text.
        """
        res = self._step_inner(rgb, target_text, depth, pose, external_dets, external_goal,
                                avoid_text, intrinsics, instruction)
        # stiction floor is now baked into bearing_to_angular's smooth ramp
        # for TRACK; AVOID commands max_angular directly and SEARCH's
        # search_angular is already comfortably above ang_min_cmd, so
        # neither needs a separate post-hoc boost (see bearing_to_angular).
        if self.cfg.invert_angular:
            res.angular = -res.angular
        # cross-tick EMA smoothing (ported from the OmniVLA node's damping).
        # Excluded for SEARCH too: search_angular is deliberately tuned low,
        # but blending it with a leftover fast TRACK turn (up to max_angular)
        # dragged the effective search speed up for several ticks after the
        # target was lost.
        a = self.cfg.smoothing
        if a > 0 and res.state not in ("STOP", "SEARCH"):
            pl, pa = self._prev_cmd
            res.linear = (1 - a) * res.linear + a * pl
            res.angular = (1 - a) * res.angular + a * pa
        # hard slew-rate limit on the final angular command, ALL states
        # included (this is what catches the SEARCH/STOP snap that the EMA
        # above deliberately skips) -- see PipelineConfig.angular_slew_max.
        slew = self.cfg.angular_slew_max
        if slew > 0:
            prev_a = self._prev_cmd[1]
            res.angular = max(prev_a - slew, min(prev_a + slew, res.angular))
        # Last step, after every other shaping above: boost a command that
        # would otherwise differential-mix down to a stalled wheel on the
        # real ESP32 rover -- see clear_wheel_deadband's docstring. Applied
        # to the FINAL command so it sees exactly what's about to be sent.
        if self.cfg.wheel_deadband_correction:
            res.linear, res.angular = clear_wheel_deadband(
                res.linear, res.angular, self.cfg.track_width_m, self.cfg.wheel_vel_deadband_ms)
        self._prev_cmd = (res.linear, res.angular)
        return res

    def _step_inner(self, rgb: np.ndarray, target_text: str, depth: Optional[np.ndarray] = None,
                     pose: Optional[tuple] = None, external_dets: Optional[list] = None,
                     external_goal: Optional[np.ndarray] = None, avoid_text: str = "",
                     intrinsics: Optional[tuple] = None, instruction: str = "") -> StepResult:
        res = StepResult()
        H, W = rgb.shape[:2]
        timing = {}

        t0 = time.time()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        motion = _estimate_motion_affine(self._prev_gray, gray) if self._prev_gray is not None else None
        self._prev_gray = gray
        timing["cmc"] = time.time() - t0

        t0 = time.time()
        if external_dets is not None:
            # Caller already resolved exactly which physical object this
            # tick's target is (e.g. REMIND's persistent per-object ID --
            # see nav_pipeline/remind_gui.py) -- skip DINO entirely and the
            # box-IoU/appearance re-lock machinery below, which exists only
            # to disambiguate BETWEEN raw same-class DINO candidates frame
            # to frame. An empty list means "not currently visible this
            # tick", same as a normal miss below.
            det = external_dets[0] if external_dets else None
            self._last_candidate_count = 0
        elif external_goal is not None:
            # Blind navigate-back leg (GOTO, see step()'s docstring) -- no
            # live detection to resolve at all this tick.
            det = None
            self._last_candidate_count = 0
        elif instruction and self._qwen_pixel_guide is not None:
            # Qwen instruction-grounded goal (see qwen_pixel_goal.py) --
            # no DINO target to resolve; the goal itself is derived further
            # down, once depth is available (mirrors the goal-from-det
            # section right below it).
            det = None
            self._last_candidate_count = 0
        else:
            if self.detector is None:
                raise RuntimeError(
                    "step() needs a text-target DINO detection this tick (no instruction/"
                    "external_dets/external_goal given), but this pipeline was built with "
                    "use_dino=False. Pass an instruction instead, or rebuild with use_dino=True."
                )
            relation = parse_relational_target(target_text)
            positional = None if relation is not None else parse_positional_target(target_text)
            if relation is not None:
                target_class, anchor_class = relation
                raw_dets = self.detector.detect(rgb, f"{target_class}. {anchor_class}.")
                target_dets = [d for d in raw_dets if clean_label(d.label) == target_class]
                anchor_dets = [d for d in raw_dets if clean_label(d.label) == anchor_class]
                initial_pick = select_by_relation(target_dets, anchor_dets) if self._last_box is None else None
                det = self._select_detection(target_dets, motion=motion, rgb=rgb, initial_pick=initial_pick)
            elif positional is not None:
                target_class, position = positional
                raw_dets = self.detector.detect(rgb, f"{target_class}.")
                target_dets = [d for d in raw_dets if clean_label(d.label) == target_class]
                initial_pick = select_by_position(target_dets, position) if self._last_box is None else None
                det = self._select_detection(target_dets, motion=motion, rgb=rgb, initial_pick=initial_pick)
            else:
                det = self._select_detection(self.detector.detect(rgb, target_text), motion=motion, rgb=rgb)
        timing["dino"] = time.time() - t0

        res.candidate_count = self._last_candidate_count
        res.ambiguous = self._last_candidate_count >= 2
        if res.ambiguous:
            if not self._ambiguity_warned:
                self._ambiguity_warned = True
                print(f"[WARN] ambiguous target '{target_text}': {self._last_candidate_count} comparably-"
                      f"scored candidates in view -- locking onto top score by default. Resend a more "
                      f"specific phrase (e.g. '{target_text} near the window') if it picks the wrong one.")
        else:
            self._ambiguity_warned = False

        if avoid_text and self.detector is None:
            if not self._avoid_no_dino_warned:
                self._avoid_no_dino_warned = True
                print(f"[WARN] avoid_text={avoid_text!r} given but this pipeline was built with "
                      f"use_dino=False -- named-obstacle detection is skipped every tick until "
                      f"a new avoid_text arrives (rebuild with use_dino=True to use this)")
        elif avoid_text:
            t0 = time.time()
            avoid_det = self.detector.detect_best(rgb, avoid_text)
            if avoid_det is not None:
                res.avoid_detection = avoid_det
                res.avoid_pixels = _bbox_to_pixel_grid(avoid_det.box, W, H)
            timing["dino_avoid"] = time.time() - t0

        t0 = time.time()
        if depth is None:
            if self.depther is None:
                raise ValueError("no depth given and depth estimator disabled")
            depth = self.depther.estimate(rgb)
        timing["depth"] = time.time() - t0

        if self.scene_tagger is not None:
            now = time.time()
            if (now - self._last_tag_t) >= self.cfg.scene_tag_period_s:
                t0 = time.time()
                objects = self.scene_tagger.tag(rgb)
                timing["scene_tag"] = time.time() - t0
                entry = {"t": now, "objects": objects}
                if pose is not None:
                    entry["pose"] = {"x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2])}
                self.scene_log.append(entry)
                self._last_tag_t = now
                summary = ", ".join(f"{k}:{v}" for k, v in sorted(objects.items())) or "(nothing detected)"
                pose_txt = f"  pose=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:+.2f})" if pose is not None else ""
                print(f"[scene] {time.strftime('%H:%M:%S', time.localtime(now))}  {summary}{pose_txt}")
                if self._scene_log_file is not None:
                    self._scene_log_file.write(json.dumps(entry) + "\n")
                    self._scene_log_file.flush()

        # update observation memory (up to policy.memory_size frames)
        rgb_p, dep_p = preprocess_rgb(rgb), preprocess_depth(depth)
        self._memory.append(rgb_p)
        self._memory_d.append(dep_p)
        self._memory = self._memory[-self._memory_size:]
        self._memory_d = self._memory_d[-self._memory_size:]

        # --- odometry delta, for belief's ego-motion propagation -------- #
        # pose is (x, y, theta) dead-reckoned odometry in a fixed world frame
        # anchored to wherever the rover was when this goal started (see
        # OdometryLogger). Convert the world-frame step since last tick into
        # the rover's OWN previous-tick local frame (forward/left), which is
        # what GoalBelief.propagate/ego_motion_update expects -- matches
        # belief_bank.py's contract exactly, just fed from real wheel
        # odometry instead of a sim step.
        odom_delta = None
        if pose is not None:
            if self._prev_pose is not None:
                x0, y0, th0 = self._prev_pose
                x1, y1, th1 = pose
                dxw, dyw = x1 - x0, y1 - y0
                dx = dxw * np.cos(th0) + dyw * np.sin(th0)
                dy = -dxw * np.sin(th0) + dyw * np.cos(th0)
                odom_delta = (dx, dy, th1 - th0)
            self._prev_pose = pose

        # --- goal ------------------------------------------------------ #
        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics
        else:
            fx, fy, cx, cy = intrinsics_from_fov(W, H, self.cfg.horizontal_fov_deg)
        goal = None
        if det is not None:
            res.detection = det
            clip_rejected = False
            mask = None
            if external_dets is not None:
                # The external resolver (e.g. REMIND, which segments every
                # detection with its own YOLO-seg backend as a normal part
                # of its own pipeline) already produced this object's mask
                # -- reuse it directly instead of a second SAM2 pass, and
                # skip CLIP entirely: target_text here is just the class
                # name that resolver already assigned, so CLIP would only
                # be re-checking that same classification with a weaker
                # model, not verifying an independent phrase. None (no mask
                # supplied) falls through to the plain bbox-depth goal
                # below, same as "no mask available" always has.
                mask = getattr(det, "mask", None)
            elif self.segmenter is not None:
                now = time.time()
                due = (now - self._last_sam_t) >= self.cfg.sam_period_s
                if due:
                    t0 = time.time()
                    mask = self.segmenter.segment_box(rgb, det.box)
                    timing["sam"] = time.time() - t0
                    if mask is not None and self.clip is not None:
                        from .sam_segmenter import mask_bbox

                        t0 = time.time()
                        score = self.clip.verify(rgb, mask_bbox(mask), target_text)
                        timing["clip"] = time.time() - t0
                        if score < self.cfg.clip_min_similarity:
                            clip_rejected = True
                            mask = None
                            size_frac = (det.box[3] - det.box[1]) / H
                            print(f"[reid-debug] CLIP REJECTED score={score:.3f} < {self.cfg.clip_min_similarity} "
                                  f"box_h_frac={size_frac:.2f}")
                    self._last_sam_t = now
                    self._last_mask = mask
                    self._last_mask_box = det.box.copy() if mask is not None else None
                elif self._last_mask is not None and _box_iou(det.box, self._last_mask_box) > 0.1:
                    # not due for a fresh SAM/CLIP pass -- reuse the cached mask as
                    # long as the current DINO box still overlaps the one it was
                    # segmented from
                    mask = self._last_mask
            if mask is not None:
                from .sam_segmenter import mask_centroid, mask_median_depth

                res.mask = mask
                d = mask_median_depth(depth, mask)
                if d is not None:
                    u, v = mask_centroid(mask)
                    from .goal_utils import pixel_depth_to_point

                    goal = pixel_depth_to_point(u, v, d, fx, fy, cx, cy)
            if not clip_rejected and goal is None:  # SAM disabled, empty mask, or no valid mask depth
                goal = goal_point_from_detection(det.box, depth, fx, fy, cx, cy)
            if goal is not None:
                self._last_goal, self._lost_count = goal, 0
                if self.cfg.use_belief_goal:
                    self.belief.observe(goal, confidence=float(det.score))
                if np.linalg.norm(goal[:2]) < self.cfg.stop_distance:
                    res.state = "STOP"
                    res.goal_point = goal
                    res.timing = timing
                    return res
        if goal is None and instruction and self._qwen_pixel_guide is not None:
            # Qwen instruction-grounded goal: throttled (qwen_instruction_
            # period_s) same as SEARCH's Qwen guidance above -- on ticks
            # it's not due, this deliberately leaves goal=None so the
            # belief-coasting fallback below propagates the LAST Qwen goal
            # by ego-motion instead of re-querying every tick or holding a
            # stale pixel (see qwen_pixel_goal.py's module docstring).
            now = time.time()
            if self._qwen_instruction_gate.due(now):
                t0 = time.time()
                candidates = self._qwen_pixel_guide.ground_candidates(
                    rgb, instruction, max_candidates=self.cfg.qwen_max_candidates)
                timing["qwen_instruction"] = time.time() - t0
                locked = (self.cfg.use_belief_goal and self.belief.initialized
                          and self.belief.sigma <= self.cfg.belief_max_sigma)
                # Obstacle points computed here too (redundant with the guard
                # section below, cheap -- a few ms) so goal SELECTION itself
                # is collision-aware, not just avoidance after the fact. See
                # PipelineConfig.qwen_max_candidates for the full rationale.
                cand_obstacle_pts = depth_to_obstacle_points(depth, fx, fy, cx, cy, self.cfg.guard)
                winner = score_instruction_candidates(
                    candidates, depth, fx, fy, cx, cy,
                    belief_mu=self.belief.mu if locked else None, belief_locked=locked,
                    obstacle_pts=cand_obstacle_pts, guard_cfg=self.cfg.guard,
                    semantic_weight=self.cfg.qwen_candidate_semantic_weight,
                    continuity_weight=self.cfg.qwen_candidate_continuity_weight,
                    continuity_scale_m=self.cfg.qwen_candidate_continuity_scale_m,
                    collision_weight=self.cfg.qwen_candidate_collision_weight,
                    collision_scale_m=self.cfg.qwen_candidate_collision_scale_m,
                    far_lookahead_m=self.cfg.qwen_far_lookahead_m,
                    depth_trust_horizon_m=self.cfg.qwen_depth_trust_horizon_m,
                )
                if winner is not None:
                    qgoal, pg, disc, far_mode = winner
                    # consistency gate -- see PipelineConfig.qwen_goal_consistency_m.
                    # Kept as a final check even with scoring above: scoring
                    # picks the best of THIS tick's candidates, but if every
                    # candidate this tick is actually a distractor (the real
                    # door wasn't offered at all, e.g. out of frame mid-turn),
                    # this still catches it rather than accepting the least-
                    # bad wrong answer. Skipped in far_mode: a bearing-only
                    # look-ahead goal is deliberately not near the stale lock
                    # (that mismatch is exactly what stranded long-range
                    # instructions -- see qwen_far_lookahead_m).
                    if locked and not far_mode:
                        dist = float(np.linalg.norm(qgoal[:2] - self.belief.mu[:2]))
                        if dist > self.cfg.qwen_goal_consistency_m:
                            print(f"[qwen-instruction-debug] REJECTED best-scoring candidate "
                                  f"{dist:.2f}m from locked goal (> "
                                  f"{self.cfg.qwen_goal_consistency_m}m) -- likely a "
                                  f"different door/target; coasting on the existing lock")
                            qgoal = None
                    elif far_mode:
                        print(f"[qwen-instruction-debug] far target -- depth untrusted, "
                              f"driving bearing to a {self.cfg.qwen_far_lookahead_m:.1f}m "
                              f"look-ahead goal and re-grounding as it closes")
                    if qgoal is not None:
                        res.qwen_pixel_goal = (float(pg.u), float(pg.v))
                        res.mask = disc  # obstacle guard excludes the goal's own surface, same as a DINO/SAM mask
                        self._last_goal, self._lost_count = qgoal, 0
                        if self.cfg.use_belief_goal:
                            self.belief.observe(qgoal, confidence=pg.confidence)
                        if np.linalg.norm(qgoal[:2]) < self.cfg.stop_distance:
                            res.state = "STOP"
                            res.goal_point = qgoal
                            res.timing = timing
                            return res
                        goal = qgoal
        if goal is None and external_goal is not None:
            # Blind navigate-back: no live detection this tick, but the
            # caller supplied a remembered local-frame point (see step()'s
            # docstring). Skip belief/SEARCH entirely -- drive toward it
            # using the exact same obstacle-guard/NavDP path as a live TRACK
            # below. Deliberately no self-declared STOP here: unlike a live
            # mask-depth goal, this point can be stale from odometry drift
            # accumulated crossing rooms, so proximity alone isn't trusted
            # as "arrived" -- that call is left to the caller (e.g.
            # remind_gui.py switching to a search spin once close).
            goal = np.asarray(external_goal, dtype=np.float32)
            if goal.shape[0] < 3:
                goal = np.array([float(goal[0]), float(goal[1]), 0.0], dtype=np.float32)
            res.state = "GOTO"

        if goal is None:
            self._lost_count += 1
            if self.cfg.use_belief_goal:
                if odom_delta is not None:
                    self.belief.propagate(*odom_delta)
                belief_ok = self.belief.initialized and self.belief.sigma <= self.cfg.belief_max_sigma
                # lost_patience (a frame-count cap) is NOT applied here on
                # purpose: it predates belief, from when the goal was
                # frozen and coasting too long on a stale point was the
                # risk. Belief's sigma already grows with real elapsed
                # ticks AND rotation (see goal_belief.py), so it's a more
                # principled cutoff than an arbitrary tick count -- ANDing
                # lost_patience=5 (~2s) on top of it was cutting off a
                # still-confident, correctly-tracking belief after just a
                # couple seconds regardless of how low sigma actually was.
            else:
                belief_ok = self._last_goal is not None and self._lost_count <= self.cfg.lost_patience
            if belief_ok:
                goal = self.belief.mu if self.cfg.use_belief_goal else self._last_goal
                if self.cfg.use_belief_goal:
                    print(f"[belief-debug] coasting tick {self._lost_count} "
                          f"sigma={self.belief.sigma:.3f}/{self.cfg.belief_max_sigma}")
                if np.linalg.norm(goal[:2]) < self.cfg.stop_distance:
                    # Same self-declared-arrival check as the live-detection
                    # path above (~line 855) -- without this, a goal only
                    # reachable via belief-coasting never gets a chance to
                    # self-declare STOP at all. Live-captured 2026-08-10: CLIP
                    # verify started rejecting a real, correctly-tracked door
                    # (score ~0.03-0.13 < 0.5) once the crop filled the whole
                    # frame at close range and no longer looked like a
                    # recognizable "door" to CLIP's zero-shot classifier --
                    # the pipeline fell to belief-coasting for the rest of the
                    # approach and, absent this check, kept creeping forward
                    # with no distance safeguard until the obstacle guard's
                    # hard_stop_dist (a much closer, collision-avoidance-only
                    # threshold, not the intended 1.5m stop) was the only
                    # thing left to catch it.
                    res.state = "STOP"
                    res.goal_point = goal
                    res.timing = timing
                    return res
            else:
                if self.cfg.use_belief_goal:
                    reason = "sigma too high" if self.belief.initialized else "belief not initialized"
                    sigma_txt = f"{self.belief.sigma:.3f}" if self.belief.initialized else "n/a"
                    print(f"[belief-debug] -> SEARCH: {reason} (sigma={sigma_txt} "
                          f"max={self.cfg.belief_max_sigma}, lost_count={self._lost_count})")
                # search: rotate in place toward the last known side. This
                # MUST come from self._last_goal (frozen at the last real
                # detection), never from self.belief.mu -- belief keeps
                # rotating every occluded tick (including all through
                # SEARCH, since propagate() above runs unconditionally), so
                # using it here made the chosen side flip mid-sweep as the
                # propagated estimate crossed y=0, and the rover reversed
                # direction before ever completing a look-around (real
                # rover test 2026-07-31: theta swung +116deg then reversed
                # to -77deg hunting for a chair, never finding it).
                res.state = "SEARCH"
                qwen_angular = None
                if self._qwen_scheduler is not None:
                    bearing = self._qwen_scheduler.step(time.time(), rgb, target_text)
                    if bearing is not None:
                        res.qwen_bearing = bearing
                        # capped at search_angular (not max_angular): this is
                        # still a searching turn, not a confident TRACK lock
                        qwen_angular = bearing_to_angular(
                            bearing, self.cfg.search_angular, self.cfg.ang_min_cmd,
                            0.0, np.radians(self.cfg.servo_ramp_deg),
                        )
                if qwen_angular is not None:
                    res.angular = qwen_angular
                else:
                    side = 1.0
                    if self._last_goal is not None and self._last_goal[1] < 0:
                        side = -1.0
                    res.angular = side * self.cfg.search_angular
                res.timing = timing
                return res

        res.goal_point = goal
        if res.state != "GOTO":  # set above when this tick is a blind navigate-back leg
            res.state = "TRACK"

        # --- obstacle guard --------------------------------------------- #
        obstacle_pts = None
        if self.cfg.avoid_enabled:
            t0 = time.time()
            # Prefer the real SAM mask; fall back to a coarse bbox exclusion
            # so the goal object's own depth is still kept out of obstacle
            # points when SAM is off/not-yet-run/stale/CLIP-rejected this
            # tick, instead of excluding nothing at all (see
            # _bbox_exclude_mask's docstring).
            exclude_mask = res.mask
            if exclude_mask is None and res.detection is not None:
                exclude_mask = _bbox_exclude_mask(res.detection.box, H, W)
            obstacle_pts = depth_to_obstacle_points(
                depth, fx, fy, cx, cy, self.cfg.guard, exclude_mask=exclude_mask
            )
            res.obstacle_points = obstacle_pts
            min_fwd, escape = forward_guard(obstacle_pts, self.cfg.guard)
            res.min_forward = min_fwd
            timing["guard"] = time.time() - t0
            if min_fwd < self.cfg.guard.hard_stop_dist:
                if (res.state == "TRACK" and goal is not None
                        and np.linalg.norm(goal[:2]) < self.cfg.guard.slow_dist):
                    # A close-in-front reading this near our OWN live tracked
                    # goal is far more likely to BE the goal than a
                    # coincidental unrelated obstacle. exclude_mask above
                    # should already keep the target out of obstacle_pts, but
                    # SAM/monocular depth both commonly degrade once the
                    # object fills/overflows the frame -- real surface points
                    # can leak through unmasked here while the STOP check's
                    # own mask-median-depth goal distance (~line 855)
                    # simultaneously reads too far to have triggered yet.
                    # Symptom this fixes: rover visibly approaching its
                    # target, then suddenly spinning full-authority away from
                    # it right before arrival. Declare arrival instead of
                    # avoiding. Real obstacles blocking the path to a still-
                    # distant goal (norm(goal) >= slow_dist) are NOT covered
                    # by this gate and still trigger AVOID exactly as before.
                    # Gated on state=="TRACK" (not "GOTO"): a blind
                    # navigate-back leg's goal is a remembered point that can
                    # be stale from drift, so proximity alone must NOT be
                    # trusted as arrival there (see the GOTO branch's own
                    # docstring above) -- AVOID still applies normally on
                    # GOTO legs.
                    res.state = "STOP"
                    res.goal_point = goal
                    self._avoid_streak = 0
                    res.timing = timing
                    return res
                # hysteresis: monocular depth is noisy — require consecutive
                # confirmations before engaging AVOID (prevents state flapping
                # that looks like random movement)
                self._avoid_streak += 1
                if self._avoid_streak >= self.cfg.avoid_confirm_ticks:
                    res.state = "AVOID"
                    res.linear = -0.5 * self.cfg.max_linear if min_fwd < self.cfg.guard.reverse_dist else 0.0
                    # graduated turn-rate: full max_angular used to fire the
                    # instant hard_stop_dist was crossed, regardless of
                    # whether the obstacle was barely inside it or right on
                    # top of the rover -- read as a sudden snap turn. Scale
                    # from ang_min_cmd (just enough to overcome stiction) up
                    # to max_angular as min_fwd shrinks from hard_stop_dist
                    # down to reverse_dist (closer than that, footprint
                    # geometry already means rotating isn't safe anyway --
                    # see reverse-first branch above -- so full authority is
                    # warranted there). Recomputed every AVOID tick, so the
                    # rate also eases back down as the rover turns clear.
                    urgency = np.clip(
                        (self.cfg.guard.hard_stop_dist - min_fwd)
                        / max(self.cfg.guard.hard_stop_dist - self.cfg.guard.reverse_dist, 1e-6),
                        0.0, 1.0,
                    )
                    angular_mag = self.cfg.ang_min_cmd + urgency * (self.cfg.max_angular - self.cfg.ang_min_cmd)
                    res.angular = escape * angular_mag
                    # latch the escape side + re-arm the cooldown (every
                    # trigger resets it to the full value, so a persistent
                    # obstacle keeps the post-escape bias alive throughout)
                    self._avoid_side = escape
                    self._avoid_cooldown = self.cfg.avoid_cooldown_ticks
                    res.timing = timing
                    return res
            else:
                self._avoid_streak = 0

        # --- NavDP ------------------------------------------------------ #
        t0 = time.time()
        M = self._memory_size
        frames = np.stack(self._memory)
        if frames.shape[0] < M:  # front-pad with zeros (matches official agent at episode start)
            pad = np.zeros((M - frames.shape[0],) + frames.shape[1:], dtype=frames.dtype)
            frames = np.concatenate([pad, frames], axis=0)
        images = torch.from_numpy(frames).unsqueeze(0)
        if self.cfg.policy_type == "crossmodal":
            depths = torch.from_numpy(dep_p).unsqueeze(0).unsqueeze(0)  # current depth only (1,1,H,W,1)
        else:
            dframes = np.stack(self._memory_d)
            if dframes.shape[0] < M:
                pad = np.zeros((M - dframes.shape[0],) + dframes.shape[1:], dtype=dframes.dtype)
                dframes = np.concatenate([pad, dframes], axis=0)
            depths = torch.from_numpy(dframes).unsqueeze(0)
        goal_native = np.array([goal[0], self._goal_y_sign * goal[1], goal[2]], dtype=np.float32)
        # Plain instance attribute, not part of NavDPStandalone's own API --
        # only nav_pipeline/s2diff_http_client.py's monkeypatched
        # sample_pointgoal reads it (via getattr(self, ...), self is this
        # same self.policy). Harmless no-op for every other policy backend.
        self.policy._pending_avoid_pixels = res.avoid_pixels
        trajs, critic = self.policy.sample_pointgoal(
            goal_native.reshape(1, 3), images, depths, sample_num=self.cfg.sample_num
        )
        trajs = trajs.cpu().numpy()
        critic = critic.cpu().numpy()
        timing["navdp"] = time.time() - t0

        clearances = None
        if obstacle_pts is not None:
            clearances = swept_clearance(trajs, obstacle_pts, self.cfg.guard)
        idx = self._select_trajectory(trajs, critic, goal, clearances)
        chosen = trajs[idx]
        res.trajectory = chosen
        res.all_trajectories = trajs
        res.critic = critic

        # Final command: continuously blend open-space goal-bearing servo with
        # clearance-vetoed trajectory-following, instead of switching outright
        # between the two control laws at slow_dist. The two don't generally
        # agree on heading even when clearances are still generous (NavDP's
        # critic-scored sample vs. a plain bearing-to-goal), so a hard switch
        # there could snap the commanded heading the instant an obstacle
        # entered the corridor even with no real collision risk yet -- this
        # is what read as "atomic" avoidance. weight ramps 0 (pure
        # open-space) at slow_dist to 1 (pure obstacle-following) at
        # hard_stop_dist, so the two commands cross-fade smoothly and
        # avoidance visibly starts influencing steering from slow_dist out.
        # AVOID's own hard trigger above is untouched by this -- once truly
        # close, decisive escape still fires outright exactly as before.
        bearing = float(np.arctan2(goal[1], goal[0]))  # +left, ROS convention
        ang_open = bearing_to_angular(
            bearing, self.cfg.max_angular, self.cfg.ang_min_cmd,
            self.cfg.servo_deadband, np.radians(self.cfg.servo_ramp_deg),
        )
        lin_open = self.cfg.max_linear * max(0.2, 1.0 - 0.8 * abs(ang_open) / self.cfg.max_angular)

        lin_obs, ang_obs = self._command_from_trajectory(chosen, res.min_forward)
        # was an unconditional < slow_dist scale-down before blending existed
        # (min_forward was always < slow_dist at this call site then); now
        # called at every distance, so clip the upper end too -- otherwise a
        # far-clear min_forward (even inf) would inflate this past 1.0.
        lin_obs *= float(np.clip(res.min_forward / self.cfg.guard.slow_dist, 0.25, 1.0))

        span = max(self.cfg.guard.slow_dist - self.cfg.guard.hard_stop_dist, 1e-6)
        weight = float(np.clip((self.cfg.guard.slow_dist - res.min_forward) / span, 0.0, 1.0))
        res.linear = (1.0 - weight) * lin_open + weight * lin_obs
        res.angular = (1.0 - weight) * ang_open + weight * ang_obs
        res.angular, self._avoid_cooldown = apply_avoid_cooldown(
            res.angular, res.state, self._avoid_side, self._avoid_cooldown,
            self.cfg.avoid_bias_gain, self.cfg.max_angular,
        )
        res.timing = timing
        return res
