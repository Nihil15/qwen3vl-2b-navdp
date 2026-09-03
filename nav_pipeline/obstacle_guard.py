"""Depth-based obstacle avoidance layers (model-agnostic, hard guarantees).

Three cooperating mechanisms, all fed by the metric depth image:

1. depth_to_obstacle_points: depth -> 2D obstacle points (x fwd, y left) in
   the robot frame, keeping only points in an obstacle height band (above
   ground, below overhead clearance). "Ground" is a LOCAL-SLOPE test, not a
   flat plane through the rover's own footing: each column is scanned from
   the closest row to the farthest, and a point counts as ground if its rise
   since the last confirmed ground point is within `max_climb_deg` over the
   distance traveled since then. A flat-plane assumption (fixed height
   threshold everywhere) reads a rising slope/hill as a wall, since the true
   ground climbs above that plane; but widening a FIXED-origin tolerance by
   range doesn't work either — at 1-2 m range the tolerance needed to admit
   a climbable slope is already as large as a real rock's height, so it goes
   blind to obstacles at exactly the range the hard-stop/slow-down thresholds
   below care about. Comparing against the local, walked-forward ground
   profile instead means cumulative slope height never matters (a rover can
   climb a slope of any total height) while a rock/step's abrupt LOCAL rise
   over a short distance still exceeds the tolerance and is still flagged,
   regardless of how far away it is. The target object's SAM mask is
   excluded so approaching the goal never counts as a collision.

2. swept_clearance: FOOTPRINT-AWARE clearance. The rover's 0.482 x 0.380 m
   rectangle is covered by two circles (radius 0.23 m at ±0.12 m along the
   body axis) and swept along every candidate trajectory using each
   waypoint's heading. Returned clearance is the gap BEYOND the hull —
   0 means the hull touches. Trajectories with gap < `margin` are vetoed.
   The current pose (0,0,0) is included so in-place rotation is also checked.

3. forward_guard: min obstacle distance inside the forward corridor ->
   hard stop below `hard_stop_dist`, linear slow-down below `slow_dist`,
   escape turn away from the obstacle side, and reverse-first escape when
   too close to rotate safely (half-diagonal 0.31 m > half-width 0.19 m).

4. apply_avoid_cooldown: purely reactive avoidance (recompute escape side
   fresh every tick, drop it the instant the corridor clears) is prone to a
   limit cycle: the rover escapes an obstacle on its right by turning left,
   clears hard_stop_dist, and goal-bearing servo immediately re-aims back at
   the same obstacle (since the goal is still past it) -> AVOID re-triggers
   -> turn left again -> repeat, with no net progress ("starvation"). This
   holds a bias toward the escape side for a few ticks after AVOID releases,
   so the rover actually drifts laterally clear of the obstacle's footprint
   before goal-seeking steering resumes at full strength.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

ROVER_LENGTH = 0.482
ROVER_WIDTH = 0.380
# two-circle cover of the footprint rectangle: centers ±dx on the body axis
_CIRCLE_DX = 0.12
_CIRCLE_R = float(np.hypot(ROVER_LENGTH / 2 - _CIRCLE_DX, ROVER_WIDTH / 2)) + 0.01  # ≈0.23


@dataclass
class GuardConfig:
    cam_height: float = 0.35        # camera height above ground (m)
    ground_band: float = 0.08       # points below this height are "ground" (m)
    max_climb_deg: float = 20.0     # terrain rising this steeply from the rover is still "ground", not an obstacle
    slope_ref_step: float = 0.15    # min forward distance before re-anchoring the local-slope ground reference (m)
    overhead: float = 1.2           # ignore points above this height (m)
    max_range: float = 4.0          # ignore points farther than this (m)
    margin: float = 0.10            # required gap beyond the swept hull (m)
    hard_stop_dist: float = 0.60    # forward obstacle closer than this -> stop (m)
    reverse_dist: float = 0.35      # too close to rotate -> back up first (m)
    slow_dist: float = 3.2          # outer edge of the avoidance blend zone (m) -- below this,
                                     # pipeline.py's final command continuously cross-fades from
                                     # pure open-space goal-bearing servo (weight 0, at slow_dist)
                                     # to NavDP's clearance-vetoed trajectory-following (weight 1,
                                     # at hard_stop_dist), rather than switching control laws
                                     # outright at a single threshold (see pipeline.py's _step_inner
                                     # final command block) -- avoidance visibly starts influencing
                                     # steering from here out, not just once already deep inside it.
                                     # Widened 2026-08-14 from 2.5 (was also the old hard
                                     # switch-over point) at user request ("gradual not atomic,
                                     # start avoiding from a distance"); kept comfortably under
                                     # max_range (4.0m) since depth reliability degrades near the
                                     # far edge of range -- don't push this past ~3.5 without also
                                     # revisiting max_range.
    corridor_half_width: float = 0.35  # forward corridor half-width ≥ half-width+margin (m)
    stride: int = 4                 # depth subsampling stride (speed)
    # Robot footprint for swept_clearance()'s two-circle cover -- defaults to
    # the ESP32 rover's real dimensions so every existing caller that doesn't
    # pass this GuardConfig into swept_clearance() is unaffected. A different
    # robot (different physical size) should pass its own GuardConfig with
    # these overridden -- see landerpi/README.md.
    footprint_length: float = ROVER_LENGTH
    footprint_width: float = ROVER_WIDTH
    footprint_circle_dx: float = _CIRCLE_DX


def depth_to_obstacle_points(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cfg: GuardConfig,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Depth image -> (N, 2) obstacle points [x fwd, y left] in robot frame."""
    s = cfg.stride
    d = depth_m[::s, ::s].astype(np.float32)
    H, W = depth_m.shape[:2]
    us = np.arange(0, W, s, dtype=np.float32)[None, :].repeat(d.shape[0], axis=0)
    vs = np.arange(0, H, s, dtype=np.float32)[:, None].repeat(d.shape[1], axis=1)

    valid = np.isfinite(d) & (d > 0.15) & (d < cfg.max_range)
    if exclude_mask is not None and exclude_mask.shape[:2] == depth_m.shape[:2]:
        valid &= ~exclude_mask[::s, ::s]

    x_cam = (us - cx) / fx * d   # right
    y_cam = (vs - cy) / fy * d   # down
    z_up = cfg.cam_height - y_cam

    # local-slope ground scan: walk each column near -> far (image bottom to
    # top), comparing each point to the last confirmed ground reference
    # rather than a flat plane through the rover's own footing (see module
    # docstring). ref_h/ref_d start at (0, 0) -- the rover's own footing.
    #
    # The reference is only re-anchored once at least slope_ref_step of
    # forward distance has actually accumulated since it was last set (NOT
    # on every accepted row). Re-anchoring every row would let ground_band's
    # noise allowance be re-granted on each sub-cm depth step, so a
    # near-vertical obstacle sampled finely enough could "staircase" past the
    # guard a fraction of a millimeter at a time even though its TOTAL rise
    # over that short run is obstacle-sized. Holding the reference fixed
    # until a real step of ground has been covered means a genuine obstacle's
    # cumulative rise is what gets compared against ground_band + slope.
    n_rows = d.shape[0]
    ref_h = np.zeros(d.shape[1], dtype=np.float32)
    ref_d = np.zeros(d.shape[1], dtype=np.float32)
    is_ground = np.zeros_like(d, dtype=bool)
    slope_tan = float(np.tan(np.radians(cfg.max_climb_deg)))
    for row in range(n_rows - 1, -1, -1):    # bottom (closest) -> top (farthest)
        rise = z_up[row] - ref_h
        run = np.maximum(d[row] - ref_d, 0.0)
        tolerance = cfg.ground_band + slope_tan * run
        ground_here = valid[row] & (rise <= tolerance)
        is_ground[row] = ground_here
        advance = ground_here & (run >= cfg.slope_ref_step)
        ref_h = np.where(advance, z_up[row], ref_h)
        ref_d = np.where(advance, d[row], ref_d)

    obstacle = valid & ~is_ground & (z_up < cfg.overhead)
    pts = np.stack([d[obstacle], -x_cam[obstacle]], axis=1)  # [x fwd, y left]
    return pts


def swept_clearance(
    trajs: np.ndarray, points: np.ndarray, cfg: Optional[GuardConfig] = None
) -> np.ndarray:
    """Footprint-swept clearance of each trajectory (N, T, 3) vs points (M, 2).

    Sweeps the two-circle cover of the robot rectangle along the waypoints
    using each waypoint's yaw (3rd channel). Returns (N,) gap beyond the
    hull in meters — 0 means the hull touches an obstacle point; inf when
    there are no points. The start pose (0, 0, 0) is prepended so rotating
    in place is covered too.

    cfg: optional GuardConfig to override the footprint (cfg.footprint_length
    / footprint_width / footprint_circle_dx). Omit to use the ESP32 rover's
    dimensions (module-level ROVER_LENGTH/ROVER_WIDTH) -- every existing
    caller that doesn't pass a cfg keeps that exact behavior.
    """
    if points.shape[0] == 0:
        return np.full(trajs.shape[0], np.inf, dtype=np.float32)
    if cfg is not None:
        dx = cfg.footprint_circle_dx
        r = float(np.hypot(cfg.footprint_length / 2 - dx, cfg.footprint_width / 2)) + 0.01
    else:
        dx, r = _CIRCLE_DX, _CIRCLE_R
    N, T, _ = trajs.shape
    start = np.zeros((N, 1, 3), dtype=trajs.dtype)
    tr = np.concatenate([start, trajs], axis=1)            # (N, T+1, 3)
    x, y, yaw = tr[:, :, 0], tr[:, :, 1], tr[:, :, 2]
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    # circle centers at ±dx along the body axis   -> (N, 2*(T+1), 2)
    cx = np.concatenate([x + cos_y * dx, x - cos_y * dx], axis=1)
    cy = np.concatenate([y + sin_y * dx, y - sin_y * dx], axis=1)
    d2 = (
        (cx[:, :, None] - points[None, None, :, 0]) ** 2
        + (cy[:, :, None] - points[None, None, :, 1]) ** 2
    )
    gap = np.sqrt(d2.min(axis=(1, 2))) - r
    return gap.astype(np.float32)  # negative = hull would overlap the obstacle


# backward-compatible alias (centerline behavior replaced by swept hull)
trajectory_clearance = swept_clearance


def forward_guard(points: np.ndarray, cfg: GuardConfig, min_points: int = 8) -> Tuple[float, float]:
    """Distance to the nearest obstacle CLUSTER in the forward corridor.

    Robust to monocular-depth noise: a single stray point cannot trigger the
    guard — at least `min_points` corridor points must agree, and the distance
    reported is the 10th percentile of the cluster (not the single minimum).
    Returns (forward_dist, escape_sign): escape +1 = turn left. (inf, 0) clear.
    """
    if points.shape[0] == 0:
        return float("inf"), 0.0
    in_corridor = (points[:, 0] > 0.0) & (np.abs(points[:, 1]) < cfg.corridor_half_width)
    if in_corridor.sum() < min_points:
        return float("inf"), 0.0
    corridor_pts = points[in_corridor]
    dist = float(np.percentile(corridor_pts[:, 0], 10))
    near = corridor_pts[corridor_pts[:, 0] < dist + 0.3]
    escape = 1.0 if float(np.median(near[:, 1])) < 0 else -1.0  # obstacle right -> turn left
    return dist, escape


def apply_avoid_cooldown(
    angular: float,
    state: str,
    avoid_side: float,
    cooldown_left: int,
    bias_gain: float,
    max_angular: float,
) -> Tuple[float, int]:
    """Bias steering toward `avoid_side` for `cooldown_left` more ticks after
    an AVOID escape, instead of letting goal-bearing servo snap straight back
    toward the just-avoided obstacle the instant the corridor clears (see
    module docstring for the oscillation this prevents).

    Call site contract: when AVOID triggers, the caller latches
    ``avoid_side = escape`` and resets ``cooldown_left = cfg.avoid_cooldown_ticks``
    (every trigger re-arms it to the full value, so a persistent obstacle
    keeps the bias alive throughout the whole encounter). This function is
    then called once per tick on whatever angular command the NON-avoid
    branches computed (the close-range trajectory-follow or open-space
    bearing-servo), nudging it toward avoid_side and counting the ticks down.
    A no-op while state == "AVOID" itself (that branch already commands the
    full escape turn) or once the cooldown has run out.

    Returns (angular, cooldown_left) -- the caller stores cooldown_left back
    onto its own instance state for the next tick.
    """
    if state == "AVOID" or cooldown_left <= 0:
        return angular, cooldown_left
    biased = float(np.clip(angular + bias_gain * avoid_side, -max_angular, max_angular))
    return biased, cooldown_left - 1
