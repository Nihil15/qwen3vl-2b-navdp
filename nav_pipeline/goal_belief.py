"""Single-slot belief for the pipeline's currently tracked 3D goal point.

Ported down from MARS/mars-habitatsim/navdp/navdp/extensions/belief_bank.py's
SubgoalBeliefBank for the real rover: that class tracks a whole named route
of subgoals (for the sim training pipeline) and keeps a full 2x2 covariance
per slot for a learned policy to condition on. This pipeline only ever
tracks ONE active target and only needs a scalar "how much do I still trust
this" number to decide when to give up and go back to SEARCH -- so this
drops the route/multi-goal/tensor-export machinery entirely rather than
carrying it unused.

Core idea (same as the ported-from class): while the target isn't detected,
propagate the goal by the rover's own measured ego-motion instead of
leaving it frozen at the last place it was seen. A frozen goal is wrong the
moment the rover moves at all; a propagated one stays roughly right as long
as the odometry it's propagated with is trustworthy.

sigma growth is split into a flat per-tick term (any occlusion adds some
doubt over time) and a rotation-proportional term. Real spin-accuracy
trials (scripts/odom_accuracy_gui.py, logged 2026-07-31 to
odometry_log/odom_accuracy_results.csv) showed dead-reckoned heading error
staying within a few degrees through roughly 90-135deg of rotation, then
growing sharply (20-90deg off, 0.2-0.35m of wheel-scrub translation) by
165-270deg. rot_noise_gain/belief_max_sigma below are a starting point
tuned to cross the distrust threshold in that 135-165deg range -- retune
against odom_accuracy_results.csv as more trials come in, the data was
noisy enough (e.g. the 90deg trial had a worse heading error than 135deg)
that these shouldn't be treated as precise physical constants.
"""
from typing import Optional

import numpy as np


class GoalBelief:
    def __init__(self, sigma_init: float = 1000.0, sigma_visible: float = 0.05,
                 odom_noise: float = 0.01, rot_noise_gain: float = 0.35,
                 decay_factor: float = 0.95):
        self.sigma_init = sigma_init
        self.sigma_visible = sigma_visible
        self.odom_noise = odom_noise
        self.rot_noise_gain = rot_noise_gain
        self.decay_factor = decay_factor
        self.reset()

    def reset(self):
        self.mu: Optional[np.ndarray] = None   # [x fwd, y left, z up], current robot-local frame
        self.sigma: float = self.sigma_init
        self.confidence: float = 0.0
        self.initialized: bool = False

    def observe(self, goal: np.ndarray, confidence: float = 1.0):
        """Fresh detection this tick -- snap to the measurement."""
        self.mu = np.asarray(goal, dtype=np.float32).copy()
        self.sigma = self.sigma_visible
        self.confidence = float(np.clip(confidence, 0.0, 1.0))
        self.initialized = True

    def propagate(self, dx: float, dy: float, dtheta: float):
        """No detection this tick -- carry mu forward by the rover's own
        measured motion instead of leaving it frozen.

        dx, dy: the rover's translation since the last tick, expressed in
        the rover's PREVIOUS-tick local frame (forward/left) -- i.e. how far
        it moved as measured from its own perspective a moment ago, not in
        world coordinates. dtheta: its heading change since then. Same
        SE(2)-inverse transform as belief_bank.py's ego_motion_update:
            p_new = R(-dtheta) @ (p_old - [dx, dy])

        z (goal height) is left untouched -- planar rover motion doesn't
        change a fixed point's height above the camera.
        """
        if not self.initialized or self.mu is None:
            return
        c, s = float(np.cos(-dtheta)), float(np.sin(-dtheta))
        p = self.mu[:2] - np.array([dx, dy], dtype=np.float32)
        self.mu[0] = c * p[0] - s * p[1]
        self.mu[1] = s * p[0] + c * p[1]
        self.sigma += self.odom_noise + self.rot_noise_gain * abs(dtheta)
        self.confidence = float(np.clip(self.confidence * self.decay_factor, 0.0, 1.0))
