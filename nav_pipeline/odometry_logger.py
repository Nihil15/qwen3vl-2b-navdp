"""Dead-reckoned odometry from the ESP32's signed wheel-encoder RPM feed.

The rover has no LiDAR and no dedicated odometry node, but the firmware
(esp32/rover_6wd_complete.ino) already publishes real encoder data on
/rover/rpm (std_msgs/Float32MultiArray [left_rpm, right_rpm], signed, +ve =
wheel drives the robot forward, 10 Hz). This integrates that into a
differential-drive pose (x, y, theta) and appends one CSV row per sample.

One file per GOAL, not per run: start_new_goal() closes whatever file is
open and starts a fresh one, named after the target text.

Pose (x, y, theta) is CONTINUOUS across goals by default -- it does NOT
reset when start_new_goal() rotates the log file. This is required for
object-location memory (nav_pipeline/object_map.py) and blind navigate-back
(pipeline.py's GOTO state) to work at all: a world location remembered from
one goal is meaningless if the very next goal restarts the origin at
wherever the rover happens to be standing. Call reset_pose() explicitly
(e.g. an operator "reset map" action) if you actually want a fresh origin --
start_new_goal(..., reset_pose=True) does both at once for callers that want
the old per-goal-origin behavior back.
"""

import csv
import math
import os
import re
import time
from collections import deque
from typing import Optional

# Must match esp32/rover_6wd_complete.ino WHEEL_RADIUS_M / TRACK_WIDTH_M.
WHEEL_RADIUS_M = 0.056
TRACK_WIDTH_M = 0.345

# Below this, both wheels count as "not turning" for IMU-heading gating (see
# _imu_theta) -- real driven rpm is always several times this on both bots,
# so it comfortably separates genuine motion from encoder-register noise
# while still catching the first tick or two of a real ramp-up/down.
MIN_MOVING_RPM = 0.5


class OdometryLogger:
    # retention for the spin_delta() rolling window -- must be >= the widest
    # window_s a caller will query
    HISTORY_WINDOW_S = 30.0

    def __init__(self, log_dir: str = "odometry_log", imu_min_mag_calib: int = 3):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path: Optional[str] = None
        self._file = None
        self._writer = None
        self.x = self.y = self.theta = 0.0
        self._last_t: Optional[float] = None
        self._history: deque = deque()  # (t, x, y, theta, |dtheta| this tick), newest last
        # Optional IMU heading fusion (see update()'s imu_heading_deg/imu_calib
        # args) -- off unless a caller actually passes those, so every
        # existing 2-arg update(left_rpm, right_rpm) call site (zenoh_node.py,
        # isaac_gui.py, remind_gui.py) is byte-for-byte unaffected.
        self.imu_min_mag_calib = imu_min_mag_calib
        self._imu_heading0_deg: Optional[float] = None
        self._last_imu_heading_deg_raw: Optional[float] = None
        self.theta_source = "enc"
        # last raw values seen in update() -- kept regardless of whether they
        # were good enough to gate theta onto the IMU (see is_imu_calibrated),
        # so a caller (e.g. remind_gui.py's status display) can show live
        # calibration state without needing its own separate rpm subscriber.
        self.last_imu_heading_deg: Optional[float] = None
        self.last_imu_calib: Optional[float] = None

    @staticmethod
    def _slug(text: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
        return s[:40] or "target"

    def start_new_goal(self, target: str, reset_pose: bool = False):
        """Close the current file (if any) and start a fresh one for this goal.

        Pose is preserved by default (see module docstring) -- a new goal is
        just a new CSV file in the same continuous world frame. Pass
        reset_pose=True to also zero (x, y, theta), e.g. an explicit operator
        "reset map" action, not an ordinary target switch.
        """
        if self._file is not None:
            self._file.close()
        fname = f"odom_{self._slug(target)}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = os.path.join(self.log_dir, fname)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", "dt", "left_rpm", "right_rpm", "v", "w", "x", "y", "theta",
                               "imu_heading_deg", "theta_src", "lateral_m_s"])
        self._last_t = None
        if reset_pose:
            self.reset_pose()
        print(f"[odometry] goal '{target}' -> logging to {self.path} "
              f"(pose {'reset' if reset_pose else f'continued at ({self.x:.2f}, {self.y:.2f}, {self.theta:+.2f})'})")

    def reset_pose(self):
        """Zero (x, y, theta) and clear the spin_delta() history -- a genuine
        fresh origin, distinct from start_new_goal()'s normal file rotation
        (see module docstring)."""
        self.x = self.y = self.theta = 0.0
        self._history.clear()
        self._imu_heading0_deg = None
        self.theta_source = "enc"

    @staticmethod
    def decode_calib(imu_calib: Optional[float]) -> str:
        """Unpack the packed calib byte (sys*1000 + gyr*100 + acc*10 + mag,
        see esp32/rover_6wd_complete.ino / landerpi/bridge.py) into a
        readable "SYS.. GYR.. ACC.. MAG.." string, each sub-score 0-3. On
        the LanderPi's BNO055 these are genuinely independent per-axis
        scores; on the rover's BNO085/BNO08x (since 2026-08-14, see
        rover_6wd_complete.ino's IMU header note) the chip only reports ONE
        combined accuracy status, broadcast into all four digits, so they
        read identically there -- decode_calib() doesn't need to know which
        chip it's looking at, it just unpacks the four digits either way."""
        if imu_calib is None:
            return "no IMU data"
        v = int(round(imu_calib))
        sys_c, gyr_c, acc_c, mag_c = (v // 1000) % 10, (v // 100) % 10, (v // 10) % 10, v % 10
        return f"SYS{sys_c} GYR{gyr_c} ACC{acc_c} MAG{mag_c}"

    def is_imu_calibrated(self) -> bool:
        """Whether the last-seen IMU sample is trustworthy enough to be
        driving theta (see _imu_theta's gating below) -- i.e. the same
        MAG-only check, but queryable at any time (e.g. remind_gui.py's
        status display) rather than only implicitly via theta_source."""
        return self._mag_calib_ok(self.last_imu_calib)

    def _mag_calib_ok(self, imu_calib: Optional[float]) -> bool:
        if imu_calib is None:
            return False
        mag_calib = int(round(imu_calib)) % 10
        return mag_calib >= self.imu_min_mag_calib

    def _imu_theta(self, imu_heading_deg: Optional[float], imu_calib: Optional[float],
                    moving: bool) -> Optional[float]:
        """Convert a raw firmware IMU heading (deg, compass CW+ -- both the
        LanderPi's BNO055 and the rover's BNO085/BNO08x report this same
        convention, see esp32/rover_6wd_complete.ino's IMU header note on
        why the BNO08x's firmware code deliberately negates its otherwise
        CCW+ yaw to match) into this run's theta convention (rad, CCW+,
        zeroed at start_new_goal()) -- or None if it isn't trustworthy yet,
        in which case the caller falls back to wheel-diff dead reckoning.

        Gated on the MAG sub-score of the packed imu_calib byte (sys*1000 +
        gyr*100 + acc*10 + mag): magnetometer calibration is what anchors the
        heading to an absolute reference and stops it drifting, so an
        uncalibrated mag reading is worse than no IMU at all. GYR/ACC/SYS
        aren't gated on -- verified live (see memory) that heading tracked a
        real 90deg turn to within ~1.6deg with SYS still stuck at 0.

        Also gated on `moving` (see MIN_MOVING_RPM): live-logged on the
        LanderPi (landerpi/README.md's IMU section), the BNO055 Euler heading
        drifted up to ~150deg total across a session while both wheel
        encoders read exactly 0 rpm -- a chassis can't physically rotate
        without its wheels turning, so that drift is sensor noise, not real
        motion, and previously got baked straight into theta (and from there
        into every subsequent x/y integration, visibly bending the path in
        home_gui).

        Whenever this tick isn't trusted (not moving, or MAG calibration
        below threshold), the reference is continuously re-synced against
        the CURRENT theta (not just the raw IMU reading) rather than left
        untouched. Live logs caught the untouched version of this bug
        directly: MAG calibration flickers below threshold and back
        mid-drive (BNO055 confidence bounces near the motors), and while it
        was below threshold theta correctly free-ran on wheel-diff dead
        reckoning up to 151deg away from wherever the session's very first
        IMU sample happened to be -- then the instant MAG recovered, theta
        snapped straight back to (stale reference - current reading),
        discarding all the real turning that had happened in between. Both
        gates funnel through this same re-sync so trust resuming -- wheels
        moving again, or calibration recovering -- always continues smoothly
        from wherever theta actually is, never jumps to a stale absolute
        reference.
        """
        if imu_heading_deg is None or not math.isfinite(imu_heading_deg):
            return None
        mag_ok = imu_calib is None or self._mag_calib_ok(imu_calib)
        if not (mag_ok and moving):
            self._imu_heading0_deg = math.degrees(self.theta) + imu_heading_deg
            return None
        if self._imu_heading0_deg is None:
            self._imu_heading0_deg = imu_heading_deg  # zero the reference at the first good sample (theta is 0 here)
        delta_deg = self._imu_heading0_deg - imu_heading_deg  # compass CW+ -> theta CCW+
        delta_deg = (delta_deg + 180.0) % 360.0 - 180.0
        return math.radians(delta_deg)

    def update(self, left_rpm: float, right_rpm: float, t: Optional[float] = None,
               imu_heading_deg: Optional[float] = None, imu_calib: Optional[float] = None,
               lateral_m_s: Optional[float] = None):
        """Feed one /rover/rpm sample; integrates and appends a CSV row.

        No-op (returns None) until start_new_goal() has opened a file.
        imu_heading_deg/imu_calib are optional (see esp32/rover_6wd_complete.ino's
        rpm_data[2]/[3]): when given and the magnetometer is calibrated, theta
        is set directly from the IMU instead of integrated from the wheel
        differential -- the IMU has no accumulating drift and doesn't get
        thrown off by wheel slip during in-place turns, both of which corrupt
        the wheel-only heading over a long run. x/y are still advanced from
        encoder-derived speed either way (translation distance is what
        encoders measure well).

        lateral_m_s is optional (default None -> every existing 2-arg/4-arg
        call site, e.g. the ESP32 skid-steer rover, is byte-for-byte
        unaffected): a body-frame sideways velocity (+left, m/s), for
        holonomic chassis (e.g. the LanderPi's Mecanum wheels, see
        landerpi/bridge.py) where real lateral motion/slip is possible and
        otherwise invisible to this rover-inherited unicycle model.
        """
        # Captured unconditionally, even before start_new_goal() has opened a
        # file (self._writer is None below) -- an operator checking IMU
        # calibration status shouldn't have to send a target first just to
        # get a live reading (see is_imu_calibrated).
        if imu_heading_deg is not None and math.isfinite(imu_heading_deg):
            self.last_imu_heading_deg = imu_heading_deg
        if imu_calib is not None:
            self.last_imu_calib = imu_calib

        if self._writer is None:
            return None
        t = time.time() if t is None else t
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        v_left = left_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v_right = right_rpm * 2.0 * math.pi * WHEEL_RADIUS_M / 60.0
        v = 0.5 * (v_left + v_right)
        w = (v_right - v_left) / TRACK_WIDTH_M

        lateral = 0.0 if lateral_m_s is None else lateral_m_s
        moving = abs(left_rpm) > MIN_MOVING_RPM or abs(right_rpm) > MIN_MOVING_RPM

        dtheta = 0.0
        if dt > 0.0:
            dtheta = w * dt
            imu_theta = self._imu_theta(imu_heading_deg, imu_calib, moving)
            if imu_theta is not None:
                self.theta = imu_theta
                self.theta_source = "imu"
            else:
                self.theta += dtheta
                self.theta_source = "enc"
            # Body-frame (v forward, lateral left) rotated into world frame.
            # lateral is 0.0 for every non-holonomic caller, reducing this to
            # the original x += v*cos(theta)*dt / y += v*sin(theta)*dt exactly.
            self.x += (v * math.cos(self.theta) - lateral * math.sin(self.theta)) * dt
            self.y += (v * math.sin(self.theta) + lateral * math.cos(self.theta)) * dt

        imu_field = f"{imu_heading_deg:.4f}" if imu_heading_deg is not None and math.isfinite(imu_heading_deg) else ""
        lateral_field = f"{lateral_m_s:.4f}" if lateral_m_s is not None else ""
        self._writer.writerow([f"{t:.6f}", f"{dt:.4f}", left_rpm, right_rpm,
                               f"{v:.4f}", f"{w:.4f}", f"{self.x:.4f}", f"{self.y:.4f}", f"{self.theta:.4f}",
                               imu_field, self.theta_source, lateral_field])
        self._file.flush()

        self._history.append((t, self.x, self.y, self.theta, abs(dtheta)))
        cutoff = t - self.HISTORY_WINDOW_S
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        return self.x, self.y, self.theta

    def spin_delta(self, window_s: float):
        """(total |dtheta| turned, net dist) over the trailing window_s, or
        None if not enough history yet.

        Used to catch a rover that's spinning in place (e.g. bouncing between
        several same-class detections around it) without net translation --
        see _select_detection's "ping-pong" note in pipeline.py. Rotation is
        the SUM of per-tick |dtheta|, not |theta_end - theta_start|: a rover
        oscillating left/right between two same-class detections (the exact
        ping-pong failure mode this watchdog exists for) can have its turns
        cancel out to a small net theta change while still spinning wildly in
        place -- verified against a real run where net rotation in any 15s
        window never crossed the 2*pi trigger threshold (peaked at 326 deg)
        while total absolute rotation crossed it in 99 separate windows over
        the same run (peaked at 513 deg/15s) for ~1m of net travel in 142s.
        """
        if not self._history:
            return None
        t_now = self._history[-1][0]
        oldest = None
        turned = 0.0
        for sample in self._history:
            if t_now - sample[0] <= window_s:
                if oldest is None:
                    oldest = sample
                turned += sample[4]
        if oldest is None or oldest[0] == t_now:
            return None
        _, x0, y0, _, _ = oldest
        _, x1, y1, _, _ = self._history[-1]
        return turned, math.hypot(x1 - x0, y1 - y0)

    def close(self):
        if self._file is not None:
            self._file.close()
