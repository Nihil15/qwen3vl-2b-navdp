#!/usr/bin/env python3
"""DINO + NavDP Zenoh navigation node (GPU side, camera-only).

Speaks the SAME Zenoh contract as the previous OmniVLA / InternVLA-N1 nodes,
so the Pi rover / Isaac Sim side is unchanged:

Subscribes (Zenoh, CDR ROS 2 msgs):
  image_raw, rt/image_raw, rover_camera, rt/rover_camera     - sensor_msgs/Image
  image_raw/compressed, rt/image_raw/compressed              - sensor_msgs/CompressedImage
  depth_raw, rt/depth_raw                                    - sensor_msgs/Image (32FC1 m, 16UC1 mm,
                                                                 or custom "png16" PNG-compressed 16UC1
                                                                 mm -- see scripts/realsense_direct_
                                                                 publisher.py) [optional]
  image_raw/camera_info, rt/image_raw/camera_info            - sensor_msgs/CameraInfo [optional]
  omnivla/goal_text                                          - std_msgs/String (target phrase)
Publishes (Zenoh, CDR):
  cmd_vel              - geometry_msgs/Twist
  omnivla/explanation  - std_msgs/String
  omnivla/waypoints    - nav_msgs/Path (selected NavDP trajectory, base_link)

Inference core: Grounding DINO (open-vocab detection -> 3D point goal) +
NavDP System-1 (trajectory diffusion + critic) with goal-directed trajectory
selection. ~0.25s/tick on the RTX 3090 Ti. If no depth topic arrives, depth is
estimated monocularly (Depth Anything V2 metric).

Run (from Nav_new root, internnav conda env):
  python -m nav_pipeline.zenoh_node --target "trash bin" [--pi-ip <IP>]
"""

import argparse
import math
import os
import struct
import sys
import time
from threading import Lock, Thread
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh Python library not found (pip install eclipse-zenoh).")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402


# ================================================================
#  CDR helpers (DDS wire format) — same contract as previous nodes
# ================================================================
class CDRReader:
    def __init__(self, data: bytes):
        self.data = data
        self.le = data[1] in (0x01, 0x11)
        self.end = "<" if self.le else ">"
        self.offset = 4
        self.base = 4

    def _align(self, n: int):
        rem = (self.offset - self.base) % n
        if rem:
            self.offset += n - rem

    def read_uint8(self) -> int:
        v = self.data[self.offset]
        self.offset += 1
        return v

    def read_int32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "i", self.data, self.offset)
        self.offset += 4
        return v

    def read_uint32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "I", self.data, self.offset)
        self.offset += 4
        return v

    def read_float32(self) -> float:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "f", self.data, self.offset)
        self.offset += 4
        return v

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8
        return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset : self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data = self.data[self.offset : self.offset + count]
        self.offset += count
        return data


class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n: int):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_int32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<i", v)

    def write_uint32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_float64(self, v: float):
        self._align(8)
        self.buf += struct.pack("<d", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self.buf += encoded

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/Image CDR -> RGB uint8 (H,W,3) or float32 depth (H,W)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    height = r.read_uint32()
    width = r.read_uint32()
    encoding = r.read_string()
    r.read_uint8(); r._align(4); r.read_uint32()  # is_bigendian, step
    pixel_data = r.read_sequence_uint8()

    enc = encoding.lower()
    if enc in ("32fc1", "depth32f"):
        img = np.frombuffer(pixel_data, dtype=np.float32)
        try:
            return img.reshape(height, width).copy()
        except ValueError:
            return None
    if enc == "16uc1":  # depth in mm
        img = np.frombuffer(pixel_data, dtype=np.uint16)
        try:
            return (img.reshape(height, width).astype(np.float32) / 1000.0)
        except ValueError:
            return None
    if enc == "png16":
        # PNG-compressed uint16 depth (mm) -- custom encoding used by
        # scripts/realsense_direct_publisher.py (the raw-SDK Pi-side
        # publisher, see its module docstring for why it exists). Published
        # at HALF color's pixel dimensions (measured 2026-08-13: full-res
        # PNG averaged ~210KB/msg vs color JPEG's ~24KB/msg and lost most
        # messages over Wi-Fi even rate-capped; halving the pixel count
        # cuts PNG size ~4x). height/width read above are the ENCODED
        # (halved) dims here, not the final ones -- upsample back to
        # color's resolution with nearest-neighbor (no invented values
        # across real depth edges) so isaac_gui.py's/zenoh_node.py's
        # depth.shape==rgb.shape freshness gate still passes. Hardcodes
        # 640x480 as the target since that's this rig's fixed color
        # profile (rgb_camera.color_profile in realsense_only_bringup.
        # launch.py / realsense_direct_publisher.py) -- update both sides
        # together if that resolution ever changes.
        if cv2 is None:
            return None
        arr = np.frombuffer(pixel_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        return (img.astype(np.float32) / 1000.0)

    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None
    if enc == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    if img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def parse_camera_info(cdr_data: bytes) -> Optional[Tuple[float, float, float, float]]:
    """sensor_msgs/CameraInfo CDR -> (fx, fy, cx, cy) from the K[9] matrix.

    Only reads through height/width/distortion_model/d/k -- r, p, binning,
    roi are never consumed by this pipeline so they're left unparsed.
    """
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    r.read_uint32()  # height
    r.read_uint32()  # width
    r.read_string()  # distortion_model
    d_count = r.read_uint32()  # d[] (variable length) -- skip its payload
    r.offset += d_count * 8
    try:
        k = [r.read_float64() for _ in range(9)]
    except (struct.error, IndexError):
        return None
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    if fx <= 0 or fy <= 0:
        return None
    return fx, fy, cx, cy


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/CompressedImage CDR -> RGB uint8 (H,W,3)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    r.read_string()  # format
    jpeg_bytes = r.read_sequence_uint8()
    try:
        if cv2 is not None:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        import io

        from PIL import Image

        return np.array(Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def parse_string(cdr_data: bytes) -> str:
    return CDRReader(cdr_data).read_string()


def parse_float32_multiarray(cdr_data: bytes) -> List[float]:
    """std_msgs/Float32MultiArray CDR -> list[float] (the .data field)."""
    r = CDRReader(cdr_data)
    dim_count = r.read_uint32()
    for _ in range(dim_count):
        r.read_string()   # label
        r.read_uint32()    # size
        r.read_uint32()    # stride
    r.read_uint32()        # data_offset
    n = r.read_uint32()
    return [r.read_float32() for _ in range(n)]


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    w = CDRWriter()
    w.write_float64(linear_x); w.write_float64(0.0); w.write_float64(0.0)
    w.write_float64(0.0); w.write_float64(0.0); w.write_float64(angular_z)
    return w.to_bytes()


def serialize_string(text: str) -> bytes:
    w = CDRWriter()
    w.write_string(text)
    return w.to_bytes()


def serialize_path(points_xy: List[Tuple[float, float]], frame_id: str = "base_link") -> bytes:
    w = CDRWriter()
    w.write_int32(0); w.write_uint32(0); w.write_string(frame_id)
    w.write_uint32(len(points_xy))
    for px, py in points_xy:
        w.write_int32(0); w.write_uint32(0); w.write_string(frame_id)
        w.write_float64(float(px)); w.write_float64(float(py)); w.write_float64(0.0)
        w.write_float64(0.0); w.write_float64(0.0); w.write_float64(0.0); w.write_float64(1.0)
    return w.to_bytes()


# ================================================================
#  Node
# ================================================================
CAMERA_KEYS = ["image_raw", "rt/image_raw", "rover_camera", "rt/rover_camera"]
CAMERA_COMPRESSED_KEYS = ["image_raw/compressed", "rt/image_raw/compressed"]
DEPTH_KEYS = ["depth_raw", "rt/depth_raw", "rover_camera_depth", "rt/rover_camera_depth"]
# Real factory-calibrated intrinsics (sensor_msgs/CameraInfo), published
# alongside image_raw by realsense_only_bringup.launch.py. Optional: falls
# back to intrinsics_from_fov(...) (goal_utils.py) when absent/stale, same
# pattern as DEPTH_KEYS falling back to monocular depth estimation.
CAMERA_INFO_KEYS = ["image_raw/camera_info", "rt/image_raw/camera_info"]
GOAL_KEYS = ["omnivla/goal_text", "rt/omnivla/goal_text"]
# ESP32 signed wheel-encoder RPM (see esp32/rover_6wd_complete.ino) -> odometry
RPM_KEYS = ["rover/rpm", "rt/rover/rpm"]


class DinoNavDPZenohNode:
    # Rover firmware zeroes velocity if no cmd_vel arrives within ~500 ms;
    # heartbeat re-sends the last command well under that deadline.
    HEARTBEAT_PERIOD_S = 0.15
    DEPTH_STALE_S = 1.0  # external depth older than this -> fall back to monocular
    INTRINSICS_STALE_S = 5.0  # camera_info is static per session, no need for depth's tight 1s window
    # Spin-stall watchdog, ported from isaac_gui.py (which has always had
    # this -- see its comment for the real 145s/17-turn incident it guards
    # against). Replayed against odom_chair_20260729_170317.csv: this run
    # alone would have tripped it in 70 separate 15s windows, peak 493deg
    # turned for 0.36m net travel -- real-rover runs currently have NO
    # protection against this at all.
    # Loosened 2026-08-14 in lockstep with isaac_gui.py's copy (15s/1 turn ->
    # 20s/1.5 turns) at user request -- see isaac_gui.py's comment for why
    # this still catches the documented incident.
    SPIN_WINDOW_S = 20.0
    SPIN_ROT_THRESH_RAD = 3.0 * np.pi
    SPIN_DIST_THRESH_M = 0.3

    def __init__(self, session: zenoh.Session, pipeline: DinoNavDPPipeline, target: str, predict_hz: float,
                 stop_confirm_count: int = 3, odometry_log_dir: str = "odometry_log",
                 imu_min_mag_calib: int = 3):
        self.session = session
        self.pipe = pipeline
        self.target = target
        self.predict_hz = predict_hz
        self.stop_confirm_count = stop_confirm_count
        self.odom = OdometryLogger(odometry_log_dir, imu_min_mag_calib=imu_min_mag_calib)
        self.odom.start_new_goal(target)

        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.latest_intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy)
        self.latest_intrinsics_t = 0.0
        self._running = True
        self._frame_count = 0
        self._infer_count = 0
        self._stop_streak = 0
        self._goal_reached = False
        self._last_cmd = (0.0, 0.0)

        self.pub_cmd = session.declare_publisher("cmd_vel")
        self.pub_explain = session.declare_publisher("omnivla/explanation")
        self.pub_path = session.declare_publisher("omnivla/waypoints")
        self.subs = (
            [session.declare_subscriber(k, self._on_image) for k in CAMERA_KEYS]
            + [session.declare_subscriber(k, self._on_image_compressed) for k in CAMERA_COMPRESSED_KEYS]
            + [session.declare_subscriber(k, self._on_depth) for k in DEPTH_KEYS]
            + [session.declare_subscriber(k, self._on_camera_info) for k in CAMERA_INFO_KEYS]
            + [session.declare_subscriber(k, self._on_goal) for k in GOAL_KEYS]
            + [session.declare_subscriber(k, self._on_rpm) for k in RPM_KEYS]
        )
        print(f"[INFO] Zenoh subs: {CAMERA_KEYS + CAMERA_COMPRESSED_KEYS + DEPTH_KEYS + CAMERA_INFO_KEYS + GOAL_KEYS + RPM_KEYS}")
        print("[INFO] Zenoh pubs: cmd_vel, omnivla/explanation, omnivla/waypoints")

        Thread(target=self._heartbeat_loop, daemon=True).start()

    # ---------------- zenoh callbacks ---------------- #
    def _on_image(self, sample):
        try:
            img = parse_image(bytes(sample.payload))
            if img is not None and img.ndim == 3:
                with self.lock:
                    self.latest_rgb = img
                    self._frame_count += 1
        except Exception as e:
            print(f"[WARN] image parse failed: {e}")

    def _on_image_compressed(self, sample):
        try:
            img = parse_compressed_image(bytes(sample.payload))
            if img is not None:
                with self.lock:
                    self.latest_rgb = img
                    self._frame_count += 1
        except Exception as e:
            print(f"[WARN] compressed image parse failed: {e}")

    def _on_depth(self, sample):
        try:
            d = parse_image(bytes(sample.payload))
            if d is not None and d.ndim == 2:
                with self.lock:
                    self.latest_depth = d
                    self.latest_depth_t = time.time()
        except Exception as e:
            print(f"[WARN] depth parse failed: {e}")

    def _on_camera_info(self, sample):
        try:
            k = parse_camera_info(bytes(sample.payload))
            if k is not None:
                with self.lock:
                    self.latest_intrinsics = k
                    self.latest_intrinsics_t = time.time()
        except Exception as e:
            print(f"[WARN] camera_info parse failed: {e}")

    def _on_rpm(self, sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) >= 2:
                imu_heading = data[2] if len(data) >= 3 else None
                imu_calib = data[3] if len(data) >= 4 else None
                lateral_m_s = data[4] if len(data) >= 5 else None  # holonomic chassis only, see landerpi/bridge.py
                self.odom.update(data[0], data[1], imu_heading_deg=imu_heading, imu_calib=imu_calib,
                                  lateral_m_s=lateral_m_s)
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    def _on_goal(self, sample):
        try:
            text = parse_string(bytes(sample.payload)).strip()
            if text and text != self.target:
                print(f"[INFO] New target: '{text}'")
                self.target = text
                self._goal_reached = False
                self._stop_streak = 0
                self.pipe.reset()
                self.odom.start_new_goal(text)
        except Exception:
            pass

    # ---------------- publishing ---------------- #
    def _heartbeat_loop(self):
        while self._running:
            time.sleep(self.HEARTBEAT_PERIOD_S)
            with self.lock:
                lin, ang = self._last_cmd
            self.pub_cmd.put(serialize_twist(lin, ang))

    def publish_cmd(self, lin: float, ang: float):
        with self.lock:
            self._last_cmd = (lin, ang)
        self.pub_cmd.put(serialize_twist(lin, ang))

    # ---------------- main loop ---------------- #
    def spin(self):
        period = 1.0 / self.predict_hz
        last_status = time.time()
        print(f"[INFO] Loop at {self.predict_hz} Hz | target: '{self.target}'")
        print("[INFO] Waiting for camera frames via Zenoh...")
        try:
            while self._running:
                t0 = time.time()
                self._tick()
                if time.time() - last_status > 10.0:
                    heading_txt = (f"{self.odom.last_imu_heading_deg:.1f}°"
                                   if self.odom.last_imu_heading_deg is not None
                                   and math.isfinite(self.odom.last_imu_heading_deg) else "n/a")
                    print(f"[STATUS] frames_rx={self._frame_count} inferences={self._infer_count} "
                          f"goal={'REACHED' if self._goal_reached else 'tracking'} "
                          f"theta_src={self.odom.theta_source} imu_heading={heading_txt} "
                          f"calib=[{OdometryLogger.decode_calib(self.odom.last_imu_calib)}]")
                    last_status = time.time()
                dt = period - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
        finally:
            self._running = False
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.1)
            self.publish_cmd(0.0, 0.0)
            self.odom.close()
            print("[INFO] Sent zero velocity. Goodbye.")

    def _tick(self):
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
            depth_age = time.time() - self.latest_depth_t
            intrinsics = self.latest_intrinsics
            intrinsics_age = time.time() - self.latest_intrinsics_t
        if rgb is None:
            return
        if self._goal_reached:
            self.publish_cmd(0.0, 0.0)
            return
        if depth is not None and (depth_age > self.DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None  # stale or mismatched -> monocular fallback
        if intrinsics is not None and intrinsics_age > self.INTRINSICS_STALE_S:
            intrinsics = None

        t0 = time.time()
        try:
            if self.pipe.cfg.use_qwen_instruction:
                # --use-qwen-instruction: self.target (set from --target /
                # the goal_text Zenoh topic) is read as a free-text
                # INSTRUCTION for Qwen to ground into NavDP's goal, not a
                # DINO object phrase -- see qwen_pixel_goal.py.
                res = self.pipe.step(rgb, "", depth=depth, pose=(self.odom.x, self.odom.y, self.odom.theta),
                                      intrinsics=intrinsics, instruction=self.target)
            else:
                res = self.pipe.step(rgb, self.target, depth=depth, pose=(self.odom.x, self.odom.y, self.odom.theta),
                                      intrinsics=intrinsics)
        except Exception as e:
            print(f"[ERROR] pipeline step failed: {e}")
            self.publish_cmd(0.0, 0.0)
            return
        infer_ms = (time.time() - t0) * 1000.0

        spin = self.odom.spin_delta(self.SPIN_WINDOW_S)
        if spin is not None and spin[0] > self.SPIN_ROT_THRESH_RAD and spin[1] < self.SPIN_DIST_THRESH_M:
            print(f"[WARN] spin-stall watchdog: turned {spin[0]:.1f}rad in {self.SPIN_WINDOW_S:.0f}s, "
                  f"only {spin[1]:.2f}m net travel -- forcing stop until a new target is sent")
            self._goal_reached = True
            self.publish_cmd(0.0, 0.0)
            self.pub_explain.put(serialize_string(
                f"SPIN STALL: {np.degrees(spin[0]):.0f} deg turned, {spin[1]:.2f}m travel -- send a new target"
            ))
            return

        if res.state == "STOP":
            self._stop_streak += 1
            if self._stop_streak >= self.stop_confirm_count:
                print(f"\n{'=' * 50}\n  GOAL REACHED: '{self.target}'\n{'=' * 50}\n")
                self._goal_reached = True
                self.publish_cmd(0.0, 0.0)
                self.pub_explain.put(serialize_string(f"GOAL REACHED: '{self.target}'. Stopping."))
                return
            self.publish_cmd(0.0, 0.0)
        else:
            self._stop_streak = 0
            self.publish_cmd(res.linear, res.angular)

        if res.trajectory is not None:
            self.pub_path.put(serialize_path([(p[0], p[1]) for p in res.trajectory]))

        score = f"{res.detection.score:.2f}" if res.detection else "-"
        goal = np.round(res.goal_point, 2).tolist() if res.goal_point is not None else None
        amb = f" [AMBIGUOUS x{res.candidate_count}]" if res.ambiguous else ""
        self.pub_explain.put(serialize_string(
            f"DINO+NavDP [{res.state}]{amb} det={score} goal={goal} -> lin={res.linear:.3f} ang={res.angular:.3f} "
            f"| target='{self.target}'"
        ))
        self._infer_count += 1
        if self._infer_count <= 5 or self._infer_count % 20 == 0:
            print(f"[PRED #{self._infer_count}] {res.state} det={score} goal={goal} "
                  f"lin={res.linear:.3f} ang={res.angular:+.3f} ({infer_ms:.0f}ms)")


def main():
    p = argparse.ArgumentParser(description="DINO+NavDP Zenoh navigation node",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pi-ip", type=str, default=None,
                   help="Pi/Isaac IP for explicit Zenoh peer; omit for multicast scouting")
    p.add_argument("--target", type=str, default="trash bin", help="object to navigate to (free text)")
    p.add_argument("--predict-hz", type=float, default=3.0)
    p.add_argument("--max-linear", type=float, default=0.15)
    p.add_argument("--max-angular", type=float, default=0.25)
    p.add_argument("--fov", type=float, default=90.0, help="camera horizontal FOV (deg)")
    p.add_argument("--stop-distance", type=float, default=1.5)
    p.add_argument("--angular-slew-max", type=float, default=0.10,
                   help="rad/s hard cap on angular command change per tick, all states (0 disables)")
    p.add_argument("--invert-angular", action="store_true",
                   help="flip turn direction (use if the rover steers away from the target)")
    p.add_argument("--no-belief-goal", action="store_true",
                   help="disable ego-motion goal belief (see goal_belief.py); reverts to "
                        "coasting on the frozen last-seen goal while the target is lost")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--odometry-log-dir", type=str, default="odometry_log",
                   help="dead-reckoned pose CSV log dir (from /rover/rpm)")
    p.add_argument("--imu-min-mag-calib", type=int, default=3,
                   help="IMU calibration digit (0-3) required before theta rides the IMU "
                        "heading instead of wheel-diff dead reckoning -- see OdometryLogger. "
                        "Was only exposed on home_gui.py before; same default/semantics here.")
    p.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length,
                   help="robot length (m) for obstacle_guard's swept-footprint clearance -- "
                        "defaults to the ESP32 rover's real size, override for a different robot "
                        "(e.g. the LanderPi, see landerpi/README.md) before trusting obstacle avoidance")
    p.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width,
                   help="robot width (m), see --footprint-length")
    p.add_argument("--no-dino", action="store_true",
                   help="skip loading Grounding DINO entirely (saves VRAM/load time) -- only "
                        "safe with --use-qwen-instruction AND no --avoid, ever, this run")
    p.add_argument("--wheel-deadband-correction", action="store_true",
                   help="ESP32 6WD rover ONLY (never --hiwonder) -- boosts a commanded "
                        "(linear, angular) that would otherwise differential-mix down to a "
                        "stalled wheel on the real hardware; see pipeline.py's "
                        "clear_wheel_deadband")
    p.add_argument("--use-qwen-search", action="store_true",
                   help="while DINO can't see --target, ask a frozen Qwen2.5-VL-7B-Instruct "
                        "to point toward the best direction to search/move next instead of "
                        "the fixed spin toward the last-known side; dropped automatically "
                        "the instant DINO reacquires the target (see qwen_search_guide.py)")
    p.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HF model id for --use-qwen-search")
    p.add_argument("--qwen-search-period-s", type=float, default=2.0,
                   help="min seconds between Qwen search-guidance calls (throttle)")
    p.add_argument("--qwen-fp16", action="store_true",
                   help="load Qwen2.5-VL-7B in fp16 instead of the 4-bit default (needs "
                        "~16.6GB VRAM vs 4-bit's ~6.2GB; use if bitsandbytes isn't installed)")
    p.add_argument("--use-qwen-instruction", action="store_true",
                   help="DINO target detection is bypassed entirely: --target is read as a "
                        "free-text INSTRUCTION for Qwen2.5-VL to ground into NavDP's goal via "
                        "depth, same as a DINO detection would. See qwen_pixel_goal.py.")
    p.add_argument("--qwen-instruction-period-s", type=float, default=1.5,
                   help="min seconds between Qwen instruction-grounding calls (throttle)")
    p.add_argument("--qwen-goal-consistency-m", type=float, default=1.5,
                   help="reject a new Qwen instruction-grounding more than this many meters "
                        "from the current confident goal instead of snapping to it (e.g. a "
                        "second door coming into view); see pipeline.py's "
                        "qwen_goal_consistency_m for the full explanation")
    p.add_argument("--qwen-max-candidates", type=int, default=3,
                   help="ask Qwen for up to this many ranked, confidence-scored waypoint "
                        "candidates instead of one, scored by semantic confidence + continuity "
                        "+ obstacle cost; 1 disables scoring. See PipelineConfig.qwen_max_candidates.")
    args = p.parse_args()

    cfg = PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        stop_distance=args.stop_distance,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        use_belief_goal=not args.no_belief_goal,
        guard=GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width),
        wheel_deadband_correction=args.wheel_deadband_correction,
        use_dino=not args.no_dino,
        use_qwen_search=args.use_qwen_search,
        qwen_model_id=args.qwen_model_id,
        qwen_search_period_s=args.qwen_search_period_s,
        qwen_load_in_4bit=not args.qwen_fp16,
        use_qwen_instruction=args.use_qwen_instruction,
        qwen_instruction_period_s=args.qwen_instruction_period_s,
        qwen_goal_consistency_m=args.qwen_goal_consistency_m,
        qwen_max_candidates=args.qwen_max_candidates,
    )
    pipeline = DinoNavDPPipeline(cfg)

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting (auto-discover)")
    session = zenoh.open(config)
    print("[INFO] Zenoh session opened.")

    node = DinoNavDPZenohNode(session, pipeline, args.target, args.predict_hz,
                              odometry_log_dir=args.odometry_log_dir,
                              imu_min_mag_calib=args.imu_min_mag_calib)
    node.spin()
    session.close()


if __name__ == "__main__":
    main()
