#!/usr/bin/env python3
"""DINO + NavDP Isaac Sim GUI — Nav_new's own control panel.

Replaces the old OmniVLA isaac_gui.py: same Zenoh contract, new inference
core (Grounding DINO -> point goal -> NavDP cross-modal -> trajectory
selection -> cmd_vel).

Features
--------
- Live camera feed from rover_camera / image_raw via Zenoh
- DINO detection bbox + score overlaid on the feed
- Target text entry (Enter or Send) + preset buttons
- State (TRACK / SEARCH / STOP), velocity readout, per-stage latency
- Top-down plot of the 32 sampled NavDP trajectories + the selected one
- STOP button (zero velocity, inference paused until new target)

Run (from Nav_new root):
    conda activate internnav
    python -m nav_pipeline.isaac_gui [--target "trash bin"] [--pi-ip <IP>]
"""

import argparse
import os
import signal
import sys
import time
import traceback
from threading import Lock, Thread
from typing import Optional

import numpy as np

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not found (pip install eclipse-zenoh)")
    sys.exit(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nav_pipeline.obstacle_guard import GuardConfig  # noqa: E402
from nav_pipeline.odometry_logger import OdometryLogger  # noqa: E402
from nav_pipeline.pipeline import DinoNavDPPipeline, PipelineConfig  # noqa: E402
from nav_pipeline.zenoh_node import (  # noqa: E402
    CAMERA_COMPRESSED_KEYS,
    CAMERA_INFO_KEYS,
    CAMERA_KEYS,
    DEPTH_KEYS,
    RPM_KEYS,
    parse_camera_info,
    parse_compressed_image,
    parse_float32_multiarray,
    parse_image,
    serialize_path,
    serialize_string,
    serialize_twist,
)

PRESETS = ["trash bin", "cardboard box", "wooden pallet", "door", "chair"]
HEARTBEAT_PERIOD_S = 0.15
DEPTH_STALE_S = 1.0
INTRINSICS_STALE_S = 5.0  # camera_info is static per session, no need for depth's tight 1s window
# Spin-stall watchdog: with a generic/multi-instance target (e.g. "chair" in a
# room full of chairs), DINO's re-acquire-on-loss can hop to a different
# physical object each time the tracked one scrolls out of frame, so the
# rover keeps turning the same way chasing "whichever chair is now in view"
# without ever closing distance on one -- a real 145s/17-turn incident this
# caught via odometry_log. If it racks up more than SPIN_ROT_THRESH_RAD within
# this window while translating less than SPIN_DIST_THRESH_M, force a stop
# instead of spinning indefinitely; latched until a new target is sent.
# Loosened 2026-08-14 (15s/1 turn -> 20s/1.5 turns) at user request -- more
# slack for a single legitimate reacquire swing to complete before being
# judged a stall. Still reliably catches the documented incident: 17 turns
# in 145s is ~2.3 turns per any 20s window, well past the new 1.5-turn bar.
SPIN_WINDOW_S = 20.0
SPIN_ROT_THRESH_RAD = 3.0 * np.pi
SPIN_DIST_THRESH_M = 0.3


class SharedState:
    def __init__(self, target: str):
        self.lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_t = 0.0
        self.latest_intrinsics: Optional[tuple] = None  # (fx, fy, cx, cy)
        self.latest_intrinsics_t = 0.0
        self.frame_count = 0
        self.mode = "manual"                    # "text" | "manual" -- starts inert: no
        #                                          target means nothing to autonomously
        #                                          navigate to; send one to switch to "text"
        self.target = target
        self.avoid = ""                          # named obstacle text, "" = disabled; see --avoid
        self.stopped = False
        self.goal_reached = False
        self.last_cmd = (0.0, 0.0)
        self.max_linear = 0.5                   # manual-drive caps; set from CLI args in main()
        self.max_angular = 0.6
        # for display
        self.display_rgb: Optional[np.ndarray] = None
        self.detection = None
        self.avoid_detection = None
        self.mask: Optional[np.ndarray] = None
        self.state_text = "waiting for camera"
        self.vel_text = "lin 0.000  ang +0.000"
        self.lat_text = ""
        self.trajs = None
        self.chosen = None
        self.goal_pt = None
        self.qwen_pixel_goal = None   # (u, v) -- set only on ticks --use-qwen-instruction fired
        self.obstacles = None
        self.min_forward = float("inf")
        self.infer_count = 0
        # IMU status (see zenoh_node.py's on_rpm) -- surfaced in the info bar
        # so an operator can see whether pose/goal-belief tracking is
        # currently riding the IMU heading or has fallen back to wheel-diff,
        # same fields home_gui.py/remind_gui.py already show.
        self.imu_heading_deg = float("nan")
        self.imu_calib = 0.0
        self.theta_source = "enc"


def zenoh_setup(session: zenoh.Session, st: SharedState, compressed_only: bool = False,
                odom: Optional[OdometryLogger] = None):
    def on_image(sample):
        try:
            img = parse_image(bytes(sample.payload))
            if img is not None and img.ndim == 3:
                with st.lock:
                    st.latest_rgb = img
                    st.frame_count += 1
        except Exception as e:
            print(f"[WARN] image parse failed: {e}")

    def on_compressed(sample):
        try:
            img = parse_compressed_image(bytes(sample.payload))
            if img is not None:
                with st.lock:
                    st.latest_rgb = img
                    st.frame_count += 1
        except Exception as e:
            print(f"[WARN] compressed image parse failed: {e}")

    def on_depth(sample):
        try:
            d = parse_image(bytes(sample.payload))
            if d is not None and d.ndim == 2:
                with st.lock:
                    st.latest_depth = d
                    st.latest_depth_t = time.time()
        except Exception as e:
            print(f"[WARN] depth parse failed: {e}")

    def on_camera_info(sample):
        try:
            k = parse_camera_info(bytes(sample.payload))
            if k is not None:
                with st.lock:
                    st.latest_intrinsics = k
                    st.latest_intrinsics_t = time.time()
        except Exception as e:
            print(f"[WARN] camera_info parse failed: {e}")

    def on_rpm(sample):
        if odom is None:
            return
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) >= 2:
                # [left_rpm, right_rpm, imu_heading_deg, imu_calib, lateral_m_s]
                # -- matches home_gui.py's/zenoh_node.py's on_rpm parsing.
                # Dropping data[2:] here (as this used to) meant IMU calib
                # never reached OdometryLogger, so callers of zenoh_setup()
                # (remind_gui.py, this file) always showed "no data received
                # yet" regardless of actual IMU calibration state.
                imu_heading = data[2] if len(data) >= 4 else None
                imu_calib = data[3] if len(data) >= 4 else None
                lateral = data[4] if len(data) >= 5 else None
                odom.update(data[0], data[1], imu_heading_deg=imu_heading,
                            imu_calib=imu_calib, lateral_m_s=lateral)
                with st.lock:
                    st.theta_source = odom.theta_source
                    if imu_heading is not None:
                        st.imu_heading_deg = imu_heading
                    if imu_calib is not None:
                        st.imu_calib = imu_calib
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    # On the real rover Wi-Fi, raw 640x480 rgb8 (~8 MB/s) saturates the link
    # (starving cmd_vel/rpm and even SSH) — subscribe compressed JPEG only.
    raw_keys = [] if compressed_only else CAMERA_KEYS
    subs = (
        [session.declare_subscriber(k, on_image) for k in raw_keys]
        + [session.declare_subscriber(k, on_compressed) for k in CAMERA_COMPRESSED_KEYS]
        + [session.declare_subscriber(k, on_depth) for k in DEPTH_KEYS]
        + [session.declare_subscriber(k, on_camera_info) for k in CAMERA_INFO_KEYS]
        + [session.declare_subscriber(k, on_rpm) for k in RPM_KEYS]
    )
    pubs = {
        "cmd": session.declare_publisher("cmd_vel"),
        "explain": session.declare_publisher("omnivla/explanation"),
        "path": session.declare_publisher("omnivla/waypoints"),
    }
    return subs, pubs


def heartbeat_loop(st: SharedState, pubs, running):
    while running["on"]:
        time.sleep(HEARTBEAT_PERIOD_S)
        with st.lock:
            lin, ang = st.last_cmd
        pubs["cmd"].put(serialize_twist(lin, ang))


def inference_loop(pipe: DinoNavDPPipeline, st: SharedState, pubs, running,
                   predict_hz: float, stop_confirm: int = 3, odom: Optional[OdometryLogger] = None):
    period = 1.0 / predict_hz
    stop_streak = 0
    last_target = None  # None so the first tick's target starts its own odometry file
    while running["on"]:
        t0 = time.time()
        with st.lock:
            rgb = st.latest_rgb
            depth = st.latest_depth
            depth_age = time.time() - st.latest_depth_t
            intrinsics = st.latest_intrinsics
            intrinsics_age = time.time() - st.latest_intrinsics_t
            mode = st.mode
            target = st.target
            avoid = st.avoid
            paused = st.stopped or st.goal_reached
        if target and target != last_target:
            if last_target is not None:
                # new goal: don't let tracked-box/goal-belief state from the
                # PREVIOUS target leak into this one (stale pose across the
                # odom reset below was corrupting belief's first ego-motion
                # delta on a target switch)
                pipe.reset()
            if odom is not None:
                odom.start_new_goal(target)
            last_target = target
        if rgb is None:
            time.sleep(0.1)
            continue
        if mode == "manual":
            # bypass detection/NavDP entirely -- last_cmd is set directly by
            # the GUI's manual-drive button/key handlers; this loop only
            # keeps the camera preview and status text current, and still
            # honors STOP as an emergency zero
            with st.lock:
                if st.stopped:
                    st.last_cmd = (0.0, 0.0)
                lin, ang = st.last_cmd
                st.display_rgb = rgb
                st.state_text = "MANUAL (stopped)" if st.stopped else "MANUAL DRIVE"
                st.vel_text = f"lin {lin:.3f}  ang {ang:+.3f}"
            time.sleep(0.05)
            continue
        if paused:
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.1)
            continue
        if depth is not None and (depth_age > DEPTH_STALE_S or depth.shape[:2] != rgb.shape[:2]):
            depth = None
        if intrinsics is not None and intrinsics_age > INTRINSICS_STALE_S:
            intrinsics = None

        try:
            pose = (odom.x, odom.y, odom.theta) if odom is not None else None
            if pipe.cfg.use_qwen_instruction:
                # --use-qwen-instruction: the same text box/--target value
                # is read as a free-text INSTRUCTION for Qwen to ground into
                # NavDP's goal, not a DINO object phrase -- see
                # qwen_pixel_goal.py and PipelineConfig.use_qwen_instruction.
                res = pipe.step(rgb, "", depth=depth, pose=pose, avoid_text=avoid,
                                 intrinsics=intrinsics, instruction=target)
            else:
                res = pipe.step(rgb, target, depth=depth, pose=pose, avoid_text=avoid, intrinsics=intrinsics)
        except Exception as e:
            print(f"[ERROR] pipeline step: {e}")
            with st.lock:
                st.last_cmd = (0.0, 0.0)
                # keep the preview live even while inference is erroring --
                # otherwise refresh() keeps showing the last successful
                # display_rgb forever (it only falls back to latest_rgb when
                # display_rgb is None), which reads as a frozen camera feed
                # even though frames are still arriving fine over Zenoh.
                st.display_rgb = rgb
                st.state_text = f"ERROR: {e}"
                st.vel_text = "lin 0.000  ang +0.000"
            time.sleep(0.5)
            continue

        spin = odom.spin_delta(SPIN_WINDOW_S) if odom is not None else None
        if spin is not None and spin[0] > SPIN_ROT_THRESH_RAD and spin[1] < SPIN_DIST_THRESH_M:
            print(f"[WARN] spin-stall watchdog: turned {spin[0]:.1f}rad in {SPIN_WINDOW_S:.0f}s, "
                  f"only {spin[1]:.2f}m net travel -- forcing stop until a new target is sent")
            with st.lock:
                st.goal_reached = True  # reuses the existing pause-until-new-target gate
                st.last_cmd = (0.0, 0.0)
                st.display_rgb = rgb
                st.state_text = (f"SPIN STALL: {np.degrees(spin[0]):.0f}° turned, "
                                 f"{spin[1]:.2f}m travel -- send a new target")
                st.vel_text = "lin 0.000  ang +0.000"
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
            continue

        if res.state == "STOP":
            stop_streak += 1
        else:
            stop_streak = 0
        reached = stop_streak >= stop_confirm

        with st.lock:
            st.display_rgb = rgb
            st.detection = res.detection
            st.avoid_detection = res.avoid_detection
            st.mask = res.mask
            st.trajs = res.all_trajectories
            st.chosen = res.trajectory
            st.goal_pt = res.goal_point
            st.qwen_pixel_goal = res.qwen_pixel_goal
            st.obstacles = res.obstacle_points
            st.min_forward = res.min_forward
            st.infer_count += 1
            if reached:
                st.goal_reached = True
                st.last_cmd = (0.0, 0.0)
                st.state_text = f"GOAL REACHED: '{target}'"
                st.vel_text = "lin 0.000  ang +0.000"
            else:
                st.last_cmd = (res.linear, res.angular) if res.state != "STOP" else (0.0, 0.0)
                st.state_text = f"{res.state} [AMBIGUOUS x{res.candidate_count}]" if res.ambiguous else res.state
                st.vel_text = f"lin {res.linear:.3f}  ang {res.angular:+.3f}"
            st.lat_text = "  ".join(f"{k} {v*1000:.0f}ms" for k, v in res.timing.items())

        if reached:
            pubs["explain"].put(serialize_string(f"GOAL REACHED: '{target}'. Stopping."))
        if res.trajectory is not None:
            pubs["path"].put(serialize_path([(p[0], p[1]) for p in res.trajectory]))
        score = f"{res.detection.score:.2f}" if res.detection else "-"
        amb = f" [AMBIGUOUS x{res.candidate_count}]" if res.ambiguous else ""
        pubs["explain"].put(serialize_string(
            f"DINO+NavDP [{res.state}]{amb} det={score} -> lin={res.linear:.3f} ang={res.angular:.3f} "
            f"| target='{target}'"
        ))

        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# ---------------------------------------------------------------------- #
class App:
    CAM_SIZE = 448          # camera panel (px)
    PLOT_SIZE = 448         # top-down plot (px)
    PLOT_RANGE = 3.5        # meters shown ahead

    def __init__(self, root: tk.Tk, st: SharedState):
        self.root = root
        self.st = st
        root.title("Nav_new — DINO + NavDP")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        # Seed a blank image so the label already occupies its final size —
        # otherwise the column is 0-width until the first camera frame
        # arrives, and the window jumps when it does.
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)
        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="Target:").pack(side="left")
        self.entry = ttk.Entry(bar, width=36)
        self.entry.insert(0, st.target)
        self.entry.pack(side="left", padx=4)
        self.entry.bind("<Return>", lambda e: self.send_target())
        ttk.Button(bar, text="Send", command=self.send_target).pack(side="left", padx=2)
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        presets = ttk.Frame(main)
        presets.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        for p in PRESETS:
            ttk.Button(presets, text=p, command=lambda t=p: self.send_target(t)).pack(side="left", padx=2)

        # Manual drive: hold a button (or an arrow key, once the window has
        # focus) to drive directly, bypassing detection/NavDP entirely.
        # Releasing zeros that axis; it does NOT hand control back to
        # whatever autonomous target was running before -- send a target
        # for that. STOP still works as usual.
        self._manual_held: set = set()
        drive = ttk.Frame(main)
        drive.grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(drive, text="Manual drive (hold, or arrow keys):").pack(side="left")
        for label, direction in (("◄", "left"), ("▲", "fwd"), ("▼", "back"), ("►", "right")):
            b = ttk.Button(drive, text=label, width=3)
            b.bind("<ButtonPress-1>", lambda e, d=direction: self.manual_press(d))
            b.bind("<ButtonRelease-1>", lambda e, d=direction: self.manual_release(d))
            b.pack(side="left", padx=2)
        for key, direction in (("Up", "fwd"), ("Down", "back"), ("Left", "left"), ("Right", "right")):
            root.bind(f"<KeyPress-{key}>", lambda e, d=direction: self.manual_press(d))
            root.bind(f"<KeyRelease-{key}>", lambda e, d=direction: self.manual_release(d))

        # Fixed character width on the two dynamic-text rows: their content
        # (state name, counters, latency numbers) changes length on every
        # refresh tick, and an unconstrained Label makes the whole window
        # resize to match on every tick. Width is a floor, not a wrap limit,
        # so longer text (e.g. a long custom target) just doesn't shrink
        # back below it.
        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=100, anchor="w")
        self.status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=100, anchor="w")
        self.info.grid(row=5, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(66, self.refresh)

    def send_target(self, text: Optional[str] = None):
        t = text if text is not None else self.entry.get().strip()
        if not t:
            return
        if text is not None:
            self.entry.delete(0, "end")
            self.entry.insert(0, t)
        self._manual_held.clear()
        with self.st.lock:
            self.st.mode = "text"
            self.st.target = t
            self.st.stopped = False
            self.st.goal_reached = False

    def stop(self):
        self._manual_held.clear()
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

    # ---------------- manual drive ---------------- #
    def manual_press(self, direction: str):
        self._manual_held.add(direction)
        self._manual_update()

    def manual_release(self, direction: str):
        self._manual_held.discard(direction)
        self._manual_update()

    def _manual_update(self):
        with self.st.lock:
            lin = 0.0
            ang = 0.0
            if "fwd" in self._manual_held:
                lin += self.st.max_linear
            if "back" in self._manual_held:
                lin -= 0.5 * self.st.max_linear
            if "left" in self._manual_held:
                ang += self.st.max_angular
            if "right" in self._manual_held:
                ang -= self.st.max_angular
            self.st.mode = "manual"
            self.st.stopped = False
            self.st.goal_reached = False
            self.st.last_cmd = (lin, ang)

    def on_close(self):
        self.closed = True
        self.root.destroy()

    # ------------------------------------------------------------------ #
    def refresh(self):
        if self.closed:
            return
        try:
            self._refresh_body()
        except Exception:
            # A single bad frame/detection/mask must never permanently kill
            # this self-rescheduling redraw loop -- before this guard, any
            # uncaught exception below silently froze the GUI (camera image
            # AND status text both stop updating) while the inference/
            # movement loop kept running fine in its own thread, with no
            # visible error beyond a traceback in the launching terminal.
            # Symptom this fixes: "GUI pauses on one frame mid run."
            traceback.print_exc()
        finally:
            if not self.closed:
                self.root.after(66, self.refresh)

    def _refresh_body(self):
        with self.st.lock:
            # Show the freshest camera frame (updated on every Zenoh image,
            # i.e. the camera's native rate), not display_rgb (only updated
            # once per inference tick, i.e. predict_hz -- ~2.5 Hz by default).
            # Overlaying the last-computed detection/mask/trajectory on a
            # newer frame can lag them slightly behind the rover's actual
            # motion, but that's far less noticeable than the whole picture
            # visibly jumping between camera frames ~400ms apart, which is
            # what showing display_rgb here produced during movement.
            rgb = self.st.latest_rgb if self.st.latest_rgb is not None else self.st.display_rgb
            det = self.st.detection
            avoid_det = self.st.avoid_detection
            mask = self.st.mask
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers, target = self.st.frame_count, self.st.infer_count, self.st.target
            drive_mode = self.st.mode
            stopped = self.st.stopped
            imu_heading, imu_calib, theta_source = self.st.imu_heading_deg, self.st.imu_calib, self.st.theta_source

        if rgb is not None:
            frame = rgb
            if mask is not None and mask.shape[:2] == rgb.shape[:2]:
                frame = rgb.copy()
                frame[mask] = (0.55 * frame[mask] + 0.45 * np.array([0, 255, 60])).astype(np.uint8)
            img = Image.fromarray(frame).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            if det is not None or avoid_det is not None:
                d = ImageDraw.Draw(img)
                if det is not None:
                    x0, y0, x1, y1 = det.box
                    d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(0, 255, 60), width=3)
                    d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), f"{det.label} {det.score:.2f}", fill=(0, 255, 60))
                if avoid_det is not None:
                    x0, y0, x1, y1 = avoid_det.box
                    d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 40, 40), width=3)
                    d.text((x0 * sx + 4, max(y0 * sy - 14, 2)), f"avoid: {avoid_det.label} {avoid_det.score:.2f}",
                            fill=(255, 40, 40))
            self._photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._photo)

        self.plot.delete("all")
        S, R = self.PLOT_SIZE, self.PLOT_RANGE

        def to_px(x, y):  # robot frame (x fwd, y left) -> canvas
            return S / 2 - (y / R) * (S / 2), S - (x / R) * S * 0.92 - 20

        self.plot.create_line(0, S - 20, S, S - 20, fill="#ddd")
        self.plot.create_oval(S / 2 - 5, S - 25, S / 2 + 5, S - 15, fill="black")
        if obstacles is not None and len(obstacles):
            for ox, oy in obstacles[:: max(1, len(obstacles) // 400)]:
                px, py = to_px(ox, oy)
                self.plot.create_rectangle(px - 1, py - 1, px + 1, py + 1, fill="#8a8a8a", outline="")
        if trajs is not None:
            for t in trajs:
                pts = [to_px(p[0], p[1]) for p in t[::2]]
                self.plot.create_line(*[c for xy in pts for c in xy], fill="#cccccc")
        if chosen is not None:
            pts = [to_px(p[0], p[1]) for p in chosen]
            self.plot.create_line(*[c for xy in pts for c in xy], fill="red", width=3)
        if goal is not None:
            gx, gy = to_px(goal[0], goal[1])
            self.plot.create_text(gx, gy, text="★", fill="#d4a017", font=("TkDefaultFont", 22))

        mode_txt = "STOPPED" if stopped else state_text
        target_txt = "manual drive" if drive_mode == "manual" else f"'{target}'"
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  target: {target_txt}   {vel_text}{fwd}")
        heading_txt = f"{imu_heading:.1f}°" if np.isfinite(imu_heading) else "n/a"
        imu_txt = (f"theta src: {theta_source}   imu heading {heading_txt}"
                   f"  calib [{OdometryLogger.decode_calib(imu_calib)}]")
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}   {imu_txt}")


def main():
    ap = argparse.ArgumentParser(description="Nav_new DINO+NavDP Isaac GUI")
    ap.add_argument("--target", default="",
                    help="starts empty -- rover stays in manual drive with nothing to "
                         "navigate to until a target is sent from the GUI (or passed here)")
    ap.add_argument("--avoid", default="",
                    help="named obstacle to steer away from, e.g. 'trash bin' -- DINO-detected "
                         "every tick like --target, but only fed into S2Diff pixel-obstacle "
                         "avoidance (nav_pipeline/s2diff_http_client.py); no effect on the "
                         "plain or in-process-S2Diff launchers. '' (default) disables it")
    ap.add_argument("--policy-type", choices=["crossmodal", "extracted"], default="crossmodal",
                    help="NavDP policy backend (see PipelineConfig.policy_type). \"crossmodal\" "
                         "(default) is the official standalone checkpoint, in-process. "
                         "nav_pipeline.s2diff_http_runner (LAUNCH/launch_rover_s2diff_http.sh) "
                         "REQUIRES \"extracted\" here -- its HTTP client monkeypatches "
                         "NavDPStandalone.sample_pointgoal, which only takes effect if this "
                         "pipeline's self.policy is actually a NavDPStandalone instance "
                         "(policy_type=\"extracted\"); left at \"crossmodal\" the HTTP server is "
                         "silently never called and NavDP runs local crossmodal inference instead.")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 0.25 on the real rover)")
    ap.add_argument("--search-angular", type=float, default=0.15,
                    help="rad/s spin rate while re-acquiring a lost target (must stay "
                         "above PipelineConfig.ang_min_cmd, the rover's stiction floor, "
                         "or it won't turn at all; lower this if fast spins sweep past "
                         "the target between detection frames)")
    ap.add_argument("--servo-ramp-deg", type=float, default=35.0,
                    help="heading error (degrees) at which TRACK steering reaches ~96%% "
                         "of max_angular. With a narrow fov and slow predict-hz, a small "
                         "ramp means edge-of-frame detections already command near-max "
                         "angular speed -- one inference tick can then sweep the target "
                         "clean out of frame before the next correction. Widen this "
                         "(e.g. 70-90 on a 60deg-fov real rover) so bearings near the "
                         "frame edge get a proportionally gentler command instead of a "
                         "near-max snap turn.")
    ap.add_argument("--angular-slew-max", type=float, default=0.10,
                    help="rad/s hard cap on how much the angular command can change per tick, "
                         "in every state -- bounds abrupt snap turns (e.g. entering/leaving "
                         "SEARCH, or AVOID's full-authority turn). 0 disables.")
    ap.add_argument("--invert-angular", action="store_true",
                    help="flip turn direction (use if the rover steers away from the target)")
    ap.add_argument("--no-belief-goal", action="store_true",
                    help="disable ego-motion goal belief (see goal_belief.py); reverts to "
                         "coasting on the frozen last-seen goal while the target is lost")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vits",
                    help="RGB-only metric depth model size (no depth sensor on the real "
                         "rover): vits is the default/fast one; vitb is more accurate "
                         "(~2x slower) -- worth it since depth error feeds directly into "
                         "the STOP distance decision. Needs checkpoints/depth_anything_v2_"
                         "metric_hypersim_vitb.pth (scripts/download_models.py --depth-encoder vitb).")
    ap.add_argument("--compressed-only", action="store_true",
                    help="subscribe only the JPEG camera stream (REQUIRED over rover Wi-Fi)")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log",
                    help="dead-reckoned pose CSV log dir (from /rover/rpm)")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3,
                    help="IMU calibration digit (0-3) required before theta rides the IMU "
                         "heading instead of wheel-diff dead reckoning -- see OdometryLogger. "
                         "Was only exposed on home_gui.py before; same default/semantics here.")
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length,
                    help="robot length (m) for obstacle_guard's swept-footprint clearance -- "
                         "defaults to the ESP32 rover's real size, override for a different robot "
                         "(e.g. the LanderPi, see landerpi/README.md) before trusting obstacle avoidance")
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width,
                    help="robot width (m), see --footprint-length")
    ap.add_argument("--no-dino", action="store_true",
                    help="skip loading Grounding DINO entirely (saves VRAM/load time) -- only "
                         "safe with --use-qwen-instruction AND no --avoid, ever, this run: "
                         "plain-text --target detection and avoid_text both need DINO and will "
                         "raise/warn respectively if this is set without --use-qwen-instruction")
    ap.add_argument("--wheel-deadband-correction", action="store_true",
                    help="ESP32 6WD rover ONLY (never --hiwonder, different firmware entirely) "
                         "-- boosts a commanded (linear, angular) that would otherwise "
                         "differential-mix down to a stalled wheel on the real hardware: "
                         "esp32/rover_6wd_complete.ino hard-zeros any per-wheel PWM target "
                         "below VEL_DEADBAND_MS=0.03 m/s, and a small linear + small angular "
                         "can land BOTH wheel targets under that even when neither value alone "
                         "looks like zero -- the pipeline reports it's driving, the rover "
                         "doesn't move. See PipelineConfig.clear_wheel_deadband.")
    ap.add_argument("--use-qwen-search", action="store_true",
                    help="while DINO can't see --target, ask a frozen Qwen2.5-VL-7B-Instruct "
                         "to point toward the best direction to search/move next (a doorway, "
                         "opening, hallway) instead of the fixed spin toward the last-known "
                         "side. DINO keeps scanning every tick regardless; the instant it "
                         "reacquires the target this path is no longer taken, so Qwen is "
                         "dropped automatically. Adds real per-call latency while searching "
                         "(VLM inference, seconds) -- see --qwen-search-period-s. Requires "
                         "transformers with Qwen2.5-VL support + qwen-vl-utils.")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HF model id for --use-qwen-search")
    ap.add_argument("--qwen-search-period-s", type=float, default=2.0,
                    help="min seconds between Qwen search-guidance calls (throttle -- the "
                         "last suggested direction is held between calls, same pattern as "
                         "SAM/CLIP's sam_period_s)")
    ap.add_argument("--qwen-fp16", action="store_true",
                    help="load Qwen2.5-VL-7B in fp16 instead of the 4-bit default (needs "
                         "~16.6GB VRAM vs 4-bit's ~6.2GB; use if bitsandbytes isn't installed "
                         "or the pointed pixel needs to match the un-quantized model exactly)")
    ap.add_argument("--use-qwen-instruction", action="store_true",
                    help="DINO target detection is bypassed entirely: whatever's sent as "
                         "--target / typed in the GUI's text box is read as a free-text "
                         "INSTRUCTION instead (e.g. 'go through the doorway and stop near "
                         "the desk') -- Qwen2.5-VL grounds it to a pixel every "
                         "--qwen-instruction-period-s, which becomes NavDP's goal via depth "
                         "exactly like a DINO detection would. See qwen_pixel_goal.py. "
                         "Mutually exclusive in intent with --use-qwen-search (that one only "
                         "steers SEARCH for a DINO text target); enabling both loads two "
                         "separate Qwen instances.")
    ap.add_argument("--qwen-instruction-period-s", type=float, default=1.5,
                    help="min seconds between Qwen instruction-grounding calls; GoalBelief "
                         "coasts the goal by ego-motion in between (see qwen_pixel_goal.py)")
    ap.add_argument("--qwen-goal-consistency-m", type=float, default=1.5,
                    help="reject a new Qwen instruction-grounding more than this many meters "
                         "from the current confident goal (e.g. a second door coming into "
                         "view mid-turn) instead of snapping to it -- coasts on the existing "
                         "lock instead. Only applies while belief is still confident (sigma <= "
                         "--belief-max-sigma); once genuinely lost, a fresh grounding is "
                         "accepted as a real reacquisition. Raise this if it's rejecting a "
                         "legitimately-updated view of the SAME door; lower it if it's still "
                         "snapping onto distractors.")
    ap.add_argument("--qwen-max-candidates", type=int, default=3,
                    help="ask Qwen for up to this many ranked, confidence-scored waypoint "
                         "candidates instead of one, then pick the best by combining semantic "
                         "confidence + continuity with the current lock + obstacle-avoidance "
                         "cost -- 1 disables scoring, back to plain single-point grounding. "
                         "See PipelineConfig.qwen_max_candidates.")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        policy_type=args.policy_type,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        search_angular=min(args.search_angular, args.max_angular),
        servo_ramp_deg=args.servo_ramp_deg,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        use_belief_goal=not args.no_belief_goal,
        depth_encoder=args.depth_encoder,
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
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState(args.target)
    st.avoid = args.avoid
    st.max_linear = args.max_linear
    st.max_angular = args.max_angular
    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)
    running = {"on": True}

    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz),
           kwargs={"odom": odom}, daemon=True).start()

    root = tk.Tk()
    App(root, st)

    # Ctrl-C in the launching terminal closes the GUI cleanly. Tk's mainloop
    # only delivers Python signals while Python code runs, so keep a periodic
    # no-op tick alive for the handler to fire promptly.
    signal.signal(signal.SIGINT, lambda *_: root.after(0, root.destroy))
    signal.signal(signal.SIGTERM, lambda *_: root.after(0, root.destroy))

    def _tick():
        root.after(200, _tick)

    _tick()
    try:
        root.mainloop()
    finally:
        running["on"] = False
        time.sleep(0.2)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        time.sleep(0.1)
        pubs["cmd"].put(serialize_twist(0.0, 0.0))
        try:
            session.close()
        except zenoh.ZError as e:
            print(f"[WARN] zenoh session close timed out/failed: {e}")
        odom.close()
        print("[INFO] zero velocity sent, session closed")


if __name__ == "__main__":
    main()
