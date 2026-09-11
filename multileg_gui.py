#!/usr/bin/env python3
"""Combined viewer + launcher for run_rover_multileg.py: camera feed with
the live DINO detection box, NavDP's own per-tick candidate trajectory
(body frame), the odometry trail (world frame), the fact3r-map live
recorder's status if one is running alongside, and an ordered subtask plan
with live progress -- plus a blank multi-instruction box that launches a
run directly from the GUI.

Two roles in one window:
  1. PASSIVE VIEWER (always on): subscribes to the rover's Zenoh camera,
     run_rover_multileg.py's own "multileg/status" feed, its NavDP
     trajectory publish ("omnivla/waypoints"), and record_rover_live.py's
     "mast3r_record/status" feed if that script is running alongside. Zero
     control surface here -- works with any run already in progress, or
     none, or several viewers watching the same run at once.
  2. LAUNCHER (opt-in, only if you type into the instruction box): spawns
     run_rover_multileg.py as a child process with your free-text
     instruction (blank by default -- nothing launches until you type one
     and click Launch), streams its stdout into the log panel, and can
     SIGINT it. This is the ONLY path by which this GUI ever commands the
     rover -- through that exact same subprocess, never a direct cmd_vel
     publish of its own. The one exception is the emergency STOP button,
     which publishes a single zero-velocity Twist directly, for the case
     where the child process is gone but the rover is still moving.

Run:
  python multileg_gui.py --pi-ip 172.18.186.125
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from threading import Lock
from tkinter import ttk
from typing import List, Optional, Tuple

import numpy as np
import zenoh
from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from nav_pipeline.zenoh_node import (  # noqa: E402
    CAMERA_COMPRESSED_KEYS, CAMERA_KEYS, CDRReader, parse_compressed_image, parse_image, parse_string,
    serialize_string,
)

STATUS_KEYS = ["multileg/status", "rt/multileg/status"]
WAYPOINTS_KEYS = ["omnivla/waypoints", "rt/omnivla/waypoints"]
RECORD_STATUS_KEYS = ["mast3r_record/status", "rt/mast3r_record/status"]
OCCUPANCY_KEYS = ["multileg/occupancy", "rt/multileg/occupancy"]
FACT3R_KEYS = ["multileg/fact3r", "rt/multileg/fact3r"]
RAW_W, RAW_H = 640, 480   # frame size DINO's box coordinates are in
RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_rover_multileg.py")
# Matches run_rover_multileg.py's own --goal-memory-path default -- read
# directly off disk rather than over Zenoh, since it's a small local file
# and this way the panel works the same for a GUI-launched or bare CLI run.
GOAL_MEMORY_PATH = os.path.join(os.path.dirname(RUNNER_PATH), "logs", "live_goal_memory.jsonl")
# Experiment log: a JSON array, one object per completed subtask
# (move/turn/object), tagged with whatever's in the "Experiment no." box at
# the moment it finished -- lets several experiment runs in one session be
# compared by path length / outcome later without re-parsing every
# logs/rover_multileg_*/result.json. Each row also carries the odometry CSV
# filename (the raw (t,x,y,theta,v,w,...) trace path_length_m was integrated
# from -- see run_rover_multileg.py's _leg_odometry_csv) and, once per plan,
# the straight-line distance from the session origin to that plan's first
# subtask (run_rover_multileg.py's origin_to_plan_start_m).
EXPERIMENT_LOG_PATH = os.path.join(os.path.dirname(RUNNER_PATH), "logs", "experiment_log.json")
RECORDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "MASt3R-SLAM", "scripts", "record_rover_live.py")
RECORDER_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "MASt3R-SLAM", "datasets", "rover")


def parse_path(cdr_data: bytes) -> List[Tuple[float, float]]:
    """nav_msgs/Path CDR -> [(x, y), ...] in whatever frame it was published
    in (body/base_link for run_rover_multileg's NavDP trajectory publish --
    see zenoh_node.py's serialize_path, which this is the exact inverse of)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # Path header
    n = r.read_uint32()
    pts = []
    for _ in range(n):
        r.read_int32(); r.read_uint32(); r.read_string()  # each PoseStamped's header
        px = r.read_float64(); py = r.read_float64(); r.read_float64()
        r.read_float64(); r.read_float64(); r.read_float64(); r.read_float64()  # orientation, unused
        pts.append((px, py))
    return pts


def serialize_twist_zero() -> bytes:
    buf = bytearray(b"\x00\x01\x00\x00")
    for _ in range(6):
        buf += struct.pack("<d", 0.0)
    return bytes(buf)


class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.rgb: Optional[np.ndarray] = None
        self.rgb_t = 0.0
        self.status: Optional[dict] = None
        self.status_t = 0.0
        self.navdp_traj: List[Tuple[float, float]] = []   # body-frame points, latest tick
        self.navdp_traj_t = 0.0
        self.record_status: Optional[dict] = None
        self.record_status_t = 0.0
        self.occupancy_png: Optional[bytes] = None   # raw PNG bytes, decoded lazily in refresh()
        self.occupancy_t = 0.0
        self.fact3r: Optional[dict] = None   # {entities:[{id,xy,obs,depth_m,spread_m}], recalled_id, ...}
        self.fact3r_t = 0.0


def zenoh_setup(session: zenoh.Session, st: SharedState):
    def on_image(sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 3:
            with st.lock:
                st.rgb, st.rgb_t = img, time.time()

    def on_image_compressed(sample):
        img = parse_compressed_image(bytes(sample.payload))
        if img is not None:
            with st.lock:
                st.rgb, st.rgb_t = img, time.time()

    def on_status(sample):
        try:
            payload = json.loads(parse_string(bytes(sample.payload)))
        except Exception:
            return
        with st.lock:
            st.status, st.status_t = payload, time.time()

    def on_waypoints(sample):
        try:
            pts = parse_path(bytes(sample.payload))
        except Exception:
            return
        with st.lock:
            st.navdp_traj, st.navdp_traj_t = pts, time.time()

    def on_record_status(sample):
        try:
            payload = json.loads(parse_string(bytes(sample.payload)))
        except Exception:
            return
        with st.lock:
            st.record_status, st.record_status_t = payload, time.time()

    def on_occupancy(sample):
        # Raw PNG bytes, no CDR/String wrapper -- a private topic only this
        # project's own scripts speak, so there's no ROS message type to match.
        with st.lock:
            st.occupancy_png, st.occupancy_t = bytes(sample.payload), time.time()

    def on_fact3r(sample):
        try:
            payload = json.loads(parse_string(bytes(sample.payload)))
        except Exception:
            return
        with st.lock:
            st.fact3r, st.fact3r_t = payload, time.time()

    subs = (
        [session.declare_subscriber(k, on_image) for k in CAMERA_KEYS]
        + [session.declare_subscriber(k, on_image_compressed) for k in CAMERA_COMPRESSED_KEYS]
        + [session.declare_subscriber(k, on_status) for k in STATUS_KEYS]
        + [session.declare_subscriber(k, on_waypoints) for k in WAYPOINTS_KEYS]
        + [session.declare_subscriber(k, on_record_status) for k in RECORD_STATUS_KEYS]
        + [session.declare_subscriber(k, on_occupancy) for k in OCCUPANCY_KEYS]
        + [session.declare_subscriber(k, on_fact3r) for k in FACT3R_KEYS]
    )
    return subs


STATE_COLOR = {
    "TRACK": "#2e7d32", "STOP": "#1565c0", "SEARCH": "#e65100",
    "GOTO": "#6a1b9a", "AVOID": "#c62828", "?": "#616161",
}


class RunnerProcess:
    """Owns (at most) one run_rover_multileg.py --serve child process. Models
    load ONCE on the first launch(); every launch() after that just publishes
    the new instruction over Zenoh to the already-running process -- no
    respawn, no reload. stop() sends a soft '__stop__' (aborts the current
    plan, server stays up); shutdown() sends '__shutdown__' (ends the
    process cleanly); kill() is the old hard SIGINT, for when the process is
    wedged (e.g. the 7B-load hang from earlier tonight) and needs a real kill."""

    def __init__(self, pi_ip: Optional[str], log_queue: "queue.Queue[str]", log_dir: str,
                 instr_pub):
        self.pi_ip = pi_ip
        self.log_queue = log_queue
        self.log_dir = log_dir
        self.instr_pub = instr_pub   # Zenoh publisher on 'multileg/instruction', owned by the GUI
        self.proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self.log_path: Optional[str] = None
        self.ready = threading.Event()   # set once "[serve] ready" is seen in the child's output

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _send_instruction(self, text: str):
        self.instr_pub.put(serialize_string(text))

    def launch(self, instruction: str, extra_args: Optional[List[str]] = None):
        """First call: spawn the persistent server (extra_args -- e.g. which
        supervisor model -- only take effect on this first spawn, since
        models aren't reloaded afterward). Every call after that: just send
        the instruction to the already-running server."""
        if self.is_running():
            self.log_queue.put(f"[gui] server already up -- sending instruction (no reload): {instruction!r}\n")
            self._send_instruction(instruction)
            return
        if self.ready.is_set():
            # process died since the last run but ready was never cleared
            self.ready.clear()
        args = [sys.executable, "-u", RUNNER_PATH, "--yes", "--serve"]
        if self.pi_ip:
            args += ["--pi-ip", self.pi_ip]
        args += extra_args or []
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"gui_serve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self.log_queue.put(f"[gui] starting persistent server (models load once): {' '.join(args)}\n"
                           f"[gui] full log: {self.log_path}\n")
        self.proc = subprocess.Popen(
            args, cwd=os.path.dirname(RUNNER_PATH), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._reader_thread = threading.Thread(target=self._pump_output, daemon=True)
        self._reader_thread.start()
        threading.Thread(target=self._send_once_ready, args=(instruction,), daemon=True).start()

    def _send_once_ready(self, instruction: str):
        # Zenoh has no delivery queue for a subscriber that doesn't exist
        # yet -- publishing before the server's own subscriber is up would
        # just be lost. Wait for the "[serve] ready" line (or give up after
        # a generous load window) before sending the first instruction.
        if self.ready.wait(timeout=180.0):
            self.log_queue.put(f"[gui] server ready -- sending instruction: {instruction!r}\n")
            self._send_instruction(instruction)
        else:
            self.log_queue.put("[gui] server never signalled ready within 180s -- "
                               "instruction NOT sent (see the log; the model load may be stuck)\n")

    def _pump_output(self):
        proc = self.proc
        try:
            with open(self.log_path, "w") as f:
                for line in proc.stdout:
                    self.log_queue.put(line)
                    f.write(line)
                    f.flush()
                    if "[serve] ready" in line:
                        self.ready.set()
        except Exception:
            pass
        rc = proc.wait()
        self.ready.clear()
        self.log_queue.put(f"[gui] server exited (code {rc}) -- full log: {self.log_path}\n")

    def stop(self):
        """Abort the current plan only -- the server stays up for the next instruction."""
        if not self.is_running():
            self.log_queue.put("[gui] no server running\n")
            return
        self.log_queue.put("[gui] sending __stop__ (abort current plan, keep server up)\n")
        self._send_instruction("__stop__")

    def shutdown(self):
        """End the persistent server cleanly (frees the GPU/VRAM)."""
        if not self.is_running():
            self.log_queue.put("[gui] no server running\n")
            return
        self.log_queue.put("[gui] sending __shutdown__\n")
        self._send_instruction("__shutdown__")

    def kill(self):
        """Hard kill for a wedged process (e.g. a model load that hangs) --
        SIGINT, escalating to SIGKILL if it doesn't exit."""
        if not self.is_running():
            self.log_queue.put("[gui] no server running\n")
            return
        self.log_queue.put("[gui] sending SIGINT (hard kill)...\n")
        self.proc.send_signal(signal.SIGINT)

    def clear_map(self):
        if not self.is_running():
            self.log_queue.put("[gui] no server running -- nothing to clear\n")
            return
        self.log_queue.put("[gui] sending __clear_map__\n")
        self._send_instruction("__clear_map__")


class RecorderProcess:
    """Owns (at most) one record_rover_live.py child process. Its progress
    is shown via the existing mast3r_record/status Zenoh subscription (see
    SharedState.record_status) regardless of who started it -- this class
    only needs to start/stop the process; the panel updates itself."""

    def __init__(self, pi_ip: Optional[str], log_queue: "queue.Queue[str]"):
        self.pi_ip = pi_ip
        self.log_queue = log_queue
        self.proc: Optional[subprocess.Popen] = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, max_seconds: float = 1800.0, query: str = "a 3D printer"):
        if self.is_running():
            self.log_queue.put("[gui] recorder already running -- stop it first\n")
            return
        if not os.path.exists(RECORDER_PATH):
            self.log_queue.put(f"[gui] recorder script not found at {RECORDER_PATH} -- "
                               f"expected MASt3R-SLAM as a sibling checkout\n")
            return
        out_dir = os.path.join(RECORDER_OUT_ROOT, f"live_gui_{time.strftime('%Y%m%d_%H%M%S')}")
        args = [sys.executable, "-u", RECORDER_PATH, "--out", out_dir,
                "--max-seconds", str(max_seconds), "--query", query]
        if self.pi_ip:
            args += ["--pi-ip", self.pi_ip]
        self.log_queue.put(f"[gui] starting fact3r-map recorder -> {out_dir}\n")
        self.proc = subprocess.Popen(args, cwd=os.path.dirname(RECORDER_PATH),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if not self.is_running():
            self.log_queue.put("[gui] recorder is not running\n")
            return
        self.log_queue.put("[gui] stopping recorder...\n")
        self.proc.send_signal(signal.SIGINT)


class MultilegViewer:
    CAM_W, CAM_H = 520, 390
    TRAIL_SIZE = 190
    TRAIL_RANGE_M = 6.0
    NAVDP_SIZE = 190
    NAVDP_RANGE_M = 4.0

    def __init__(self, root: tk.Tk, st: SharedState, pi_ip: Optional[str], session: zenoh.Session,
                 default_instruction: str = "", supervisor_model: str = "Qwen/Qwen3-VL-2B-Instruct"):
        self.root = root
        self.st = st
        self.session = session
        root.title("Nav_new — multileg viewer + launcher")

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        # ---- column 1: camera + live status text -----------------------
        col1 = ttk.Frame(main)
        col1.grid(row=0, column=0, sticky="n", padx=(0, 10))
        self.cam_label = ttk.Label(col1)
        self.cam_label.pack()
        self._blank = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_W, self.CAM_H), "#222"))
        self.cam_label.configure(image=self._blank)
        self.state_line = ttk.Label(col1, text="—", font=("TkDefaultFont", 13, "bold"))
        self.state_line.pack(anchor="w", pady=(6, 0))
        self.detector_line = ttk.Label(col1, text="", width=68, anchor="w")
        self.detector_line.pack(anchor="w")
        self.supervisor_line = ttk.Label(col1, text="", width=68, anchor="w", wraplength=520, justify="left")
        self.supervisor_line.pack(anchor="w", pady=(2, 0))
        self.pose_line = ttk.Label(col1, text="")
        self.pose_line.pack(anchor="w", pady=(4, 0))

        # ---- column 2: subtask plan + two top-down plots ----------------
        col2 = ttk.Frame(main)
        col2.grid(row=0, column=1, sticky="n", padx=(0, 10))
        ttk.Label(col2, text="Subtask plan", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        # tk.Text (not ttk.Label rows) so the plan is selectable + copy-pastable
        # (click-drag or Ctrl+A then Ctrl+C) -- state="disabled" still allows
        # selection/copy, it only blocks typing, same as self.log_text below.
        self.plan_text = tk.Text(col2, width=54, height=8, bg="#111", fg="#ddd",
                                 font=("Courier", 9), wrap="none", state="disabled",
                                 relief="flat", padx=4, pady=2)
        self.plan_text.pack(anchor="w", fill="x", pady=(2, 8))
        self.plan_text.tag_configure("done_ok", foreground="#2e7d32")
        self.plan_text.tag_configure("done_bad", foreground="#c62828")
        self.plan_text.tag_configure("running", foreground="#f9a825", font=("Courier", 9, "bold"))
        self.plan_text.tag_configure("pending", foreground="#888888")
        self._plan_len = -1
        self._logged_subtask_keys: set = set()   # (label, dur_s) already written to EXPERIMENT_LOG_PATH,
        # so a reconnect / replayed status doesn't double-log the same completed subtask

        # Odometry-trail panel removed: the BEV occupancy panel below already
        # carries a live pose+heading marker in the same world frame, so a
        # separate blank-except-for-a-line trail canvas was redundant.
        plots = ttk.Frame(col2)
        plots.pack(anchor="w")
        right_plot = ttk.Frame(plots)
        right_plot.grid(row=0, column=0, padx=(0, 6))
        ttk.Label(right_plot, text="NavDP trajectory (body, ±%.0fm)" % self.NAVDP_RANGE_M).pack(anchor="w")
        self.navdp_canvas = tk.Canvas(right_plot, width=self.NAVDP_SIZE, height=self.NAVDP_SIZE, bg="#111")
        self.navdp_canvas.pack()
        occ_plot = ttk.Frame(plots)
        occ_plot.grid(row=0, column=1)
        occ_head = ttk.Frame(occ_plot)
        occ_head.pack(anchor="w", fill="x")
        ttk.Label(occ_head, text="fact3r BEV occupancy (world, ±%.0fm)" % self.TRAIL_RANGE_M).pack(side="left")
        ttk.Button(occ_head, text="Clear map", command=self.on_clear_map).pack(side="left", padx=(8, 0))
        self.occ_canvas = tk.Canvas(occ_plot, width=self.TRAIL_SIZE, height=self.TRAIL_SIZE, bg="#111")
        self.occ_canvas.pack(anchor="w")

        # ---- column 3: fact3r-map recorder status + instruction launcher --
        col3 = ttk.Frame(main)
        col3.grid(row=0, column=2, sticky="n", padx=(0, 10))
        ttk.Label(col3, text="fact3r-map live recorder", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.record_line = ttk.Label(col3, text="not running", foreground="#888", width=48, anchor="w",
                                     wraplength=380, justify="left")
        self.record_line.pack(anchor="w", pady=(2, 4))
        rec_btns = ttk.Frame(col3)
        rec_btns.pack(anchor="w", pady=(0, 10))
        self.rec_start_btn = ttk.Button(rec_btns, text="Start recorder", command=self.on_recorder_start)
        self.rec_start_btn.pack(side="left")
        self.rec_stop_btn = ttk.Button(rec_btns, text="Stop recorder", command=self.on_recorder_stop)
        self.rec_stop_btn.pack(side="left", padx=6)

        ttk.Label(col3, text="Instruction (blank until you type one)",
                  font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.instr_entry = ttk.Entry(col3, width=48)
        if default_instruction:
            self.instr_entry.insert(0, default_instruction)   # pre-filled, NOT auto-launched
        self.instr_entry.pack(anchor="w", pady=(2, 4), fill="x")
        self.instr_entry.bind("<Return>", lambda e: self.on_launch())

        exp_row = ttk.Frame(col3)
        exp_row.pack(anchor="w", pady=(0, 4), fill="x")
        ttk.Label(exp_row, text="Experiment no.:").pack(side="left")
        self.experiment_var = tk.StringVar(value="1")
        ttk.Entry(exp_row, textvariable=self.experiment_var, width=10).pack(side="left", padx=(4, 8))
        ttk.Label(exp_row, text="-- tags every subtask's path-length row below and in "
                                f"{os.path.basename(EXPERIMENT_LOG_PATH)}",
                  foreground="#666", wraplength=280, justify="left").pack(side="left")

        sup_row = ttk.Frame(col3)
        sup_row.pack(anchor="w", pady=(0, 4), fill="x")
        ttk.Label(sup_row, text="Supervisor:").pack(side="left")
        # 2B first/default: it loads in <1s from local cache and has been the
        # reliable choice all session. 7B/8B are heavier VLM judges but the
        # 7B fp16 fallback (no bitsandbytes here) hung indefinitely on this
        # box at least once -- offered, not defaulted to.
        _sup_choices = ["Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct",
                        "Qwen/Qwen2.5-VL-7B-Instruct"]
        if supervisor_model and supervisor_model not in _sup_choices:
            _sup_choices.insert(0, supervisor_model)
        self.supervisor_var = tk.StringVar(value=supervisor_model or _sup_choices[0])
        ttk.Combobox(sup_row, textvariable=self.supervisor_var, state="readonly", width=30,
                    values=_sup_choices).pack(side="left", padx=(4, 0))

        self.server_line = ttk.Label(col3, text="server: not started", foreground="#888")
        self.server_line.pack(anchor="w", pady=(0, 2))
        btns = ttk.Frame(col3)
        btns.pack(anchor="w", pady=(0, 2))
        self.launch_btn = ttk.Button(btns, text="Launch", command=self.on_launch)
        self.launch_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Stop plan", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.shutdown_btn = ttk.Button(btns, text="Shutdown server", command=self.on_shutdown,
                                       state="disabled")
        self.shutdown_btn.pack(side="left", padx=6)
        btns2 = ttk.Frame(col3)
        btns2.pack(anchor="w", pady=(0, 6))
        self.kill_btn = ttk.Button(btns2, text="Kill (hard, if wedged)", command=self.on_kill,
                                   state="disabled")
        self.kill_btn.pack(side="left")
        ttk.Button(btns2, text="EMERGENCY STOP (zero cmd_vel)",
                  command=self.on_emergency_stop).pack(side="left", padx=6)

        ttk.Label(col3, text="Run log").pack(anchor="w")
        self.log_text = tk.Text(col3, width=48, height=22, bg="#111", fg="#ddd",
                                font=("Courier", 9), state="disabled", wrap="word")
        self.log_text.pack(anchor="w")

        # ---- column 4: fact3r-map goal memory (recall & remember) ---------
        col4 = ttk.Frame(main)
        col4.grid(row=0, column=3, sticky="n")
        ttk.Label(col4, text="Recall & remember", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(col4, text="fact3r-map goal memory -- \"go to X\" writes a confirmed "
                             "DINO arrival here; a later \"go back to X\" recalls it and "
                             "approaches before searching.",
                 wraplength=280, justify="left", foreground="#666").pack(anchor="w", pady=(0, 4))
        self.memory_event_line = ttk.Label(col4, text="no recall/remember activity yet",
                                           foreground="#888", wraplength=280, justify="left")
        self.memory_event_line.pack(anchor="w", pady=(0, 6))
        ttk.Label(col4, text="Remembered locations").pack(anchor="w")
        self.memory_list = tk.Listbox(col4, width=36, height=20, bg="#111", fg="#ddd",
                                      font=("Courier", 9), selectbackground="#333")
        self.memory_list.pack(anchor="w")
        self._memory_mtime = -1.0   # != any real mtime (or the 0.0 used when the file doesn't exist
        self._memory_records: list = []   # yet) so the panel populates ("no memory yet") on the first tick

        # ---- column 5: fact3r LIVE entity map (id + location) ------------
        # Distinct from column 4: that is the JSONL goal-memory of confirmed
        # DINO arrivals; THIS is the SAM2+SigLIP2 entity map built from every
        # frame while driving (run_rover_memfirst_dino.py, published on
        # 'multileg/fact3r'). "go to X" queries this first.
        col5 = ttk.Frame(main)
        col5.grid(row=0, column=4, sticky="n", padx=(10, 0))
        self.fact3r_head = ttk.Label(col5, text="fact3r entities (live map)",
                                     font=("TkDefaultFont", 12, "bold"))
        self.fact3r_head.pack(anchor="w")
        ttk.Label(col5, text="every object SAM2 sees while driving, with a metric "
                             "position. ★ = the one the last “go to X” recalled.",
                  wraplength=280, justify="left", foreground="#666").pack(anchor="w", pady=(0, 4))
        self.fact3r_list = tk.Listbox(col5, width=40, height=30, bg="#111", fg="#ddd",
                                      font=("Courier", 9), selectbackground="#333")
        self.fact3r_list.pack(anchor="w")

        self.conn_line = ttk.Label(root, text="waiting for data...", foreground="#888")
        self.conn_line.grid(row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(4, 4))

        self.pi_ip = pi_ip
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        instr_pub = session.declare_publisher("multileg/instruction")
        self.runner = RunnerProcess(pi_ip, self.log_queue,
                                   log_dir=os.path.join(os.path.dirname(RUNNER_PATH), "logs"),
                                   instr_pub=instr_pub)
        self.recorder = RecorderProcess(pi_ip, self.log_queue)
        self._emergency_session: Optional[zenoh.Session] = None

        self._photo = None
        self.root.after(120, self.refresh)
        self.root.after(150, self.drain_log)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- launcher controls -----------------------------------------------
    def on_launch(self):
        text = self.instr_entry.get().strip()
        if not text:
            self._log("[gui] type an instruction first -- the box is blank on purpose, "
                       "nothing launches until you do\n")
            return
        # First click spawns the persistent server (--supervisor-model-id
        # only matters here, since models load once); every click after
        # that just sends this instruction to the already-running server --
        # RunnerProcess.launch() decides which case this is.
        self.runner.launch(text, extra_args=["--supervisor-model-id", self.supervisor_var.get(),
                                             "--no-supervisor-4bit",
                                             # Live-observed 2026-09-08: a real, supervisor-confirmed
                                             # "chair" scoring 0.02-0.6 against both floors (defaults
                                             # 0.5 CLIP / 0.65 appearance-lock) for almost an entire
                                             # approach, starving TRACK and leaving the pipeline stuck
                                             # in SEARCH's fixed scan the whole time. Lowered so a real
                                             # object's own weak scores still clear the bar.
                                             "--dino-clip-min-sim", "0.4",
                                             "--dino-appearance-min-sim", "0.4",
                                             # Was --encoder-only (hard-forced) from 2026-09-09's
                                             # earlier frozen-IMU diagnosis. Per user request
                                             # (2026-09-09, later same day): use the IMU again --
                                             # run_rover_multileg.py now self-detects a frozen/dead
                                             # heading channel LIVE and falls back to encoder-only
                                             # automatically the first time it actually catches the
                                             # fault (--auto-encoder-only, on by default, see
                                             # nav_pipeline/odometry_logger.py's _check_imu_alive) --
                                             # no manual flag needed either way now. Pass
                                             # --encoder-only explicitly again only if the detector
                                             # itself turns out to be the problem.
                                             # Live-observed 2026-09-09: --max-angular defaults to
                                             # 1.2 rad/s (run_rover_multileg.py's own object-leg CLI
                                             # default), but bearing_to_angular's ramp constants
                                             # (servo_ramp_deg=35deg, angular_slew_max=0.10) were
                                             # tuned assuming the pipeline's real-rover max_angular
                                             # default of 0.25 -- with the cap at 1.2 the same few-
                                             # degree bearing jitter commands a >5x larger angular
                                             # rate, which read as the rover spinning hard even while
                                             # cleanly TRACKing a real detection. Bring it back down
                                             # to the tuned value. (2026-09-10: +20% per user
                                             # request, 0.25 -> 0.30.)
                                             "--max-angular", "0.30",
                                             # Per user request 2026-09-09: query goal-memory on
                                             # EVERY object leg -- recall and approach the remembered
                                             # point first if one matches this phrase, otherwise fall
                                             # straight through to a fresh DINO search (see
                                             # _recall_and_approach: returns False immediately on no
                                             # hit, no behavior change for a never-seen target).
                                             "--goal-memory-approach",
                                             # fact3r-map entity-memory integration (2026-09-09):
                                             # bank a SigLIP2 view of each confirmed arrival, and on
                                             # a later recall, appearance-check the first live
                                             # re-detection against it -- advisory signal into the
                                             # Qwen supervisor's context, doesn't override DINO/
                                             # supervisor arrival on its own. See
                                             # nav_pipeline/siglip2_embedder.py.
                                             "--goal-memory-appearance",
                                             # Give the Qwen supervisor NavDP's own last 2 buffered
                                             # frames alongside the live view each check() call, so
                                             # "is progress being made" isn't judged from one frozen
                                             # snapshot every 4s. See TaskSupervisor.check()'s
                                             # history_frames param / DinoNavDPPipeline.get_recent_frames.
                                             "--supervisor-history-frames", "2"])

    def on_stop(self):
        self.runner.stop()

    def on_shutdown(self):
        self.runner.shutdown()

    def on_kill(self):
        self.runner.kill()

    def on_clear_map(self):
        self.runner.clear_map()

    def on_recorder_start(self):
        self.recorder.start()

    def on_recorder_stop(self):
        self.recorder.stop()

    def on_emergency_stop(self):
        try:
            if self._emergency_session is None:
                cfg = zenoh.Config()
                if self.pi_ip:
                    cfg.insert_json5("connect/endpoints", f'["tcp/{self.pi_ip}:7447"]')
                self._emergency_session = zenoh.open(cfg)
            pub = self._emergency_session.declare_publisher("cmd_vel")
            for _ in range(6):
                pub.put(serialize_twist_zero())
                time.sleep(0.05)
            self._log("[gui] EMERGENCY STOP: zero cmd_vel sent x6\n")
        except Exception as e:
            self._log(f"[gui] emergency stop failed: {e}\n")

    def _log(self, text: str):
        self.log_queue.put(text)

    def drain_log(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        up = self.runner.is_running()
        ready = up and self.runner.ready.is_set()
        self.server_line.configure(
            text=("server: running, models loaded -- Launch just sends new instructions" if ready
                  else "server: starting (loading models)..." if up
                  else "server: not started -- Launch will start it"),
            foreground="#2e7d32" if ready else "#f9a825" if up else "#888")
        self.stop_btn.configure(state="normal" if up else "disabled")
        self.shutdown_btn.configure(state="normal" if up else "disabled")
        self.kill_btn.configure(state="normal" if up else "disabled")
        self.root.after(150, self.drain_log)

    def on_close(self):
        try:
            self.runner.shutdown()
        except Exception:
            pass
        try:
            self.recorder.stop()
        except Exception:
            pass
        self.root.destroy()

    # -- rendering ------------------------------------------------------
    def _draw_camera(self, rgb: Optional[np.ndarray], detector: Optional[dict]):
        if rgb is None:
            img = Image.new("RGB", (self.CAM_W, self.CAM_H), "#222")
        else:
            img = Image.fromarray(rgb.astype("uint8")).resize((self.CAM_W, self.CAM_H))
        if detector and detector.get("box"):
            sx, sy = self.CAM_W / RAW_W, self.CAM_H / RAW_H
            x0, y0, x1, y1 = detector["box"]
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            draw = ImageDraw.Draw(img)
            color = "#00e676" if detector.get("detected") else "#ff5252"
            draw.rectangle(box, outline=color, width=3)
            score = detector.get("score")
            label = f"{detector.get('target', '?')} {score:.2f}" if score is not None else str(detector.get("target"))
            ty = max(box[1] - 16, 0)
            draw.rectangle((box[0], ty, box[0] + 8 * len(label) + 6, ty + 15), fill=color)
            draw.text((box[0] + 3, ty + 1), label, fill="#000000")
        self._photo = ImageTk.PhotoImage(img)
        self.cam_label.configure(image=self._photo)

    def _update_plan(self, status: dict):
        plan = status.get("plan") or []
        cur = status.get("current_index", -1)
        results = status.get("results") or []
        cur_path_m = status.get("current_path_length_m")
        good = ("ok", "reached_dino", "reached_supervisor", "reached_close_loss", "reached_supervisor_approx")
        lines = []   # (text, tag)
        for i, desc in enumerate(plan):
            matching = [r for r in results if str(r.get("label", "")).startswith(f"{i}_")]
            if matching:
                r = matching[-1]
                outcome = r.get("outcome", "?")
                path_m = r.get("path_length_m")
                path_txt = f"  path={path_m:.2f}m" if path_m is not None else ""
                lines.append((f"done {i + 1}. {desc}  -> {outcome}{path_txt}\n",
                              "done_ok" if outcome in good else "done_bad"))
            elif i == cur:
                path_txt = f"  path={cur_path_m:.2f}m" if cur_path_m is not None else ""
                lines.append((f"-> {i + 1}. {desc}  (running){path_txt}\n", "running"))
            else:
                lines.append((f"   {i + 1}. {desc}\n", "pending"))
        # Only rewrite when the rendered text actually changed -- re-inserting
        # every ~120ms refresh tick (even with identical content) would drop
        # whatever the user just selected to copy, mid-selection.
        rendered = "".join(t for t, _ in lines)
        if rendered != getattr(self, "_plan_rendered", None):
            self._plan_rendered = rendered
            self.plan_text.configure(state="normal", height=max(3, min(len(lines), 20)))
            self.plan_text.delete("1.0", "end")
            for text, tag in lines:
                self.plan_text.insert("end", text, tag)
            self.plan_text.configure(state="disabled")
        self._plan_len = len(plan)
        self._log_new_subtasks(results, status.get("origin_to_plan_start_m"))

    def _log_new_subtasks(self, results: list, origin_to_plan_start_m: Optional[float]):
        """Append one JSON row per newly-completed subtask to EXPERIMENT_LOG_PATH
        (a JSON array on disk -- read, extend, rewrite), tagged with the
        current 'Experiment no.' box, so several experiments in one session
        can be compared by path length / odometry trace later. Each
        (label, dur_s) is logged once; dur_s in the key distinguishes a RETRY
        of the same label (see run_rover_multileg.py's orchestrator retry).
        odometry_csv + origin_to_plan_start_m come straight from the backend
        (run_rover_multileg.py's _leg_odometry_csv / origin_to_plan_start_m) --
        the raw (t,x,y,theta,v,w,...) trace and the distance already
        travelled from the session origin before this plan's first subtask."""
        new_rows = []
        for r in results:
            key = (r.get("label"), r.get("dur_s"))
            if key in self._logged_subtask_keys:
                continue
            self._logged_subtask_keys.add(key)
            new_rows.append(r)
        if not new_rows:
            return
        exp_no = self.experiment_var.get().strip() or "1"
        instruction = self.instr_entry.get().strip()
        now_s = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for r in new_rows:
            target = r.get("instruction") or r.get("target_m") or r.get("target_deg") or ""
            rows.append({
                "experiment_no": exp_no, "logged_at": now_s, "instruction": instruction,
                "subtask_label": r.get("label"), "kind": r.get("kind"), "target": target,
                "path_length_m": r.get("path_length_m"), "dur_s": r.get("dur_s"),
                "outcome": r.get("outcome"), "odometry_csv": r.get("odometry_csv"),
                "origin_to_plan_start_m": origin_to_plan_start_m,
            })
        try:
            os.makedirs(os.path.dirname(EXPERIMENT_LOG_PATH), exist_ok=True)
            existing = []
            if os.path.exists(EXPERIMENT_LOG_PATH):
                try:
                    with open(EXPERIMENT_LOG_PATH) as f:
                        loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing = loaded
                except Exception:
                    pass   # corrupt/foreign file -- don't lose new rows over it, start fresh below
            existing.extend(rows)
            with open(EXPERIMENT_LOG_PATH, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            self._log(f"[gui] experiment log write failed: {e}\n")

    def _draw_navdp(self, traj: List[Tuple[float, float]], age: Optional[float]):
        c = self.navdp_canvas
        c.delete("all")
        cx, r = self.NAVDP_SIZE / 2, self.NAVDP_SIZE / 2
        scale = r / self.NAVDP_RANGE_M
        # robot at bottom-centre, facing "up" (body frame: x forward, y left)
        rx, ry = cx, self.NAVDP_SIZE - 12
        c.create_line(0, ry, self.NAVDP_SIZE, ry, fill="#333")
        c.create_polygon(rx, ry - 8, rx - 5, ry + 4, rx + 5, ry + 4, fill="#00e676")
        stale = age is not None and age > 2.0
        color = "#666" if stale else "#ffca28"
        pts = []
        for (x, y) in traj:
            pts.extend((rx - y * scale, ry - x * scale))
        if len(pts) >= 4:
            c.create_line(*pts, fill=color, width=2)
        for i in range(0, len(pts), 2):
            c.create_oval(pts[i] - 2, pts[i + 1] - 2, pts[i] + 2, pts[i + 1] + 2, fill=color, outline="")

    def _draw_occupancy(self, png_bytes: Optional[bytes], pose, stale: bool):
        """The server publishes a grid already cropped to ±TRAIL_RANGE_M
        around the SAME world origin _draw_trail uses, so pose overlays with
        the identical transform -- no separate scale/offset bookkeeping."""
        c = self.occ_canvas
        c.delete("all")
        size = self.TRAIL_SIZE
        if png_bytes:
            try:
                img = Image.open(io.BytesIO(png_bytes)).convert("L")
                img = img.transpose(Image.FLIP_TOP_BOTTOM)   # grid row grows with +y; screen y is inverted
                img = img.resize((size, size), Image.NEAREST)
                if stale:
                    img = Image.eval(img, lambda p: int(p * 0.5))   # dim it if the feed's gone quiet
                self._occ_photo = ImageTk.PhotoImage(img)
                c.create_image(0, 0, anchor="nw", image=self._occ_photo)
            except Exception:
                pass
        cx, cy, r = size / 2, size / 2, size / 2
        scale = r / self.TRAIL_RANGE_M
        c.create_line(cx, 0, cx, size, fill="#333")
        c.create_line(0, cy, size, cy, fill="#333")
        if pose:
            px, py = cx + pose[0] * scale, cy - pose[1] * scale
            th = np.radians(pose[2]) if len(pose) > 2 else 0.0
            c.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#00e676", outline="")
            c.create_line(px, py, px + 12 * np.cos(th), py - 12 * np.sin(th), fill="#00e676", width=2)

    def _refresh_memory_panel(self, memory_event: Optional[dict]):
        if memory_event:
            phrase = memory_event.get("phrase", "?")
            wxy = memory_event.get("world_xy")
            age_s = time.time() - float(memory_event.get("ts", time.time()))
            loc = f"({wxy[0]:+.2f},{wxy[1]:+.2f})" if wxy else "?"
            if memory_event.get("event") == "recalled":
                text = f"RECALLED '{phrase}' from {loc} ({age_s:.0f}s old) -- approaching before search"
                color = "#4fc3f7"
            else:
                text = f"REMEMBERED '{phrase}' -> {loc}"
                color = "#2e7d32"
            self.memory_event_line.configure(text=text, foreground=color)

        # Re-read the JSONL only when its mtime changes -- cheap, and works
        # whether the record was written by this GUI's own server or a bare
        # CLI run, since both write the same file.
        try:
            mtime = os.path.getmtime(GOAL_MEMORY_PATH)
        except OSError:
            mtime = 0.0
        if mtime != self._memory_mtime:
            self._memory_mtime = mtime
            records = []
            try:
                with open(GOAL_MEMORY_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception:
                pass
            self._memory_records = records
            self.memory_list.delete(0, "end")
            if not records:
                self.memory_list.insert("end", "  (no memory yet)")
            for r in reversed(records[-100:]):
                wxy = r.get("world_xy")
                loc = f"({wxy[0]:+.2f},{wxy[1]:+.2f})" if wxy else "?"
                age_min = (time.time() - float(r.get("ts", 0))) / 60.0
                self.memory_list.insert("end", f"{r.get('query_text', '?')[:22]:22} {loc:16} {age_min:5.1f}m ago")

    # -- main loop --------------------------------------------------------
    def _refresh_fact3r_panel(self, fact3r: Optional[dict], age: Optional[float]):
        ents = (fact3r or {}).get("entities") or []
        recalled = (fact3r or {}).get("recalled_id")
        # The entity map lives only in the --serve process. Once that's gone
        # (server shut down, GUI about to be closed) the feed stops -- drop
        # the rows so stale entities don't linger on screen.
        gone = age is not None and age > 12.0
        if gone:
            ents = []
        n = len(ents)
        stale = age is not None and 8.0 < age <= 12.0
        self.fact3r_head.configure(
            text=(f"fact3r entities (live map)  —  {n}"
                  + ("  [stale]" if stale else "")
                  + ("  (server closed — map erased)" if gone else "")
                  + ("  (no map yet)" if fact3r is None else "")))
        self.fact3r_list.delete(0, "end")
        if not ents:
            self.fact3r_list.insert(
                "end", "  (server closed — map erased)" if gone
                else "  (nothing mapped yet — drive to populate)")
            return
        self.fact3r_list.insert("end", f"   {'id':8} {'x':>6} {'y':>6} {'obs':>4} {'d(m)':>5}")
        for e in ents:
            xy = e.get("xy") or [0.0, 0.0]
            is_recalled = e.get("id") == recalled
            mark = "★ " if is_recalled else "  "
            d = e.get("depth_m")
            self.fact3r_list.insert(
                "end",
                f"{mark}{str(e.get('id', '?')):8} {xy[0]:6.2f} {xy[1]:6.2f} "
                f"{e.get('obs', 0):4d} {('%.1f' % d) if d is not None else '-':>5}")
            if is_recalled:
                self.fact3r_list.itemconfig(self.fact3r_list.size() - 1, foreground="#7fdfff")

    def refresh(self):
        with self.st.lock:
            rgb = self.st.rgb
            rgb_age = time.time() - self.st.rgb_t if self.st.rgb_t else None
            status = self.st.status
            status_age = time.time() - self.st.status_t if self.st.status_t else None
            navdp_traj = list(self.st.navdp_traj)
            navdp_age = time.time() - self.st.navdp_traj_t if self.st.navdp_traj_t else None
            rec = self.st.record_status
            rec_age = time.time() - self.st.record_status_t if self.st.record_status_t else None
            occ_png = self.st.occupancy_png
            occ_age = time.time() - self.st.occupancy_t if self.st.occupancy_t else None
            fact3r = self.st.fact3r
            fact3r_age = time.time() - self.st.fact3r_t if self.st.fact3r_t else None

        self._refresh_fact3r_panel(fact3r, fact3r_age)

        detector = status.get("detector") if status else None
        self._draw_camera(rgb, detector)
        self._draw_navdp(navdp_traj, navdp_age)
        self._draw_occupancy(occ_png, status.get("pose") if status else None,
                             stale=(occ_age is None or occ_age > 3.0))

        if status:
            self._update_plan(status)
            state = (detector or {}).get("state", "-")
            color = STATE_COLOR.get(state, "#888")
            self.state_line.configure(
                text=f"[{status.get('current_kind', '?')}] {status.get('current_desc', '')}  --  {state}",
                foreground=color)
            if detector:
                gp = detector.get("goal_point")
                d2g = f"{(gp[0] ** 2 + gp[1] ** 2) ** 0.5:.2f}m" if gp else "n/a"
                self.detector_line.configure(
                    text=f"target='{detector.get('target')}' detected={detector.get('detected')} "
                         f"score={detector.get('score')} dist={d2g}"
                         f"{'  AMBIGUOUS' if detector.get('ambiguous') else ''}")
            else:
                self.detector_line.configure(text="")
            sup = status.get("supervisor")
            self.supervisor_line.configure(
                text=(f"supervisor: {sup.get('status')} act={sup.get('action')} "
                      f"target='{sup.get('target')}' -- {sup.get('reason', '')}") if sup else "")
            pose = status.get("pose")
            if pose:
                self.pose_line.configure(text=f"pose  x={pose[0]:+.2f}  y={pose[1]:+.2f}  θ={pose[2]:+.0f}°")

        if rec:
            if rec.get("recording"):
                self.record_line.configure(
                    text=f"RECORDING  t={rec.get('t', 0):.0f}s  frames={rec.get('frames')}  "
                         f"odom={rec.get('rpm_samples')}  -> {rec.get('out')}",
                    foreground="#2e7d32" if (rec_age or 0) < 3 else "#c62828")
            else:
                self.record_line.configure(text=f"stopped -- last take: {rec.get('out')}", foreground="#888")

        self._refresh_memory_panel(status.get("memory") if status else None)

        bits = [f"camera {rgb_age:.1f}s ago" if rgb_age is not None else "camera: no data",
                f"status {status_age:.1f}s ago" if status_age is not None else "status: no data"]
        stale = (rgb_age is not None and rgb_age > 3) or (status_age is not None and status_age > 3)
        self.conn_line.configure(text="  |  ".join(bits), foreground="#c62828" if stale else "#4caf50")

        self.root.after(120, self.refresh)


def main():
    global RUNNER_PATH, GOAL_MEMORY_PATH
    ap = argparse.ArgumentParser(description="Live viewer + launcher for run_rover_multileg.py")
    ap.add_argument("--pi-ip", type=str, default=None, help="Pi IP for the Zenoh peer; omit for multicast")
    ap.add_argument("--default-instruction", type=str, default="",
                    help="pre-fill the instruction box with this text (still requires clicking "
                         "Launch -- never auto-launches)")
    ap.add_argument("--runner", type=str, default=None,
                    help="path to the --serve runner the Run button spawns (default: "
                         "run_rover_multileg.py). Point at run_rover_occluded_recall.py for the "
                         "go-behind + recall pipeline -- it accepts the same --serve / --supervisor-* "
                         "/ --goal-memory-* / --max-angular args, plus directional + recall clauses.")
    ap.add_argument("--supervisor-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                    help="pre-select the Supervisor combobox (still switchable in the GUI). "
                         "run_rover_memfirst_dino.py's fact3r recall verifies crops with this "
                         "model -- 7B/8B are far better judges than the 2B default.")
    args = ap.parse_args()

    if args.runner:
        RUNNER_PATH = os.path.abspath(args.runner)
        GOAL_MEMORY_PATH = os.path.join(os.path.dirname(RUNNER_PATH), "logs", "live_goal_memory.jsonl")
        print(f"[INFO] Run button will spawn: {RUNNER_PATH}")

    cfg = zenoh.Config()
    if args.pi_ip:
        cfg.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[INFO] Zenoh: connecting to tcp/{args.pi_ip}:7447")
    else:
        print("[INFO] Zenoh: multicast scouting")
    session = zenoh.open(cfg)

    st = SharedState()
    root = tk.Tk()
    root._subs = zenoh_setup(session, st)   # keep subscriber refs alive for root's lifetime
    MultilegViewer(root, st, args.pi_ip, session, default_instruction=args.default_instruction,
                   supervisor_model=args.supervisor_model)
    try:
        root.mainloop()
    finally:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
