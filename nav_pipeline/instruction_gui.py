"""Qwen instruction -> NavDP GUI: camera feed, top-down trajectory plot,
free-text instruction box, Send, STOP. No DINO target box, no object
presets, no manual-drive pad -- this GUI has exactly one way to command the
rover: type an instruction and Send.

Reuses nav_pipeline.isaac_gui's Zenoh subscription plumbing, odometry
wiring, and inference loop UNCHANGED (SharedState/zenoh_setup/
heartbeat_loop/inference_loop) -- inference_loop already reads
pipe.cfg.use_qwen_instruction and, when set, sends SharedState.target as a
free-text instruction to DinoNavDPPipeline.step() instead of a DINO target
phrase (see isaac_gui.py). This module only supplies a different Tk layout
around that same pipeline, forced into instruction mode.
"""
import argparse
import signal
import time
import tkinter as tk
from threading import Thread
from tkinter import ttk
from typing import Optional

import numpy as np
import zenoh
from PIL import Image, ImageDraw, ImageTk

from .isaac_gui import SharedState, heartbeat_loop, inference_loop, zenoh_setup
from .obstacle_guard import GuardConfig
from .odometry_logger import OdometryLogger
from .pipeline import DinoNavDPPipeline, PipelineConfig
from .zenoh_node import serialize_twist

INSTRUCTION_PRESETS = [
    "go to the door",
    "walk through the doorway ahead",
    "go straight and stop before the wall",
]


class InstructionApp:
    CAM_SIZE = 448
    PLOT_SIZE = 448
    PLOT_RANGE = 3.5   # meters shown ahead

    def __init__(self, root: tk.Tk, st: SharedState):
        self.root = root
        self.st = st
        root.title("Nav_new — Qwen instruction + NavDP")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.closed = False

        main = ttk.Frame(root, padding=8)
        main.grid(sticky="nsew")

        self.cam_label = ttk.Label(main)
        self.cam_label.grid(row=0, column=0, padx=4, pady=4)
        self._blank_photo = ImageTk.PhotoImage(Image.new("RGB", (self.CAM_SIZE, self.CAM_SIZE), "#222"))
        self.cam_label.configure(image=self._blank_photo)
        self.plot = tk.Canvas(main, width=self.PLOT_SIZE, height=self.PLOT_SIZE, bg="white")
        self.plot.grid(row=0, column=1, padx=4, pady=4)

        bar = ttk.Frame(main)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="Instruction:").pack(side="left")
        self.entry = ttk.Entry(bar, width=52)
        self.entry.insert(0, st.target)
        self.entry.pack(side="left", padx=4, fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send_instruction())
        ttk.Button(bar, text="Send", command=self.send_instruction).pack(side="left", padx=2)
        ttk.Button(bar, text="STOP", command=self.stop).pack(side="left", padx=10)

        presets = ttk.Frame(main)
        presets.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        for p in INSTRUCTION_PRESETS:
            ttk.Button(presets, text=p, command=lambda t=p: self.send_instruction(t)).pack(side="left", padx=2)

        # Fixed character width for the same reason as isaac_gui.App: the
        # text length changes every refresh tick, and an unconstrained
        # Label makes the whole window resize on every tick.
        self.status = ttk.Label(main, text="starting...", font=("TkDefaultFont", 11, "bold"),
                                 width=100, anchor="w")
        self.status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.info = ttk.Label(main, text="", width=100, anchor="w")
        self.info.grid(row=4, column=0, columnspan=2, sticky="w")

        self._photo = None
        self.root.after(66, self.refresh)

    def send_instruction(self, text: Optional[str] = None):
        t = text if text is not None else self.entry.get().strip()
        if not t:
            return
        if text is not None:
            self.entry.delete(0, "end")
            self.entry.insert(0, t)
        with self.st.lock:
            self.st.mode = "text"
            self.st.target = t
            self.st.stopped = False
            self.st.goal_reached = False

    def stop(self):
        with self.st.lock:
            self.st.stopped = True
            self.st.last_cmd = (0.0, 0.0)

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
            import traceback
            traceback.print_exc()
        finally:
            if not self.closed:
                self.root.after(66, self.refresh)

    def _refresh_body(self):
        with self.st.lock:
            # freshest camera frame (camera's native rate), not display_rgb
            # (only updated once per inference tick) -- see isaac_gui.App
            # for why: showing display_rgb made the picture visibly jump
            # between frames ~400ms apart during movement.
            rgb = self.st.latest_rgb if self.st.latest_rgb is not None else self.st.display_rgb
            qwen_pixel_goal = self.st.qwen_pixel_goal
            trajs, chosen, goal = self.st.trajs, self.st.chosen, self.st.goal_pt
            obstacles, min_fwd = self.st.obstacles, self.st.min_forward
            state_text, vel_text, lat = self.st.state_text, self.st.vel_text, self.st.lat_text
            frames, infers, instruction = self.st.frame_count, self.st.infer_count, self.st.target
            stopped = self.st.stopped
            imu_heading, imu_calib, theta_source = self.st.imu_heading_deg, self.st.imu_calib, self.st.theta_source

        if rgb is not None:
            img = Image.fromarray(rgb).convert("RGB")
            sx, sy = self.CAM_SIZE / img.width, self.CAM_SIZE / img.height
            img = img.resize((self.CAM_SIZE, self.CAM_SIZE))
            if qwen_pixel_goal is not None:
                u, v = qwen_pixel_goal
                d = ImageDraw.Draw(img)
                cx, cy = u * sx, v * sy
                r = 8
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 176, 32), width=3)
                d.line([cx - r - 4, cy, cx + r + 4, cy], fill=(255, 176, 32), width=1)
                d.line([cx, cy - r - 4, cx, cy + r + 4], fill=(255, 176, 32), width=1)
                d.text((cx + r + 4, cy - 8), "qwen goal", fill=(255, 176, 32))
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
        fwd = f"   fwd-clear {min_fwd:.2f}m" if np.isfinite(min_fwd) else ""
        self.status.configure(text=f"[{mode_txt}]  instruction: '{instruction}'   {vel_text}{fwd}")
        heading_txt = f"{imu_heading:.1f}°" if np.isfinite(imu_heading) else "n/a"
        imu_txt = (f"theta src: {theta_source}   imu heading {heading_txt}"
                   f"  calib [{OdometryLogger.decode_calib(imu_calib)}]")
        self.info.configure(text=f"frames {frames}   inferences {infers}   {lat}   {imu_txt}")


def main():
    ap = argparse.ArgumentParser(description="Nav_new Qwen-instruction + NavDP GUI")
    ap.add_argument("--instruction", default="",
                    help="starts empty -- rover stays inert until an instruction is sent "
                         "from the GUI (or passed here)")
    ap.add_argument("--pi-ip", default=None)
    ap.add_argument("--predict-hz", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-linear", type=float, default=0.5,
                    help="m/s cap (sim default; use 0.15 on the real rover)")
    ap.add_argument("--max-angular", type=float, default=0.4,
                    help="rad/s cap (sim default; use 0.25 on the real rover)")
    ap.add_argument("--servo-ramp-deg", type=float, default=35.0)
    ap.add_argument("--angular-slew-max", type=float, default=0.10)
    ap.add_argument("--invert-angular", action="store_true")
    ap.add_argument("--no-belief-goal", action="store_true")
    ap.add_argument("--depth-encoder", choices=["vits", "vitb"], default="vits")
    ap.add_argument("--compressed-only", action="store_true",
                    help="subscribe only the JPEG camera stream (REQUIRED over rover Wi-Fi)")
    ap.add_argument("--odometry-log-dir", type=str, default="odometry_log")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3)
    ap.add_argument("--footprint-length", type=float, default=GuardConfig().footprint_length)
    ap.add_argument("--footprint-width", type=float, default=GuardConfig().footprint_width)
    ap.add_argument("--wheel-deadband-correction", action="store_true",
                    help="ESP32 6WD rover ONLY (never --hiwonder) -- boosts a commanded "
                         "(linear, angular) that would otherwise differential-mix down to a "
                         "stalled wheel on the real hardware; see pipeline.py's "
                         "clear_wheel_deadband")
    ap.add_argument("--qwen-model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--qwen-instruction-period-s", type=float, default=1.5,
                    help="min seconds between Qwen instruction-grounding calls; GoalBelief "
                         "coasts the goal by ego-motion in between")
    ap.add_argument("--qwen-goal-consistency-m", type=float, default=1.5,
                    help="reject a new Qwen instruction-grounding more than this many meters "
                         "from the current confident goal instead of snapping to it (e.g. a "
                         "second door coming into view mid-turn) -- coasts on the existing "
                         "lock instead. See pipeline.py's qwen_goal_consistency_m.")
    ap.add_argument("--qwen-max-candidates", type=int, default=3,
                    help="ask Qwen for up to this many ranked, confidence-scored waypoint "
                         "candidates instead of one, then pick the best by combining semantic "
                         "confidence + continuity with the current lock + obstacle-avoidance "
                         "cost (arxiv 2605.19420-inspired) -- 1 disables scoring, back to plain "
                         "single-point grounding. See PipelineConfig.qwen_max_candidates.")
    ap.add_argument("--qwen-fp16", action="store_true",
                    help="load Qwen2.5-VL-7B in fp16 instead of the 4-bit default (needs "
                         "~16.6GB VRAM vs 4-bit's ~6.2GB; use if bitsandbytes isn't installed)")
    ap.add_argument("--stop-distance", type=float, default=PipelineConfig().stop_distance,
                    help="metres from the goal point at which to declare arrival and stop "
                         "(depth-estimated; see pipeline.py's stop_distance). Collision "
                         "avoidance (guard.hard_stop_dist) is separate and still active.")
    args = ap.parse_args()

    print("[INFO] loading models...")
    pipe = DinoNavDPPipeline(PipelineConfig(
        device=args.device,
        horizontal_fov_deg=args.fov,
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        servo_ramp_deg=args.servo_ramp_deg,
        angular_slew_max=args.angular_slew_max,
        invert_angular=args.invert_angular,
        use_belief_goal=not args.no_belief_goal,
        depth_encoder=args.depth_encoder,
        guard=GuardConfig(footprint_length=args.footprint_length, footprint_width=args.footprint_width),
        wheel_deadband_correction=args.wheel_deadband_correction,
        # This GUI has no --avoid and no scene-tagger flag -- Grounding
        # DINO/SAM/CLIP/DINOv2 would sit in VRAM fully loaded and never
        # once called. Lean footprint: only Qwen + NavDP + the shared depth
        # estimator actually run here.
        use_dino=False,
        use_sam=False,
        use_appearance_reid=False,
        use_qwen_instruction=True,   # this GUI has exactly one navigation mode
        stop_distance=args.stop_distance,
        qwen_model_id=args.qwen_model_id,
        qwen_instruction_period_s=args.qwen_instruction_period_s,
        qwen_goal_consistency_m=args.qwen_goal_consistency_m,
        qwen_max_candidates=args.qwen_max_candidates,
        qwen_load_in_4bit=not args.qwen_fp16,
    ))

    config = zenoh.Config()
    if args.pi_ip:
        config.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
    session = zenoh.open(config)
    print("[INFO] zenoh session opened")

    st = SharedState(args.instruction)
    st.mode = "text" if args.instruction else "manual"
    st.max_linear, st.max_angular = args.max_linear, args.max_angular
    odom = OdometryLogger(args.odometry_log_dir, imu_min_mag_calib=args.imu_min_mag_calib)
    # inference_loop starts the odometry file itself on its first tick
    # (last_target starts as None specifically so a starting --instruction
    # triggers this the same way a GUI Send does -- see isaac_gui.py's
    # inference_loop) -- calling start_new_goal here too raced it: two
    # opens of the same CSV in quick succession, and an in-flight RPM
    # callback wrote to the first handle just as it closed ("write to
    # closed file"). Let inference_loop own this exclusively.
    _subs, pubs = zenoh_setup(session, st, compressed_only=args.compressed_only, odom=odom)

    running = {"on": True}
    Thread(target=heartbeat_loop, args=(st, pubs, running), daemon=True).start()
    Thread(target=inference_loop, args=(pipe, st, pubs, running, args.predict_hz),
           kwargs={"odom": odom}, daemon=True).start()

    root = tk.Tk()
    InstructionApp(root, st)

    # Ctrl-C in the launching terminal closes the GUI cleanly -- same
    # pattern as isaac_gui.py's main(), including the periodic no-op tick
    # (Tk's mainloop only delivers Python signals while Python code runs).
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
        # Double zero-velocity publish on the way out -- SIGTERM is NOT
        # caught by Python by default (only SIGINT is), so anything that
        # relies on a try/finally around mainloop to send a stop command
        # needs SIGTERM converted to a clean Tk shutdown first (above);
        # this is the actual stop command that results from it. See the
        # nav_pipeline.zenoh_node live-rover session for why this distinction
        # matters -- a bare `timeout`/`kill` on this process would otherwise
        # skip this entirely and leave the rover's last command in effect.
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
