#!/usr/bin/env python3
"""Real-time viewer for run_ab_position_recall_test.py.

The test drops an atomic snapshot (frame.jpg + topdown.jpg + state.json) into
its --live-dir every tick; this GUI polls that directory and shows the camera
view, the carved top-down map, a status strip, and the test's stdout log.

Two ways to use it:

  # 1. spawn the run from here -- everything after `--` goes to the test
  python ab_recall_viewer_gui.py -- \
      --scene-glb /path/2azQ1b91cZZ_yup.ply --mp3d-house /path/2azQ1b91cZZ.house \
      --category-a chair --category-b table --no-start-instance-filter

  # 2. just watch a run that's already writing somewhere
  python ab_recall_viewer_gui.py --watch /tmp/ab_live

Runs in any env with tkinter + pillow (the test itself needs the `habitat`
env; when spawning, pass --python /path/to/habitat/python if this GUI runs
under a different interpreter).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

POLL_MS = 120
CAM_W = 640
MAP_W = 460


class Viewer:
    def __init__(self, root: tk.Tk, live_dir: Path, proc: subprocess.Popen | None):
        self.root = root
        self.live_dir = live_dir
        self.proc = proc
        self.log_q: queue.Queue[str] = queue.Queue()
        self._last_state_ts = 0.0
        self._cam_tk = None
        self._map_tk = None

        root.title(f"A/B recall viewer -- {live_dir}")
        root.configure(bg="#1e1e1e")

        top = tk.Frame(root, bg="#1e1e1e")
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.cam_lbl = tk.Label(top, bg="black", text="waiting for frame.jpg ...",
                                fg="#888", width=CAM_W, height=480)
        self.cam_lbl.pack(side=tk.LEFT, padx=6, pady=6)

        right = tk.Frame(top, bg="#1e1e1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.map_lbl = tk.Label(right, bg="black", text="waiting for topdown.jpg ...",
                                fg="#888", width=MAP_W)
        self.map_lbl.pack(side=tk.TOP)
        tk.Label(right, bg="#1e1e1e", fg="#666", font=("TkFixedFont", 8),
                 text="orange = path   green A   red B   yellow start   "
                      "magenta recalled   cyan = rover").pack(side=tk.TOP, anchor="w")

        self.status = tk.Label(root, bg="#111", fg="#0f0", justify=tk.LEFT,
                               anchor="w", font=("TkFixedFont", 11), padx=8, pady=6)
        self.status.pack(side=tk.TOP, fill=tk.X)

        self.bar = ttk.Progressbar(root, maximum=1.0)
        self.bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        logf = tk.Frame(root, bg="#1e1e1e")
        logf.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.log = tk.Text(logf, height=12, bg="#0a0a0a", fg="#bbb",
                           font=("TkFixedFont", 9), wrap=tk.NONE)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if proc is not None:
            threading.Thread(target=self._pump_proc, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(POLL_MS, self._tick)

    # -- subprocess stdout -> queue -------------------------------------- #
    def _pump_proc(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.log_q.put(line.rstrip("\n"))
        self.log_q.put(f"[viewer] run process exited with code {self.proc.wait()}")

    def _drain_log(self):
        appended = False
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            # skip the habitat/magnum boilerplate spam
            low = line.lower()
            if any(s in low for s in ("glob path result", "no_lights", "deprecated",
                                      "gl_", "workaround", "no-forward-compatible",
                                      "no-layout-qualifiers")):
                continue
            self.log.insert(tk.END, line + "\n")
            appended = True
        if appended:
            self.log.see(tk.END)

    # -- snapshot polling ---------------------------------------------- #
    def _load_img(self, name: str, target_w: int):
        p = self.live_dir / name
        try:
            data = p.read_bytes()
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            return None
        if im.width != target_w:
            h = int(im.height * target_w / im.width)
            im = im.resize((target_w, h), Image.NEAREST if "topdown" in name else Image.BILINEAR)
        return ImageTk.PhotoImage(im)

    def _tick(self):
        self._drain_log()

        cam = self._load_img("frame.jpg", CAM_W)
        if cam is not None:
            self._cam_tk = cam
            self.cam_lbl.configure(image=cam, text="", width=cam.width(), height=cam.height())
        mp = self._load_img("topdown.jpg", MAP_W)
        if mp is not None:
            self._map_tk = mp
            self.map_lbl.configure(image=mp, text="", width=mp.width(), height=mp.height())

        try:
            st = json.loads((self.live_dir / "state.json").read_text())
        except Exception:
            st = None
        if st and st.get("ts", 0) != self._last_state_ts:
            self._last_state_ts = st["ts"]
            self._render_status(st)

        if self.proc is not None and self.proc.poll() is not None and self.log_q.empty():
            # keep showing the final frame; stop hammering the poller as fast
            self.root.after(600, self._tick)
            return
        self.root.after(POLL_MS, self._tick)

    def _render_status(self, st: dict):
        age = time.time() - st.get("ts", 0)
        lines = []
        lines.append(f"scene: {st.get('scene', '?')}")
        lines.append(f"PHASE: {st.get('phase', '?')}")
        step, mx = st.get("step", 0), st.get("max_steps", 0)
        if mx:
            lines.append(f"step {step}/{mx}   state={st.get('state', '?')}"
                         f"   {'  <ESCAPING>' if st.get('escaping') else ''}")
            self.bar.configure(value=min(1.0, step / mx))
        else:
            lines.append(f"state={st.get('state', '?')}")
        if st.get("mode"):
            lines.append(f"drive mode: {st['mode']}   waypoint {st.get('waypoint', '-')}")
        if "dist_to_A_m" in st:
            lines.append(f"dist to A: {st['dist_to_A_m']:.2f} m")
        if "dist_to_recalled_m" in st:
            lines.append(f"dist to recalled B: {st['dist_to_recalled_m']:.2f} m")
        if "entities" in st:
            lines.append(f"entities mapped: {st['entities']}")
        if st.get("instruction"):
            lines.append(f"instruction: {st['instruction']!r}")
        if st.get("done"):
            lines.append("")
            lines.append(f"LEG1 {st.get('leg1_outcome')}  err={st.get('leg1_err_m')} m  "
                         f"{'OK' if st.get('leg1_success') else 'FAIL'}")
            lines.append(f"RECALL err={st.get('recall_err_m')} m")
            lines.append(f"LEG2 {st.get('leg2_outcome')}  err={st.get('leg2_err_m')} m  "
                         f"{'OK' if st.get('leg2_success') else 'FAIL'}")
            lines.append(f"SUCCESS RATE = {st.get('success_rate')}")
        lines.append(f"  (snapshot age {age:.1f}s)")
        self.status.configure(text="\n".join(lines),
                              fg="#5f5" if age < 3 else "#fa5")

    def _on_close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", metavar="DIR",
                    help="watch an existing run's --live-dir instead of spawning one")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run the test with when spawning "
                         "(default: this one; use the habitat env's python)")
    ap.add_argument("--test-script", default=str(Path(__file__).with_name("run_ab_position_recall_test.py")))
    ap.add_argument("run_args", nargs=argparse.REMAINDER,
                    help="after `--`, all remaining args go to run_ab_position_recall_test.py")
    args = ap.parse_args()

    proc = None
    if args.watch:
        live_dir = Path(args.watch)
        live_dir.mkdir(parents=True, exist_ok=True)
    else:
        extra = args.run_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        if not extra:
            ap.error("nothing to run: pass test args after `--`, or use --watch DIR")
        live_dir = Path("logs/_live") / time.strftime("%Y%m%d_%H%M%S")
        live_dir.mkdir(parents=True, exist_ok=True)
        cmd = [args.python, args.test_script, *extra, "--live-dir", str(live_dir)]
        env = dict(os.environ, HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet",
                   PYTHONUNBUFFERED="1")
        print("spawning:", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)

    root = tk.Tk()
    Viewer(root, live_dir, proc)
    root.mainloop()


if __name__ == "__main__":
    main()
