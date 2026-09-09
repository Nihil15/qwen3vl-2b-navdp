#!/usr/bin/env python3
"""Manual-instruction GUI for the A/B recall test on an HM3D scene.

Walk the scene yourself (arrow keys / WASD), decide which objects to use,
type the two instructions, and launch run_ab_position_recall_test.py from
the pose you are standing at:

    A  --  initial navigation instruction  (leg 1, free text)   -> --instruction-a
    B  --  recall instruction              (leg 2, free text)   -> --query-b

The scene's own semantic categories are listed with instance counts so you
can see what is actually nameable; a click on the top-down map tells you the
nearest object's category and can drop it straight into the A / B category
box (used only for ground-truth scoring + start placement -- the wording the
pipeline follows is the free text you type).

Run inside the `habitat` conda env (habitat_sim + tkinter + PIL).
    python ab_recall_gui.py
    python ab_recall_gui.py --scene wcojb4TFT35
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections import Counter, defaultdict
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from run_ab_recall_hm3d_test import (
    FOV_DEG, H, W, build_sim, find_dataset_config, find_scene_glb, largest_island,
)

DEFAULT_HM3D_ROOT = "/home/gpu/Desktop/pineapple/mars-habitatsim/assets/hm3d/versioned_data/hm3d-0.2/hm3d"
STRUCTURAL = {
    "wall", "ceiling", "floor", "window", "door", "door frame", "beam", "pipe",
    "air duct", "smoke detector", "light switch", "air vent", "fire sprinkler",
    "window shutters", "curtain rod", "unknown", "misc", "objects",
}
STEP_M = 0.25
TURN_RAD = math.radians(9.0)


class Scene:
    """Thin habitat_sim wrapper: free-fly render + semantic inventory."""

    def __init__(self, hm3d_root: str, scene: str, cam_height: float):
        root = Path(hm3d_root)
        self.scene_glb = find_scene_glb(root, scene)
        self.dataset_cfg = find_dataset_config(root)
        self.sim = build_sim(self.scene_glb, self.dataset_cfg)
        self.pf = self.sim.pathfinder
        self.island = largest_island(self.pf)
        self.cam_height = cam_height

        start = np.array(self.pf.get_random_navigable_point(island_index=self.island))
        self.pos_xz = [float(start[0]), float(start[2])]
        self.floor_y = float(start[1])
        self.yaw = 0.0

        self.cat_count: Counter = Counter()
        self.cat_pos: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for obj in self.sim.semantic_scene.objects or []:
            if obj is None or obj.category is None:
                continue
            name = obj.category.name()
            snapped = np.array(self.pf.snap_point(obj.aabb.center(), self.island))
            if np.all(np.isfinite(snapped)):
                self.cat_count[name] += 1
                self.cat_pos[name].append((float(snapped[0]), float(snapped[2])))
        lo, hi = self.pf.get_bounds()
        self.bounds = (float(lo[0]), float(lo[2]), float(hi[0]), float(hi[2]))

    # -- movement -------------------------------------------------------- #
    def turn(self, d: float):
        self.yaw = (self.yaw + d + math.pi) % (2 * math.pi) - math.pi

    def move(self, fwd: float, strafe: float = 0.0):
        x, z = self.pos_xz
        nx = x + fwd * math.cos(self.yaw) + strafe * math.cos(self.yaw + math.pi / 2)
        nz = z + fwd * math.sin(self.yaw) + strafe * math.sin(self.yaw + math.pi / 2)
        snapped = np.array(self.pf.snap_point(np.array([nx, self.floor_y, nz]), self.island))
        if np.all(np.isfinite(snapped)):
            self.pos_xz = [float(snapped[0]), float(snapped[2])]
            self.floor_y = float(snapped[1])

    def teleport_random(self):
        p = np.array(self.pf.get_random_navigable_point(island_index=self.island))
        self.pos_xz = [float(p[0]), float(p[2])]
        self.floor_y = float(p[1])

    def face_category(self, cat: str):
        """Jump to a standoff pose looking at the nearest instance of `cat`."""
        pts = self.cat_pos.get(cat)
        if not pts:
            return
        x, z = self.pos_xz
        tx, tz = min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - z) ** 2)
        yaw = math.atan2(tz - z, tx - x)
        bx, bz = tx - 1.6 * math.cos(yaw), tz - 1.6 * math.sin(yaw)
        snapped = np.array(self.pf.snap_point(np.array([bx, self.floor_y, bz]), self.island))
        if np.all(np.isfinite(snapped)):
            self.pos_xz = [float(snapped[0]), float(snapped[2])]
            self.floor_y = float(snapped[1])
        self.yaw = yaw

    def nearest_category(self, x: float, z: float, max_d: float = 1.5):
        best, best_d = None, max_d ** 2
        for cat, pts in self.cat_pos.items():
            for px, pz in pts:
                d = (px - x) ** 2 + (pz - z) ** 2
                if d < best_d:
                    best, best_d = cat, d
        return best, math.sqrt(best_d) if best else None

    # -- render --------------------------------------------------------- #
    def frame(self) -> np.ndarray:
        cam_y = self.floor_y + self.cam_height
        habitat_yaw = -(self.yaw + math.pi / 2.0)
        st = habitat_sim.AgentState()
        st.position = np.array([self.pos_xz[0], cam_y, self.pos_xz[1]], dtype=np.float32)
        st.rotation = quat_from_angle_axis(habitat_yaw, np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(st)
        obs = self.sim.get_sensor_observations()
        return np.asarray(obs["rgb"])[:, :, :3].copy()


class App:
    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.args = args
        root.title(f"A/B recall -- manual instructions -- {args.scene}")
        self.scene = Scene(args.hm3d_root, args.scene, args.cam_height)
        self.proc: subprocess.Popen | None = None
        self.log_q: queue.Queue[str] = queue.Queue()

        outer = ttk.Frame(root, padding=6)
        outer.pack(fill="both", expand=True)

        # --- left: camera view --------------------------------------- #
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nw")
        self.view = ttk.Label(left)
        self.view.pack()
        self.pose_lbl = ttk.Label(left, font=("TkFixedFont", 10))
        self.pose_lbl.pack(anchor="w", pady=(4, 0))
        ttk.Label(
            left,
            foreground="#555",
            text=("↑/W forward   ↓/S back   ←→ / A D turn   "
                  "Q/E strafe   [ ] cam height   T random teleport"),
        ).pack(anchor="w")

        # --- middle: top-down map ---------------------------------------- #
        mid = ttk.Frame(outer)
        mid.grid(row=0, column=1, sticky="n", padx=8)
        ttk.Label(mid, text="top-down  (click = nearest object)").pack(anchor="w")
        self.MAP = 360
        self.canvas = tk.Canvas(mid, width=self.MAP, height=self.MAP, bg="#111",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_map_click)
        self.click_lbl = ttk.Label(mid, font=("TkFixedFont", 10),
                                   text="clicked: --")
        self.click_lbl.pack(anchor="w", pady=(3, 0))
        crow = ttk.Frame(mid)
        crow.pack(anchor="w", pady=2)
        ttk.Button(crow, text="▶ walk to it", command=self.walk_to_clicked).pack(side="left")
        ttk.Button(crow, text="set as A cat", command=lambda: self.set_cat("a")).pack(side="left", padx=3)
        ttk.Button(crow, text="set as B cat", command=lambda: self.set_cat("b")).pack(side="left")

        # --- right: categories + instructions + run -------------------- #
        right = ttk.Frame(outer)
        right.grid(row=0, column=2, sticky="n")
        ttk.Label(right, text="categories in scene (count)").pack(anchor="w")
        lb_frame = ttk.Frame(right)
        lb_frame.pack()
        self.cat_lb = tk.Listbox(lb_frame, height=14, width=30, font=("TkFixedFont", 9),
                                 exportselection=False)
        sb = ttk.Scrollbar(lb_frame, command=self.cat_lb.yview)
        self.cat_lb.configure(yscrollcommand=sb.set)
        self.cat_lb.pack(side="left")
        sb.pack(side="left", fill="y")
        self.cat_lb.bind("<<ListboxSelect>>", self.on_cat_select)
        self.cats_sorted = sorted(self.scene.cat_count,
                                  key=lambda c: (c in STRUCTURAL, -self.scene.cat_count[c], c))
        for c in self.cats_sorted:
            tag = "  " if c in STRUCTURAL else "* "
            self.cat_lb.insert("end", f"{tag}{c}  ({self.scene.cat_count[c]})")

        form = ttk.Frame(right, padding=(0, 8))
        form.pack(fill="x")
        pickable = [c for c in self.cats_sorted if c not in STRUCTURAL]

        ttk.Label(form, text="A — initial navigation instruction (leg 1)").grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.instr_a = ttk.Entry(form, width=42)
        self.instr_a.insert(0, "go to the ")
        self.instr_a.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 2))
        ttk.Label(form, text="A object category (scoring)").grid(row=2, column=0, sticky="w")
        self.cat_a = ttk.Combobox(form, width=22, values=pickable)
        self.cat_a.grid(row=2, column=1, sticky="e")

        ttk.Label(form, text="B — recall instruction (leg 2)").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.instr_b = ttk.Entry(form, width=42)
        self.instr_b.insert(0, "the ")
        self.instr_b.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 2))
        ttk.Label(form, text="B object category (scoring)").grid(row=5, column=0, sticky="w")
        self.cat_b = ttk.Combobox(form, width=22, values=pickable)
        self.cat_b.grid(row=5, column=1, sticky="e")

        self.instr_a.bind("<KeyRelease>", lambda e: self.autofill_cat(self.instr_a, self.cat_a))
        self.instr_b.bind("<KeyRelease>", lambda e: self.autofill_cat(self.instr_b, self.cat_b))

        opt = ttk.Frame(right)
        opt.pack(fill="x")
        self.no_avoid = tk.BooleanVar(value=True)
        self.reground = tk.BooleanVar(value=True)
        self.dry = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="no obstacle guard (sim: navmesh already blocks walls)",
                        variable=self.no_avoid).pack(anchor="w")
        ttk.Checkbutton(opt, text="re-ground every tick (instruction-period 0)",
                        variable=self.reground).pack(anchor="w")
        gc = ttk.Frame(opt)
        gc.pack(anchor="w")
        ttk.Label(gc, text="goal-consistency m (reject fresh grounding past this):").pack(side="left")
        self.gc_m = ttk.Entry(gc, width=5)
        self.gc_m.insert(0, "6.0")
        self.gc_m.pack(side="left")
        ttk.Checkbutton(opt, text="dry run (print command only)", variable=self.dry).pack(anchor="w")
        self.run_btn = ttk.Button(right, text="▶  Run A/B episode from this pose",
                                  command=self.run_episode)
        self.run_btn.pack(fill="x", pady=4)

        ttk.Label(right, text="run log").pack(anchor="w")
        self.log = tk.Text(right, height=12, width=52, font=("TkFixedFont", 8),
                           bg="#0c0c0c", fg="#c8c8c8", wrap="word")
        self.log.pack(fill="both", expand=True)

        for seq in ("<Up>", "<w>", "<W>"):
            root.bind(seq, lambda e: self.step(fwd=STEP_M))
        for seq in ("<Down>", "<s>", "<S>"):
            root.bind(seq, lambda e: self.step(fwd=-STEP_M))
        for seq in ("<Left>", "<a>", "<A>"):
            root.bind(seq, lambda e: self.step(turn=TURN_RAD))
        for seq in ("<Right>", "<d>", "<D>"):
            root.bind(seq, lambda e: self.step(turn=-TURN_RAD))
        root.bind("<q>", lambda e: self.step(strafe=STEP_M))
        root.bind("<e>", lambda e: self.step(strafe=-STEP_M))
        root.bind("<bracketleft>", lambda e: self.adj_cam(-0.1))
        root.bind("<bracketright>", lambda e: self.adj_cam(0.1))
        root.bind("<t>", lambda e: (self.scene.teleport_random(), self.refresh()))

        self.refresh()
        self.root.after(120, self.pump_log)

    # -- viewer --------------------------------------------------------- #
    def step(self, fwd=0.0, strafe=0.0, turn=0.0):
        if turn:
            self.scene.turn(turn)
        if fwd or strafe:
            self.scene.move(fwd, strafe)
        self.refresh()

    def adj_cam(self, d):
        self.scene.cam_height = max(0.3, min(2.2, self.scene.cam_height + d))
        self.refresh()

    def refresh(self):
        rgb = self.scene.frame()
        self._imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.configure(image=self._imgtk)
        x, z = self.scene.pos_xz
        self.pose_lbl.configure(
            text=(f"x={x:+.2f}  z={z:+.2f}  yaw={math.degrees(self.scene.yaw):+6.1f}°"
                  f"  floor_y={self.scene.floor_y:+.2f}  cam_h={self.scene.cam_height:.2f}"))
        self.draw_map()

    # -- top-down map ------------------------------------------------- #
    def _to_canvas(self, x, z):
        lox, loz, hix, hiz = self.scene.bounds
        m = 12
        cx = m + (x - lox) / max(hix - lox, 1e-6) * (self.MAP - 2 * m)
        cz = m + (z - loz) / max(hiz - loz, 1e-6) * (self.MAP - 2 * m)
        return cx, cz

    def _from_canvas(self, cx, cz):
        lox, loz, hix, hiz = self.scene.bounds
        m = 12
        x = lox + (cx - m) / (self.MAP - 2 * m) * (hix - lox)
        z = loz + (cz - m) / (self.MAP - 2 * m) * (hiz - loz)
        return x, z

    def draw_map(self):
        c = self.canvas
        c.delete("all")
        for cat, pts in self.scene.cat_pos.items():
            col = "#444" if cat in STRUCTURAL else "#3d8bfd"
            for px, pz in pts:
                cx, cz = self._to_canvas(px, pz)
                c.create_oval(cx - 2, cz - 2, cx + 2, cz + 2, fill=col, outline="")
        x, z = self.scene.pos_xz
        cx, cz = self._to_canvas(x, z)
        hx = cx + 11 * math.cos(self.scene.yaw)
        hz = cz + 11 * math.sin(self.scene.yaw)
        c.create_line(cx, cz, hx, hz, fill="#ff5252", width=2, arrow="last")
        c.create_oval(cx - 4, cz - 4, cx + 4, cz + 4, outline="#ff5252", width=2)

    def on_map_click(self, ev):
        x, z = self._from_canvas(ev.x, ev.y)
        cat, d = self.scene.nearest_category(x, z, max_d=2.0)
        if cat:
            self.clicked_cat = cat
            self.click_lbl.configure(text=f"clicked: {cat}  ({d:.2f} m)")
        else:
            self.clicked_cat = None
            self.click_lbl.configure(text="clicked: (nothing near)")

    def walk_to_clicked(self):
        if getattr(self, "clicked_cat", None):
            self.scene.face_category(self.clicked_cat)
            self.refresh()

    def set_cat(self, which):
        cat = getattr(self, "clicked_cat", None)
        if not cat:
            return
        box = self.cat_a if which == "a" else self.cat_b
        box.set(cat)
        entry = self.instr_a if which == "a" else self.instr_b
        cur = entry.get().strip()
        if cur in ("", "go to the", "the"):
            entry.delete(0, "end")
            entry.insert(0, (f"go to the {cat}" if which == "a" else f"the {cat}"))

    def on_cat_select(self, _ev):
        sel = self.cat_lb.curselection()
        if sel:
            self.clicked_cat = self.cats_sorted[sel[0]]
            self.click_lbl.configure(text=f"clicked: {self.clicked_cat}  "
                                          f"({self.scene.cat_count[self.clicked_cat]} inst)")

    def autofill_cat(self, entry: ttk.Entry, box: ttk.Combobox):
        if box.get().strip():
            return
        text = entry.get().lower()
        hits = [c for c in self.cats_sorted if c not in STRUCTURAL and c in text]
        if hits:
            box.set(max(hits, key=len))

    # -- run ---------------------------------------------------------- #
    def run_episode(self):
        if self.proc is not None:
            self._log("[a run is already in progress]\n")
            return
        instr_a = self.instr_a.get().strip()
        instr_b = self.instr_b.get().strip()
        cat_a = self.cat_a.get().strip()
        cat_b = self.cat_b.get().strip()
        if not (instr_a and instr_b and cat_a and cat_b):
            self._log("[need: A instruction, B instruction, and both categories]\n")
            return
        if cat_a not in self.scene.cat_count or cat_b not in self.scene.cat_count:
            self._log("[category not present in this scene -- pick from the list]\n")
            return
        x, z = self.scene.pos_xz
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"logs/ab_gui_{ts}"
        try:
            gc_m = float(self.gc_m.get())
        except ValueError:
            gc_m = 6.0
        cmd = [
            sys.executable, "-u", "run_ab_position_recall_test.py",
            "--scene-dataset", "hm3d", "--hm3d-root", self.args.hm3d_root,
            "--scene", self.args.scene,
            "--category-a", cat_a, "--instruction-a", instr_a,
            "--category-b", cat_b, "--query-b", instr_b,
            "--start-xy", f"{x:.3f}", f"{z:.3f}",
            "--min-ab", "1.2", "--max-ab", "9.0", "--min-clearance", "0.6",
            "--cam-height", f"{self.scene.cam_height:.2f}",
            "--goal-consistency-m", f"{gc_m:.2f}",
            "--no-qwen-appearance-lock", "--no-qwen-grounding-crops",
            "--stop-distance", "0.6", "--reached-radius", "0.8",
            "--max-steps-leg1", "260",
            "--device", self.args.device, "--out", outdir,
        ]
        if self.no_avoid.get():
            cmd.append("--no-avoid")
        if self.reground.get():
            cmd += ["--instruction-period-s", "0.0"]
        self.log.delete("1.0", "end")
        self._log("$ " + " ".join(
            (f'"{a}"' if " " in a else a) for a in cmd) + "\n\n")
        if self.dry.get():
            return
        self.run_btn.configure(state="disabled")
        self.proc = subprocess.Popen(cmd, cwd=str(Path(__file__).parent),
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        self._outdir = outdir
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen):
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_q.put(line)
        proc.wait()
        self.log_q.put(f"\n[process exited rc={proc.returncode}]\n")
        self.log_q.put("__DONE__")

    def pump_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                if line == "__DONE__":
                    self.proc = None
                    self.run_btn.configure(state="normal")
                    self._show_result()
                else:
                    self._log(line)
        except queue.Empty:
            pass
        self.root.after(150, self.pump_log)

    def _show_result(self):
        import json
        rp = Path(getattr(self, "_outdir", "")) / "result.json"
        if not rp.is_file():
            self._log("[no result.json written]\n")
            return
        d = json.loads(rp.read_text())
        g = d.get("goat_metrics", {})
        l1, l2 = d.get("leg1", {}), d.get("leg2") or {}
        rc = d.get("recall") or {}
        self._log(
            "\n==== RESULT ====\n"
            f"leg1  {l1.get('outcome')}  err_to_A={l1.get('error_to_nearest_instance_m', -1):.2f} m  "
            f"success={l1.get('success')}\n"
            f"recall  err_vs_true_B={rc.get('position_error_vs_true_B_m', float('nan')):.2f} m  "
            f"spread={rc.get('position_spread_m', float('nan')):.2f} m\n"
            f"leg2  {l2.get('outcome')}  err_to_recalled={l2.get('error_to_recalled_m', float('nan')):.2f} m  "
            f"err_to_B={l2.get('error_to_nearest_instance_m', float('nan')):.2f} m  success={l2.get('success')}\n"
            f"SR={g.get('success_rate')}   mean_dist={g.get('mean_distance_to_goal_m', float('nan')):.2f} m\n"
            f"dir: {self._outdir}\n")

    def _log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT)
    ap.add_argument("--scene", default="wcojb4TFT35")
    ap.add_argument("--cam-height", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
