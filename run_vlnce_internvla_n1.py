#!/usr/bin/env python3
"""One real R2R-CE (VLN-CE) episode, driven by the native InternVLA-N1-DualVLN
dual-system model, against ANY Matterport3D scan -- continuous control
throughout, scored with the actual VLN-CE metrics (NE / SR / OSR / SPL).

Sibling of ``run_vlnce_episode_test.py`` (Qwen pixel-goal + NavDP). Same
episode loader, same navmesh-baking ``build_sim``, same dead-reckon +
navmesh-snap kinematics, same metrics -- the ONLY thing swapped is the
brain: instead of this repo's Qwen+NavDP pipeline, every decision comes from
the real InternVLA-N1-DualVLN checkpoint ("Ground Slow, Move Fast", arXiv
2512.08186), run natively as its own dual system:

    RGB(+depth) + instruction --System-2 (QwenVL-2.5-7B, ~2Hz)--> either a
    discrete VLN-CE token {STOP, forward, turn-left, turn-right, look-down}
    OR a pixel goal + latent goal --System-1 (Diffusion Transformer, every
    tick)--> a body-frame local trajectory --steer to last waypoint-->
    (v, w) --dead-reckon (navmesh-snapped)--> new pose, repeat.

Two-process split (habitat-sim and internnav can't share one conda env here):
  * MODEL   -- `internvla` env, InternNav repo root:
        conda activate internvla
        cd /home/gpu/Desktop/pineapple/InternNav
        python internvla_n1_server.py            # resident, TCP :8791
  * THIS    -- `habitat` env (has habitat-sim 0.3.3):
        conda activate habitat
        cd qwen3vl-2b-navdp
        python run_vlnce_internvla_n1.py --scene <mp3d_scan_id> --split val_unseen

Pass ``--autostart-server`` to have this script spawn/stop the model server
itself (slower: reloads the ~17GB checkpoint each run -- prefer a resident
server for repeated episodes).

"Any scan id": raw Matterport3D v1 research-release scans live at
``--mp3d-raw-root`` (default the Extreme SSD). The first time a scan is used,
its ``house_segmentations/<id>.ply`` is extracted and rotated Z-up -> Y-up
(the exact fix documented in the fact3r-navdp-bridge notes -- raw MP3D is
Z-up, habitat / R2R-CE episodes are Y-up) and cached under
``--mesh-cache-root``. The trimesh rotation runs in the `navdp` env (the
only one here with trimesh). A navmesh is baked at load time by
``build_sim`` (no ``.navmesh`` ships in the raw dump).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import math
import socket
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
except ImportError as exc:  # pragma: no cover
    raise SystemExit("habitat_sim is required -- run this under the `habitat` conda env") from exc

W, H, FOV_DEG = 640, 480, 90.0
EYE_HEIGHT_M = 1.2
SUCCESS_DISTANCE_M = 3.0  # VLN-CE's own convention

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8792  # internvla_n1_server_vlnce.py (discretised-trajectory + S2-sampling variant)
_HEADER_SIZE = 4

# InternVLA-N1's own discrete-action convention (internnav/model/utils/
# vln_utils.py trajectory_to_discrete_actions_close_to_goal defaults) -- the
# model was trained against this action space, these units aren't a guess.
DISCRETE_FORWARD_M = 0.25
DISCRETE_TURN_DEG = 15.0
DISCRETE_SUBSTEPS = 5  # subdivide one discrete action for a smooth (non-teleported) trace

ACTIONS2IDX_INV = {0: "STOP", 1: "forward", 2: "left", 3: "right", 5: "look_down"}

NAVDP_ENV_PYTHON = "/home/gpu/miniconda3/envs/navdp/bin/python"
INTERNNAV_REPO = "/home/gpu/Desktop/pineapple/InternNav"


# ------------------------------------------------------------------ geometry --
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def habitat_yaw_from_quat(rot_xyzw) -> float:
    x, y, z, w = rot_xyzw
    return 2.0 * math.atan2(y, w)  # pure yaw-about-Y rotations, as R2R-CE start poses are


import re as _re


def split_clauses(instruction: str) -> list[str]:
    """Break a compound R2R-CE instruction into ordered sub-goals. R2R-CE
    phrasing is sentences ("Walk through the living room. Turn right. Stop by
    the door.") and comma lists ("walk past living room, walk past dining
    room, turn right, wait by gold room"), not "then" -- so split on sentence
    ends first, and on commas / "and then" only if that leaves a single very
    long clause. Merges a trailing bare "turn X" / "wait/stop ..." fragment
    onto the previous clause so a sub-goal always carries a place to go."""
    s = instruction.strip()
    parts = [p.strip(" .,") for p in _re.split(r"(?<=[.!?])\s+", s) if p.strip(" .,")]
    if len(parts) < 2:
        parts = [p.strip(" .,") for p in _re.split(r",\s*(?:and\s+then\s+|then\s+|and\s+)?|;\s*", s) if p.strip(" .,")]
    merged: list[str] = []
    for p in parts:
        bare = len(p.split()) <= 3 and _re.match(r"^(turn|wait|stop|continue|keep going)\b", p, _re.I)
        if bare and merged:
            merged[-1] = merged[-1] + ", " + p
        else:
            merged.append(p)
    return merged or [instruction]


# -------------------------------------------------------- raw-scan -> Y-up ply --
def ensure_yup_mesh(scan_id: str, raw_root: Path, cache_root: Path) -> Path:
    """Return a habitat-loadable Y-up mesh for ``scan_id``, converting the raw
    Matterport3D v1 dump on first use. Cached, so this is a no-op after the
    first episode in a given house.

    Recipe (fact3r-navdp-bridge notes): use the vertex-coloured
    ``house_segmentations/<id>.ply`` (the textured ``matterport_mesh`` obj
    segfaults habitat-sim with two sensors), rotate its vertices
    (x, y, z) -> (x, z, -y) so height moves from Z to Y.
    """
    out_dir = cache_root / scan_id
    yup = out_dir / f"{scan_id}_yup.ply"
    if yup.is_file():
        return yup

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_ply = out_dir / f"{scan_id}.ply"
    if not raw_ply.is_file():
        zip_path = raw_root / scan_id / "house_segmentations.zip"
        if not zip_path.is_file():
            raise SystemExit(
                f"no converted mesh and no raw dump for {scan_id!r}\n"
                f"  looked for cached : {yup}\n"
                f"  looked for raw zip: {zip_path}\n"
                f"pass --mp3d-raw-root / --mesh-cache-root, or pre-convert the mesh."
            )
        print(f"[convert] extracting {scan_id}.ply from {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            member = next(m for m in zf.namelist() if m.endswith(f"/house_segmentations/{scan_id}.ply"))
            with zf.open(member) as src, open(raw_ply, "wb") as dst:
                dst.write(src.read())
            # grab the .house too (handy for later semantic work; not needed here)
            try:
                hmember = next(m for m in zf.namelist() if m.endswith(f"/{scan_id}.house"))
                with zf.open(hmember) as src, open(out_dir / f"{scan_id}.house", "wb") as dst:
                    dst.write(src.read())
            except StopIteration:
                pass

    print(f"[convert] rotating {scan_id}.ply Z-up -> Y-up via the navdp env's trimesh ...")
    code = (
        "import sys, numpy as np, trimesh\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "m = trimesh.load(src, process=False)\n"
        "R = np.array([[1,0,0],[0,0,1],[0,-1,0]])\n"
        "m.vertices = m.vertices @ R.T\n"
        "m.export(dst)\n"
        "print('wrote', dst, 'verts', len(m.vertices))\n"
    )
    py = NAVDP_ENV_PYTHON if Path(NAVDP_ENV_PYTHON).is_file() else "python"
    subprocess.run([py, "-c", code, str(raw_ply), str(yup)], check=True)
    if not yup.is_file():
        raise SystemExit(f"trimesh rotation did not produce {yup}")
    return yup


# ------------------------------------------------------------- episode loading --
def _split_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / f"{split}.json.gz"


def episodes_for_scene(dataset_root: Path, split: str, scene: str) -> list[dict]:
    data = json.load(gzip.open(_split_path(dataset_root, split), "rt"))
    return [ep for ep in data["episodes"] if scene in ep["scene_id"]]


def load_episode(dataset_root: Path, split: str, scene: str,
                 episode_id: str | None, episode_index: int) -> dict:
    eps = episodes_for_scene(dataset_root, split, scene)
    if not eps:
        raise SystemExit(f"no {split} episodes reference scene {scene!r} in {_split_path(dataset_root, split)}")
    if episode_id is not None:
        for ep in eps:
            if str(ep["episode_id"]) == str(episode_id):
                return ep
        raise SystemExit(f"episode {episode_id} for scene {scene} not in {split} "
                         f"(available: {[e['episode_id'] for e in eps][:20]}{' ...' if len(eps) > 20 else ''})")
    return eps[episode_index]


# --------------------------------------------------------------- habitat sim --
def build_sim(scene_mesh: str, hfov_deg: float = FOV_DEG):
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [H, W]
    rgb_spec.position = np.array([0.0, 0.0, 0.0])  # camera AT agent y (cam_y already = floor + EYE_HEIGHT_M); do NOT double-add
    rgb_spec.hfov = hfov_deg

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [H, W]
    depth_spec.position = np.array([0.0, 0.0, 0.0])  # camera AT agent y (cam_y already = floor + EYE_HEIGHT_M); do NOT double-add
    depth_spec.hfov = hfov_deg

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_mesh
    sim_cfg.enable_physics = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    if not sim.pathfinder.is_loaded:
        navmesh = str(Path(scene_mesh).with_suffix(".navmesh"))
        if Path(navmesh).is_file():
            sim.pathfinder.load_nav_mesh(navmesh)
    if not sim.pathfinder.is_loaded:
        # Raw MP3D v1 research-release mesh -- no shipped .navmesh, bake one.
        # Bare defaults only (agent_height overridden to the render eye
        # height); include_static_objects / custom agent_radius segfault on
        # these meshes (fact3r-navdp-bridge notes).
        settings = habitat_sim.nav.NavMeshSettings()
        settings.set_defaults()
        settings.agent_height = EYE_HEIGHT_M
        ok = sim.recompute_navmesh(sim.pathfinder, settings)
        print(f"[note] no shipped navmesh for {scene_mesh} -- baked one at runtime "
              f"(recompute_navmesh ok={ok})")
    return sim


def _agent_rotation(pose_yaw: float, pitch_deg: float = 0.0):
    habitat_yaw = -(pose_yaw + math.pi / 2.0)  # same convention as run_vlnce_episode_test.py
    rot = quat_from_angle_axis(habitat_yaw, np.array([0.0, 1.0, 0.0]))
    if pitch_deg:
        rot = rot * quat_from_angle_axis(math.radians(pitch_deg), np.array([1.0, 0.0, 0.0]))
    return rot


def render(sim, pose, cam_y: float, pitch_deg: float = 0.0):
    X, Yp, yaw = pose
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([X, cam_y, Yp], dtype=np.float32)
    state.rotation = _agent_rotation(yaw, pitch_deg)
    agent.set_state(state)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[:, :, :3].copy()
    depth = np.asarray(obs["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return rgb, depth


# ----------------------------------------------------- stuck-recovery override --
class StuckRecovery:
    """Detects a stalled local minimum and overrides the commanded (v, w).
    Ported verbatim (same defaults) from run_vlnce_episode_test.py -- see its
    docstring: navmesh-snap silently eats a step budget as ~0 realized motion
    while the commanded v looks normal; this rotates a fixed sweep then forces
    a few ticks of forward motion to actually relocate."""

    def __init__(self, window: int = 30, min_progress_m: float = 0.15,
                 turn_ticks: int = 12, push_ticks: int = 15,
                 escape_w: float = 0.8, push_v: float = 0.15):
        self.window = window
        self.min_progress_m = min_progress_m
        self.turn_ticks = turn_ticks
        self.push_ticks = push_ticks
        self.escape_w = escape_w
        self.push_v = push_v
        self._history: list[tuple[float, float]] = []
        self._phase = None
        self._phase_left = 0
        self._escape_dir = 1.0
        self._escape_start_xy: Optional[tuple[float, float]] = None
        self.trigger_count = 0
        self.escalation = 0
        self.just_triggered = False

    def override(self, pose, v: float, w: float) -> tuple[float, float, bool]:
        self.just_triggered = False
        self._history.append((pose[0], pose[1]))
        if len(self._history) > self.window:
            self._history.pop(0)

        if self._phase == "turn":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase, self._phase_left = "push", self.push_ticks
            return 0.0, self.escape_w * self._escape_dir, True

        if self._phase == "push":
            self._phase_left -= 1
            if self._phase_left <= 0:
                self._phase = None
                if self._escape_start_xy is not None:
                    moved = float(np.hypot(pose[0] - self._escape_start_xy[0],
                                           pose[1] - self._escape_start_xy[1]))
                    self.escalation = 0 if moved > self.min_progress_m else self.escalation + 1
                self._history.clear()
            return self.push_v, 0.0, True

        if len(self._history) == self.window:
            p0 = np.array(self._history[0])
            p1 = np.array(self._history[-1])
            if float(np.linalg.norm(p1 - p0)) < self.min_progress_m:
                self._escape_dir *= -1.0
                self._escape_start_xy = (pose[0], pose[1])
                turn_ticks = self.turn_ticks * (1 + min(self.escalation, 3))
                self._phase, self._phase_left = "turn", turn_ticks
                self.trigger_count += 1
                self.just_triggered = True
                return 0.0, self.escape_w * self._escape_dir, True

        return v, w, False


# ---------------------------------------------------------- N1 server client --
class N1Client:
    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT, timeout: float = 120.0):
        self.host, self.port, self.timeout = host, port, timeout

    def _request(self, mode: str, payload: dict) -> dict:
        msg = json.dumps({"mode": mode, "payload": payload}).encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as conn:
            conn.sendall(struct.pack(">I", len(msg)) + msg)
            (body_len,) = struct.unpack(">I", _recv_exact(conn, _HEADER_SIZE))
            body = _recv_exact(conn, body_len)
        resp = json.loads(body.decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(f"internvla_n1_server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._request("ping", {})
            return True
        except OSError:
            return False

    def reset(self, seed: int | None = None) -> None:
        self._request("reset", {"seed": seed} if seed is not None else {})

    def step(self, rgb: np.ndarray, depth: np.ndarray, pose, cam_y: float,
             instruction: str, look_down: bool = False,
             rgb_ld: np.ndarray | None = None, depth_ld: np.ndarray | None = None,
             bon: bool = True) -> dict:
        # Sanitise depth to what InternVLA-N1 expects (metres, finite, clamped
        # -- habitat_vln_evaluator.py filters + clamps identically).
        def _clean(dp):
            dp = np.nan_to_num(np.asarray(dp, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(dp, 0.0, 10.0)
        depth = _clean(depth)
        payload = {
            "rgb_b64": _encode_rgb(rgb),
            "depth_b64": _encode_depth(depth),
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
            "pose": _pose_to_4x4(pose, cam_y),
            "instruction": instruction,
            "look_down": look_down,
            "bon": bool(bon),
        }
        if rgb_ld is not None:
            depth_ld = _clean(depth_ld)
            payload.update({
                "rgb_ld_b64": _encode_rgb(rgb_ld),
                "depth_ld_b64": _encode_depth(depth_ld),
                "ld_height": int(depth_ld.shape[0]),
                "ld_width": int(depth_ld.shape[1]),
            })
        return self._request("step", payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks, remaining = [], n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before expected bytes were received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_rgb(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_depth(depth: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(depth, dtype=np.float32).tobytes()).decode("ascii")


def _pose_to_4x4(pose, cam_y: float) -> list:
    # Bookkeeping only -- internvla_n1_agent_realworld never reads `pose`.
    # Kept consistent with run_internvla_n1_compare.py's convention.
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return [[c, 0.0, s, float(pose[0])],
            [0.0, 1.0, 0.0, float(cam_y)],
            [-s, 0.0, c, float(pose[1])],
            [0.0, 0.0, 0.0, 1.0]]


def start_server() -> subprocess.Popen:
    print("[server] --autostart-server: launching internvla_n1_server.py in the `internvla` env "
          "(loads the ~17GB checkpoint, be patient) ...")
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", "internvla", "python", "internvla_n1_server.py"],
        cwd=INTERNNAV_REPO,
    )
    client = N1Client(timeout=10.0)
    for _ in range(600):  # up to ~10 min for a cold checkpoint load
        if proc.poll() is not None:
            raise SystemExit(f"internvla_n1_server exited early (code {proc.returncode})")
        if client.ping():
            print("[server] up.")
            return proc
        time.sleep(1.0)
    proc.terminate()
    raise SystemExit("internvla_n1_server did not become reachable within 10 min")


# -------------------------------------------------------------------- driving --
def integrate(pose, pf, cam_y: float, v: float, w: float, dt: float):
    """One dead-reckon sub-step with navmesh snapping (same as run_vlnce_episode_test.py)."""
    nx = pose[0] + v * math.cos(pose[2]) * dt
    ny = pose[1] + v * math.sin(pose[2]) * dt
    if pf.is_loaded:
        snapped = np.array(pf.snap_point(np.array([nx, cam_y, ny])))
        pose[0], pose[1] = float(snapped[0]), float(snapped[2])
    else:
        pose[0], pose[1] = nx, ny
    pose[2] = wrap(pose[2] + w * dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="Matterport3D scan id, e.g. 2azQ1b91cZZ")
    ap.add_argument("--split", default="val_unseen", choices=("val_unseen", "val_seen", "train", "test"))
    ap.add_argument("--episode-id", default=None, help="specific R2R-CE episode_id for this scene")
    ap.add_argument("--episode-index", type=int, default=0,
                    help="0-based index into this scene's episodes for the split (used when "
                         "--episode-id is not given)")
    ap.add_argument("--list-episodes", action="store_true",
                    help="print every R2R-CE episode_id + instruction for --scene/--split and exit")
    ap.add_argument("--convert-only", action="store_true",
                    help="just produce the Y-up mesh for --scene and exit (no model, no sim run)")

    ap.add_argument("--vlnce-root", default="/home/gpu/Desktop/pineapple/mp3d_data/datasets/R2R_VLNCE_v1-3",
                    help="dir holding <split>/<split>.json.gz")
    ap.add_argument("--mp3d-raw-root", default="/media/gpu/Extreme SSD/matterport3d/v1/scans",
                    help="raw Matterport3D v1 research-release scans (<id>/house_segmentations.zip)")
    ap.add_argument("--mesh-cache-root", default="/home/gpu/Desktop/Nihil/MASt3R-SLAM/datasets/mp3d",
                    help="where converted <id>/<id>_yup.ply are cached")
    ap.add_argument("--scene-mesh", default=None,
                    help="explicit Y-up mesh path -- skip the raw-dump conversion entirely")

    ap.add_argument("--instruction", default=None, help="override the episode's own instruction text")
    ap.add_argument("--start-yaw-override-deg", type=float, default=None,
                    help="use this world-frame heading instead of the episode's start_rotation "
                         "(R2R-CE start headings are occasionally degenerate)")

    ap.add_argument("--autostart-server", action="store_true",
                    help="spawn + stop internvla_n1_server.py automatically (reloads the "
                         "checkpoint each run -- prefer a resident server for repeated episodes)")
    ap.add_argument("--server-host", default=SERVER_HOST)
    ap.add_argument("--server-port", type=int, default=SERVER_PORT)

    ap.add_argument("--hfov", type=float, default=FOV_DEG,
                    help="horizontal FOV of the RGB+depth sensors (deg). InternVLA-N1's VLN "
                         "training/eval (InternNav scripts/eval/configs/vln_r2r.yaml) uses 79.")
    ap.add_argument("--s1-lookdown-deg", type=float, default=30.0,
                    help="render a separate frame pitched this many deg DOWN and feed it to "
                         "System-1 (the DiT trajectory generator) -- InternNav's habitat "
                         "evaluator does LOOKDOWN x2 (=30deg) every step for exactly this. "
                         "0 = feed S1 the same forward frame as S2 (old behaviour).")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0,
                    help="base camera pitch, NEGATIVE = tilt DOWN. InternNav's habitat_vln_evaluator "
                         "renders with a 30 deg downward tilt (pass -30); the harness default 0 "
                         "(level) is what A-E used and is a likely cause of the model never "
                         "committing to a forward step.")
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--plan-step-gap", type=int, default=4,
                    help="sub-steps to hold one System-1 trajectory command (matches the model's "
                         "own re-plan cadence, internvla_n1_server Args.plan_step_gap)")
    ap.add_argument("--max-linear", type=float, default=0.5, help="clip on |trajectory| -> linear vel")
    ap.add_argument("--max-angular", type=float, default=0.5, help="clip on waypoint bearing -> angular vel")
    ap.add_argument("--max-decisions", type=int, default=250,
                    help="server decisions (S2/S1 queries) before the episode is called at max_steps")
    ap.add_argument("--max-traj-actions", type=int, default=16,
                    help="cap on discrete steps executed from one System-1 trajectory before "
                         "re-querying (the trajectory->action list can be 30+ long; InternNav's "
                         "own evaluator caps it similarly with MAX_LOCAL_STEPS)")
    ap.add_argument("--frozen-window", type=int, default=14,
                    help="frozen-escape: if net displacement over this many decisions stays under "
                         "--frozen-move-m, force a turn+push and reset model history")
    ap.add_argument("--frozen-move-m", type=float, default=0.5,
                    help="frozen-escape displacement threshold (m) over --frozen-window decisions")
    ap.add_argument("--frozen-push-m", type=float, default=1.0,
                    help="frozen-escape forward push distance (m) after the escape turn")
    ap.add_argument("--max-consecutive-turns", type=int, default=0,
                    help="after this many turn-only decisions in a row, force one 0.25 m forward "
                         "step (the model will otherwise reorient forever without committing). "
                         "0 disables. Try 6-8.")
    ap.add_argument("--turn-creep-m", type=float, default=0.0,
                    help="small forced forward creep on a turn-only decision (m). A perfectly "
                         "stationary sim turn re-renders an identical frame; feeding identical "
                         "frames into the model's rolling history is what locks the L/R flip-flop. "
                         "Try 0.06-0.1.")
    ap.add_argument("--sequence-clauses", action="store_true",
                    help="split a compound instruction into ordered sub-goals and drive them one "
                         "at a time (advance on a stall / STOP / --max-clause-decisions). System-2 "
                         "has no 'which landmark am I on' state, so feeding a long multi-room "
                         "instruction whole makes it fixate on the first landmark -- this is the "
                         "fix for the geo>=8m multi-room episodes.")
    ap.add_argument("--clause-advance-k", type=int, default=3,
                    help="advance to the next clause after this many consecutive no-forward-motion "
                         "decisions (the model has done what the current clause asks)")
    ap.add_argument("--max-clause-decisions", type=int, default=45,
                    help="hard cap on decisions per clause before force-advancing")
    ap.add_argument("--arrival-noop-k", type=int, default=3,
                    help="declare arrival (STOP) after this many consecutive no-forward-motion "
                         "decisions WHILE within --arrival-radius-m of the goal -- the model's "
                         "implicit 'I'm at my predicted goal' signal (it rarely emits STOP). "
                         "0 disables.")
    ap.add_argument("--arrival-radius-m", type=float, default=4.0,
                    help="the noop-streak arrival heuristic only accrues within this distance of "
                         "the goal (outside it, no-motion = reorientation/stall, not arrival)")
    ap.add_argument("--seed", type=int, default=-1,
                    help="diffusion seed for the episode (server torch.manual_seed). -1 = use the "
                         "episode_id, so reruns are reproducible and config changes are measurable.")
    ap.add_argument("--best-of-n", choices=("auto", "on", "off"), default="auto",
                    help="System-1 trajectory selection: 'off' = mean of the 32 diffusion samples "
                         "(smooth, stops well, can't navigate multi-room); 'on' = pick the "
                         "forward-most heading-consistent sample (navigates multi-room, overshoots "
                         "short paths); 'auto' = on iff optimal geodesic >= --best-of-n-geo-m.")
    ap.add_argument("--best-of-n-geo-m", type=float, default=6.5,
                    help="'auto' best-of-N threshold on the episode's optimal geodesic length")
    ap.add_argument("--max-path-mult", type=float, default=0.0,
                    help="stop (take final position) once travelled path exceeds this multiple of "
                         "the optimal geodesic -- backstop against best-of-N overshoot/wander on "
                         "short episodes. 0 disables.")
    ap.add_argument("--dwell-k", type=int, default=5,
                    help="STOP after this many consecutive decisions where System-1's trajectory "
                         "collapsed to <=2 discrete steps ('basically at my predicted goal') -- "
                         "catches the 'reached the goal then wandered off' failure. 0 disables.")
    ap.add_argument("--oracle-stop", action="store_true",
                    help="stop the episode the moment NE < success radius -- the VLN-CE "
                         "'oracle stop' convention; reports the SR/SPL the navigation would get "
                         "with a perfect arrival detector (upper bound, not leaderboard-legal)")
    ap.add_argument("--ignore-stop", action="store_true",
                    help="don't end on a self-declared STOP -- reset the agent and keep driving "
                         "(oracle_navigation_error_m still reports true closest approach)")
    ap.add_argument("--no-realtime-pacing", dest="realtime_pacing", action="store_false",
                    help="don't sleep to wall-clock dt between decisions")
    ap.add_argument("--success-distance", type=float, default=SUCCESS_DISTANCE_M,
                    help="NE below this counts as success. Default 3.0 = VLN-CE convention; lower (e.g. 0.3) to require physically reaching the object.")
    ap.add_argument("--out", default=None,
                    help="output dir (default logs/vlnce_internvla_n1_<scene>_ep<id>)")
    args = ap.parse_args()

    vlnce_root = Path(args.vlnce_root)

    if args.list_episodes:
        eps = episodes_for_scene(vlnce_root, args.split, args.scene)
        print(f"{len(eps)} {args.split} episode(s) for scene {args.scene}:")
        for ep in eps:
            txt = ep["instruction"]["instruction_text"].strip().replace("\n", " ")
            print(f"  ep {str(ep['episode_id']):>6}  geo={float(ep['info']['geodesic_distance']):5.1f}m  {txt}")
        return

    # ---- mesh (convert on first use) ----
    if args.scene_mesh:
        scene_mesh = args.scene_mesh
    else:
        scene_mesh = str(ensure_yup_mesh(args.scene, Path(args.mp3d_raw_root), Path(args.mesh_cache_root)))
    print(f"scene mesh: {scene_mesh}")
    if args.convert_only:
        print("--convert-only: done.")
        return

    # ---- episode ----
    episode = load_episode(vlnce_root, args.split, args.scene, args.episode_id, args.episode_index)
    instruction = episode["instruction"]["instruction_text"].strip()
    if args.instruction:
        print(f"[note] overriding episode instruction {instruction!r} with {args.instruction!r}")
        instruction = args.instruction
    optimal_length = float(episode["info"]["geodesic_distance"])
    goal_pos = episode["goals"][0]["position"]
    goal_radius = float(args.success_distance)
    start_pos = episode["start_position"]
    if args.start_yaw_override_deg is not None:
        start_yaw = math.radians(args.start_yaw_override_deg)
        print(f"[note] overriding start heading with {args.start_yaw_override_deg:.0f} deg")
    else:
        start_yaw = -habitat_yaw_from_quat(episode["start_rotation"]) - math.pi / 2.0

    out_dir = Path(args.out) if args.out else Path(f"logs/vlnce_internvla_n1_{args.scene}_ep{episode['episode_id']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Episode {episode['episode_id']} (scene {args.scene}, split {args.split})")
    print(f'Instruction: "{instruction}"')
    print(f"Optimal (geodesic) length: {optimal_length:.2f} m   success radius: {goal_radius:.1f} m")

    # ---- model server ----
    server_proc = start_server() if args.autostart_server else None
    client = N1Client(args.server_host, args.server_port)
    if not client.ping():
        raise SystemExit(
            "internvla_n1_server not reachable on "
            f"{args.server_host}:{args.server_port}.\nStart it first:\n"
            "  conda activate internvla && cd /home/gpu/Desktop/pineapple/InternNav && "
            "python internvla_n1_server.py\n(or pass --autostart-server)"
        )
    seed = args.seed if args.seed >= 0 else int(episode["episode_id"])
    client.reset(seed=seed)
    bon = (args.best_of_n == "on" or
           (args.best_of_n == "auto" and optimal_length >= args.best_of_n_geo_m))
    print(f"[seed] episode diffusion seed = {seed}   best-of-N = {bon} "
          f"({args.best_of_n}, geo {optimal_length:.1f} vs {args.best_of_n_geo_m})")

    # ---- sim ----
    sim = build_sim(scene_mesh, args.hfov)
    print(f"navmesh loaded: {sim.pathfinder.is_loaded}  (hfov={args.hfov} cam_pitch={args.cam_pitch_deg} deg)")
    pf = sim.pathfinder

    pose = np.array([float(start_pos[0]), float(start_pos[2]), wrap(start_yaw)])
    cam_y = float(start_pos[1]) + EYE_HEIGHT_M
    goal_xy = (float(goal_pos[0]), float(goal_pos[2]))

    path_xy = [(pose[0], pose[1])]
    log: list[dict] = []
    min_dist = dist((pose[0], pose[1]), goal_xy)
    outcome = "max_steps"
    stuck = StuckRecovery()
    stop_count = 0
    action_counts: dict[str, int] = {}
    traj_decisions = 0

    def record(step_i: int, state: str, v: float, w: float, escaping: bool, extra: dict | None = None):
        d = dist((pose[0], pose[1]), goal_xy)
        rec = {"step": step_i, "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
               "state": state, "v": float(v), "w": float(w), "dist_to_goal": d, "escaping": escaping}
        if extra:
            rec.update(extra)
        log.append(rec)

    recent_xy: list[tuple[float, float]] = []   # (x,y) per decision, for the frozen-escape
    escape_dir = 1.0
    escapes = 0
    noop_streak = 0
    settle_streak = 0   # consecutive decisions where S1's trajectory collapsed (near its own goal)
    turn_run = 0        # consecutive turn-only decisions
    forced_forwards = 0

    clauses = split_clauses(instruction) if args.sequence_clauses else [instruction]
    clause_idx = 0
    clause_dec = 0
    clause_noop = 0
    clause_moved = False   # has THIS clause driven at least one forward step yet?
    clause_advances = 0
    if args.sequence_clauses:
        print(f"[clauses] {len(clauses)} sub-goal(s):")
        for k, c in enumerate(clauses):
            print(f"   {k}: {c}")

    def advance_clause(reason: str) -> bool:
        """Move to the next sub-goal. Returns True if there was one (episode
        continues), False if this was the last clause (caller ends normally)."""
        nonlocal clause_idx, clause_dec, clause_noop, clause_advances, clause_moved
        if clause_idx >= len(clauses) - 1:
            return False
        clause_idx += 1
        clause_advances += 1
        clause_dec = 0
        clause_noop = 0
        clause_moved = False
        print(f"  [dec {i:3d}] [clause {clause_idx}/{len(clauses) - 1}] ({reason}) -> "
              f"\"{clauses[clause_idx][:60]}\"")
        client.reset()
        recent_xy.clear()
        return True

    def apply_tokens(tok_list: list[int], tag: str, pixel) -> int:
        """Integrate a sequence of 1/forward-0.25m, 2/left-15deg, 3/right-15deg
        steps (navmesh-snapped, sub-stepped for a smooth trace). Returns the
        net turn sign (#left - #right, clamped to -1/0/+1)."""
        nonlocal min_dist
        nL = nR = 0
        n_fwd = 0
        for tok in tok_list:
            if tok == 1:
                n_fwd += 1
                for _ in range(DISCRETE_SUBSTEPS):
                    integrate(pose, pf, cam_y, DISCRETE_FORWARD_M / (DISCRETE_SUBSTEPS * args.dt), 0.0, args.dt)
                    min_dist = min(min_dist, dist((pose[0], pose[1]), goal_xy))
                action_counts["forward"] = action_counts.get("forward", 0) + 1
            elif tok in (2, 3):
                sign = 1.0 if tok == 2 else -1.0
                nL += tok == 2
                nR += tok == 3
                for _ in range(DISCRETE_SUBSTEPS):
                    integrate(pose, pf, cam_y, 0.0,
                              sign * math.radians(DISCRETE_TURN_DEG) / (DISCRETE_SUBSTEPS * args.dt), args.dt)
                    min_dist = min(min_dist, dist((pose[0], pose[1]), goal_xy))
                action_counts["left" if tok == 2 else "right"] = \
                    action_counts.get("left" if tok == 2 else "right", 0) + 1
            path_xy.append((pose[0], pose[1]))
        # turn-creep: a pure-rotation decision on a real robot still nudges the
        # camera position -- in sim a perfectly stationary turn re-renders a
        # byte-identical frame, and feeding many identical frames into the
        # model's rolling history is what locks it into the left/right
        # flip-flop. A small forced forward creep on turn-only decisions keeps
        # every viewpoint genuinely new. Opt-in (--turn-creep-m).
        if args.turn_creep_m > 0 and n_fwd == 0 and (nL + nR) > 0:
            for _ in range(DISCRETE_SUBSTEPS):
                integrate(pose, pf, cam_y, args.turn_creep_m / (DISCRETE_SUBSTEPS * args.dt), 0.0, args.dt)
                min_dist = min(min_dist, dist((pose[0], pose[1]), goal_xy))
            path_xy.append((pose[0], pose[1]))
        record(i, tag, 0.0, 0.0, False, {"pixel": pixel, "n_tokens": len(tok_list)})
        return (nL > nR) - (nL < nR)

    print(f'\n=== Driving (InternVLA-N1 dual-system): "{instruction[:70]}'
          f'{"..." if len(instruction) > 70 else ""}" ===')

    for i in range(args.max_decisions):
        tick_start = time.time()
        active_instr = clauses[clause_idx]
        clause_dec += 1
        rgb, depth = render(sim, pose, cam_y, pitch_deg=args.cam_pitch_deg)
        if args.s1_lookdown_deg > 0:
            rgb_ld, depth_ld = render(sim, pose, cam_y, pitch_deg=args.cam_pitch_deg - args.s1_lookdown_deg)
        else:
            rgb_ld = depth_ld = None
        resp = client.step(rgb, depth, pose, cam_y, active_instr, rgb_ld=rgb_ld, depth_ld=depth_ld, bon=bon)

        # ---- look-down sub-flow: pitch the camera, re-query with look_down ----
        ld_retries = 0
        while (resp.get("output_action") and list(resp["output_action"]) == [5] and ld_retries < 3):
            ld_rgb, ld_depth = render(sim, pose, cam_y, pitch_deg=args.cam_pitch_deg - 40.0)
            resp = client.step(ld_rgb, ld_depth, pose, cam_y, active_instr, look_down=True,
                               rgb_ld=rgb_ld, depth_ld=depth_ld, bon=bon)
            ld_retries += 1

        d = dist((pose[0], pose[1]), goal_xy)
        min_dist = min(min_dist, d)

        # ---- path cap: best-of-N keeps the robot moving, which is what fixes
        #      the multi-room episodes -- but on a short episode with nowhere
        #      left to go it overshoots and wanders. If the travelled path
        #      exceeds a generous multiple of the optimal geodesic, the run is
        #      lost; stop and take the final position.
        if (args.max_path_mult > 0 and clause_idx == len(clauses) - 1
                and path_length(path_xy) > args.max_path_mult * optimal_length):
            record(i, "PATH_CAP", 0.0, 0.0, False, {"path_m": path_length(path_xy)})
            outcome = "path_cap"
            break

        out_action = resp.get("output_action")
        out_traj = resp.get("output_trajectory")
        out_traj_actions = resp.get("output_traj_actions")
        out_pixel = resp.get("output_pixel")

        net_turn = None
        made_progress = False  # a real forward-carrying decision this tick?

        if out_action:
            tokens = [t for t in out_action if t != 5]
            names = [ACTIONS2IDX_INV.get(t, str(t)) for t in tokens]
            if i % 10 == 0 or 0 in tokens:
                print(f"  [dec {i:3d}] discrete {names}  dist_to_goal={d:5.2f}m "
                      f"pose=({pose[0]:+.2f},{pose[1]:+.2f})")
            if 0 in tokens:  # STOP
                record(i, "STOP", 0.0, 0.0, False, {"tokens": names, "pixel": out_pixel,
                                                    "clause": clause_idx})
                stop_count += 1
                if advance_clause("model STOP"):
                    continue
                if not args.ignore_stop:
                    outcome = "stopped"
                    break
                client.reset()
                recent_xy.clear()
                continue
            net_turn = apply_tokens(tokens, "discrete:" + "+".join(names), out_pixel)
            made_progress = 1 in tokens

        elif out_traj_actions:
            traj_decisions += 1
            capped = out_traj_actions[:args.max_traj_actions]
            nm = [ACTIONS2IDX_INV.get(t, str(t)) for t in capped]
            if i % 10 == 0:
                print(f"  [dec {i:3d}] traj->actions {len(out_traj_actions)} "
                      f"({nm.count('forward')}F/{nm.count('left')}L/{nm.count('right')}R, "
                      f"cap {len(capped)})  dist_to_goal={d:5.2f}m pose=({pose[0]:+.2f},{pose[1]:+.2f})")
            net_turn = apply_tokens(capped, "traj_actions", out_pixel)
            # discretiser collapsing to <=1 action = its own "no forward step
            # gets closer to my predicted goal" early-out = implicit arrival.
            made_progress = len(out_traj_actions) >= 2

        elif out_traj is not None and len(out_traj) >= 2:
            # trajectory returned but discretiser gave nothing actionable
            # (its own "moving forward would leave the goal" early-out) -- hold.
            traj_decisions += 1
            if i % 10 == 0:
                print(f"  [dec {i:3d}] trajectory ({len(out_traj)} wp) but 0 discrete actions -- holding")
            record(i, "traj_noop", 0.0, 0.0, False, {"waypoints": len(out_traj), "pixel": out_pixel})
        else:
            if i % 10 == 0:
                print(f"  [dec {i:3d}] no actionable output (pixel-only latent stage) -- holding")
            record(i, "hold", 0.0, 0.0, False, {"pixel": out_pixel})

        # ---- arrival detection (the model almost never emits STOP itself on
        #      this path). Two independent signals:
        #   --oracle-stop : true VLN-CE "oracle stop" ceiling -- stop the moment
        #                   NE < success radius (reports the SR the navigation
        #                   would get with a perfect stop detector).
        #   noop-streak   : K consecutive decisions where S1's own discretiser
        #                   produced no real forward motion = the model thinks
        #                   it's at its predicted goal -> declare arrival.
        # ---- clause sequencing: advance to the next sub-goal on a per-clause
        #      stall (model has done what this clause asks) or a hard cap.
        #      A "stall" only counts AFTER the clause has driven >=1 forward
        #      step -- otherwise the initial reorientation phase (turn-only)
        #      trivially "stalls" and every clause is skipped in ~5 decisions.
        # "settle" = S1 produced a trajectory but it collapsed to <=2 discrete
        # steps (its own "no forward step gets me closer to my predicted goal")
        # -- the model saying "I'm basically there".
        settled_this_tick = (
            (out_traj_actions is not None and 0 < len(out_traj_actions) <= 2)
            or (not out_action and not out_traj_actions and out_traj is not None and len(out_traj) >= 2)
        )

        if made_progress:
            clause_moved = True
        if args.sequence_clauses and clause_idx < len(clauses) - 1:
            clause_noop = clause_noop + 1 if (clause_moved and not made_progress) else 0
            if clause_noop >= args.clause_advance_k:
                if advance_clause(f"stalled {clause_noop}"):
                    continue
            elif clause_dec >= args.max_clause_decisions:
                if advance_clause(f"cap {clause_dec}"):
                    continue

        if args.oracle_stop and d < goal_radius:
            record(i, "ORACLE_STOP", 0.0, 0.0, False, {"dist_to_goal": d})
            outcome = "oracle_stopped"
            break
        # noop-streak arrival: only meaningful WHEN ALREADY NEAR THE GOAL --
        # away from it, a run of turn-only decisions is reorientation (or a
        # stall the frozen-escape handles), not arrival.
        near_goal = d < args.arrival_radius_m
        # anti-oscillation: the model will turn-only forever without committing
        # to a forward step. After K consecutive turn-only decisions, force one
        # 0.25 m forward -- changes the scene enough to unstick the grounding.
        turn_run = turn_run + 1 if (net_turn is not None and not made_progress) else 0
        if args.max_consecutive_turns > 0 and turn_run >= args.max_consecutive_turns:
            forced_forwards += 1
            for _ in range(DISCRETE_SUBSTEPS):
                integrate(pose, pf, cam_y, DISCRETE_FORWARD_M / (DISCRETE_SUBSTEPS * args.dt), 0.0, args.dt)
                min_dist = min(min_dist, dist((pose[0], pose[1]), goal_xy))
            path_xy.append((pose[0], pose[1]))
            record(i, "forced_forward", 0.0, 0.0, True, {"after_turns": turn_run})
            turn_run = 0

        # ---- dwell-stop (#3): S1 keeps saying "basically there" for K decisions
        #      -> arrival, even if the model never emits STOP and even if it's
        #      still emitting small trajectories (the ep97 "reached then wandered"
        #      failure). Only on the final clause; not in the first 15 decisions.
        settle_streak = settle_streak + 1 if settled_this_tick else 0
        if (args.dwell_k > 0 and settle_streak >= args.dwell_k and i >= 15
                and clause_idx == len(clauses) - 1):
            record(i, "DWELL_STOP", 0.0, 0.0, False,
                   {"settle_streak": settle_streak, "dist_to_goal": d})
            stop_count += 1
            outcome = "dwell_stop"
            break

        noop_streak = (noop_streak + 1) if (not made_progress and near_goal) else 0
        if args.arrival_noop_k > 0 and noop_streak >= args.arrival_noop_k:
            record(i, "ARRIVED_NOOP", 0.0, 0.0, False, {"streak": noop_streak, "dist_to_goal": d})
            stop_count += 1
            if not args.ignore_stop:
                outcome = "arrived_noop"
                break
            client.reset()
            recent_xy.clear()
            noop_streak = 0
            continue

        # ---- frozen-escape: the greedy-decode flip-flop lock (documented in
        #      reference/internvla_dualvln_zenoh_node.py) leaves net displacement
        #      ~0 for a long stretch. Clearing the model's history alone does
        #      NOT break it -- the robot is still looking at the same view, so
        #      it re-locks within a decision or two (measured). Force a real
        #      displacement (turn + push, alternating direction) so the next
        #      S2 call is conditioned on a genuinely new viewpoint.
        recent_xy.append((float(pose[0]), float(pose[1])))
        recent_xy[:] = recent_xy[-args.frozen_window:]
        if len(recent_xy) >= args.frozen_window:
            net = math.hypot(recent_xy[-1][0] - recent_xy[0][0],
                             recent_xy[-1][1] - recent_xy[0][1])
            if net < args.frozen_move_m:
                escapes += 1
                escape_dir *= -1.0
                print(f"  [dec {i:3d}] [frozen-escape #{escapes}] net {net:.2f}m over "
                      f"{args.frozen_window} decisions -- forced turn+push "
                      f"({'left' if escape_dir > 0 else 'right'})")
                for _ in range(24):   # ~55 deg
                    integrate(pose, pf, cam_y, 0.0,
                              escape_dir * math.radians(55) / (24 * args.dt), args.dt)
                for _ in range(32):   # push
                    integrate(pose, pf, cam_y, args.frozen_push_m / (32 * args.dt), 0.0, args.dt)
                    min_dist = min(min_dist, dist((pose[0], pose[1]), goal_xy))
                path_xy.append((pose[0], pose[1]))
                record(i, f"frozen_escape#{escapes}", 0.0, 0.0, True, {})
                client.reset()
                recent_xy.clear()
                settle_streak = 0
                noop_streak = 0

        if args.realtime_pacing:
            remaining = args.dt * max(1, args.plan_step_gap) - (time.time() - tick_start)
            if remaining > 0:
                time.sleep(remaining)

    final_xy = (float(pose[0]), float(pose[1]))
    navigation_error = dist(final_xy, goal_xy)
    travelled = path_length(path_xy)
    success = navigation_error < goal_radius
    oracle_success = min_dist < goal_radius
    spl = (optimal_length / max(optimal_length, travelled)) if (success and travelled > 0) else 0.0

    metrics = {
        "model": "InternVLA-N1-DualVLN (native dual-system)",
        "episode_id": episode["episode_id"], "scene": args.scene, "split": args.split,
        "instruction": instruction, "outcome": outcome, "decisions": len(log),
        "start_xy": [float(path_xy[0][0]), float(path_xy[0][1])],
        "goal_xy": list(goal_xy), "final_xy": list(final_xy),
        "optimal_length_m": optimal_length, "path_length_m": travelled,
        "navigation_error_m": navigation_error, "oracle_navigation_error_m": min_dist,
        "success": success, "oracle_success": oracle_success, "spl": spl,
        "success_radius_m": goal_radius,
        "stuck_recovery_triggers": stuck.trigger_count,
        "frozen_escapes": escapes,
        "forced_forwards": forced_forwards,
        "sequence_clauses": args.sequence_clauses,
        "clauses": clauses,
        "clause_advances": clause_advances,
        "final_clause_idx": clause_idx,
        "stop_count": stop_count,
        "trajectory_decisions": traj_decisions,
        "discrete_action_counts": action_counts,
        "scene_mesh": scene_mesh,
    }
    print("\n=== VLN-CE metrics ===")
    for k in ("outcome", "decisions", "navigation_error_m", "oracle_navigation_error_m",
              "success", "oracle_success", "spl", "path_length_m", "optimal_length_m"):
        print(f"  {k:26s}: {metrics[k]}")

    (out_dir / "result.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "trajectory.json").write_text(json.dumps(log, indent=2))
    print(f"\nwrote {out_dir/'result.json'} and {out_dir/'trajectory.json'}")

    sim.close()
    if server_proc is not None:
        print("[server] --autostart-server: stopping internvla_n1_server.py")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
