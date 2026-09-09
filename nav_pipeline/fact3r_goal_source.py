"""Bridge: a Fact3R-Map semantic entity location -> a NavDP point goal.

Fact3R-Map (``MASt3R-SLAM/fact3r-map``) is built offline from a monocular
walk through the scene and answers "where is <the black chair>?" with a
single metric 2D location in its own BEV floor-plane frame. This module
turns that one static answer into the per-tick ``external_goal`` that
``pipeline.DinoNavDPPipeline.step`` already consumes for blind
navigate-back legs (state ``"GOTO"``): the rover drives toward a remembered
point with the full obstacle-guard + NavDP machinery live, and the caller
decides when it has arrived. Qwen pixel-grounding stays the per-frame goal
source for targets that are actually in view / not in the map -- this is
the "REMEMBER -> GO" half only.

Two frames meet here:

* **Fact3R map frame** -- ``resolve_semantic_goal.py`` reports the winning
  entity as ``winner["centroid_yx"] = [y_map, x_map]`` in metres, and the
  mapping camera's own trace as ``rover_track_yx`` (list of ``[y, x]``) in
  the same frame.
* **Rover odom frame** -- ``pipeline.step(pose=(x, y, theta))`` expects a
  continuous planar world frame anchored wherever this run started.

``MapToOdom`` is the 2D similarity transform (rotation, translation,
optional scale) between them. Choose it with:

* ``identity``  -- the map was built from THIS run's frames and the axes
  already coincide (the common offline / same-trajectory case);
* ``anchor``    -- pin ``rover_track_yx[0]`` to a known odom pose and take
  the map->odom rotation from the first metres of each trace;
* ``umeyama``   -- least-squares similarity fit of the whole
  ``rover_track_yx`` against a supplied odom trajectory of the same walk;
* ``explicit``  -- caller passes ``tx, ty, phi_deg, scale`` directly.

Nothing here imports torch / transformers / fact3r; ``from_map_query``
shells out to ``resolve_semantic_goal.py`` in its own SigLIP env once and
then this module only does arithmetic.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

Vec2 = np.ndarray


def _yx_to_xy(pair: Sequence[float]) -> Vec2:
    """Fact3R reports points in planner order ``[y, x]``; NavDP works in
    ``[x, y]``. One place does the swap."""
    return np.array([float(pair[1]), float(pair[0])], dtype=np.float64)


def _track_yx_to_xy(track: Sequence[Sequence[float]]) -> np.ndarray:
    return np.array([_yx_to_xy(p) for p in track], dtype=np.float64)


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Arc-length resample an (N, 2) polyline to ``count`` points so two
    traces of the same walk can be compared index-for-index."""
    if len(points) == count:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(dist[-1])
    if total <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    want = np.linspace(0.0, total, count)
    return np.stack(
        [np.interp(want, dist, points[:, 0]), np.interp(want, dist, points[:, 1])],
        axis=1,
    )


@dataclass
class MapToOdom:
    """``p_odom = scale * R(phi_rad) @ p_map + [tx, ty]`` for planar points."""

    tx: float = 0.0
    ty: float = 0.0
    phi_rad: float = 0.0
    scale: float = 1.0

    def apply(self, xy: Sequence[float]) -> Vec2:
        c, s = math.cos(self.phi_rad), math.sin(self.phi_rad)
        x, y = float(xy[0]), float(xy[1])
        return np.array(
            [
                self.scale * (c * x - s * y) + self.tx,
                self.scale * (s * x + c * y) + self.ty,
            ],
            dtype=np.float64,
        )

    def describe(self) -> str:
        return (
            f"MapToOdom(t=({self.tx:.3f}, {self.ty:.3f}) m, "
            f"phi={math.degrees(self.phi_rad):+.1f} deg, scale={self.scale:.4f})"
        )

    # -- constructors ---------------------------------------------------- #

    @classmethod
    def identity(cls) -> "MapToOdom":
        return cls()

    @classmethod
    def explicit(
        cls, tx: float, ty: float, phi_deg: float, scale: float = 1.0
    ) -> "MapToOdom":
        return cls(tx=float(tx), ty=float(ty), phi_rad=math.radians(phi_deg), scale=float(scale))

    @classmethod
    def from_anchor(
        cls,
        map_track_xy: np.ndarray,
        odom_anchor_xy: Sequence[float] = (0.0, 0.0),
        odom_heading_rad: float = 0.0,
        span_m: float = 1.0,
        scale: float = 1.0,
    ) -> "MapToOdom":
        """Pin the first mapped pose to ``odom_anchor_xy`` and rotate so the
        map trace's initial heading lines up with ``odom_heading_rad`` (the
        direction the rover set off in). ``span_m`` is how far along each
        trace to look for that initial heading -- long enough to be past
        SLAM start-up jitter, short enough to still be "the first move"."""
        track = np.asarray(map_track_xy, dtype=np.float64)
        if len(track) < 2:
            raise ValueError("from_anchor needs at least two mapped poses")
        start = track[0]
        rel = track - start
        far = np.argmax(np.linalg.norm(rel, axis=1) >= span_m)
        if far == 0:  # never reached span_m; use the whole trace
            far = len(track) - 1
        map_heading = math.atan2(rel[far, 1], rel[far, 0])
        phi = odom_heading_rad - map_heading
        c, s = math.cos(phi), math.sin(phi)
        rot_start = scale * np.array([c * start[0] - s * start[1], s * start[0] + c * start[1]])
        return cls(
            tx=float(odom_anchor_xy[0] - rot_start[0]),
            ty=float(odom_anchor_xy[1] - rot_start[1]),
            phi_rad=phi,
            scale=float(scale),
        )

    @classmethod
    def umeyama(
        cls, map_track_xy: np.ndarray, odom_track_xy: np.ndarray, fixed_scale: bool = False
    ) -> "MapToOdom":
        """Least-squares 2D similarity mapping the map trace onto the odom
        trace (Umeyama 1991). Both are resampled to a common length first."""
        src = np.asarray(map_track_xy, dtype=np.float64)
        dst = np.asarray(odom_track_xy, dtype=np.float64)
        n = min(len(src), len(dst))
        if n < 2:
            raise ValueError("umeyama needs at least two points in each trace")
        src = _resample_polyline(src, n)
        dst = _resample_polyline(dst, n)
        mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
        sc, dc = src - mu_src, dst - mu_dst
        cov = (dc.T @ sc) / n
        u, d, vt = np.linalg.svd(cov)
        s_mat = np.eye(2)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            s_mat[1, 1] = -1.0
        r = u @ s_mat @ vt
        var_src = (sc**2).sum() / n
        scale = 1.0 if fixed_scale else float((d * np.diag(s_mat)).sum() / max(var_src, 1e-12))
        t = mu_dst - scale * (r @ mu_src)
        return cls(tx=float(t[0]), ty=float(t[1]), phi_rad=float(math.atan2(r[1, 0], r[0, 0])), scale=scale)


class Fact3rGoalSource:
    """Holds one resolved entity location and emits the per-tick point goal."""

    def __init__(
        self,
        goal_map_xy: Sequence[float],
        map_to_odom: Optional[MapToOdom] = None,
        z: float = 0.0,
        reached_radius: float = 1.0,
        query: str = "",
    ):
        self.goal_map_xy = np.asarray(goal_map_xy, dtype=np.float64)
        self.map_to_odom = map_to_odom or MapToOdom.identity()
        self.z = float(z)
        self.reached_radius = float(reached_radius)
        self.query = query
        self._goal_odom_xy = self.map_to_odom.apply(self.goal_map_xy)

    @property
    def goal_odom_xy(self) -> Vec2:
        return self._goal_odom_xy

    def tick(self, pose: Sequence[float]) -> Vec2:
        """``pose = (x, y, theta)`` odom -> ``[x_fwd, y_left, z_up]`` goal in
        the rover's current local frame, ready for ``step(external_goal=...)``.
        Same SE(2) convention as ``goal_belief.propagate`` / the pipeline's
        own ``odom_delta``."""
        gx, gy = self._goal_odom_xy
        dx, dy = gx - float(pose[0]), gy - float(pose[1])
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        fwd = dx * c + dy * s
        left = -dx * s + dy * c
        return np.array([fwd, left, self.z], dtype=np.float32)

    def distance(self, pose: Sequence[float]) -> float:
        gx, gy = self._goal_odom_xy
        return float(math.hypot(gx - float(pose[0]), gy - float(pose[1])))

    def reached(self, pose: Sequence[float]) -> bool:
        return self.distance(pose) <= self.reached_radius

    def describe(self) -> str:
        gx, gy = self._goal_odom_xy
        return (
            f"Fact3rGoalSource(query={self.query!r} "
            f"map_xy=({self.goal_map_xy[0]:.2f}, {self.goal_map_xy[1]:.2f}) "
            f"odom_xy=({gx:.2f}, {gy:.2f}) reached_radius={self.reached_radius:.2f}m); "
            f"{self.map_to_odom.describe()}"
        )

    # -- constructors -------------------------------------------------- #

    @classmethod
    def from_goal_request(
        cls,
        path: str | Path,
        *,
        map_to_odom: Optional[MapToOdom] = None,
        align: str = "identity",
        odom_track_xy: Optional[np.ndarray] = None,
        odom_anchor_xy: Sequence[float] = (0.0, 0.0),
        odom_heading_rad: float = 0.0,
        reached_radius: float = 1.0,
        fixed_scale: bool = True,
    ) -> "Fact3rGoalSource":
        """Load a ``fact3r-semantic-goal-request`` JSON (the output of
        ``resolve_semantic_goal.py``) and build the goal source.

        ``map_to_odom`` wins if given. Otherwise ``align`` selects how the
        transform is derived from the request's own ``rover_track_yx`` plus,
        for ``umeyama``, the caller-supplied ``odom_track_xy``.
        """
        request = json.loads(Path(path).read_text(encoding="utf-8"))
        if request.get("format") != "fact3r-semantic-goal-request":
            raise ValueError(f"{path}: not a fact3r-semantic-goal-request ({request.get('format')!r})")
        winner = request["winner"]
        if not winner.get("cell_count"):
            raise ValueError(f"{path}: winning entity {winner.get('group_id')} has no BEV footprint / position")
        goal_map_xy = _yx_to_xy(winner["centroid_yx"])
        track_yx = request.get("rover_track_yx")
        map_track_xy = _track_yx_to_xy(track_yx) if track_yx else None

        if map_to_odom is None:
            if align == "identity":
                map_to_odom = MapToOdom.identity()
            elif align == "anchor":
                if map_track_xy is None:
                    raise ValueError("align='anchor' needs rover_track_yx in the goal request")
                map_to_odom = MapToOdom.from_anchor(
                    map_track_xy, odom_anchor_xy=odom_anchor_xy, odom_heading_rad=odom_heading_rad
                )
            elif align == "umeyama":
                if map_track_xy is None or odom_track_xy is None:
                    raise ValueError("align='umeyama' needs rover_track_yx AND odom_track_xy")
                map_to_odom = MapToOdom.umeyama(map_track_xy, odom_track_xy, fixed_scale=fixed_scale)
            else:
                raise ValueError(f"unknown align={align!r} (identity|anchor|umeyama|explicit-via-map_to_odom)")

        return cls(
            goal_map_xy,
            map_to_odom=map_to_odom,
            reached_radius=reached_radius,
            query=str(request.get("query", "")),
        )

    @classmethod
    def from_map_query(
        cls,
        *,
        fact3r_map_root: str | Path,
        semantic_map: str | Path,
        query: str,
        siglip_env: str,
        work_dir: str | Path,
        device: str = "0",
        observed_between: Optional[tuple[int, int]] = None,
        min_score: Optional[float] = None,
        extra_args: Optional[Sequence[str]] = None,
        **from_goal_request_kwargs,
    ) -> "Fact3rGoalSource":
        """Run ``resolve_semantic_goal.py`` in ``siglip_env`` (via
        ``conda run``) to turn ``query`` into a goal request, then delegate
        to :meth:`from_goal_request`.

        ``semantic_map`` is the ``*_semantic.json`` stem produced by the
        depth-semantic-BEV stage. ``work_dir`` receives ``goal_request.json``.
        """
        fact3r_map_root = Path(fact3r_map_root).resolve()
        script = fact3r_map_root / "scripts" / "resolve_semantic_goal.py"
        if not script.is_file():
            raise FileNotFoundError(f"resolve_semantic_goal.py not found under {fact3r_map_root}")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        out_json = work_dir / "goal_request.json"

        cmd = [
            "conda", "run", "--no-capture-output", "-n", siglip_env,
            "python3", str(script),
            "--map", str(semantic_map),
            "--query", query,
            "--output", str(out_json),
            "--device", str(device),
        ]
        if observed_between is not None:
            cmd += ["--observed-between", str(observed_between[0]), str(observed_between[1])]
        if min_score is not None:
            cmd += ["--min-score", str(min_score)]
        if extra_args:
            cmd += list(extra_args)

        print(f"[fact3r-bridge] resolving goal: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return cls.from_goal_request(out_json, **from_goal_request_kwargs)
