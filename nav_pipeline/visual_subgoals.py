"""Visual subgoals recorded along the outbound leg -- a waypoint spine with a
forced keyframe at every SHARP TURN (the corners the rover makes to get around
an opaque obstacle), each node carrying the visual context seen there.

Why this is not just A* at recall time: A* over the occupancy grid re-derives
*a* route, but it does not know where the rover actually cornered on the way
out, and its shortest-path corners shave exactly the clearance the outbound
drive proved was needed. Recording the real path -- and marking the points
where the heading of travel swung hard -- lets recall replay the same cornering
motion, and lets each corner be visually re-confirmed (the SigLIP embedding of
the frame at that node) rather than trusted blind.

A node is emitted when EITHER:
  * arc length since the last node reaches ``spacing_m`` (ordinary spine
    sampling), or
  * the heading of travel over the recent trail swings by at least
    ``sharp_turn_deg`` (a corner) -- this node gets ``sharp_turn=True`` and is
    never thinned out of the returned spine.

Heading is measured from DISPLACEMENT between trail points, not raw yaw, so a
stationary in-place wiggle or noisy IMU does not fake a turn.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Subgoal:
    xy: Tuple[float, float]
    heading_in: float          # bearing of travel arriving at this node (rad)
    heading_out: float         # bearing of travel leaving the previous node (rad)
    sharp_turn: bool
    turn_rad: float            # signed heading change that triggered a sharp node (0 otherwise)
    ts: float
    obstacle_id: Optional[str] = None
    embedding: Optional[List[float]] = None   # SigLIP of the frame seen here

    def as_xy(self) -> Tuple[float, float]:
        return (float(self.xy[0]), float(self.xy[1]))


@dataclass
class VisualSubgoalTrack:
    spacing_m: float = 0.6
    sharp_turn_deg: float = 35.0
    min_move_m: float = 0.12          # ignore trail shorter than this (noise floor)
    min_turn_arc_m: float = 0.20      # a turn only counts once this much path supports it
    nodes: List[Subgoal] = field(default_factory=list)
    poses: List[Tuple[float, float, float, float]] = field(default_factory=list)  # full x,y,theta,ts log
    _trail: List[Tuple[float, float, float, float]] = field(default_factory=list, repr=False)

    # -- recording ------------------------------------------------- #
    def maybe_add(self, pose: Sequence[float], ts: float, rgb: Optional[np.ndarray] = None,
                  embed_fn: Optional[Callable[[np.ndarray], Optional[Sequence[float]]]] = None,
                  obstacle_id: Optional[str] = None) -> Optional[Subgoal]:
        """Feed one pose ``(x, y, theta)`` + timestamp. Returns the new
        ``Subgoal`` if this tick emitted one, else ``None``. ``embed_fn`` (if
        given, with ``rgb``) produces the node's stored embedding -- wrap the
        encode-once cache here."""
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        self.poses.append((x, y, th, float(ts)))
        if not self.nodes:
            self._emit((x, y, th), ts, sharp=False, turn=0.0, b_in=th, b_out=th,
                       rgb=rgb, embed_fn=embed_fn, obstacle_id=obstacle_id)
            return self.nodes[-1]
        self._trail.append((x, y, th, float(ts)))
        arc = self._trail_arc()
        if arc < self.min_move_m:
            return None
        b_in, b_out, turn = self._trail_turn()
        sharp = arc >= self.min_turn_arc_m and abs(turn) >= math.radians(self.sharp_turn_deg)
        if sharp or arc >= self.spacing_m:
            self._emit((x, y, th), ts, sharp=sharp, turn=(turn if sharp else 0.0),
                       b_in=b_in, b_out=b_out, rgb=rgb, embed_fn=embed_fn,
                       obstacle_id=obstacle_id)
            self._trail = [(x, y, th, float(ts))]
            return self.nodes[-1]
        return None

    def force_node(self, pose: Sequence[float], ts: float, rgb: Optional[np.ndarray] = None,
                   embed_fn: Optional[Callable] = None, obstacle_id: Optional[str] = None) -> Subgoal:
        """Emit a node unconditionally (e.g. at outbound arrival) so the spine
        always ends where the rover actually stopped."""
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        b_in, b_out, turn = self._trail_turn() if len(self._trail) >= 3 else (th, th, 0.0)
        self._emit((x, y, th), ts, sharp=False, turn=0.0, b_in=b_in, b_out=b_out,
                   rgb=rgb, embed_fn=embed_fn, obstacle_id=obstacle_id)
        self._trail = [(x, y, th, float(ts))]
        return self.nodes[-1]

    def _emit(self, pose, ts, *, sharp, turn, b_in, b_out, rgb, embed_fn, obstacle_id) -> None:
        emb = None
        if embed_fn is not None and rgb is not None:
            try:
                out = embed_fn(rgb)
                if out is not None:
                    emb = [round(float(v), 6) for v in np.asarray(out).ravel()]
            except Exception:
                emb = None
        self.nodes.append(Subgoal(
            xy=(float(pose[0]), float(pose[1])), heading_in=float(b_in),
            heading_out=float(b_out), sharp_turn=bool(sharp), turn_rad=float(turn),
            ts=float(ts), obstacle_id=obstacle_id, embedding=emb))

    def _trail_arc(self) -> float:
        if len(self._trail) < 2:
            return 0.0
        pts = np.asarray([(p[0], p[1]) for p in self._trail], dtype=np.float64)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    def _trail_turn(self) -> Tuple[float, float, float]:
        """Bearing of the first half of the trail vs the second half, and the
        wrapped difference. Returns ``(b_in, b_out, turn)``; a half with no
        real displacement yields turn 0 (can't define a bearing)."""
        if len(self._trail) < 3:
            return 0.0, 0.0, 0.0
        pts = np.asarray([(p[0], p[1]) for p in self._trail], dtype=np.float64)
        mid = len(pts) // 2
        d1 = pts[mid] - pts[0]
        d2 = pts[-1] - pts[mid]
        if np.hypot(*d1) < 1e-3 or np.hypot(*d2) < 1e-3:
            return 0.0, 0.0, 0.0
        b_in = math.atan2(d1[1], d1[0])
        b_out = math.atan2(d2[1], d2[0])
        return b_in, b_out, _wrap(b_out - b_in)

    # -- recall spine -------------------------------------------- #
    def reverse_nodes(self) -> List[Subgoal]:
        """The recorded nodes, newest-first -- the return order. Sharp-turn
        nodes are all present (nothing is dropped here)."""
        return list(reversed(self.nodes))

    def reverse_spine(self) -> List[Tuple[float, float]]:
        """Return-leg waypoint list: node xy's newest-first. Consecutive
        non-sharp nodes closer together than ``spacing_m`` are thinned; every
        sharp-turn node and both endpoints are kept."""
        rev = self.reverse_nodes()
        if not rev:
            return []
        spine: List[Tuple[float, float]] = [rev[0].as_xy()]
        for n in rev[1:-1]:
            p = n.as_xy()
            if n.sharp_turn or math.hypot(p[0] - spine[-1][0], p[1] - spine[-1][1]) >= self.spacing_m:
                spine.append(p)
        if len(rev) >= 2:
            end = rev[-1].as_xy()
            if end != spine[-1]:
                spine.append(end)
        return spine

    # -- persistence ------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "format": "visual-subgoal-track",
            "spacing_m": self.spacing_m,
            "sharp_turn_deg": self.sharp_turn_deg,
            "nodes": [asdict(n) for n in self.nodes],
            "poses": [list(p) for p in self.poses],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "VisualSubgoalTrack":
        t = cls(spacing_m=float(d.get("spacing_m", 0.6)),
                sharp_turn_deg=float(d.get("sharp_turn_deg", 35.0)))
        t.nodes = [Subgoal(**{**n, "xy": tuple(n["xy"])}) for n in d.get("nodes", [])]
        t.poses = [tuple(p) for p in d.get("poses", [])]
        return t

    @classmethod
    def load(cls, path: str | Path) -> "VisualSubgoalTrack":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
