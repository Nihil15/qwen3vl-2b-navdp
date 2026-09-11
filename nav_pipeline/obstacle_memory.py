"""Appearance memory for the OPAQUE OBSTACLES seen while travelling.

The live occupancy grid records that a cell is blocked; it does not record what
the blocker looked like. When the rover has to round an opaque obstacle to
reach a goal behind it, that obstacle is exactly the landmark recall needs: on
the way back the rover should be able to say "that is the cabinet I went around"
from a SigLIP look, not just "a cell here is occupied".

This is the object twin of ``fact3r_live_memory`` but deliberately lighter and
SAM2-free (the rover env has no SAM2): it takes an embedding + a world position
+ the goal that was blocked, associates by appearance-AND-position into an
``ObstacleEntity`` (``is_obstacle=True``, per the design decision to flag the
entity rather than keep a parallel store), OR-accumulates the ``opaque`` flag
and the list of blocked goals, and can re-identify one from a fresh embedding.

Crops are produced caller-side: a Grounding-DINO box when the obstacle is the
named anchor of a "go behind the X" instruction, or ``bearing_crop_box`` below
as the fallback for an unnamed obstacle the occlusion check flagged.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def bearing_crop_box(width: int, height: int, bearing_rad: float, fov_deg: float,
                     width_frac: float = 0.35, height_frac: float = 0.9) -> np.ndarray:
    """A vertical strip of the frame centred on ``bearing_rad`` (angle of the
    obstacle LEFT of the optical axis, +ve = left), for embedding an unnamed
    obstacle. Clamped to the image; always at least 1px wide."""
    half = math.radians(fov_deg) / 2.0
    frac = max(-1.0, min(1.0, bearing_rad / half)) if half > 0 else 0.0
    cx = width * (0.5 - 0.5 * frac)            # +bearing (left) -> smaller column
    bw = max(2.0, width * width_frac)
    x0 = int(round(cx - bw / 2.0))
    x1 = int(round(cx + bw / 2.0))
    x0 = max(0, min(x0, width - 1))
    x1 = max(x0 + 1, min(x1, width))
    bh = max(1, int(round(height * height_frac)))
    y0 = max(0, (height - bh) // 2)
    y1 = min(height, y0 + bh)
    return np.array([x0, y0, x1, y1], dtype=float)


@dataclass
class ObstacleEntity:
    obstacle_id: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    is_obstacle: bool = True
    opaque: bool = False
    blocked_goals: List[str] = field(default_factory=list)
    first_ts: float = 0.0
    last_ts: float = 0.0

    @property
    def observations(self) -> int:
        return len(self.positions)

    @property
    def position(self) -> Tuple[float, float]:
        arr = np.asarray(self.positions, dtype=np.float64)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    @property
    def position_spread(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.asarray(self.positions, dtype=np.float64).std(axis=0)))

    def similarity(self, emb: np.ndarray) -> float:
        if not self.embeddings:
            return -1.0
        return max(float(np.dot(emb, e)) for e in self.embeddings)


class ObstacleMemory:
    def __init__(self, associate_sim: float = 0.82, associate_pos_gate_m: float = 1.2,
                 max_views: int = 6):
        self.associate_sim = float(associate_sim)
        self.associate_pos_gate_m = float(associate_pos_gate_m)
        self.max_views = int(max_views)
        self.entities: List[ObstacleEntity] = []

    def _select(self, emb: np.ndarray, xy: Sequence[float]) -> Optional[ObstacleEntity]:
        ox, oy = float(xy[0]), float(xy[1])
        best, best_sim = None, -1.0
        for ent in self.entities:
            if math.hypot(ox - ent.position[0], oy - ent.position[1]) > self.associate_pos_gate_m:
                continue
            sim = ent.similarity(emb)
            if sim > best_sim:
                best, best_sim = ent, sim
        return best if (best is not None and best_sim >= self.associate_sim) else None

    def observe(self, emb: Sequence[float], xy: Sequence[float], opaque: bool,
                goal_key: str, ts: Optional[float] = None) -> ObstacleEntity:
        """Fold one obstacle sighting into memory; returns the entity it landed
        in (new or existing)."""
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        ts = time.time() if ts is None else float(ts)
        ent = self._select(emb, xy)
        if ent is None:
            ent = ObstacleEntity(obstacle_id=f"obs-{len(self.entities):04d}", first_ts=ts)
            self.entities.append(ent)
        ent.positions.append((float(xy[0]), float(xy[1])))
        if len(ent.embeddings) < self.max_views:
            ent.embeddings.append(emb)
        ent.opaque = ent.opaque or bool(opaque)
        if goal_key and goal_key not in ent.blocked_goals:
            ent.blocked_goals.append(goal_key)
        ent.last_ts = ts
        return ent

    def reidentify(self, emb: Sequence[float], threshold: float = 0.85
                   ) -> Optional[Tuple[ObstacleEntity, float]]:
        """Best obstacle entity matching `emb` above `threshold`, or None."""
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        best, best_sim = None, threshold
        for ent in self.entities:
            sim = ent.similarity(emb)
            if sim > best_sim:
                best, best_sim = ent, sim
        return (best, best_sim) if best is not None else None

    def query_text(self, text_emb: Sequence[float]) -> Optional[Tuple[ObstacleEntity, float]]:
        """Best obstacle entity by SigLIP text-vs-crop similarity (no
        threshold -- SigLIP cross-modal scores are small; caller ranks)."""
        text_emb = np.asarray(text_emb, dtype=np.float32)
        ranked = [(ent.similarity(text_emb), ent) for ent in self.entities if ent.embeddings]
        if not ranked:
            return None
        ranked.sort(key=lambda t: -t[0])
        return ranked[0][1], float(ranked[0][0])

    def goals_blocked_by(self, goal_key: str) -> List[ObstacleEntity]:
        return [e for e in self.entities if goal_key in e.blocked_goals]

    # -- persistence ------------------------------------------------ #
    def to_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for e in self.entities:
                f.write(json.dumps({
                    "obstacle_id": e.obstacle_id,
                    "is_obstacle": e.is_obstacle,
                    "opaque": e.opaque,
                    "blocked_goals": e.blocked_goals,
                    "position_xy": list(e.position),
                    "position_spread_m": round(e.position_spread, 4),
                    "observations": e.observations,
                    "first_ts": e.first_ts,
                    "last_ts": e.last_ts,
                    "embeddings": [[round(float(v), 6) for v in emb] for emb in e.embeddings],
                }) + "\n")
        return p

    @classmethod
    def from_jsonl(cls, path: str | Path, **kw) -> "ObstacleMemory":
        mem = cls(**kw)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            e = ObstacleEntity(
                obstacle_id=d["obstacle_id"],
                embeddings=[np.asarray(v, dtype=np.float32) for v in d.get("embeddings", [])],
                positions=[tuple(d["position_xy"])],
                is_obstacle=bool(d.get("is_obstacle", True)),
                opaque=bool(d.get("opaque", False)),
                blocked_goals=list(d.get("blocked_goals", [])),
                first_ts=float(d.get("first_ts", 0.0)),
                last_ts=float(d.get("last_ts", 0.0)),
            )
            mem.entities.append(e)
        return mem
