"""Association / opaque-accumulation / re-id / persistence for
nav_pipeline.obstacle_memory. Pure vectors -- no SigLIP.

Run:  python -m pytest tests/test_obstacle_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.obstacle_memory import (ObstacleMemory, bearing_crop_box)  # noqa: E402


def _unit(*vals) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


CAB = _unit(1, 0, 0, 0)
CAB2 = _unit(0.98, 0.1, 0.1, 0.05)     # ~same look
CHAIR = _unit(0, 1, 0, 0)              # different look


def test_similar_and_close_sightings_merge_into_one_entity():
    mem = ObstacleMemory(associate_sim=0.82, associate_pos_gate_m=1.2)
    mem.observe(CAB, (2.0, 0.0), opaque=True, goal_key="behind::cabinet", ts=1.0)
    mem.observe(CAB2, (2.3, 0.1), opaque=True, goal_key="behind::cabinet", ts=2.0)
    assert len(mem.entities) == 1
    assert mem.entities[0].observations == 2


def test_different_look_or_far_position_opens_a_new_entity():
    mem = ObstacleMemory(associate_sim=0.82, associate_pos_gate_m=1.2)
    mem.observe(CAB, (2.0, 0.0), opaque=True, goal_key="g", ts=1.0)
    mem.observe(CHAIR, (2.1, 0.0), opaque=False, goal_key="g", ts=2.0)   # same spot, different look
    mem.observe(CAB, (8.0, 0.0), opaque=True, goal_key="g", ts=3.0)      # same look, far away
    assert len(mem.entities) == 3


def test_opaque_flag_and_blocked_goals_accumulate():
    mem = ObstacleMemory()
    mem.observe(CAB, (2.0, 0.0), opaque=False, goal_key="behind::cabinet", ts=1.0)
    ent = mem.observe(CAB2, (2.1, 0.0), opaque=True, goal_key="left::cabinet", ts=2.0)
    assert ent.opaque is True
    assert ent.blocked_goals == ["behind::cabinet", "left::cabinet"]
    assert ent.is_obstacle is True


def test_reidentify_matches_the_same_object_and_rejects_a_different_one():
    mem = ObstacleMemory()
    mem.observe(CAB, (2.0, 0.0), opaque=True, goal_key="g", ts=1.0)
    hit = mem.reidentify(CAB2, threshold=0.85)
    assert hit is not None and hit[1] >= 0.85
    assert mem.reidentify(CHAIR, threshold=0.85) is None


def test_goals_blocked_by():
    mem = ObstacleMemory()
    mem.observe(CAB, (2.0, 0.0), opaque=True, goal_key="behind::cabinet", ts=1.0)
    mem.observe(CHAIR, (5.0, 0.0), opaque=True, goal_key="behind::plant", ts=2.0)
    assert [e.obstacle_id for e in mem.goals_blocked_by("behind::cabinet")] == ["obs-0000"]


def test_jsonl_round_trip(tmp_path):
    mem = ObstacleMemory()
    mem.observe(CAB, (2.0, 0.0), opaque=True, goal_key="behind::cabinet", ts=1.0)
    mem.observe(CAB2, (2.2, 0.1), opaque=False, goal_key="left::cabinet", ts=2.0)
    p = mem.to_jsonl(tmp_path / "obstacles.jsonl")

    back = ObstacleMemory.from_jsonl(p)
    assert len(back.entities) == 1
    e = back.entities[0]
    assert e.obstacle_id == "obs-0000"
    assert e.opaque is True
    assert e.blocked_goals == ["behind::cabinet", "left::cabinet"]
    assert back.reidentify(CAB, threshold=0.8) is not None


def test_bearing_crop_box_shifts_with_bearing_and_stays_in_bounds():
    w, h = 640, 480
    centre = bearing_crop_box(w, h, 0.0, fov_deg=60.0)
    left = bearing_crop_box(w, h, 0.4, fov_deg=60.0)      # +bearing = left of axis
    right = bearing_crop_box(w, h, -0.4, fov_deg=60.0)
    cc = (centre[0] + centre[2]) / 2
    lc = (left[0] + left[2]) / 2
    rc = (right[0] + right[2]) / 2
    assert abs(cc - w / 2) < 2
    assert lc < cc < rc
    for b in (centre, left, right):
        assert 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
