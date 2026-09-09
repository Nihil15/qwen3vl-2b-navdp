"""Frame-math checks for nav_pipeline.fact3r_goal_source.

Pure arithmetic -- no torch / fact3r / models. Run:

    python -m pytest tests/test_fact3r_goal_source.py
    # or plain:
    python tests/test_fact3r_goal_source.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.fact3r_goal_source import (  # noqa: E402
    Fact3rGoalSource,
    MapToOdom,
    _yx_to_xy,
)

TOL = 1e-6


def test_yx_to_xy_swaps():
    assert np.allclose(_yx_to_xy([3.0, 7.0]), [7.0, 3.0])


def test_identity_is_passthrough():
    m = MapToOdom.identity()
    for p in ([0, 0], [1.5, -2.0], [-4.0, 9.0]):
        assert np.allclose(m.apply(p), p, atol=TOL)


def test_explicit_rotation_then_translation():
    # 90 deg: (1, 0) -> (0, 1); then + (10, -5)
    m = MapToOdom.explicit(tx=10.0, ty=-5.0, phi_deg=90.0, scale=1.0)
    assert np.allclose(m.apply([1.0, 0.0]), [10.0, -4.0], atol=1e-6)
    assert np.allclose(m.apply([0.0, 2.0]), [8.0, -5.0], atol=1e-6)


def test_explicit_scale():
    m = MapToOdom.explicit(tx=0.0, ty=0.0, phi_deg=0.0, scale=2.0)
    assert np.allclose(m.apply([3.0, -1.5]), [6.0, -3.0], atol=TOL)


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, size=(40, 2))
    true = MapToOdom.explicit(tx=2.0, ty=-3.0, phi_deg=37.0, scale=1.4)
    dst = np.array([true.apply(p) for p in src])
    est = MapToOdom.umeyama(src, dst, fixed_scale=False)
    assert abs(est.tx - 2.0) < 1e-6
    assert abs(est.ty + 3.0) < 1e-6
    assert abs(math.degrees(est.phi_rad) - 37.0) < 1e-4
    assert abs(est.scale - 1.4) < 1e-6
    # and it actually maps src onto dst
    mapped = np.array([est.apply(p) for p in src])
    assert np.allclose(mapped, dst, atol=1e-6)


def test_umeyama_resamples_unequal_traces():
    line_src = np.stack([np.linspace(0, 10, 5), np.zeros(5)], axis=1)
    line_dst = np.stack([np.linspace(0, 10, 50), np.full(50, 3.0)], axis=1)  # shifted +3 in y
    est = MapToOdom.umeyama(line_src, line_dst, fixed_scale=True)
    assert np.allclose(est.apply([5.0, 0.0]), [5.0, 3.0], atol=1e-6)


def test_from_anchor_pins_start_and_heading():
    # map trace sets off heading +x; we want it to set off heading +y (90 deg) from odom origin
    map_track = np.stack([np.linspace(0, 5, 20), np.zeros(20)], axis=1) + np.array([100.0, 50.0])
    m = MapToOdom.from_anchor(map_track, odom_anchor_xy=(0.0, 0.0), odom_heading_rad=math.pi / 2)
    assert np.allclose(m.apply(map_track[0]), [0.0, 0.0], atol=1e-6)      # start pinned
    forward = m.apply(map_track[-1])
    assert forward[1] > 4.0 and abs(forward[0]) < 1e-6                    # trace now runs +y


def test_tick_world_to_local_frame():
    # goal at odom (5, 0); rover at origin facing +x -> straight ahead
    src = Fact3rGoalSource([5.0, 0.0], MapToOdom.identity())
    fwd, left, _ = src.tick((0.0, 0.0, 0.0))
    assert abs(fwd - 5.0) < TOL and abs(left) < TOL

    # rover now facing +y (90 deg): the same goal is 5 m to its RIGHT -> left = -5
    fwd, left, _ = src.tick((0.0, 0.0, math.pi / 2))
    assert abs(fwd) < 1e-6 and abs(left + 5.0) < 1e-6

    # rover at (5, -3) facing +x: goal is 3 m to the left, 0 ahead
    fwd, left, _ = src.tick((5.0, -3.0, 0.0))
    assert abs(fwd) < TOL and abs(left - 3.0) < TOL


def test_distance_and_reached():
    src = Fact3rGoalSource([3.0, 4.0], MapToOdom.identity(), reached_radius=1.0)
    assert abs(src.distance((0.0, 0.0, 0.0)) - 5.0) < TOL
    assert not src.reached((0.0, 0.0, 0.0))
    assert src.reached((3.0, 3.5, 1.0))


def test_from_goal_request_identity(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path("/tmp/_fact3r_req.json")
    request = {
        "format": "fact3r-semantic-goal-request",
        "version": 1,
        "query": "the black chair",
        "winner": {"group_id": "e7", "cell_count": 12, "centroid_yx": [4.0, -2.0]},
        "rover_track_yx": [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    }
    out.write_text(json.dumps(request))
    src = Fact3rGoalSource.from_goal_request(out, align="identity", reached_radius=0.75)
    assert src.query == "the black chair"
    assert np.allclose(src.goal_map_xy, [-2.0, 4.0])       # yx -> xy swap
    assert np.allclose(src.goal_odom_xy, [-2.0, 4.0])      # identity
    assert src.reached_radius == 0.75


def test_from_goal_request_rejects_unmapped():
    out = Path("/tmp/_fact3r_req_bad.json")
    out.write_text(json.dumps({
        "format": "fact3r-semantic-goal-request",
        "winner": {"group_id": "e0", "cell_count": 0, "centroid_yx": [0, 0]},
    }))
    try:
        Fact3rGoalSource.from_goal_request(out)
    except ValueError as e:
        assert "no BEV footprint" in str(e)
    else:
        raise AssertionError("expected ValueError for a winner with no footprint")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
