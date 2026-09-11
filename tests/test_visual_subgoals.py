"""Sharp-turn detection + spine reversal for nav_pipeline.visual_subgoals.

Pure geometry on synthetic pose traces -- no camera, no torch. Run:

    python -m pytest tests/test_visual_subgoals.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.visual_subgoals import VisualSubgoalTrack  # noqa: E402


def _feed(track: VisualSubgoalTrack, poses):
    for i, (x, y, th) in enumerate(poses):
        track.maybe_add((x, y, th), ts=float(i) * 0.1)


def test_l_shaped_path_emits_exactly_one_sharp_turn_at_the_corner():
    track = VisualSubgoalTrack(spacing_m=0.6, sharp_turn_deg=35.0)
    poses = [(i * 0.1, 0.0, 0.0) for i in range(11)]                # 1.0 m east
    poses += [(1.0, j * 0.1, math.pi / 2) for j in range(1, 12)]    # then 1.1 m north
    _feed(track, poses)

    sharp = [n for n in track.nodes if n.sharp_turn]
    assert len(sharp) == 1, [(round(n.xy[0], 2), round(n.xy[1], 2), n.sharp_turn) for n in track.nodes]
    cx, cy = sharp[0].xy
    assert abs(cx - 1.0) < 0.25 and -0.05 < cy < 0.45          # node sits at the corner
    assert abs(sharp[0].turn_rad) >= math.radians(35)          # a real turn, past the threshold


def test_straight_path_has_no_sharp_turns_but_is_sampled_at_spacing():
    track = VisualSubgoalTrack(spacing_m=0.6, sharp_turn_deg=35.0)
    _feed(track, [(i * 0.1, 0.0, 0.0) for i in range(31)])          # 3 m straight
    assert not any(n.sharp_turn for n in track.nodes)
    assert len(track.nodes) >= 4                                    # ~5 spine nodes over 3 m
    gaps = [math.hypot(b.xy[0] - a.xy[0], b.xy[1] - a.xy[1])
            for a, b in zip(track.nodes, track.nodes[1:])]
    assert max(gaps) < 1.0


def test_reverse_spine_runs_from_last_pose_back_to_the_start():
    track = VisualSubgoalTrack(spacing_m=0.6)
    poses = [(i * 0.1, 0.0, 0.0) for i in range(11)]
    poses += [(1.0, j * 0.1, math.pi / 2) for j in range(1, 12)]
    _feed(track, poses)
    track.force_node((1.0, 1.1, math.pi / 2), ts=99.0)

    spine = track.reverse_spine()
    assert spine[0] == track.nodes[-1].as_xy()
    assert spine[-1] == track.nodes[0].as_xy() == (0.0, 0.0)
    # monotone-ish: first leg back is southward (y decreasing) before turning west
    assert spine[1][1] <= spine[0][1] + 1e-6


def test_sharp_turn_node_is_never_thinned_from_the_spine():
    track = VisualSubgoalTrack(spacing_m=5.0, sharp_turn_deg=35.0)   # huge spacing: only turns survive
    poses = [(i * 0.1, 0.0, 0.0) for i in range(11)]
    poses += [(1.0, j * 0.1, math.pi / 2) for j in range(1, 12)]
    _feed(track, poses)
    corner = next(n for n in track.nodes if n.sharp_turn).as_xy()
    assert corner in track.reverse_spine()


def test_save_load_round_trip(tmp_path):
    track = VisualSubgoalTrack(spacing_m=0.6, sharp_turn_deg=40.0)
    poses = [(i * 0.1, 0.0, 0.0) for i in range(11)]
    poses += [(1.0, j * 0.1, math.pi / 2) for j in range(1, 12)]
    _feed(track, poses)
    p = track.save(tmp_path / "subgoals.json")

    back = VisualSubgoalTrack.load(p)
    assert [n.xy for n in back.nodes] == [n.xy for n in track.nodes]
    assert [n.sharp_turn for n in back.nodes] == [n.sharp_turn for n in track.nodes]
    assert back.reverse_spine() == track.reverse_spine()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
