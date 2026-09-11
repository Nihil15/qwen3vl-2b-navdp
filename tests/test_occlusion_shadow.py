"""LiveOccupancy.occlusion_shadow: does the pose->goal ray cross an OCCUPIED
cell, and is the space directly behind it unseen (opaque) or already visible?

Run:  python -m pytest tests/test_occlusion_shadow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav_pipeline.live_occupancy import FREE, OCCUPIED, LiveOccupancy  # noqa: E402


def _occ():
    return LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)


def test_opaque_when_unknown_is_directly_behind_the_hit():
    occ = _occ()
    r, c = occ.to_cell(1.0, 0.0)
    occ.grid[r, c] = OCCUPIED                       # wall at x=1, nothing mapped beyond
    hit = occ.occlusion_shadow((0.0, 0.0), (3.0, 0.0))
    assert hit is not None
    (hx, hy), opaque = hit
    assert abs(hx - 1.0) < 0.11 and abs(hy) < 0.11
    assert opaque is True


def test_not_opaque_when_free_space_is_visible_behind_the_hit():
    occ = _occ()
    r, c = occ.to_cell(1.0, 0.0)
    occ.grid[r, c] = OCCUPIED
    for x in (1.15, 1.25, 1.35, 1.45):             # FREE cells directly behind the hit
        rr, cc = occ.to_cell(x, 0.0)
        occ.grid[rr, cc] = FREE
    (_, opaque) = occ.occlusion_shadow((0.0, 0.0), (3.0, 0.0))
    assert opaque is False


def test_none_when_the_ray_never_hits_an_obstacle():
    occ = _occ()
    r0, c0 = occ.to_cell(-0.5, -0.5)
    r1, c1 = occ.to_cell(3.5, 0.5)
    occ.grid[r0:r1, c0:c1] = FREE
    assert occ.occlusion_shadow((0.0, 0.0), (3.0, 0.0)) is None


def test_obstacle_off_the_ray_is_ignored():
    occ = _occ()
    r, c = occ.to_cell(1.0, 2.0)                    # occupied, but well off the y=0 ray
    occ.grid[r, c] = OCCUPIED
    assert occ.occlusion_shadow((0.0, 0.0), (3.0, 0.0)) is None


def test_hit_reported_is_the_nearest_one_along_the_ray():
    occ = _occ()
    for x in (1.0, 2.0):
        r, c = occ.to_cell(x, 0.0)
        occ.grid[r, c] = OCCUPIED
    (hx, _), opaque = occ.occlusion_shadow((0.0, 0.0), (3.0, 0.0))
    assert abs(hx - 1.0) < 0.11 and opaque is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
