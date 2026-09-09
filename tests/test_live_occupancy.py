"""Geometry and planning checks for the live occupancy grid.

These are the parts that silently produce a plausible-but-wrong answer if a
sign or an axis is flipped, which is exactly the class of bug that cost this
pipeline two full simulator runs.
"""

import math

import numpy as np
import pytest

from nav_pipeline.live_occupancy import (FREE, OCCUPIED, UNKNOWN, LiveOccupancy,
                                         thin_waypoints)


def _make_free_disc(occ: LiveOccupancy, center_xy, radius_m: float) -> None:
    """Fill a disc of FREE cells directly (no ray-casting needed) -- these
    tests are about frontier_points' cell logic, not integrate()'s carving."""
    cx, cy = center_xy
    rad_cells = int(radius_m / occ.res)
    r0, c0 = occ.to_cell(cx, cy)
    for dr in range(-rad_cells, rad_cells + 1):
        for dc in range(-rad_cells, rad_cells + 1):
            if dr * dr + dc * dc <= rad_cells * rad_cells:
                r, c = r0 + dr, c0 + dc
                if occ._in_bounds(r, c):
                    occ.grid[r, c] = FREE


def test_cell_world_roundtrip():
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    for x, y in [(0.0, 0.0), (3.3, -2.7), (-4.1, 5.9)]:
        r, c = occ.to_cell(x, y)
        wx, wy = occ.to_world(r, c)
        assert abs(wx - x) <= 0.1 and abs(wy - y) <= 0.1


def test_carving_marks_free_between_rover_and_hit():
    """A wall straight ahead: cells in front of it become FREE, the wall
    becomes OCCUPIED, and cells behind it stay UNKNOWN."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    occ._carve_free(*occ.to_cell(0.0, 0.0), *occ.to_cell(2.0, 0.0))
    hr, hc = occ.to_cell(2.0, 0.0)
    occ.grid[hr, hc] = OCCUPIED

    mid_r, mid_c = occ.to_cell(1.0, 0.0)
    assert occ.grid[mid_r, mid_c] == FREE
    assert occ.grid[hr, hc] == OCCUPIED
    beyond_r, beyond_c = occ.to_cell(3.0, 0.0)
    assert occ.grid[beyond_r, beyond_c] == UNKNOWN


def test_carve_never_erases_an_obstacle():
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    r, c = occ.to_cell(1.0, 0.0)
    occ.grid[r, c] = OCCUPIED
    occ._carve_free(*occ.to_cell(0.0, 0.0), *occ.to_cell(2.0, 0.0))
    assert occ.grid[r, c] == OCCUPIED


def _open_room(occ, x0, x1, y0, y1):
    r0, c0 = occ.to_cell(x0, y0)
    r1, c1 = occ.to_cell(x1, y1)
    occ.grid[r0:r1, c0:c1] = FREE


def test_astar_routes_through_a_doorway_not_the_straight_line():
    """Two rooms joined by one gap, with the gap off the straight line. A
    local push toward the goal cannot solve this; a search can."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _open_room(occ, -4.0, -0.2, -4.0, 4.0)     # left room
    _open_room(occ, 0.2, 4.0, -4.0, 4.0)       # right room
    _open_room(occ, -0.4, 0.4, 3.0, 3.6)       # doorway, up near y=+3

    path, info = occ.astar((-3.0, -3.0), (3.0, -3.0), robot_radius=0.1)
    assert path is not None, info
    # the route has to detour north through the doorway
    assert max(p[1] for p in path) > 2.5
    assert info["length_m"] > 6.0               # longer than the 6 m straight line


def test_astar_reports_when_no_route_exists():
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _open_room(occ, -4.0, -0.2, -4.0, 4.0)
    _open_room(occ, 0.2, 4.0, -4.0, 4.0)       # no doorway
    path, info = occ.astar((-3.0, -3.0), (3.0, -3.0), robot_radius=0.1)
    assert path is None
    assert "no route" in info["reason"]


def test_unknown_is_not_navigable():
    """Planning through never-observed cells is how you drive into a wall
    that was never seen."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _open_room(occ, -4.0, -2.0, -1.0, 1.0)
    path, info = occ.astar((-3.0, 0.0), (3.0, 0.0), robot_radius=0.1)
    assert path is None


def test_astar_nav_search_m_reaches_past_a_dense_occupied_pocket():
    """Real case (2026-09-05, MP3D scene): the rover's own final pose and
    the goal it needed to reach (0.48m apart) both sat inside a genuinely
    dense pocket of OCCUPIED cells (close furniture) -- clearance was
    negative in every cell out to the old hardcoded 2.0m search radius, so
    nearest_navigable found no anchor for either start or goal and astar()
    refused outright, even with allow_unknown=True. Reproduce that directly
    on nearest_navigable (astar()'s own anchor step, and what actually
    failed): an occupied disc wide enough that clearance stays negative
    everywhere within a 2.0m search of the goal, with a real FREE opening
    only reachable by searching further."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    goal_xy = (0.0, 0.0)
    occ.grid[:, :] = FREE   # real free space surrounds the pocket, not unmapped void
    r0, c0 = occ.to_cell(*goal_xy)
    rad_cells = int(4.5 / occ.res)   # occupied out to 4.5m -- well past both search radii below
    for dr in range(-60, 61):
        for dc in range(-60, 61):
            r, c = r0 + dr, c0 + dc
            if occ._in_bounds(r, c) and (dr * dr + dc * dc) <= rad_cells * rad_cells:
                occ.grid[r, c] = OCCUPIED
    clear = occ.clearance(robot_radius=0.1)

    assert occ.nearest_navigable(*goal_xy, clear, max_search_m=2.0) is None
    assert occ.nearest_navigable(*goal_xy, clear, max_search_m=6.0) is not None, \
        "widened search should find the real opening just past the occupied pocket"


def test_nearest_navigable_steps_out_of_an_object():
    """A recalled position is an object centroid -- inside the object. The
    rover must stand next to it, not in it."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _open_room(occ, -2.0, 2.0, -2.0, 2.0)
    r, c = occ.to_cell(0.0, 0.0)
    occ.grid[r - 2:r + 3, c - 2:c + 3] = OCCUPIED   # the table itself
    clear = occ.clearance(robot_radius=0.1)
    cell = occ.nearest_navigable(0.0, 0.0, clear)
    assert cell is not None
    wx, wy = occ.to_world(*cell)
    assert 0.15 < math.hypot(wx, wy) < 1.5


def test_thin_waypoints_keeps_endpoints_and_spacing():
    path = [(0.1 * i, 0.0) for i in range(31)]     # 3 m at 0.1 m spacing
    wps = thin_waypoints(path, spacing=0.6)
    assert wps[0] == path[0] and wps[-1] == path[-1]
    gaps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(wps, wps[1:])]
    assert all(g >= 0.55 for g in gaps[:-1])
    assert len(wps) < len(path)


def test_frontier_points_found_on_the_edge_of_a_disc():
    """A disc of FREE surrounded by UNKNOWN: frontier cells should be near
    the disc's edge, not its interior or outside it entirely."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _make_free_disc(occ, (0.0, 0.0), radius_m=2.0)
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.1, max_radius=6.0, k=3)
    assert len(pts) > 0
    for x, y in pts:
        d = math.hypot(x, y)
        assert 1.5 < d < 2.3   # near the disc's radius, not the center or far outside


def test_frontier_points_empty_when_fully_explored():
    """No UNKNOWN anywhere adjacent to FREE -- nothing left to explore
    toward, so this should return an empty list, not fabricate a point."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=4.0, res=0.1)
    occ.grid[:, :] = FREE   # the whole grid is known
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.1, max_radius=6.0, k=3)
    assert pts == []


def test_frontier_points_respects_max_radius():
    """A frontier exists, but only far beyond max_radius -- should not be
    offered as a target (this is 'walk a bit further', not 'explore the
    whole house')."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _make_free_disc(occ, (0.0, 0.0), radius_m=8.0)
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.1, max_radius=2.0, k=3)
    assert pts == []


def test_frontier_points_spread_apart_not_clustered():
    """k=3 on a full ring of frontier should not return 3 adjacent cells on
    the same tiny arc -- farthest-point sampling should spread them out."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _make_free_disc(occ, (0.0, 0.0), radius_m=3.0)
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.1, max_radius=6.0, k=3)
    assert len(pts) == 3
    pairwise = [math.hypot(a[0] - b[0], a[1] - b[1])
               for i, a in enumerate(pts) for b in pts[i + 1:]]
    assert min(pairwise) > 1.5   # not all bunched on one small arc


def test_frontier_points_not_starved_by_unknown_credited_as_wall():
    """The actual bug this guards against: a real cluttered-room grid
    returned ZERO frontier points at robot_radius=0.25 because the OLD
    clearance test (grid == FREE only) counted UNKNOWN the same as a wall,
    so every cell adjacent to UNKNOWN -- i.e. every frontier cell, by
    definition -- read as having no clearance, even in a corridor plenty
    wide enough to actually walk (0.25m clearance needs >=0.5m of open
    width; this one is 0.7m, walled on both sides, opening onto UNKNOWN at
    its far end -- should still offer a frontier point there)."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=10.0, res=0.1)
    r0, c0 = occ.to_cell(0.0, 0.0)
    occ.grid[r0 - 3:r0 + 4, c0:c0 + 20] = FREE
    occ.grid[r0 - 4, c0:c0 + 20] = OCCUPIED       # corridor's left wall
    occ.grid[r0 + 4, c0:c0 + 20] = OCCUPIED       # corridor's right wall
    occ.grid[r0 - 4:r0 + 5, c0 - 1] = OCCUPIED    # wall off the NEAR end too --
                                                   # only the far end should open onto UNKNOWN
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.25, max_radius=6.0, k=1)
    assert len(pts) == 1
    x, _ = pts[0]
    assert x > 1.5   # near the corridor's open (far) end, not its walled sides


def test_frontier_points_still_avoids_real_walls():
    """The fix must not become 'anything not OCCUPIED is navigable' --
    a frontier point should never be reported ON TOP of a wall cell."""
    occ = LiveOccupancy((0.0, 0.0), extent_m=10.0, res=0.1)
    _make_free_disc(occ, (0.0, 0.0), radius_m=2.0)
    r0, c0 = occ.to_cell(1.5, 0.0)
    occ.grid[r0 - 1:r0 + 2, c0 - 1:c0 + 2] = OCCUPIED   # a wall poking into the disc
    pts = occ.frontier_points((0.0, 0.0), robot_radius=0.15, max_radius=6.0, k=5)
    for x, y in pts:
        r, c = occ.to_cell(x, y)
        assert occ.grid[r, c] != OCCUPIED


def test_frontier_points_ordered_nearest_first():
    occ = LiveOccupancy((0.0, 0.0), extent_m=20.0, res=0.1)
    _make_free_disc(occ, (1.0, 0.0), radius_m=3.0)   # off-center disc
    pts = occ.frontier_points((1.0, 0.0), robot_radius=0.1, max_radius=6.0, k=3)
    dists = [math.hypot(x - 1.0, y - 0.0) for x, y in pts]
    assert dists == sorted(dists)
