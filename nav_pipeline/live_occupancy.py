"""A metric occupancy grid carved live from depth + odometry, and A* over it.

Why this exists: NavDP is a LOCAL policy. Handing it a goal 4 m away behind
furniture and a doorway does not work, and the failure is not subtle -- it
oscillates GOTO/AVOID at a fixed radius from the goal and makes no net
progress (measured: 260 ticks, 0.05 m travelled). Reaching a remembered
position across a room is a global problem: it needs a route, and then NavDP
follows that route one short hop at a time, which is the regime it is good at.

This is also what fact3r's own return path does -- resolve the query to a
metric position, A* over the occupancy layer, follow the waypoints -- so leg 2
here is fact3r's method, not an improvisation.

The occupancy layer is carved from the SAME depth frames the entity mapper
sees, with no ground truth: each frame becomes a virtual 2D laser scan (the
obstacle-band points from obstacle_guard, reduced to a min-range per azimuth
bin), and each ray is carved FREE up to its hit and marked OCCUPIED at it.
Bins with no obstacle-band return are open floor and carve free to max_range.

An earlier attempt at camera-only occupancy came out ~0.1% free and was
written off as unusable. That measurement was taken while the sim camera was
accidentally at 2.4 m (see run_ab_recall_hm3d_test.build_sim): from up there a
level camera sees no floor nearer than 3.3 m, so nearly every return landed on
wall inside the obstacle band and nothing was ever carved. At a real camera
height the same code carves normally.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .obstacle_guard import GuardConfig, depth_to_obstacle_points

UNKNOWN, FREE, OCCUPIED = -1, 0, 1

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class LiveOccupancy:
    """World-frame occupancy grid. World (x, y) maps to (col, row) via
    col = (x - origin_x) / res, row = (y - origin_y) / res -- the same planar
    frame the rest of this pipeline uses (x forward-ish, y left-ish, both in
    odom), so no habitat axis juggling leaks in here."""

    def __init__(self, center_xy: Sequence[float], extent_m: float = 30.0,
                 res: float = 0.10, guard: Optional[GuardConfig] = None,
                 n_bins: int = 181, max_range: float = 4.0):
        self.res = float(res)
        self.guard = guard or GuardConfig()
        self.n_bins = int(n_bins)
        self.max_range = float(max_range)
        n = int(round(extent_m / self.res))
        self.grid = np.full((n, n), UNKNOWN, dtype=np.int8)
        self.origin_xy = (float(center_xy[0]) - extent_m / 2.0,
                          float(center_xy[1]) - extent_m / 2.0)

    # -- frame <-> cell ------------------------------------------------- #
    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int((y - self.origin_xy[1]) / self.res),
                int((x - self.origin_xy[0]) / self.res))

    def to_world(self, row: int, col: int) -> Tuple[float, float]:
        return (self.origin_xy[0] + (col + 0.5) * self.res,
                self.origin_xy[1] + (row + 0.5) * self.res)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    # -- carving -------------------------------------------------------- #
    def integrate(self, depth: np.ndarray, pose: Sequence[float], intrinsics) -> int:
        """Carve one depth frame. Returns the number of rays carved."""
        fx, fy, cx, cy = intrinsics
        pts = depth_to_obstacle_points(depth, fx, fy, cx, cy, self.guard)

        half = math.radians(self.guard_fov_deg() / 2.0)
        edges = np.linspace(-half, half, self.n_bins + 1)
        ranges = np.full(self.n_bins, self.max_range, dtype=np.float64)
        hit = np.zeros(self.n_bins, dtype=bool)
        if pts.shape[0]:
            ang = np.arctan2(pts[:, 1], pts[:, 0])          # +left
            rng = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.digitize(ang, edges) - 1
            ok = (idx >= 0) & (idx < self.n_bins) & (rng < self.max_range)
            for b, r in zip(idx[ok], rng[ok]):
                if r < ranges[b]:
                    ranges[b], hit[b] = r, True

        x0, y0, th = float(pose[0]), float(pose[1]), float(pose[2])
        r0, c0 = self.to_cell(x0, y0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        carved = 0
        for b in range(self.n_bins):
            a = th + centers[b]
            reach = ranges[b] - (self.res if hit[b] else 0.0)
            r1, c1 = self.to_cell(x0 + reach * math.cos(a), y0 + reach * math.sin(a))
            self._carve_free(r0, c0, r1, c1)
            if hit[b]:
                hr, hc = self.to_cell(x0 + ranges[b] * math.cos(a),
                                      y0 + ranges[b] * math.sin(a))
                if self._in_bounds(hr, hc):
                    self.grid[hr, hc] = OCCUPIED
            carved += 1
        return carved

    def _carve_free(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """Bresenham from the rover to the ray's end, marking FREE. An
        OCCUPIED cell is never overwritten: a wall seen once should not be
        erased by a later ray that grazes past it."""
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        for _ in range(dr + dc + 2):
            if (r, c) == (r1, c1):
                break
            if self._in_bounds(r, c) and self.grid[r, c] == UNKNOWN:
                self.grid[r, c] = FREE
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def guard_fov_deg(self) -> float:
        return getattr(self, "_fov_deg", 90.0)

    def set_fov(self, fov_deg: float) -> None:
        self._fov_deg = float(fov_deg)

    # -- planning ------------------------------------------------------- #
    def _clearance_from_mask(self, not_wall: np.ndarray, robot_radius: float) -> np.ndarray:
        """Shared distance-transform core for `clearance` (mask = FREE only)
        and the `allow_unknown` paths in `astar`/`frontier_points` (mask =
        not OCCUPIED, i.e. FREE-or-UNKNOWN) -- factored out so both stay
        exactly consistent instead of two near-identical inline copies."""
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(not_wall) * self.res - robot_radius

    def clearance(self, robot_radius: float = 0.25) -> np.ndarray:
        """Metres of free space around each cell, already netting off the
        robot radius, so `clearance > 0` is exactly where the centre may go.
        UNKNOWN counts as blocked -- planning a route through cells the rover
        has never observed is how you drive into a wall it never saw."""
        return self._clearance_from_mask(self.grid == FREE, robot_radius)

    def stats(self) -> dict:
        total = self.grid.size
        return {"free": float((self.grid == FREE).mean()),
                "occupied": float((self.grid == OCCUPIED).mean()),
                "unknown": float((self.grid == UNKNOWN).mean()),
                "free_cells": int((self.grid == FREE).sum()), "cells": int(total)}

    def nearest_navigable(self, x: float, y: float, clear: np.ndarray,
                          max_search_m: float = 2.0) -> Optional[Tuple[int, int]]:
        """The closest cell the robot centre may occupy. The recalled position
        is the object's own centroid, which is by definition inside an
        obstacle -- the rover has to stand NEXT to a table, not in it."""
        r0, c0 = self.to_cell(x, y)
        nav = clear > 0
        rad = int(max_search_m / self.res)
        best, best_d = None, None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r, c = r0 + dr, c0 + dc
                if not self._in_bounds(r, c) or not nav[r, c]:
                    continue
                d = dr * dr + dc * dc
                if best_d is None or d < best_d:
                    best, best_d = (r, c), d
        return best

    def frontier_points(self, current_xy, robot_radius: float = 0.25,
                        max_radius: float = 6.0, k: int = 3) -> List[Tuple[float, float]]:
        """Up to `k` reachable points on the boundary between what's been
        seen and what hasn't, spread apart, within `max_radius` of
        `current_xy` -- targets for "walk a bit further before giving up",
        not a full frontier-exploration planner. A cell counts as frontier
        if it's navigable (FREE with clearance for `robot_radius`) and has
        at least one 8-neighbour that's still UNKNOWN -- the boundary the
        rover could actually walk up to and see past.

        Selection is greedy farthest-point sampling seeded at the frontier
        cell nearest `current_xy` (so the first target is a cheap, close
        one), then each subsequent point maximizes distance to already-
        picked points -- spreads the k targets around instead of clustering
        them on one wall. Returned nearest-first by distance from
        `current_xy` so a caller visiting them in order doesn't backtrack
        more than necessary.

        Deliberately NOT `self.clearance()` for the navigability test: that
        method's distance-transform runs over `grid == FREE` only, crediting
        neither OCCUPIED nor UNKNOWN as space the robot could occupy --
        correct for A* (a route must stay inside CONFIRMED-safe cells), but
        it means every cell adjacent to UNKNOWN (which is exactly what a
        frontier cell is, by definition) reads as having artificially low
        clearance, since the unknown space beyond it isn't credited as
        possibly-free. Measured: on a real cluttered-room grid, this
        silently returned ZERO frontier points at the default
        `robot_radius=0.25` even though the free region was large --
        dropping to `robot_radius=0.05` (no real margin at all) recovered 3.
        Fixed here the same way `astar`'s `allow_unknown` treats the
        planning problem: clearance is measured against `grid == OCCUPIED`
        only (FREE and UNKNOWN both count as "not a confirmed wall"), so a
        frontier cell isn't penalized for what hasn't been seen yet on its
        far side."""
        clear = self._clearance_from_mask(self.grid != OCCUPIED, robot_radius)
        nav = (self.grid == FREE) & (clear > 0)
        unknown = self.grid == UNKNOWN

        # 8-neighbour dilation of `unknown` via array shifts (no SciPy dep
        # needed for a one-cell dilation).
        adj_unknown = np.zeros_like(unknown)
        for dr, dc in _NEIGHBOURS:
            shifted = np.zeros_like(unknown)
            r_src = slice(max(0, -dr), unknown.shape[0] - max(0, dr))
            c_src = slice(max(0, -dc), unknown.shape[1] - max(0, dc))
            r_dst = slice(max(0, dr), unknown.shape[0] - max(0, -dr))
            c_dst = slice(max(0, dc), unknown.shape[1] - max(0, -dc))
            shifted[r_dst, c_dst] = unknown[r_src, c_src]
            adj_unknown |= shifted

        frontier_mask = nav & adj_unknown
        rows, cols = np.nonzero(frontier_mask)
        if rows.size == 0:
            return []

        pts = np.array([self.to_world(r, c) for r, c in zip(rows, cols)])
        cur = np.array(current_xy, dtype=np.float64)
        d_to_cur = np.linalg.norm(pts - cur, axis=1)
        in_range = d_to_cur <= max_radius
        if not in_range.any():
            return []
        pts, d_to_cur = pts[in_range], d_to_cur[in_range]

        chosen = [pts[int(np.argmin(d_to_cur))]]
        remaining = np.ones(len(pts), dtype=bool)
        remaining[int(np.argmin(d_to_cur))] = False
        while len(chosen) < k and remaining.any():
            rem_pts = pts[remaining]
            min_d_to_chosen = np.min(
                [np.linalg.norm(rem_pts - c, axis=1) for c in chosen], axis=0)
            pick_local = int(np.argmax(min_d_to_chosen))
            pick_global = np.nonzero(remaining)[0][pick_local]
            chosen.append(pts[pick_global])
            remaining[pick_global] = False

        chosen.sort(key=lambda p: float(np.linalg.norm(np.array(p) - cur)))
        return [(float(p[0]), float(p[1])) for p in chosen]

    def astar(self, start_xy, goal_xy, robot_radius: float = 0.25,
              clearance_ref: float = 0.5, clearance_weight: float = 4.0,
              allow_unknown: bool = False, unknown_penalty: float = 3.0,
              nav_search_m: float = 2.0):
        """Least-cost route, biased toward the middle of free space.

        nav_search_m: how far `nearest_navigable` may search around start/
        goal for a cell to actually anchor on (see its own docstring).
        Widened from a hardcoded 2.0 after a real "no navigable cell nearby"
        failure (2026-09-05, MP3D scene): both the rover's own final pose
        AND the recalled position it needed to reach (only 0.48m apart) sat
        inside a real, DENSE pocket of OCCUPIED cells -- clearance measured
        exactly -robot_radius (i.e. 0 raw distance-to-non-occupied) in every
        sampled cell out to the old 2.0m radius. Not a bug: the rover had
        genuinely wedged itself among close furniture (matches the AVOID-
        heavy state counts logged the same run). But a fixed 2.0m gives up
        entirely rather than searching a little further for the nearest real
        opening -- exposed as a parameter (default unchanged) so a caller
        whose start/goal legitimately sits in a tight pocket can widen it
        instead of falling straight to blind dead-reckon.

        Same formulation as the project's existing HM3D planner
        (thirdparty/safediffuser/diffuser/hm3d/planner.py::astar): 8-connected,
        step cost scaled by a penalty that rises as clearance falls below
        clearance_ref, Euclidean heuristic (admissible because the penalty is
        >= 1 everywhere). Hugging a shortest path would shave corners, which
        is where the guard's hard-stop lives.

        allow_unknown: cells the rover has never observed are normally
        refused (see `clearance`'s docstring) -- correct for a route that's
        supposed to stay inside what's already confirmed safe, but it also
        means "go back to B" routinely comes back with no route at all,
        since B by definition usually isn't anywhere leg 1 walked. When
        True, UNKNOWN cells (not OCCUPIED -- a real wall is still a wall)
        are also passable, at a flat `unknown_penalty` step cost (there's no
        real clearance estimate to scale by since nothing was observed
        there). This only changes which route A* proposes; the live
        obstacle guard still runs every tick during execution and reacts to
        whatever is actually there, same as always. Off by default so every
        existing caller is unaffected -- callers wanting a route-of-last-
        resort call this a second time with it True after a strict (False)
        attempt returns no route.

        Also affects the base `clear` used for FREE cells, not just literally-
        unknown ones: `clearance()`'s distance-transform runs over `grid ==
        FREE` only, so a genuinely FREE cell near an unknown boundary reads
        as low-clearance too (the transform can't credit space it doesn't
        know is open) -- the same issue `frontier_points` has, and it means
        the approach to a frontier can fail this test well before reaching
        any cell that's actually UNKNOWN. When `allow_unknown`, `clear` is
        computed against `grid != OCCUPIED` (matching `frontier_points`) so
        FREE cells aren't penalized for unmapped space beyond them."""
        clear = (self._clearance_from_mask(self.grid != OCCUPIED, robot_radius)
                if allow_unknown else self.clearance(robot_radius))
        penalty = 1.0 + clearance_weight * np.clip(
            (clearance_ref - clear) / clearance_ref, 0.0, 1.0)

        if allow_unknown:
            unknown = self.grid == UNKNOWN
            # synthetic positive clearance just so `nearest_navigable`'s
            # `clear > 0` test and this method's own `nav` mask admit these
            # cells too; the real cost signal is `penalty`, set separately.
            clear_for_nav = np.where(unknown, np.maximum(clear, 0.01), clear)
            penalty = np.where(unknown, unknown_penalty, penalty)
        else:
            clear_for_nav = clear

        nav = clear_for_nav > 0
        start = self.nearest_navigable(start_xy[0], start_xy[1], clear_for_nav, max_search_m=nav_search_m)
        goal = self.nearest_navigable(goal_xy[0], goal_xy[1], clear_for_nav, max_search_m=nav_search_m)
        if start is None or goal is None:
            return None, {"reason": "start or goal has no navigable cell nearby",
                          "start_cell": start, "goal_cell": goal}

        goal_arr = np.array(goal, dtype=float)

        def h(rc):
            return float(np.hypot(*(np.array(rc, dtype=float) - goal_arr))) * self.res

        open_heap = [(h(start), 0.0, start)]
        came, g_score, closed = {}, {start: 0.0}, set()
        found = False
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                found = True
                break
            closed.add(cur)
            r, c = cur
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc) or not nav[nr, nc]:
                    continue
                step = self.res * (1.41421356 if dr and dc else 1.0)
                ng = g + step * 0.5 * (penalty[r, c] + penalty[nr, nc])
                if ng < g_score.get((nr, nc), np.inf):
                    g_score[(nr, nc)] = ng
                    came[(nr, nc)] = cur
                    heapq.heappush(open_heap, (ng + h((nr, nc)), ng, (nr, nc)))
        if not found:
            return None, {"reason": "no route through observed free space",
                          "start_cell": start, "goal_cell": goal,
                          "navigable_cells": int(nav.sum())}

        cells = [goal]
        while cells[-1] != start:
            cells.append(came[cells[-1]])
        cells.reverse()
        path = [self.to_world(r, c) for r, c in cells]
        length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                     for i in range(len(path) - 1))
        return path, {"cells": len(cells), "length_m": length,
                      "goal_cell": goal, "start_cell": start}


def thin_waypoints(path: List[Tuple[float, float]], spacing: float = 0.6
                   ) -> List[Tuple[float, float]]:
    """Cell-by-cell A* output is far too dense to steer to -- consecutive
    points are 0.1 m apart, so the bearing to the 'next' one is noise. Keep
    points roughly `spacing` apart, always keeping the last."""
    if not path:
        return []
    out = [path[0]]
    for p in path[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= spacing:
            out.append(p)
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out
