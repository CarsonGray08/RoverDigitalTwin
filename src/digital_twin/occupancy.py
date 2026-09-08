"""Occupancy / traversability grid.

Cell states (planner semantics):

- ``UNKNOWN``  no evidence -> unsafe unless explicitly exploring.
- ``FREE``     ray-carved empty space -> drivable but not floor-confirmed.
- ``FLOOR``    floor-confirmed -> traversable.
- ``OCCUPIED`` obstacle evidence -> blocked.

Classification is **relative to the fitted floor plane**, not an absolute world-Z
band. After ``M_axis`` the twin is gravity-aligned, so the floor is usually level
(normal +Z), but a slight slope is handled by carrying the plane normal: the
height used for classification is the signed distance to the plane,
``n . p - offset``. This also handles floors whose elevation is not Z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

Cell = tuple[int, int]
Point3 = tuple[float, float, float]


class CellState(Enum):
    UNKNOWN = "unknown"
    FREE = "free"
    FLOOR = "floor"
    OCCUPIED = "occupied"


@dataclass(frozen=True)
class FloorPlane:
    """Floor as a plane ``{p : normal . p = offset}`` (unit normal)."""

    normal: Point3 = (0.0, 0.0, 1.0)
    offset: float = 0.0

    @classmethod
    def level(cls, height_m: float) -> "FloorPlane":
        """A level floor at the given twin-Z height (the common case)."""
        return cls(normal=(0.0, 0.0, 1.0), offset=height_m)

    def height_above(self, point: Point3) -> float:
        x, y, z = point
        nx, ny, nz = self.normal
        return nx * x + ny * y + nz * z - self.offset


@dataclass(frozen=True)
class OccupancyConfig:
    cell_m: float = 0.30
    floor_band_m: float = 0.05      # within +/- this of the plane -> floor
    obstacle_min_m: float = 0.10    # at least this above the plane -> obstacle
    obstacle_max_m: float = 2.0     # ignore geometry above this (ceilings)
    min_obstacle_points: int = 3    # reject single-point obstacle noise
    min_floor_points: int = 1
    proxy_height_m: float = 0.5     # height of generated obstacle proxy boxes


@dataclass(frozen=True)
class ObstacleProxy:
    """Axis-aligned box proxy for Isaac validation collisions."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


@dataclass
class OccupancyGrid:
    floor: FloorPlane = field(default_factory=FloorPlane)
    config: OccupancyConfig = field(default_factory=OccupancyConfig)

    _floor_counts: dict[Cell, int] = field(default_factory=dict)
    _obstacle_counts: dict[Cell, int] = field(default_factory=dict)
    _free: set[Cell] = field(default_factory=set)

    def cell_of(self, x: float, y: float) -> Cell:
        c = self.config.cell_m
        return (math.floor(x / c), math.floor(y / c))

    def ingest_points(self, points: Iterable[Point3]) -> None:
        cfg = self.config
        for p in points:
            h = self.floor.height_above(p)
            cell = self.cell_of(p[0], p[1])
            if abs(h) <= cfg.floor_band_m:
                self._floor_counts[cell] = self._floor_counts.get(cell, 0) + 1
            elif cfg.obstacle_min_m <= h <= cfg.obstacle_max_m:
                self._obstacle_counts[cell] = self._obstacle_counts.get(cell, 0) + 1
            # Points in the gap (floor_band, obstacle_min) or below the floor or
            # above obstacle_max are ambiguous and contribute no evidence.

    def carve_ray(self, origin_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
        """Mark cells along a depth ray (exclusive of the endpoint cell) FREE.

        Free-space carving: the segment from the sensor to a measured hit is
        empty up to the hit. Does not override floor/obstacle evidence (applied
        at classify time). The endpoint cell is the hit, so it is excluded.
        """

        for cell in self._supercover_cells(origin_xy, end_xy):
            self._free.add(cell)

    def state(self, cell: Cell) -> CellState:
        cfg = self.config
        if self._obstacle_counts.get(cell, 0) >= cfg.min_obstacle_points:
            return CellState.OCCUPIED
        if self._floor_counts.get(cell, 0) >= cfg.min_floor_points:
            return CellState.FLOOR
        if cell in self._free:
            return CellState.FREE
        return CellState.UNKNOWN

    def classify(self) -> dict[Cell, CellState]:
        cells = set(self._floor_counts) | set(self._obstacle_counts) | self._free
        result: dict[Cell, CellState] = {}
        for cell in cells:
            st = self.state(cell)
            if st is not CellState.UNKNOWN:
                result[cell] = st
        return result

    def occupied_cells(self) -> set[Cell]:
        return {cell for cell in self._obstacle_counts if self.state(cell) is CellState.OCCUPIED}

    def floor_cells(self) -> set[Cell]:
        return {cell for cell in self._floor_counts if self.state(cell) is CellState.FLOOR}

    def traversable_cells(self) -> set[Cell]:
        """Floor-confirmed plus ray-carved free cells (not blocked)."""
        classified = self.classify()
        return {cell for cell, st in classified.items() if st in (CellState.FLOOR, CellState.FREE)}

    def known_cells(self) -> set[Cell]:
        """Every cell with positive evidence (for coverage/bootstrap)."""
        return set(self.classify())

    def counts(self) -> dict[str, int]:
        classified = self.classify()
        out = {state.value: 0 for state in CellState if state is not CellState.UNKNOWN}
        for st in classified.values():
            out[st.value] += 1
        return out

    def obstacle_proxies(self) -> list[ObstacleProxy]:
        """Merge occupied cells into per-row boxes (fewer, larger proxies)."""
        cfg = self.config
        c = cfg.cell_m
        floor_z = self.floor.offset  # level-floor top; good enough for collisions
        min_z = floor_z
        max_z = floor_z + cfg.proxy_height_m

        by_row: dict[int, list[int]] = {}
        for (i, j) in self.occupied_cells():
            by_row.setdefault(j, []).append(i)

        proxies: list[ObstacleProxy] = []
        for j, cols in by_row.items():
            cols.sort()
            run_start = cols[0]
            prev = cols[0]
            for i in cols[1:] + [None]:
                if i is not None and i == prev + 1:
                    prev = i
                    continue
                proxies.append(
                    ObstacleProxy(
                        min_x=run_start * c,
                        max_x=(prev + 1) * c,
                        min_y=j * c,
                        max_y=(j + 1) * c,
                        min_z=min_z,
                        max_z=max_z,
                    )
                )
                if i is not None:
                    run_start = i
                    prev = i
        return proxies

    # -- internal -----------------------------------------------------------

    def _supercover_cells(self, origin_xy: tuple[float, float], end_xy: tuple[float, float]) -> list[Cell]:
        """Cells a segment passes through, excluding the endpoint cell."""
        c = self.config.cell_m
        start = self.cell_of(*origin_xy)
        end = self.cell_of(*end_xy)
        (x0, y0), (x1, y1) = origin_xy, end_xy
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        if steps == 0:
            return []
        cells: list[Cell] = []
        seen: set[Cell] = set()
        for s in range(steps + 1):
            t = s / steps
            cell = self.cell_of(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if cell == end:
                break
            if cell not in seen:
                seen.add(cell)
                cells.append(cell)
        return cells
