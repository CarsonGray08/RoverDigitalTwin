"""Startup 360 scan: numeric coverage gate + in-place spin plan.

On startup the rover rotates slowly in place to seed the local map before any
navigation is allowed. Navigation stays disabled until a *numeric* coverage gate
passes: the fraction of grid cells within radius ``R`` of the rover that have
been observed (are "known") must exceed a configured threshold. The gate latches
once complete so driving into fresh, partially-unknown space later does not
re-disable navigation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from digital_twin.models import CommandName, CommandStep

Cell = tuple[int, int]


@dataclass(frozen=True)
class CoverageResult:
    fraction: float
    known_cells: int
    total_cells: int


def cells_within_radius(rover_xy: tuple[float, float], radius_m: float, cell_m: float) -> Iterable[Cell]:
    """Yield cell indices whose center lies within ``radius_m`` of the rover.

    Cell indexing matches ``live_scan.rasterize_xy_points`` (``floor(x / cell)``),
    so the known-cell set and this disk use one coordinate system.
    """

    if cell_m <= 0.0:
        raise ValueError("cell_m must be positive")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    rx, ry = rover_xy
    i_min = math.floor((rx - radius_m) / cell_m)
    i_max = math.floor((rx + radius_m) / cell_m)
    j_min = math.floor((ry - radius_m) / cell_m)
    j_max = math.floor((ry + radius_m) / cell_m)
    radius_sq = radius_m * radius_m
    for i in range(i_min, i_max + 1):
        cx = (i + 0.5) * cell_m
        for j in range(j_min, j_max + 1):
            cy = (j + 0.5) * cell_m
            if (cx - rx) ** 2 + (cy - ry) ** 2 <= radius_sq:
                yield (i, j)


def coverage_fraction(
    known_cells: Iterable[Cell],
    rover_xy: tuple[float, float],
    radius_m: float,
    cell_m: float,
    exclusion_radius_m: float = 0.0,
) -> CoverageResult:
    """Fraction of directly observed cells in a rover-centred annulus.

    ``exclusion_radius_m`` is the expected sensor/body dead zone.  It is
    excluded from both numerator and denominator, never silently counted as
    observed coverage.
    """

    known = set(known_cells)
    total = 0
    covered = 0
    rx, ry = rover_xy
    for cell in cells_within_radius(rover_xy, radius_m, cell_m):
        cx, cy = (cell[0] + 0.5) * cell_m, (cell[1] + 0.5) * cell_m
        if (cx - rx) ** 2 + (cy - ry) ** 2 < exclusion_radius_m ** 2:
            continue
        total += 1
        if cell in known:
            covered += 1
    fraction = covered / total if total else 0.0
    return CoverageResult(fraction=fraction, known_cells=covered, total_cells=total)


@dataclass
class BootstrapGate:
    """Latching gate that enables navigation once startup coverage is reached."""

    radius_m: float = 1.5
    cell_m: float = 0.30
    coverage_threshold: float = 0.6
    exclusion_radius_m: float = 0.45

    complete: bool = False
    last_fraction: float = 0.0
    last_total: int = 0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0 or self.cell_m <= 0.0:
            raise ValueError("coverage radius and cell size must be positive")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage threshold must be in [0, 1]")
        if not 0.0 <= self.exclusion_radius_m < self.radius_m:
            raise ValueError("exclusion radius must be non-negative and below coverage radius")

    def update(self, known_cells: Iterable[Cell], rover_xy: tuple[float, float]) -> bool:
        result = coverage_fraction(
            known_cells, rover_xy, self.radius_m, self.cell_m, self.exclusion_radius_m)
        self.last_fraction = result.fraction
        self.last_total = result.total_cells
        if not self.complete and result.total_cells > 0 and result.fraction >= self.coverage_threshold:
            self.complete = True
        return self.complete

    @property
    def navigation_enabled(self) -> bool:
        return self.complete

    def status(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "navigation_enabled": self.navigation_enabled,
            "coverage_fraction": round(self.last_fraction, 4),
            "coverage_threshold": self.coverage_threshold,
            "coverage_radius_m": self.radius_m,
            "coverage_annulus_inner_radius_m": self.exclusion_radius_m,
            "cells_considered": self.last_total,
        }


def navigation_blocked_by_bootstrap(observation_payload: dict[str, Any]) -> bool:
    """True only when an observation carries a bootstrap block that explicitly
    disables navigation. Absent/zeroed blocks do not block (backward-safe)."""

    bootstrap = observation_payload.get("bootstrap")
    if not isinstance(bootstrap, dict):
        return False
    if not bootstrap.get("enabled", False):
        return False
    return not bootstrap.get("navigation_enabled", True)


def spin_in_place_plan(
    *,
    total_deg: float = 360.0,
    step_deg: float = 30.0,
    speed: int = 2,
    turn: CommandName = CommandName.LEFT,
    settle: bool = True,
    degrees_to_distance=None,
) -> tuple[CommandStep, ...]:
    """Build a slow in-place rotation as a sequence of small turn steps.

    The bootstrap spins slowly so ARKit tracking initializes and keyframes are
    collected at many headings. ``degrees_to_distance`` converts a per-step yaw
    into the rover's command ``distance`` units. Without a converter, the numeric
    distance is in degrees; physical commands need a calibrated converter.
    A ``STOP`` is interleaved between turns (``settle``) so each
    viewpoint is captured without motion blur.
    """

    if total_deg <= 0.0 or step_deg <= 0.0:
        raise ValueError("total_deg and step_deg must be positive")
    steps = max(1, math.ceil(total_deg / step_deg))
    per_step = total_deg / steps
    convert = degrees_to_distance or (lambda deg: deg)
    plan: list[CommandStep] = []
    for _ in range(steps):
        plan.append(CommandStep(turn, speed, int(round(convert(per_step)))))
        if settle:
            plan.append(CommandStep(CommandName.STOP, 0, 0))
    return tuple(plan)
