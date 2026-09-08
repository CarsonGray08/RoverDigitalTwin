"""Gravity-frame floor-height estimation.

In the twin frame the up-axis is +Z (the fixed ``M_axis`` swap makes it
gravity-aligned), so the floor-plane fit the plan calls for reduces to a robust
estimate of the floor **height** ``z0`` near the rover: the normal is already
known (gravity), only the offset has to be solved. This decouples the estimate
from world-pose error during the startup 360.

Robustness strategy (rejecting rover body, walls, clutter, glitches):

- Consider only points in a horizontal annulus around the rover: inside
  ``exclude_radius_m`` is the rover's own body/wheels; beyond ``radius_m`` is too
  far to trust for the local support surface.
- Histogram the candidate Z values and take the **lowest dense cluster** (the
  floor is the lowest dominant horizontal surface; tables/clutter sit above).
  A sparse band below the floor — a depth glitch or spike — never reaches the
  cluster threshold, so it is rejected rather than dragging the estimate down.
- Report the median Z within that cluster (median, not min, absorbs noise).

``FloorEstimator`` accumulates measurements with an EMA so ``SupportGround`` can
be moved *slowly* toward the estimate, and tracks whether the estimate has
settled (the "stable after the startup 360" acceptance criterion).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class FloorMeasurement:
    height_m: float
    support_points: int
    # Points that fell below the chosen floor cluster (sparse low outliers that
    # were rejected). High counts hint the radius/threshold need tuning.
    rejected_below: int


def estimate_floor_height(
    points: Iterable[Point3],
    rover_xy: tuple[float, float],
    *,
    radius_m: float = 2.0,
    exclude_radius_m: float = 0.15,
    bin_size_m: float = 0.02,
    min_cluster_points: int = 20,
) -> FloorMeasurement | None:
    """Robustly estimate the floor height near ``rover_xy``.

    Returns ``None`` when there is not enough evidence (the planner keeps the
    region unknown rather than inventing a floor).
    """

    if bin_size_m <= 0.0:
        raise ValueError("bin_size_m must be positive")
    if exclude_radius_m < 0.0 or radius_m <= exclude_radius_m:
        raise ValueError("require 0 <= exclude_radius_m < radius_m")

    rx, ry = rover_xy
    inner_sq = exclude_radius_m * exclude_radius_m
    outer_sq = radius_m * radius_m
    zs: list[float] = []
    for x, y, z in points:
        dx, dy = x - rx, y - ry
        dist_sq = dx * dx + dy * dy
        if inner_sq <= dist_sq <= outer_sq:
            zs.append(z)

    if len(zs) < min_cluster_points:
        return None

    # Histogram by bin index. Use a 3-bin sliding window so a floor split across
    # adjacent bins still clears the threshold.
    counts: dict[int, int] = {}
    for z in zs:
        b = int(math.floor(z / bin_size_m))
        counts[b] = counts.get(b, 0) + 1

    bins = sorted(counts)
    chosen: int | None = None
    for b in bins:
        window = counts.get(b - 1, 0) + counts.get(b, 0) + counts.get(b + 1, 0)
        if window >= min_cluster_points:
            chosen = b
            break
    if chosen is None:
        return None

    # Points within the chosen 3-bin window define the floor; median is the
    # height. Everything strictly below the window is a rejected low outlier.
    lo = (chosen - 1) * bin_size_m
    hi = (chosen + 2) * bin_size_m
    cluster = [z for z in zs if lo <= z < hi]
    rejected_below = sum(1 for z in zs if z < lo)
    if not cluster:
        return None
    return FloorMeasurement(
        height_m=statistics.median(cluster),
        support_points=len(cluster),
        rejected_below=rejected_below,
    )


@dataclass
class FloorEstimator:
    """Accumulates floor-height measurements into a slowly-moving estimate.

    ``update`` blends each new robust measurement into an EMA (so the physics
    ``SupportGround`` is nudged, not snapped) and tracks stability.
    """

    radius_m: float = 2.0
    exclude_radius_m: float = 0.15
    bin_size_m: float = 0.02
    min_cluster_points: int = 20
    # EMA weight for each new measurement (0..1). Smaller = slower/steadier.
    smoothing: float = 0.25
    # The estimate is "stable" when the last ``stable_window`` blended values
    # span no more than ``stable_tol_m``.
    stable_tol_m: float = 0.01
    stable_window: int = 5

    height_m: float | None = None
    samples: int = 0
    last_measurement: FloorMeasurement | None = None
    _recent: list[float] = field(default_factory=list)

    def measure(self, points: Iterable[Point3], rover_xy: tuple[float, float]) -> FloorMeasurement | None:
        return estimate_floor_height(
            points,
            rover_xy,
            radius_m=self.radius_m,
            exclude_radius_m=self.exclude_radius_m,
            bin_size_m=self.bin_size_m,
            min_cluster_points=self.min_cluster_points,
        )

    def update(self, points: Iterable[Point3], rover_xy: tuple[float, float]) -> float | None:
        measurement = self.measure(points, rover_xy)
        if measurement is None:
            return self.height_m  # no new evidence; hold the current estimate
        self.last_measurement = measurement
        if self.height_m is None:
            self.height_m = measurement.height_m
        else:
            self.height_m = (1.0 - self.smoothing) * self.height_m + self.smoothing * measurement.height_m
        self.samples += 1
        self._recent.append(self.height_m)
        if len(self._recent) > self.stable_window:
            self._recent.pop(0)
        return self.height_m

    @property
    def has_estimate(self) -> bool:
        return self.height_m is not None

    @property
    def is_stable(self) -> bool:
        if len(self._recent) < self.stable_window:
            return False
        return (max(self._recent) - min(self._recent)) <= self.stable_tol_m

    def status(self) -> dict:
        return {
            "has_estimate": self.has_estimate,
            "height_m": None if self.height_m is None else round(self.height_m, 4),
            "samples": self.samples,
            "is_stable": self.is_stable,
            "support_points": None if self.last_measurement is None else self.last_measurement.support_points,
        }
