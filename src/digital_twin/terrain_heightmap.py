"""Terrain height lookup for belly-clearance, pure numpy.

Belly clearance is the gap between the lowest point of the chassis and the
terrain directly beneath it. The robust way to get the terrain height under the
rover, without a PhysX scene query (whose signature is finicky on this Isaac
build) or self-hit filtering, is to read it straight from the geometry we
already author: the nvblox tile vertices. This bins them into a 2.5-D max-height
grid and, for a rover footprint, returns the HIGHEST terrain point under it --
the thing the rover could belly out on. The validator then computes
``clearance = belly_z - terrain_height_under_footprint``.

Pure numpy so it is unit-testable off-GPU/off-Isaac; the validator builds it from
the same tile vertices it authors as colliders.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroundSupport:
    """A footprint-height result with explicit provenance.

    ``inferred_support`` is a local physics aid only.  It is never inserted into
    the heightmap and callers must not treat it as traversable map evidence.
    """

    source: str  # observed | inferred_support | unknown
    height_m: float | None
    evidence_cells: int = 0


class TerrainHeightmap:
    """A 2.5-D max-height grid over terrain points (twin frame, Z up)."""

    def __init__(self, cell_m: float, min_x: float, min_y: float, grid: np.ndarray) -> None:
        self.cell_m = float(cell_m)
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.grid = grid  # (nx, ny) float32, -inf where empty

    @property
    def is_empty(self) -> bool:
        return self.grid.size == 0

    @classmethod
    def from_points(cls, points, cell_m: float = 0.1) -> "TerrainHeightmap":
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3 or cell_m <= 0:
            return cls(cell_m if cell_m > 0 else 0.1, 0.0, 0.0, np.zeros((0, 0), np.float32))
        mn = pts[:, :2].min(axis=0)
        ij = np.floor((pts[:, :2] - mn) / cell_m).astype(np.int64)
        nx = int(ij[:, 0].max()) + 1
        ny = int(ij[:, 1].max()) + 1
        grid = np.full((nx, ny), -np.inf, dtype=np.float64)
        np.maximum.at(grid, (ij[:, 0], ij[:, 1]), pts[:, 2])
        return cls(cell_m, float(mn[0]), float(mn[1]), grid.astype(np.float32))

    def max_height_in_footprint(self, x: float, y: float, radius: float) -> float | None:
        """Highest terrain z within ``radius`` (m) of ``(x, y)``, or ``None`` if no
        terrain has been observed in that footprint."""
        if self.is_empty:
            return None
        nx, ny = self.grid.shape
        i0 = int(np.floor((x - radius - self.min_x) / self.cell_m))
        i1 = int(np.floor((x + radius - self.min_x) / self.cell_m))
        j0 = int(np.floor((y - radius - self.min_y) / self.cell_m))
        j1 = int(np.floor((y + radius - self.min_y) / self.cell_m))
        i0, i1 = max(0, i0), min(nx - 1, i1)
        j0, j1 = max(0, j0), min(ny - 1, j1)
        if i0 > i1 or j0 > j1:
            return None
        sub = self.grid[i0:i1 + 1, j0:j1 + 1]
        finite = sub[np.isfinite(sub)]
        return float(finite.max()) if finite.size else None

    def observed_cells(self, cell_m: float) -> set[tuple[int, int]]:
        """Return cells with direct terrain evidence in a caller grid."""
        if self.is_empty or cell_m <= 0:
            return set()
        ii, jj = np.where(np.isfinite(self.grid))
        xs = self.min_x + (ii + 0.5) * self.cell_m
        ys = self.min_y + (jj + 0.5) * self.cell_m
        return set(zip(np.floor(xs / cell_m).astype(int), np.floor(ys / cell_m).astype(int)))

    def ground_support(
        self,
        x: float,
        y: float,
        footprint_radius_m: float,
        *,
        inference_ring_outer_m: float | None = None,
        min_ring_cells: int = 8,
    ) -> GroundSupport:
        """Return observed ground or a robust *local* ring estimate.

        The query never fills the global map.  A median/MAD-filtered ring avoids
        selecting an isolated rock or wall as the provisional floor height.
        """
        observed = self.max_height_in_footprint(x, y, footprint_radius_m)
        if observed is not None:
            return GroundSupport("observed", observed)
        if self.is_empty:
            return GroundSupport("unknown", None)
        outer = inference_ring_outer_m or max(footprint_radius_m * 2.0, footprint_radius_m + 0.35)
        if outer <= footprint_radius_m:
            raise ValueError("inference ring must extend beyond the footprint")
        ii, jj = np.where(np.isfinite(self.grid))
        if not len(ii):
            return GroundSupport("unknown", None)
        xs = self.min_x + (ii + 0.5) * self.cell_m
        ys = self.min_y + (jj + 0.5) * self.cell_m
        distances = np.hypot(xs - x, ys - y)
        values = self.grid[ii, jj].astype(np.float64)
        ring = values[(distances > footprint_radius_m) & (distances <= outer)]
        if len(ring) < min_ring_cells:
            return GroundSupport("unknown", None, int(len(ring)))
        median = float(np.median(ring))
        mad = float(np.median(np.abs(ring - median)))
        if mad > 0.0:
            ring = ring[np.abs(ring - median) <= max(0.03, 3.0 * mad)]
        if len(ring) < min_ring_cells:
            return GroundSupport("unknown", None, int(len(ring)))
        return GroundSupport("inferred_support", float(np.median(ring)), int(len(ring)))
