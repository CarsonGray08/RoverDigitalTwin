"""Map fidelity metrics: floor flatness, duplicate floor sheets, the seam
where a scan comes back to its start, and the size of known objects.

Every function scores a surface point set, so one implementation measures any
map on equal terms: the elevation grid (cell centres) or the visual mesh.
Heights are taken from a floor plane fitted to the map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Floor:
    normal: np.ndarray
    offset: float

    def height(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal + self.offset


def fit_floor(points: np.ndarray, band_m: float = 0.05, iterations: int = 8) -> tuple[Floor, np.ndarray]:
    """The dominant low plane and the mask of points within ``band_m`` of it.

    Seeded from the lowest third of the map, then refit on the band around the
    current plane, so walls, boxes and a stray second floor sheet don't tilt
    it; a duplicated sheet inside the band shows up as error instead of being
    averaged into the plane.
    """
    z = points[:, 2]
    mask = z <= np.percentile(z, 35) + band_m
    floor = _plane(points[mask])
    for _ in range(iterations):
        new = np.abs(floor.height(points)) <= band_m
        if new.sum() < 3 or np.array_equal(new, mask):
            break
        mask = new
        floor = _plane(points[mask])
    return floor, mask


def floor_rms(points: np.ndarray, floor: Floor, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(floor.height(points[mask]) ** 2)))


def floor_step(points: np.ndarray, floor: Floor, mask: np.ndarray, cell_m: float = 0.05) -> float:
    """Height discontinuity of the floor, the duplicate-sheet metric: the p99 of
    the steps between neighbouring cells' median heights and within cells
    (p90 - p10, two stacked sheets). A duplicated sheet covers whole regions;
    p99 rather than the maximum keeps a handful of noisy cells from deciding."""
    h = floor.height(points[mask])
    ij = np.floor(points[mask, :2] / cell_m).astype(np.int64)
    ij -= ij.min(axis=0)
    cells, inverse = np.unique(ij, axis=0, return_inverse=True)
    q10, q50, q90 = _group_quantiles(inverse.ravel(), h, (0.1, 0.5, 0.9), len(cells))
    grid = np.full(cells.max(axis=0) + 1, np.nan)
    grid[cells[:, 0], cells[:, 1]] = q50
    steps = np.concatenate([q90 - q10, *(np.abs(np.diff(grid, axis=a)).ravel() for a in (0, 1))])
    return float(np.nanpercentile(steps, 99))


def seam(points: np.ndarray, frames: np.ndarray, fraction: float = 0.2, voxel_m: float = 0.2) -> float | None:
    """How far apart the surfaces seen at the start and at the end of a scan are.

    Each sample of one end is measured against the other end's surface along
    its normal (point-to-plane, so it does not depend on how densely either
    side was sampled, and a surface merely sliding along itself is no seam).
    The signed distances are reduced to a median per ``voxel_m`` voxel, which
    averages depth noise away and leaves the misregistration; the seam is the
    p95 over voxels. ``None`` if start and end never overlap.
    """
    lo, hi = frames.min(), frames.max()
    span = hi - lo + 1
    early, late = points[frames < lo + fraction * span], points[frames > hi - fraction * span]
    if min(len(early), len(late)) < 50:
        return None
    distances, where = [], []
    for source, target in ((late, early), (early, late)):
        d, q = _point_to_plane(source, target)
        distances.append(d)
        where.append(q)
    d, q = np.concatenate(distances), np.concatenate(where)
    if len(d) < 50:
        return None
    _, voxel, counts = np.unique(np.floor(q / voxel_m).astype(np.int64), axis=0, return_inverse=True,
                                 return_counts=True)
    voxel = voxel.ravel()
    (median,) = _group_quantiles(voxel, d, (0.5,), len(counts))
    median = median[counts >= 30]
    return float(np.percentile(np.abs(median), 95)) if len(median) else None


def _point_to_plane(source: np.ndarray, target: np.ndarray, max_distance: float = 0.3, k: int = 16):
    """Signed distance of each source point to the target surface along the
    nearest target point's normal, and that target point. Only targets whose
    neighbourhood is a plane count: a collinear one has no normal."""
    tree = cKDTree(target)
    dist, idx = tree.query(source, distance_upper_bound=max_distance)
    ok = np.isfinite(dist)
    q, p = target[idx[ok]], source[ok]
    _, nbrs = tree.query(q, k=k)
    centered = target[nbrs] - target[nbrs].mean(axis=1, keepdims=True)
    eigval, eigvec = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
    planar = eigval[:, 1] > 0.05 * eigval[:, 2]
    d = np.einsum("ij,ij->i", p - q, eigvec[:, :, 0])
    return d[planar], q[planar]


def box_size(points: np.ndarray, floor: Floor, center_xy, search_m: float, cell_m: float = 0.02,
             min_height_m: float = 0.04) -> tuple[float, float, float] | None:
    """Measured (length, width, height) of the object nearest ``center_xy``:
    the connected footprint of cells standing above the floor, the rectangle
    that fits it best, and its p90 height. The object must stand clear of walls
    and other objects within ``search_m``."""
    h = floor.height(points)
    near = (np.linalg.norm(points[:, :2] - center_xy, axis=1) <= search_m) & (h > min_height_m)
    if near.sum() < 3:
        return None
    ij = np.floor((points[near, :2] - (np.asarray(center_xy) - search_m)) / cell_m).astype(int)
    n = int(np.ceil(2 * search_m / cell_m)) + 1
    top = np.full((n, n), -np.inf)
    np.maximum.at(top, (ij[:, 0], ij[:, 1]), h[near])
    labels, _ = ndimage.label(np.isfinite(top), np.ones((3, 3)))   # a slanted face touches diagonally
    occupied = np.argwhere(labels > 0)
    nearest = occupied[np.argmin(np.linalg.norm(occupied - n // 2, axis=1))]
    cells = np.argwhere(labels == labels[tuple(nearest)])
    length, width = _fit_rectangle(cells * cell_m, cell_m)
    return length, width, float(np.percentile(top[cells[:, 0], cells[:, 1]], 90))


def _fit_rectangle(xy: np.ndarray, resolution: float) -> tuple[float, float]:
    """Sides of the rectangle whose edges the points hug best (L-shape fitting,
    Zhang et al. 2017). Unlike the minimum-area rectangle it is not fooled
    when only two faces of a box were seen. Extents are of cell centres:
    partly covered cells count as covered, so centres are the unbiased edge."""
    best, sides = -1.0, (0.0, 0.0)
    for angle in np.radians(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(angle), np.sin(angle)
        r = xy @ np.array([[c, -s], [s, c]])
        lo, hi = r.min(axis=0), r.max(axis=0)
        d = np.minimum(r - lo, hi - r).min(axis=1)
        closeness = np.sum(1.0 / np.maximum(d, resolution))
        if closeness > best:
            best, sides = closeness, tuple(hi - lo)
    return float(max(sides)), float(min(sides))


def _plane(points: np.ndarray) -> Floor:
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][-1]
    normal = normal if normal[2] >= 0 else -normal
    return Floor(normal, float(-normal @ centroid))


def _group_quantiles(groups: np.ndarray, values: np.ndarray, qs, n_groups: int) -> list[np.ndarray]:
    order = np.lexsort((values, groups))
    g, v = groups[order], values[order]
    starts = np.searchsorted(g, np.arange(n_groups))
    counts = np.bincount(g, minlength=n_groups)
    return [v[starts + np.floor((counts - 1) * q).astype(int)] for q in qs]
