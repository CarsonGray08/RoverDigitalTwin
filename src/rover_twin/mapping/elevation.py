"""2.5D elevation grid: the one terrain model for physics.

Sparse square tiles of ``tile_cells`` x ``tile_cells`` cells. Each cell holds
a height, the inverse-variance weight behind it, and a state: unknown,
observed, or inferred (only the patch under the rover's footprint, which no
sensor can see).

Fusion follows elevation mapping (Fankhauser et al. 2014). A frame's points
become one measurement per cell: the mean of the points within three noise
sigmas of the cell's highest point, so a cell on a box edge reads as the box
top while a floor cell keeps all its points and stays unbiased. A point's
vertical sigma grows with range: depth noise acts along the ray, so it counts
only through the ray's vertical component, plus attitude noise, which tilts
every ray. The measurement is fused by inverse-variance weighting, except
that one more than ``step_m`` above the cell is a new surface on top (an
obstacle, a wall) and replaces it, and one more than ``step_m`` below is a
surface hidden under it and is ignored.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from rover_twin.config import Elevation

UNKNOWN, OBSERVED, INFERRED = 0, 1, 2

Key = tuple[int, int]


class ElevationGrid:
    def __init__(self, cfg: Elevation) -> None:
        self.cfg = cfg
        self.n = cfg.tile_cells
        self.tiles: dict[Key, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # height, weight, state

    def clear(self) -> None:
        self.tiles.clear()

    def integrate(self, points: np.ndarray, camera: np.ndarray) -> set[Key]:
        """Fuse world points seen from ``camera`` (world position); returns the
        tiles whose collider patch changed."""
        cfg = self.cfg
        z = points[:, 2]
        keep = (z >= cfg.min_height_m) & (z <= cfg.max_height_m)
        points, z = points[keep], z[keep]
        if len(z) == 0:
            return set()
        ray = points - camera
        rng = np.linalg.norm(ray, axis=1)
        sigma = rng * np.hypot(cfg.noise_per_m * ray[:, 2] / rng, cfg.attitude_noise_rad)
        inv_var = 1.0 / sigma ** 2
        keys, inverse = np.unique(_pack(np.floor(points[:, :2] / cfg.cell_m).astype(np.int64)), return_inverse=True)
        cells = _unpack(keys)
        top = np.full(len(cells), -np.inf)
        np.maximum.at(top, inverse, z)
        band = z >= top[inverse] - 3.0 * sigma
        w = np.bincount(inverse[band], inv_var[band], len(cells))
        m = np.bincount(inverse[band], (z * inv_var)[band], len(cells)) / w

        h, weight, state = self._get(cells)
        replace = (state != OBSERVED) | (m > h + cfg.step_m)
        changed = replace | (m >= h - cfg.step_m)
        new_h = np.where(replace, m, (h * weight + m * w) / (weight + w))
        new_w = np.where(replace, w, weight + w)
        return self._set(cells[changed], new_h[changed], new_w[changed], np.full(changed.sum(), OBSERVED))

    def ground(self, center_xy) -> float | None:
        """The ground the rover stands on at ``center_xy``: the median observed
        height of the ring just outside its footprint, None until a quarter of
        that ring has been seen."""
        cells, dist, h, state = self._around(center_xy)
        r = self.cfg.footprint_radius_m
        in_ring = (dist > r) & (dist <= r + self.cfg.footprint_ring_m)
        observed = in_ring & (state == OBSERVED)
        if observed.sum() < 0.25 * in_ring.sum():
            return None
        return float(np.float32(np.median(h[observed])))   # as stored

    def infer_footprint(self, center_xy) -> set[Key]:
        """Fill the never-seen cells under the rover's footprint with the
        ground around it, marked inferred. The camera never sees the ground
        the rover stands on, wherever it is."""
        height = self.ground(center_xy)
        if height is None:
            return set()
        cells, dist, h, state = self._around(center_xy)
        fill = (dist <= self.cfg.footprint_radius_m) & ((state == UNKNOWN) | ((state == INFERRED) & (h != height)))
        return self._set(cells[fill], np.full(fill.sum(), height), np.zeros(fill.sum()), np.full(fill.sum(), INFERRED))

    def _around(self, center_xy):
        """Cells within the footprint's ring of ``center_xy``: (cells, distance, height, state)."""
        cfg = self.cfg
        ring = cfg.footprint_radius_m + cfg.footprint_ring_m
        lo = np.floor((np.asarray(center_xy) - ring) / cfg.cell_m).astype(np.int64)
        span = int(np.ceil(2 * ring / cfg.cell_m)) + 1
        cells = (lo + np.stack(np.mgrid[0:span, 0:span], -1).reshape(-1, 2))
        dist = np.linalg.norm((cells + 0.5) * cfg.cell_m - center_xy, axis=1)
        h, _, state = self._get(cells)
        return cells, dist, h, state

    def patch(self, key: Key) -> tuple[np.ndarray, np.ndarray]:
        """Collider patch of a tile: its cells plus the first row and column of
        its +x / +y neighbours, so adjacent patches share their edge vertices.
        Returns ((n+1, n+1) heights, NaN where unknown; (n+1, n+1) states)."""
        k = self.n + 1
        cells = np.asarray(key) * self.n + np.stack(np.mgrid[0:k, 0:k], -1).reshape(-1, 2)
        h, _, state = self._get(cells)
        return np.where(state == UNKNOWN, np.nan, h).reshape(k, k).astype(np.float32), state.reshape(k, k)

    def points(self, state: int = OBSERVED) -> np.ndarray:
        """Cell centres (x, y, height) of every cell in ``state``."""
        out = []
        for (tx, ty), (h, _, s) in self.tiles.items():
            i, j = np.nonzero(s == state)
            out.append(np.column_stack([(tx * self.n + i + 0.5) * self.cfg.cell_m,
                                        (ty * self.n + j + 0.5) * self.cfg.cell_m, h[i, j]]))
        return np.concatenate(out) if out else np.zeros((0, 3))

    def save(self, path: str | Path) -> None:
        keys = sorted(self.tiles)
        stacked = [np.stack([self.tiles[k][i] for k in keys]) if keys else np.zeros((0, self.n, self.n))
                   for i in range(3)]
        np.savez_compressed(path, keys=np.asarray(keys, np.int64).reshape(-1, 2), height=stacked[0],
                            weight=stacked[1], state=stacked[2], cell_m=self.cfg.cell_m)

    @classmethod
    def load(cls, path: str | Path, cfg: Elevation | None = None) -> "ElevationGrid":
        data = np.load(path)
        grid = cls(replace(cfg or Elevation(), cell_m=float(data["cell_m"]), tile_cells=data["height"].shape[1]))
        for key, h, w, s in zip(data["keys"], data["height"], data["weight"], data["state"]):
            grid.tiles[(int(key[0]), int(key[1]))] = (h, w, s)
        return grid

    def _get(self, cells: np.ndarray):
        n = len(cells)
        h, w, s = np.zeros(n), np.zeros(n), np.zeros(n, np.uint8)
        keys, local = np.divmod(cells, self.n)
        for key, idx in _group(keys):
            tile = self.tiles.get(key)
            if tile is not None:
                i, j = local[idx, 0], local[idx, 1]
                h[idx], w[idx], s[idx] = tile[0][i, j], tile[1][i, j], tile[2][i, j]
        return h, w, s

    def _set(self, cells: np.ndarray, h, w, s) -> set[Key]:
        keys, local = np.divmod(cells, self.n)
        for key, idx in _group(keys):
            tile = self.tiles.get(key)
            if tile is None:
                tile = self.tiles[key] = (np.zeros((self.n, self.n), np.float32),
                                          np.zeros((self.n, self.n), np.float32),
                                          np.zeros((self.n, self.n), np.uint8))
            i, j = local[idx, 0], local[idx, 1]
            tile[0][i, j], tile[1][i, j], tile[2][i, j] = h[idx], w[idx], s[idx]
        # A cell on a tile's first row or column is also in the apron of the
        # neighbouring patch on that side.
        edge = local == 0
        touched = np.concatenate([keys, keys - edge * [1, 0], keys - edge * [0, 1], keys - edge.all(1, keepdims=True)])
        return {(int(a), int(b)) for a, b in _unpack(np.unique(_pack(touched)))} & self.tiles.keys()


_OFFSET = 1 << 30       # indices within +-2^30 cells: +-21000 km at 2 cm


def _pack(ij: np.ndarray) -> np.ndarray:
    """(N, 2) int cell or tile indices as one sortable int64 each."""
    return ((ij[:, 0] + _OFFSET) << 31) | (ij[:, 1] + _OFFSET)


def _unpack(packed: np.ndarray) -> np.ndarray:
    return np.column_stack([(packed >> 31) - _OFFSET, (packed & ((1 << 31) - 1)) - _OFFSET])


def _group(keys: np.ndarray):
    """(key, row indices) for each distinct row of an (N, 2) int array."""
    if len(keys) == 0:
        return
    packed = _pack(keys)
    order = np.argsort(packed, kind="stable")
    sorted_ = packed[order]
    starts = np.flatnonzero(np.r_[True, sorted_[1:] != sorted_[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(order)]):
        tx, ty = _unpack(sorted_[start:start + 1])[0]
        yield (int(tx), int(ty)), order[start:end]
