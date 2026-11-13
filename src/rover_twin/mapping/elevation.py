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
surface hidden under it and is ignored, unless the frame shows the cell's
surface gone. Its column, from just above the lower ground up to its top, is
projected into the frame's depth image. Anything the rover can hit stands
on the ground, so the foot of the column must be seen past: depth beyond it
around every point there (someone who stood there has walked away; a dark
thing whose own pixels have no depth is not seen past). Above that, a point
without depth around it is let be, but a nearer surface or one at the column
anywhere in it keeps the cell (an overhang the camera sees under is in its
own column, so it stays), and so does part of it the camera cannot see,
except above its eye line. The highest point it sees must be seen past too:
a dark chair seat on an open base has no depth of its own, and the floor
seen under it does not make it gone.
After ``CLEAR_FRAMES`` such frames in a row the ground replaces it.

A rebuild re-fuses old frames, which still hold whatever has left. The
mapper hands the grid a veto (:meth:`ElevationGrid.set_veto`): the cells
seen gone and when, so a frame from before then is not evidence of anything
standing there. Later frames count as ever, so someone who comes back is mapped.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage

from rover_twin.config import Elevation
from rover_twin.frames import Intrinsics, rigid_inverse

UNKNOWN, OBSERVED, INFERRED = 0, 1, 2
SPIKE_NEIGHBOURS = 5     # observed of a cell's 8 before it can be judged a lone spike
NEIGHBOURHOOD = np.stack(np.mgrid[-1:2, -1:2], -1).reshape(-1, 2)
CLEAR_FRAMES = 3         # frames in a row that see a surface gone before it is
COLUMN_DZ_M = 0.03       # spacing of the points checked up a cell's column
# A point is seen past only if the pixels this close with depth all see
# beyond it: (rows, columns). Wide, for a pose a few cm off sideways at a
# face's edge; not tall, or the floor in front of every point would hide it.
WINDOW_PX = (5, 15)
MIN_WITH_DEPTH = 0.6     # ... and at least this share of them has depth: grazing floor beyond is patchy
SEEN_CONFIDENCE = 1      # ARKit medium: beyond is all it asks of a pixel, not where exactly
PAST_M, PAST_PER_M = 0.05, 0.03   # how far beyond: depth noise, pose and cell size
FOOT = 3                 # points at the bottom of a column (from step_m up) that must be seen past


@dataclass(frozen=True, eq=False)
class View:
    """The frame points came from: its depth and ARKit confidence (None: all
    trusted), its intrinsics and its camera pose (CV axes, in the world)."""

    depth: np.ndarray
    confidence: np.ndarray | None
    intrinsics: Intrinsics
    T_W_C: np.ndarray

Key = tuple[int, int]


class ElevationGrid:
    def __init__(self, cfg: Elevation) -> None:
        self.cfg = cfg
        self.n = cfg.tile_cells
        self.tiles: dict[Key, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}   # height, weight, state
        self._gone: dict[int, int] = {}     # packed cell: frames in a row that saw its surface gone
        self.cleared = 0                    # cells cleared so far: a surface seen gone
        self.last_cleared = (np.zeros((0, 2), np.int64), np.zeros(0))   # by the last integrate: cells, ground
        self._veto = None                   # see set_veto

    def clear(self) -> None:
        """Forget every cell; the veto stays (a rebuild clears, then re-fuses)."""
        self.tiles.clear()
        self._gone.clear()

    def set_veto(self, cells: np.ndarray, t: np.ndarray, ground: np.ndarray) -> None:
        """Cells whose surface was seen gone at phone time ``t``, down to
        ``ground``: a frame from before then is not evidence of anything
        standing there, so a rebuild that re-fuses old frames does not bring
        back someone who has left. Later frames count as ever."""
        if len(cells) == 0:
            self._veto = None
            return
        keys = _pack(cells)
        order = np.lexsort((-t, keys))            # per cell, its latest clearing first
        keys, t, ground = keys[order], t[order], ground[order]
        first = np.r_[True, keys[1:] != keys[:-1]]
        self._veto = (keys[first], t[first], ground[first])

    def _vetoed(self, points: np.ndarray, t: float) -> np.ndarray:
        """Points above the ground of a cell seen gone after ``t``."""
        keys, times, ground = self._veto
        if t >= times.max():
            return np.zeros(len(points), bool)
        packed = _pack(np.floor(points[:, :2] / self.cfg.cell_m).astype(np.int64))
        at = np.minimum(np.searchsorted(keys, packed), len(keys) - 1)
        return (keys[at] == packed) & (t < times[at]) & (points[:, 2] > ground[at] + self.cfg.step_m)

    def integrate(self, points: np.ndarray, camera: np.ndarray, view: View | None = None,
                  t: float | None = None) -> set[Key]:
        """Fuse world points seen from ``camera`` (world position); returns the
        tiles whose collider patch changed. With the ``view`` they came from,
        a surface the frame shows gone is cleared (see the module docstring).
        ``t``, the frame's phone time, is checked against the veto."""
        cfg = self.cfg
        self.last_cleared = (np.zeros((0, 2), np.int64), np.zeros(0))
        z = points[:, 2]
        keep = (z >= cfg.min_height_m) & (z <= cfg.max_height_m)
        points, z = points[keep], z[keep]
        if self._veto is not None and t is not None and len(z):
            keep = ~self._vetoed(points, t)
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
        below = (state == OBSERVED) & (m < h - cfg.step_m)
        gone = np.zeros(len(cells), bool)
        if view is not None and below.any():
            gone[below] = self._gone_in(view, cells[below], m[below], h[below])
        cleared = self._count_gone(keys, gone)
        self.cleared += int(cleared.sum())
        self.last_cleared = (cells[cleared], m[cleared])
        replace = (state != OBSERVED) | (m > h + cfg.step_m) | cleared
        changed = replace | (m >= h - cfg.step_m)
        new_h = np.where(replace, m, (h * weight + m * w) / (weight + w))
        new_w = np.where(replace, w, weight + w)
        return self._set(cells[changed], new_h[changed], new_w[changed], np.full(changed.sum(), OBSERVED))

    def _gone_in(self, view: View, cells: np.ndarray, ground: np.ndarray, top: np.ndarray) -> np.ndarray:
        """For each cell, whether ``view`` sees past the whole of its column
        from ``ground`` up to ``top`` that it could see."""
        cfg = self.cfg
        n = np.maximum(np.ceil((top - ground - cfg.step_m) / COLUMN_DZ_M), 1).astype(np.int64)
        which = np.repeat(np.arange(len(cells)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        z = np.minimum(ground[which] + cfg.step_m + k * COLUMN_DZ_M, top[which])
        T_C_W = rigid_inverse(view.T_W_C)
        camera = np.column_stack([(cells[which] + 0.5) * cfg.cell_m, z]) @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        intr, depth = view.intrinsics, view.depth
        ahead = camera[:, 2] > 0.1
        safe = np.where(ahead, camera[:, 2], 1.0)
        u = np.floor(intr.fx * camera[:, 0] / safe + intr.cx).astype(np.int64)
        v = np.floor(intr.fy * camera[:, 1] / safe + intr.cy).astype(np.int64)
        in_view = ahead & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
        valid = np.isfinite(depth) & (depth > 0)
        if view.confidence is not None:
            valid &= view.confidence >= SEEN_CONFIDENCE
        nearest = ndimage.minimum_filter(np.where(valid, depth, np.inf), size=WINDOW_PX, mode="nearest")
        share = ndimage.uniform_filter(valid.astype(np.float32), size=WINDOW_PX, mode="nearest")
        vv, uu = np.clip(v, 0, depth.shape[0] - 1), np.clip(u, 0, depth.shape[1] - 1)
        beyond = nearest[vv, uu] > camera[:, 2] + PAST_M + PAST_PER_M * camera[:, 2]
        past = in_view & (share[vv, uu] >= MIN_WITH_DEPTH) & beyond
        # Something at or before the point, or a point below the camera's eye
        # line it cannot see (it sees all of the column the ground is seen
        # from), keeps the cell; so does a foot point not seen past.
        keeps = (in_view & ~beyond) | (~in_view & (z <= view.T_W_C[2, 3])) | ((k < FOOT) & ~past)
        # So does the top of what it sees of the column, unless seen past: a
        # surface on a thin stem whose own pixels have no depth (a dark chair
        # seat over an open base) is not gone because the floor is seen under it.
        highest = np.full(len(cells), -1)
        np.maximum.at(highest, which[in_view], k[in_view])
        keeps |= in_view & (k == highest[which]) & ~past
        return np.bincount(which, keeps, len(cells)) == 0

    def _count_gone(self, keys: np.ndarray, gone: np.ndarray) -> np.ndarray:
        """Count frames in a row that saw each cell's surface gone (``keys``
        packed); any other look at a cell starts it over. True where it reaches
        CLEAR_FRAMES."""
        if self._gone:
            held = np.fromiter(self._gone, np.int64, len(self._gone))
            for key in held[np.isin(held, keys[~gone])]:
                del self._gone[int(key)]
        cleared = np.zeros(len(keys), bool)
        for i in np.flatnonzero(gone):
            count = self._gone.get(int(keys[i]), 0) + 1
            if count >= CLEAR_FRAMES:
                cleared[i] = True
                self._gone.pop(int(keys[i]), None)
            else:
                self._gone[int(keys[i])] = count
        return cleared

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

    def infer_footprint(self, center_xy, fallback_z: float | None = None) -> set[Key]:
        """Fill the never-seen cells under the rover's footprint with the
        ground around it, marked inferred. The camera never sees the ground
        the rover stands on, wherever it is. Until the ring around it is seen
        (a camera looking ahead does not see it from where the rover starts),
        ``fallback_z``, the ground the rover is known to stand on, fills the
        cells still unknown, so the rover can turn in place to look around."""
        height = self.ground(center_xy)
        cells, dist, h, state = self._around(center_xy)
        inside = dist <= self.cfg.footprint_radius_m
        if height is not None:
            fill = inside & ((state == UNKNOWN) | ((state == INFERRED) & (h != height)))
        elif fallback_z is not None:
            height, fill = float(np.float32(fallback_z)), inside & (state == UNKNOWN)
        else:
            return set()
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
        Returns ((n+1, n+1) heights, NaN where unknown; (n+1, n+1) states).
        A lone spike is lowered to its neighbours (see :func:`_despeckle`)."""
        k = self.n + 1
        # One cell more all round, so every cell is judged against all its neighbours.
        cells = np.asarray(key) * self.n - 1 + np.stack(np.mgrid[0:k + 2, 0:k + 2], -1).reshape(-1, 2)
        h, _, state = self._get(cells)
        h = _despeckle(h.reshape(k + 2, k + 2), state.reshape(k + 2, k + 2), self.cfg.step_m)[1:-1, 1:-1]
        state = state.reshape(k + 2, k + 2)[1:-1, 1:-1]
        return np.where(state == UNKNOWN, np.nan, h).astype(np.float32), state

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
        # A cell is in its own tile's patch and, on a first row or column, also
        # in the apron of the neighbouring patch on that side. A lone spike is
        # judged against its 8 neighbours, so the patches holding those change too.
        keys, local = np.divmod((cells[:, None, :] + NEIGHBOURHOOD).reshape(-1, 2), self.n)
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


def _despeckle(h: np.ndarray, state: np.ndarray, step: float) -> np.ndarray:
    """Heights with each lone spike lowered to its highest neighbour: an
    observed cell more than ``step`` above every observed one of its 8 (and
    most of them observed). One cell is narrower than anything the rover meets
    and than the depth can resolve; a stray reading is not, and in physics it
    is a jolt and to validation an obstacle. Anything two cells wide is kept."""
    observed = state == OBSERVED
    ring = np.ones((3, 3), bool)
    ring[1, 1] = False
    highest = ndimage.maximum_filter(np.where(observed, h, -np.inf), footprint=ring, mode="constant", cval=-np.inf)
    count = ndimage.convolve(observed.astype(np.int32), ring.astype(np.int32), mode="constant")
    return np.where(observed & (count >= SPIKE_NEIGHBOURS) & (h > highest + step), highest, h)
