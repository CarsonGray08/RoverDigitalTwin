"""3D occupancy of the space the rover's body moves through: the latest view wins.

An obstacle layer above the ground, from ``z_min_m`` (above a step the rover
drives over) to ``z_max_m``, in voxels ``voxel_m`` across and ``voxel_z_m``
tall. The band is world height from the start floor: flat floors only. Each
voxel holds the log-odds that something is there. Each frame:

- where its depth points land, a voxel is occupied (one hit is enough, so
  someone walking in shows at once);
- where every pixel of a voxel's image has depth, all of it beyond the
  voxel, the camera sees through it: a few such frames and it is empty;
- where a voxel is in view, nothing nearer hides it, but the depth there
  neither confirms nor refutes it (no depth, low confidence, too patchy), it
  fades by ``fade`` per frame: an obstacle nothing confirms is forgotten after
  a few seconds in view;
- hidden behind something nearer, or out of view, it stays as it was.

So the space in front of the rover shows what the camera sees now, and what
it cannot see keeps its last state. The cost: something that stays in view
and never returns depth (a dark seat, a black pole) fades too; the depth
never showed the pole at all, so no rule could keep it.

Each voxel keeps the keyframe it was last observed from (``owner``), when
(``t``), and where it was then in that keyframe's ARKit frame (``local``).
When the pose graph corrects keyframes, :meth:`OccupancyGrid.move` puts
every voxel where its keyframe's current pose puts ``local``: computed from
the original each time, so small corrections add up instead of rounding
away. Where two land in one voxel, the later observation wins. This remains
an approximation: accumulated evidence from earlier keyframes moves with its
latest owner, rather than being replayed at each observation's corrected pose.

The grid is dense over the area seen so far, grown as the rover goes.
"""

from __future__ import annotations

import numpy as np

from rover_twin.config import Occupancy
from rover_twin.frames import Intrinsics, rigid_inverse

GROW_M = 2.0          # margin added when the grid grows
MOSTLY_IN_VIEW = 0.5  # of a voxel's image, in the frame, for depth beyond it to clear it


class OccupancyGrid:
    def __init__(self, cfg: Occupancy) -> None:
        self.cfg = cfg
        self.nz = int(round((cfg.z_max_m - cfg.z_min_m) / cfg.voxel_z_m))
        self.origin = np.zeros(2)                       # world xy of voxel (0, 0)'s corner
        self.clear()

    def clear(self) -> None:
        self._allocate((0, 0))

    def _allocate(self, size) -> None:
        shape = (*size, self.nz)
        self.logodds = np.zeros(shape, np.float32)
        self.owner = np.zeros(shape, np.int32)
        self.t = np.full(shape, -np.inf)
        self.local = np.zeros((*shape, 3), np.float32)
        self.shown = np.full(size, np.nan)              # each column's top as changed() last reported it
        self._touched: list[np.ndarray] = []            # columns updated since

    # ---- fusion ----

    def integrate(self, hits: np.ndarray, free_depth: np.ndarray, intrinsics: Intrinsics,
                  T_W_C: np.ndarray, owner: int, W_T_A: np.ndarray, t: float) -> None:
        """Fuse one frame, seen at phone time ``t``: ``hits`` (N, 3) world points
        it measured; ``free_depth`` its z-depth image, 0 where a pixel is not to
        be trusted to see through anything; ``owner`` the keyframe its pose
        comes from, which puts its ARKit world in the twin at ``W_T_A``."""
        cfg = self.cfg
        camera = T_W_C[:3, 3]
        self._cover(camera[:2] - cfg.max_range_m, camera[:2] + cfg.max_range_m)
        if len(hits):
            self._cover(hits[:, :2].min(0), hits[:, :2].max(0))
        # A measured hit wins even if projection discretization also suggests
        # a miss. Never subtract free evidence from a hit in the same frame.
        band = (hits[:, 2] >= cfg.z_min_m) & (hits[:, 2] < cfg.z_max_m)
        hit_keys = np.unique(np.ravel_multi_index(self._index(hits[band]).T, self.logodds.shape))
        hit_idx = np.column_stack(np.unravel_index(hit_keys, self.logodds.shape))
        updated = []
        i, j, k = self._in_reach(camera, intrinsics, T_W_C)
        if len(i):
            past, fade = self._look(self._centres(i, j, k), free_depth, intrinsics, T_W_C)
            unhit = ~np.isin(np.ravel_multi_index((i, j, k), self.logodds.shape), hit_keys)
            past, fade = past & unhit, fade & unhit
            self.logodds[i[past], j[past], k[past]] -= cfg.miss
            fade &= self.logodds[i, j, k] > 0
            a, b, c = i[fade], j[fade], k[fade]
            self.logodds[a, b, c] = np.maximum(self.logodds[a, b, c] - cfg.fade, 0.0)
            seen = past | fade
            updated.append(np.column_stack([i[seen], j[seen], k[seen]]))
        if len(hit_idx):
            self.logodds[tuple(hit_idx.T)] += cfg.hit
            updated.append(hit_idx)
        np.clip(self.logodds, cfg.clamp[0], cfg.clamp[1], out=self.logodds)
        if updated:
            self._touched.append(np.concatenate(updated)[:, :2])
            i, j, k = np.concatenate(updated).T
            A_T_W = rigid_inverse(W_T_A)
            self.owner[i, j, k] = owner
            self.t[i, j, k] = t
            self.local[i, j, k] = self._centres(i, j, k) @ A_T_W[:3, :3].T + A_T_W[:3, 3]

    def _look(self, centres, depth, intr: Intrinsics, T_W_C) -> tuple[np.ndarray, np.ndarray]:
        """(seen past, fades) for each voxel centre, judged over the whole voxel.

        The sphere round it gives a pixel rectangle that encloses its image. Seen
        past: most of the rectangle is in the image, every pixel of it there
        has depth, and all of it is beyond the voxel's far side by the margin;
        rays above and below a thin dark surface inside it do not count,
        because its own pixels have no depth. Fades: some of the rectangle is
        in the image and nothing there is at or in front of the voxel, but it
        is not seen past (missing depth in it, or mostly out of the image).
        """
        cfg = self.cfg
        T_C_W = rigid_inverse(T_W_C)
        camera = centres @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        # The sphere round the voxel bounds it: its image lies in the square of
        # half-size fx * radius / (z - radius) round the centre's image, and its
        # depth between z - radius and z + radius. A larger rectangle than the
        # voxel's own only asks for more depth before clearing it.
        h, w = depth.shape
        radius = np.linalg.norm([cfg.voxel_m, cfg.voxel_m, cfg.voxel_z_m]) / 2
        z = camera[:, 2]
        zmin, zmax = z - radius, z + radius
        near = zmin > 0.1
        safe = np.where(near, z, 1.0)
        r = intr.fx * radius / np.where(near, zmin, 1.0)
        u, v = intr.fx * camera[:, 0] / safe + intr.cx, intr.fy * camera[:, 1] / safe + intr.cy
        lo = np.floor(np.column_stack([u - r, v - r]).clip(-1e6, 1e6)).astype(np.int64)
        hi = np.ceil(np.column_stack([u + r, v + r]).clip(-1e6, 1e6)).astype(np.int64) + 1
        area = np.prod(hi - lo, axis=1).astype(float)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, [w, h])            # the part in the image
        mostly = np.prod(np.maximum(hi - lo, 0), axis=1) >= MOSTLY_IN_VIEW * area
        visible = (zmin > 0.1) & (hi > lo).all(1)
        past, fade = np.zeros(len(centres), bool), np.zeros(len(centres), bool)
        if visible.any():
            valid = np.isfinite(depth) & (depth > 0)
            nearest = _rectangle_min(np.where(valid, depth, np.inf), lo[visible], hi[visible])
            holes = np.pad(np.cumsum(np.cumsum(~valid, 0), 1), ((1, 0), (1, 0)))   # missing pixels, summed
            (x0, y0), (x1, y1) = lo[visible].T, hi[visible].T
            missing = holes[y1, x1] - holes[y0, x1] - holes[y1, x0] + holes[y0, x0]
            far = zmax[visible]
            beyond = nearest > far + cfg.margin_m + cfg.margin_per_m * far
            past[visible] = beyond & mostly[visible] & (missing == 0)
            fade[visible] = beyond & ~past[visible]
        return past, fade

    def move(self, W_T_A: list[np.ndarray]) -> None:
        """Put every voxel where its keyframe now puts it: ``W_T_A[owner]`` is
        where that keyframe's correction places its ARKit world in the twin.
        Where two land in one voxel, the later observation wins."""
        known = np.isfinite(self.t)
        if not known.any():
            return
        i, j, k = np.nonzero(known)
        value, owner, t, local = self.logodds[i, j, k], self.owner[i, j, k], self.t[i, j, k], self.local[i, j, k]
        poses = np.stack(W_T_A)
        world = np.einsum("nij,nj->ni", poses[owner, :3, :3], local.astype(np.float64)) + poses[owner, :3, 3]
        self.clear()
        self._cover(world[:, :2].min(0), world[:, :2].max(0))
        idx = self._index(world, clamp_z=True)
        order = np.argsort(t, kind="stable")              # the latest written last, so it wins
        a, b, c = idx[order].T
        self.logodds[a, b, c], self.owner[a, b, c] = value[order], owner[order]
        self.t[a, b, c], self.local[a, b, c] = t[order], local[order]

    # ---- what it holds ----

    def occupied(self) -> np.ndarray:
        """Centres (x, y, z) of every voxel believed occupied."""
        i, j, k = np.nonzero(self.logodds > self.cfg.occupied)
        return self._centres(i, j, k)

    def tops(self, xy: np.ndarray) -> np.ndarray:
        """The top of what is occupied in the column over each world point; NaN where nothing is."""
        cfg, s = self.cfg, self.cfg.voxel_m
        out = np.full(len(xy), np.nan)
        ij = np.floor((xy - self.origin) / s).astype(np.int64)
        ok = (ij >= 0).all(1) & (ij < self.logodds.shape[:2]).all(1)
        if ok.any():
            occ = self.logodds[ij[ok, 0], ij[ok, 1]] > cfg.occupied
            top = self.nz - 1 - np.argmax(occ[:, ::-1], axis=1)
            out[ok] = np.where(occ.any(axis=1), cfg.z_min_m + (top + 1) * cfg.voxel_z_m, np.nan)
        return out

    def columns(self) -> tuple[np.ndarray, np.ndarray]:
        """(cells, top) of every column with something occupied in it: cells
        as world column indices (x, y) of size ``voxel_m``."""
        cfg = self.cfg
        occ = self.logodds > cfg.occupied
        i, j = np.nonzero(occ.any(axis=2))
        top = self.nz - 1 - np.argmax(occ[i, j, ::-1], axis=1)
        offset = np.round(self.origin / cfg.voxel_m).astype(np.int64)
        return np.column_stack([i, j]) + offset, cfg.z_min_m + (top + 1) * cfg.voxel_z_m

    def changed(self) -> np.ndarray:
        """World column indices (x, y) whose top changed since the last call,
        including those emptied, so what shows them can be updated. Only the
        columns fused since are looked at."""
        if not self._touched:
            return np.zeros((0, 2), np.int64)
        shape = self.logodds.shape[:2]
        ij = np.concatenate(self._touched)
        ij = np.column_stack(np.unravel_index(np.unique(np.ravel_multi_index(ij.T, shape)), shape))
        self._touched = []
        occ = self.logodds[ij[:, 0], ij[:, 1]] > self.cfg.occupied
        top = self.nz - 1 - np.argmax(occ[:, ::-1], axis=1)
        now = np.where(occ.any(axis=1), self.cfg.z_min_m + (top + 1) * self.cfg.voxel_z_m, np.nan)
        then = self.shown[ij[:, 0], ij[:, 1]]
        differ = ~((now == then) | (np.isnan(now) & np.isnan(then)))
        self.shown[ij[:, 0], ij[:, 1]] = now
        return ij[differ] + np.round(self.origin / self.cfg.voxel_m).astype(np.int64)

    def settle(self) -> None:
        """Take every column as shown, as it is now (after everything was sent)."""
        self._touched = []
        self.shown = np.full(self.logodds.shape[:2], np.nan)
        cells, top = self.columns()
        ij = cells - np.round(self.origin / self.cfg.voxel_m).astype(np.int64)
        self.shown[ij[:, 0], ij[:, 1]] = top

    def seen_empty(self, points: np.ndarray) -> np.ndarray:
        """Whether each world point is in a voxel the camera has seen past
        more than it saw anything in. Unseen, faded and occupied voxels, and
        points outside the band, are not."""
        out = np.zeros(len(points), bool)
        if not self.logodds.size or not len(points):
            return out
        s = self.cfg.voxel_m
        ij = np.floor((points[:, :2] - self.origin) / s).astype(np.int64)
        k = np.floor((points[:, 2] - self.cfg.z_min_m) / self.cfg.voxel_z_m).astype(np.int64)
        ok = (k >= 0) & (k < self.nz) & (ij >= 0).all(1) & (ij < self.logodds.shape[:2]).all(1)
        out[ok] = self.logodds[ij[ok, 0], ij[ok, 1], k[ok]] < 0.0
        return out

    def count_in(self, x0: float, x1: float, y0: float, y1: float) -> int:
        """Occupied columns whose centre is inside the box."""
        p = self.occupied()
        at = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        return len(np.unique(p[at, :2], axis=0))

    # ---- geometry ----

    def _in_reach(self, camera: np.ndarray, intr: Intrinsics, T_W_C: np.ndarray):
        """Voxel indices within range whose column is in the image sideways."""
        cfg, s = self.cfg, self.cfg.voxel_m
        lo = np.floor((camera[:2] - cfg.max_range_m - self.origin) / s).astype(np.int64)
        hi = np.ceil((camera[:2] + cfg.max_range_m - self.origin) / s).astype(np.int64)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, self.logodds.shape[:2])
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        xy = self.origin + (np.column_stack([ii, jj]) + 0.5) * s
        near = np.hypot(*(xy - camera[:2]).T) <= cfg.max_range_m
        ii, jj, xy = ii[near], jj[near], xy[near]
        # A column is a candidate if its middle is ahead and within the image's width, with slack.
        mid = np.column_stack([xy, np.full(len(xy), 0.5 * (cfg.z_min_m + cfg.z_max_m))])
        T_C_W = rigid_inverse(T_W_C)
        c = mid @ T_C_W[:3, :3].T + T_C_W[:3, 3]
        ahead = c[:, 2] > 0.05
        u = intr.fx * c[:, 0] / np.where(ahead, c[:, 2], 1.0) + intr.cx
        slack = 0.25 * intr.width
        keep = ahead & (u > -slack) & (u < intr.width + slack)
        ii, jj = ii[keep], jj[keep]
        k = np.tile(np.arange(self.nz), len(ii))
        return np.repeat(ii, self.nz), np.repeat(jj, self.nz), k

    def _centres(self, i, j, k) -> np.ndarray:
        s = self.cfg.voxel_m
        return np.column_stack([self.origin[0] + (i + 0.5) * s, self.origin[1] + (j + 0.5) * s,
                                self.cfg.z_min_m + (k + 0.5) * self.cfg.voxel_z_m])

    def _index(self, points: np.ndarray, clamp_z: bool = False) -> np.ndarray:
        """Voxel indices of ``points``; outside the band dropped, or with
        ``clamp_z`` kept in its nearest layer (a correction lifted it a little)."""
        s = self.cfg.voxel_m
        ij = np.floor((points[:, :2] - self.origin) / s).astype(np.int64)
        k = np.floor((points[:, 2] - self.cfg.z_min_m) / self.cfg.voxel_z_m).astype(np.int64)
        if clamp_z:
            k = np.clip(k, 0, self.nz - 1)
        ok = (k >= 0) & (k < self.nz) & (ij >= 0).all(1) & (ij < self.logodds.shape[:2]).all(1)
        return np.column_stack([ij, k])[ok]

    def _cover(self, lo_xy: np.ndarray, hi_xy: np.ndarray) -> None:
        """Grow the grid to cover the box (world xy), with GROW_M to spare."""
        s = self.cfg.voxel_m
        n = np.array(self.logodds.shape[:2])
        if n.all():
            top = self.origin + n * s
            if (lo_xy >= self.origin).all() and (hi_xy < top).all():
                return
            new_lo = np.minimum(self.origin, np.floor((lo_xy - GROW_M) / s) * s)
            new_hi = np.maximum(top, np.ceil((hi_xy + GROW_M) / s) * s)
        else:
            new_lo, new_hi = np.floor((lo_xy - GROW_M) / s) * s, np.ceil((hi_xy + GROW_M) / s) * s
        size = np.round((new_hi - new_lo) / s).astype(np.int64)
        old = (self.logodds, self.owner, self.t, self.local, self.shown)
        touched = self._touched
        self._allocate(size)
        if n.all():
            at = np.round((self.origin - new_lo) / s).astype(np.int64)
            for new, previous in zip((self.logodds, self.owner, self.t, self.local, self.shown), old):
                new[at[0]:at[0] + n[0], at[1]:at[1] + n[1]] = previous
            self._touched = [ij + at for ij in touched]
        self.origin = new_lo


def _rectangle_min(depth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Minimum in each half-open pixel rectangle, via power-of-two blocks.

    Four overlapping blocks cover any rectangle. Build only the block sizes
    queried this frame; no per-voxel pixel loop and no missing-pixel averaging.
    """
    sizes = hi - lo
    powers = np.floor(np.log2(sizes)).astype(np.int64)
    cache = {(0, 0): depth}

    def blocks(px, py):
        if (px, py) not in cache:
            if px:
                a = blocks(px - 1, py)
                stride = 1 << (px - 1)
                cache[px, py] = np.minimum(a[:, :-stride], a[:, stride:])
            else:
                a = blocks(px, py - 1)
                stride = 1 << (py - 1)
                cache[px, py] = np.minimum(a[:-stride], a[stride:])
        return cache[px, py]

    result = np.empty(len(lo), depth.dtype)
    stride = int(powers[:, 0].max()) + 1
    groups = powers[:, 0] + stride * powers[:, 1]
    order = np.argsort(groups, kind="stable")          # each size's rectangles together, found once
    sizes, starts = np.unique(groups[order], return_index=True)
    for group, a0, a1 in zip(sizes, starts, np.r_[starts[1:], len(order)]):
        py, px = divmod(int(group), stride)
        at = order[a0:a1]
        x0, y0 = lo[at].T
        x1, y1 = (hi[at] - [1 << px, 1 << py]).T
        a = blocks(px, py)
        result[at] = np.minimum(np.minimum(a[y0, x0], a[y0, x1]), np.minimum(a[y1, x0], a[y1, x1]))
    return result
