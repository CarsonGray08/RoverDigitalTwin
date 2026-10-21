"""Keyframe pose graph: nodes are keyframe rover poses (x, y, yaw) and a
height correction; ARKit odometry links consecutive keyframes, and
registering a keyframe onto an earlier keyframe that sees the same place
again (the start of a 360, a revisited corner) adds a loop closure: its walls
measure the error in the plane. Height is measured more densely: any two
keyframes that see the same floor measure how far apart their heights have
drifted, the floor being the same spots in both, flat or not, so each new
keyframe is compared with a few earlier ones at spaced intervals. A robust
least squares solve gives every keyframe's corrected pose; height,
independent of the plane once roll and pitch come from gravity, is a linear
solve of its own.

Poses are arrays ``[x, y, yaw_rad]``. A keyframe's surface is kept in its own
rover frame, so it moves with the keyframe's estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

ODOMETRY_SIGMA = (0.01, math.radians(0.3))    # m, rad per keyframe step
CLOSURE_SIGMA = (0.02, math.radians(0.3))     # along the best-constrained direction
HEIGHT_SIGMA = (0.005, 0.005)                 # m: ARKit's height step between keyframes, a closure's floor gap
CLOSURE_TRAVEL_M = 1.0          # a closure is with a keyframe this far back along the path ...
CLOSURE_TURN_RAD = math.pi      # ... or this much turning back
OVERLAP_M = 1.0                 # and close enough now to see the same surfaces
OVERLAP_RAD = math.radians(40.0)
MIN_INLIERS = 0.6               # registration accepted above this matched fraction ...
MAX_RESIDUAL_M = 0.03           # ... and below this median residual
WALL_Z_M = 0.10                 # only points above the floor constrain (x, y, yaw); clear of ARKit's height drift
FLOOR_RANGE_M = 2.5             # the floor a closure compares: near enough to read flat in one frame
MATCH_M = 0.03                  # floor points of the two keyframes this close in the plane are the same spot
MAX_HEIGHT_GAP_M = 0.15         # a larger floor gap is a bad match, not drift
HEIGHT_LAGS = (1, 2, 4, 8, 16, 32)   # earlier keyframes each new one compares floors with


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``b`` in ``a``'s frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    d = b[:2] - a[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], _wrap(b[2] - a[2])])


def apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Points (N, 3) in ``pose``'s frame into the frame ``pose`` is in; z is kept."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    xy = points[:, :2] @ np.array([[c, s], [-s, c]]) + pose[:2]
    return np.column_stack([xy, points[:, 2]])


def register(fixed: np.ndarray, moving: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Point-to-plane ICP in the plane: the transform ``[x, y, yaw]`` that puts
    ``moving`` onto ``fixed`` (both (N, 3), centred near the rotation's pivot),
    the unit-free information of its (x, y) part, the matched fraction and the
    median residual. Facing a single wall the information is near zero along
    it: the match cannot tell where along the wall it is."""
    normals, vertical = _normals(fixed)
    fixed, normals = fixed[vertical], normals[vertical]
    tree, D = cKDTree(fixed), np.zeros(3)
    J = np.zeros((0, 3))
    for k in range(iterations):
        p = apply(D, moving)
        dist, idx = tree.query(p, distance_upper_bound=0.20 if k < iterations // 2 else 0.08)
        ok = np.isfinite(dist)
        if ok.sum() < 3:
            return D, np.zeros((2, 2)), 0.0, math.inf
        n, q, p = normals[idx[ok]], fixed[idx[ok], :2], p[ok, :2]
        J = np.column_stack([n, n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1]])
        step = np.linalg.lstsq(J, -np.sum(n * (p - q), axis=1), rcond=None)[0]
        D = compose(step, D)
    dist, _ = tree.query(apply(D, moving), distance_upper_bound=0.08)
    ok = np.isfinite(dist)
    residual = float(np.median(dist[ok])) if ok.any() else math.inf
    return D, J[:, :2].T @ J[:, :2] / len(J), float(ok.mean()), residual


@dataclass(frozen=True)
class Closure:
    i: int
    j: int
    z: np.ndarray            # j in i's frame, as registered
    weight: np.ndarray       # (3, 3): residual (world x, y; yaw) -> whitened

    def residual(self, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        e = _minus(between(xi, xj), self.z)
        c, s = math.cos(xi[2]), math.sin(xi[2])
        world = np.array([c * e[0] - s * e[1], s * e[0] + c * e[1], e[2]])
        return self.weight @ world


@dataclass
class PoseGraph:
    odometry: list[np.ndarray] = field(default_factory=list)   # ARKit pose of each keyframe
    estimate: list[np.ndarray] = field(default_factory=list)   # corrected pose of each keyframe
    surfaces: list[np.ndarray] = field(default_factory=list)   # each keyframe's surface in its rover frame
    closures: list[Closure] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)        # height correction of each keyframe
    height_closures: list[tuple[int, int, float]] = field(default_factory=list)   # (i, j, gap j-i)

    def add(self, odometry: np.ndarray, points_W: np.ndarray) -> bool:
        """Add a keyframe at its ARKit pose with its surface (twin frame, at
        that pose). True when a closure was found and the graph re-solved."""
        guess = compose(self.estimate[-1], between(self.odometry[-1], odometry)) if self.odometry else odometry
        self.odometry.append(odometry)
        self.estimate.append(guess)
        self.heights.append(self.heights[-1] if self.heights else 0.0)
        self.surfaces.append(apply(_inverse(odometry), points_W))
        j = len(self.odometry) - 1
        found, compared = False, set()
        for i in self._candidates(j):
            pivot = np.array([guess[0], guess[1], 0.0])
            fixed = apply(self.estimate[i], _walls(self.surfaces[i])) - pivot
            moving = apply(guess, _walls(self.surfaces[j])) - pivot
            if len(fixed) < 100 or len(moving) < 100:
                continue
            D, info, inliers, residual = register(fixed, moving)
            if inliers >= MIN_INLIERS and residual <= MAX_RESIDUAL_M:
                registered = np.array([guess[0] + D[0], guess[1] + D[1], guess[2] + D[2]])
                self.closures.append(Closure(i, j, between(self.estimate[i], registered), _weight(info)))
                self._compare_floors(i, j, registered)
                compared.add(i)
                found = True
        if found:
            self._solve()
        for lag in HEIGHT_LAGS:
            if j - lag >= 0 and j - lag not in compared:
                self._compare_floors(j - lag, j, self.estimate[j])
        self._solve_heights()
        return found

    def correction(self, k: int = -1) -> np.ndarray:
        """Planar transform taking keyframe ``k``'s ARKit pose to its estimate."""
        return compose(self.estimate[k], _inverse(self.odometry[k]))

    def height(self, k: int = -1) -> float:
        """What to add to keyframe ``k``'s ARKit height."""
        return self.heights[k]

    def _candidates(self, j: int) -> list[int]:
        travel, turn, out = 0.0, 0.0, []
        for i in range(j - 1, -1, -1):
            step = between(self.odometry[i], self.odometry[i + 1])
            travel, turn = travel + math.hypot(step[0], step[1]), turn + abs(step[2])
            if travel < CLOSURE_TRAVEL_M and turn < CLOSURE_TURN_RAD:
                continue
            rel = between(self.estimate[i], self.estimate[j])
            if math.hypot(rel[0], rel[1]) < OVERLAP_M and abs(rel[2]) < OVERLAP_RAD:
                out.append(i)
        return out

    def _solve(self) -> None:
        n = len(self.estimate)
        odo = [between(self.odometry[k], self.odometry[k + 1]) for k in range(n - 1)]
        w_odo = 1.0 / np.array([ODOMETRY_SIGMA[0], ODOMETRY_SIGMA[0], ODOMETRY_SIGMA[1]])
        first = self.estimate[0]

        def residuals(flat):
            x = flat.reshape(n, 3)
            out = [(x[0] - first) * 1e3]
            out += [_minus(between(x[k], x[k + 1]), odo[k]) * w_odo for k in range(n - 1)]
            out += [c.residual(x[c.i], x[c.j]) for c in self.closures]
            return np.concatenate(out)

        # Each residual involves one or two keyframes: telling the solver so makes
        # a solve ~10x faster at a few hundred keyframes (0.6 s at 300, from 6.9 s).
        sparsity = lil_matrix((3 * (n + len(self.closures)), 3 * n), dtype=np.int8)
        sparsity[0:3, 0:3] = 1
        for k in range(n - 1):
            sparsity[3 * k + 3:3 * k + 6, 3 * k:3 * k + 6] = 1
        for m, c in enumerate(self.closures):
            rows = slice(3 * (n + m), 3 * (n + m) + 3)
            sparsity[rows, 3 * c.i:3 * c.i + 3] = sparsity[rows, 3 * c.j:3 * c.j + 3] = 1
        result = least_squares(residuals, np.concatenate(self.estimate), loss="soft_l1", f_scale=3.0,
                               jac_sparsity=sparsity)
        self.estimate = list(result.x.reshape(n, 3))


    def _compare_floors(self, i: int, j: int, pose_j: np.ndarray) -> None:
        gap = _floor_gap(apply(self.estimate[i], _floor(self.surfaces[i])), apply(pose_j, _floor(self.surfaces[j])))
        if gap is not None:
            self.height_closures.append((i, j, gap))

    def _solve_heights(self) -> None:
        """Heights that keep ARKit's steps between keyframes and meet the
        floor gaps the closures measured, the first keyframe fixed."""
        n = len(self.heights)
        rows, rhs = [np.eye(1, n, 0)[0] * 1e3], [0.0]
        for k in range(n - 1):
            rows.append((np.eye(1, n, k + 1)[0] - np.eye(1, n, k)[0]) / HEIGHT_SIGMA[0])
            rhs.append(0.0)
        for i, j, gap in self.height_closures:
            rows.append((np.eye(1, n, j)[0] - np.eye(1, n, i)[0]) / HEIGHT_SIGMA[1])
            rhs.append(gap / HEIGHT_SIGMA[1])
        self.heights = list(np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0])


def _walls(surface: np.ndarray) -> np.ndarray:
    return surface[surface[:, 2] > WALL_Z_M]


def _floor(surface: np.ndarray) -> np.ndarray:
    """The near floor of a keyframe's surface (rover frame: the rover is at the origin)."""
    return surface[(surface[:, 2] <= WALL_Z_M) & (np.hypot(surface[:, 0], surface[:, 1]) <= FLOOR_RANGE_M)]


def _floor_gap(fixed: np.ndarray, moving: np.ndarray) -> float | None:
    """How much higher ``moving``'s floor must go to meet ``fixed``'s, where
    both saw the same spots; None without enough of them."""
    if len(fixed) < 200 or len(moving) < 200:
        return None
    dist, idx = cKDTree(fixed[:, :2]).query(moving[:, :2], distance_upper_bound=MATCH_M)
    ok = np.isfinite(dist)
    if ok.sum() < 200:
        return None
    gap = float(np.median(fixed[idx[ok], 2] - moving[ok, 2]))
    return gap if abs(gap) <= MAX_HEIGHT_GAP_M else None


def _normals(points: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Each point's surface normal in the plane, and whether its surface is
    near vertical: only those constrain (x, y, yaw)."""
    _, idx = cKDTree(points).query(points, k=k)
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    n = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]
    flat = np.linalg.norm(n[:, :2], axis=1)
    return n[:, :2] / np.maximum(flat, 1e-9)[:, None], flat > 0.9


def _weight(info: np.ndarray) -> np.ndarray:
    """Whitening for a closure: CLOSURE_SIGMA along the best-constrained
    direction, looser along weaker ones in proportion to the information."""
    lam, vec = np.linalg.eigh(info)
    lam = np.maximum(lam, 1e-6 * lam[-1] + 1e-12)
    W = np.zeros((3, 3))
    W[:2, :2] = (vec * np.sqrt(lam / lam[-1])).T / CLOSURE_SIGMA[0]
    W[2, 2] = 1.0 / CLOSURE_SIGMA[1]
    return W


def _inverse(a: np.ndarray) -> np.ndarray:
    return between(a, np.zeros(3))


def _minus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    d[2] = _wrap(d[2])
    return d


def _wrap(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi
