"""Routes to a goal on the live terrain: where the rover can go, the shortest
way there, and the commands that drive it, ready to rehearse.

The planner works on a coarse copy of the elevation grid (``route.cell_m``).
A cell is free when the rover, centred there, can turn on the spot without
its footprint coming within ``route.clearance_m`` of an obstacle, stands on
known ground at every heading as validation requires, and is inside the lab
bounds. A* finds the shortest 8-connected path over free cells; the path is
then pulled straight wherever the line between two of its points stays free,
so the rover turns only at the corners left. Each corner becomes a turn and
the straight after it one or more drives of at most ``limits.max_distance``,
split into plans of at most ``limits.max_commands`` that each end on a pose
validation can check.

Free is conservative: turning anywhere needs the footprint's half diagonal
(about 0.40 m) plus the clearance on every side, so a gap narrower than about
1 m is closed even where the rover could drive straight through. The twin
still rehearses every plan before a route is approved.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from rover_twin.config import Config
from rover_twin.frames import Pose2D, wrap_deg
from rover_twin.mapping.elevation import UNKNOWN
from rover_twin.twin.terrain import TerrainMap

FREE, OBSTACLE, UNSEEN, OUTSIDE = 0, 1, 2, 3
HEADINGS_DEG = range(0, 180, 15)     # the footprint is the same turned 180 deg
MAX_POINTS = 32                      # of one route_costs request
WHY = {OBSTACLE: "within {reach:.2f} m of an obstacle, too close to turn there",
       UNSEEN: "on or beside ground the map has not observed",
       OUTSIDE: "outside the configured lab bounds"}


@dataclass(frozen=True)
class FreeSpace:
    """Where the rover's centre may be. ``why[i, j]`` is FREE or the reason it is not."""

    origin: tuple[float, float]      # corner of cell [0, 0]
    cell_m: float
    why: np.ndarray
    reach_m: float                   # obstacles stay farther than this from a free cell

    @property
    def free(self) -> np.ndarray:
        return self.why == FREE

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        i, j = int(math.floor((x - self.origin[0]) / self.cell_m)), int(math.floor((y - self.origin[1]) / self.cell_m))
        return (i, j) if 0 <= i < self.why.shape[0] and 0 <= j < self.why.shape[1] else None

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return self.origin[0] + (i + 0.5) * self.cell_m, self.origin[1] + (j + 0.5) * self.cell_m

    def blocked(self, x: float, y: float) -> str | None:
        """Why the rover cannot be centred at (x, y), or None if it can."""
        c = self.cell(x, y)
        if c is None:
            return "outside the mapped area"
        why = int(self.why[c])
        return None if why == FREE else WHY[why].format(reach=self.reach_m)

    def clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Whether every cell on the straight line from a to b is free."""
        n = max(1, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / (0.25 * self.cell_m)))
        t = np.linspace(0.0, 1.0, n + 1)
        i = np.floor((a[0] + t * (b[0] - a[0]) - self.origin[0]) / self.cell_m).astype(int)
        j = np.floor((a[1] + t * (b[1] - a[1]) - self.origin[1]) / self.cell_m).astype(int)
        inside = (i >= 0) & (i < self.why.shape[0]) & (j >= 0) & (j < self.why.shape[1])
        return bool(inside.all() and (self.why[i, j] == FREE).all())


def free_space(terrain: TerrainMap, cfg: Config) -> FreeSpace | None:
    """The planning grid for the terrain as it is now; None before anything is mapped."""
    grid = terrain.grid(cfg.rover.max_step_m)
    if grid is None:
        return None
    origin, _, state, obstacle = grid
    fine, rc = terrain.cell_m, cfg.rover
    reach = math.hypot(rc.footprint_half_length_m, rc.footprint_half_width_m) + cfg.route.clearance_m
    # Distance from each cell centre to the nearest obstacle cell centre: the
    # footprint, turned any way, stays within its half diagonal of its centre.
    near = (ndimage.distance_transform_edt(~obstacle) * fine <= reach) if obstacle.any() \
        else np.zeros(obstacle.shape, bool)
    # The least known fraction under the footprint at any heading, as validation measures it.
    known = (state != UNKNOWN).astype(float)
    least = np.ones_like(known)
    for yaw in HEADINGS_DEG:
        kernel = _footprint(rc.footprint_half_length_m, rc.footprint_half_width_m, yaw, fine)
        least = np.minimum(least, signal.fftconvolve(known, kernel, mode="same") / kernel.sum())
    unseen = least < cfg.limits.min_known_fraction - 1e-6

    k = max(1, round(cfg.route.cell_m / fine))      # fine cells per planning cell, each way
    shape = (-(-state.shape[0] // k), -(-state.shape[1] // k))
    why = np.full(shape, FREE, np.uint8)
    why[_any(unseen, k, shape)] = UNSEEN
    why[_any(near, k, shape)] = OBSTACLE
    cell = k * fine
    x0, x1, y0, y1 = cfg.limits.lab_bounds_m
    xs = origin[0] + (np.arange(shape[0]) + 0.5) * cell
    ys = origin[1] + (np.arange(shape[1]) + 0.5) * cell
    outside = ~(((xs >= x0) & (xs <= x1))[:, None] & ((ys >= y0) & (ys <= y1))[None, :])
    why[outside] = OUTSIDE
    return FreeSpace((float(origin[0]), float(origin[1])), cell, why, reach)


@dataclass(frozen=True)
class RouteRequest:
    start: Pose2D
    goal_x: float
    goal_y: float
    goal_yaw_deg: float | None       # None: arrive facing along the last straight
    speed: int | None = None         # simulated command speed; None is route.speed
    rehearse: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "RouteRequest":
        """docs/twin-api.md's schema; raises ValueError (KeyError, TypeError) on anything else."""
        s, g = data["start_pose"], data["goal"]
        yaw = g.get("yaw_deg")
        speed = data.get("speed")
        rehearse = data.get("rehearse", True)
        if not isinstance(rehearse, bool):
            raise ValueError(f"rehearse must be true or false, got {rehearse!r}")
        return cls(Pose2D(_number(s["x"]), _number(s["y"]), _number(s["yaw_deg"])), _number(g["x"]),
                   _number(g["y"]), None if yaw is None else _number(yaw),
                   None if speed is None else _integer(speed), rehearse)


@dataclass(frozen=True)
class Route:
    found: bool
    reason: str
    failure_type: str | None = None
    waypoints: tuple[tuple[float, float], ...] = ()
    plans: tuple[dict, ...] = ()     # {"commands": [...], "goal": {...}}, as /validate_plan takes them
    length_m: float = 0.0
    turn_deg: float = 0.0


def plan(space: FreeSpace | None, cfg: Config, request: RouteRequest) -> Route:
    """The shortest safe route from the request's start to its goal, as plans."""
    speed = cfg.route.speed if request.speed is None else request.speed
    if not cfg.limits.min_speed <= speed <= cfg.limits.max_speed:
        return Route(False, f"Speed {speed} is outside {cfg.limits.min_speed}..{cfg.limits.max_speed}", "speed_limit")
    if space is None:
        return Route(False, "Nothing has been mapped yet", "unknown_not_ready")
    start, goal = request.start, (request.goal_x, request.goal_y)
    blocked = space.blocked(start.x, start.y)
    if blocked:
        return Route(False, f"Start ({start.x:.2f}, {start.y:.2f}) is {blocked}", "start_blocked")
    blocked = space.blocked(*goal)
    if blocked:
        return Route(False, f"Goal ({goal[0]:.2f}, {goal[1]:.2f}) is {blocked}", "goal_blocked")
    s, g = space.cell(start.x, start.y), space.cell(*goal)
    cost, parent = _search(space.free, s, [g])
    if not math.isfinite(cost[g]):
        return Route(False, "No route: free ground does not connect the start to the goal "
                            f"with {space.reach_m:.2f} m to turn", "no_route")
    waypoints = _straighten(space, _points(space, parent, s, g, (start.x, start.y), goal))
    commands, turned = _commands(cfg, start, waypoints, request.goal_yaw_deg, speed)
    plans = _plans(cfg, start, commands, request)
    length = sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:]))
    return Route(True, f"Route of {length:.2f} m with {len(waypoints) - 2} corners", None,
                 tuple(waypoints), tuple(plans), length, turned)


def costs(space: FreeSpace | None, points: list[tuple[float, float]]) -> dict:
    """Route length and the turning at its corners between every ordered pair
    of points: what a scheduler needs to cost a leg. Turning to leave or to
    arrive at a heading is not included; it depends on the legs either side.
    None where no route exists."""
    n = len(points)
    length = [[None] * n for _ in range(n)]
    turn = [[None] * n for _ in range(n)]
    why = [space.blocked(*p) if space is not None else "outside the mapped area" for p in points]
    for a in range(n):
        if why[a]:
            continue
        s = space.cell(*points[a])
        targets = [space.cell(*points[b]) for b in range(n) if not why[b]]
        cost, parent = _search(space.free, s, targets)
        for b in range(n):
            if why[b] or not math.isfinite(cost[space.cell(*points[b])]):
                continue
            if a == b:
                length[a][b], turn[a][b] = 0.0, 0.0
                continue
            ways = _straighten(space, _points(space, parent, s, space.cell(*points[b]), points[a], points[b]))
            length[a][b] = round(sum(math.dist(p, q) for p, q in zip(ways, ways[1:])), 3)
            turn[a][b] = round(sum((abs(wrap_deg(_bearing(q, r) - _bearing(p, q)))
                                    for p, q, r in zip(ways, ways[1:], ways[2:])), 0.0), 1)
    return {"length_m": length, "turn_deg": turn,
            "blocked": [None if w is None else f"({p[0]:.2f}, {p[1]:.2f}) is {w}" for p, w in zip(points, why)]}


def _search(free: np.ndarray, start: tuple[int, int], goals: list[tuple[int, int]]):
    """Dijkstra (A* toward a single goal) over free cells, 8-connected, no
    corner cutting. Returns the cost (cells) to every cell reached, inf
    elsewhere, and each reached cell's parent."""
    ni, nj = free.shape
    cost = np.full(free.shape, np.inf)
    parent = np.full(free.shape, -1, np.int64)
    left = set(goals)
    one = goals[0] if len(goals) == 1 else None

    def h(c):
        if one is None:
            return 0.0
        di, dj = abs(c[0] - one[0]), abs(c[1] - one[1])
        return max(di, dj) + (math.sqrt(2) - 1) * min(di, dj)

    cost[start] = 0.0
    heap = [(h(start), 0.0, start)]
    while heap and left:
        _, c, (i, j) = heapq.heappop(heap)
        if c > cost[i, j]:
            continue
        left.discard((i, j))
        for di, dj, step in _NEIGHBOURS:
            a, b = i + di, j + dj
            if not (0 <= a < ni and 0 <= b < nj) or not free[a, b]:
                continue
            if di and dj and not (free[i + di, j] and free[i, j + dj]):
                continue
            if c + step < cost[a, b]:
                cost[a, b], parent[a, b] = c + step, i * nj + j
                heapq.heappush(heap, (c + step + h((a, b)), c + step, (a, b)))
    return cost, parent


_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def _points(space: FreeSpace, parent: np.ndarray, s, g, start_xy, goal_xy) -> list[tuple[float, float]]:
    """The searched path as points: the exact start, the cell centres between, the exact goal."""
    nj, cells, c = space.why.shape[1], [], g
    while c != s:
        cells.append(c)
        c = divmod(int(parent[c]), nj)
    middle = [space.centre(*c) for c in reversed(cells[1:])]
    return [tuple(start_xy)] + middle + [tuple(goal_xy)]


def _straighten(space: FreeSpace, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep only the corners: from each kept point, jump to the farthest
    point of the path a straight, free line reaches."""
    out, i = [points[0]], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not space.clear(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return _merge_corners(space, out)


def _merge_corners(space: FreeSpace, ways: list[tuple[float, float]], close_m: float = 0.5):
    """Going round the clearance circle of an obstacle's corner leaves a few
    short legs, each a turn for the rover. Two corners joined by a short leg
    become one where the legs either side, extended, meet: outside the
    circle, so usually still free; kept only if both new legs are clear."""
    changed = True
    while changed:
        changed = False
        for k in range(1, len(ways) - 2):
            p, a, b, q = ways[k - 1], ways[k], ways[k + 1], ways[k + 2]
            if math.dist(a, b) > close_m:
                continue
            x = _meet(p, a, b, q)
            if x is not None and math.dist(x, a) <= close_m and space.clear(p, x) and space.clear(x, q):
                ways = ways[:k] + [x] + ways[k + 2:]
                changed = True
                break
    return ways


def _meet(p, a, b, q):
    """Where the ray p->a, beyond a, meets the ray q->b, beyond b; None if they do not."""
    d1, d2 = (a[0] - p[0], a[1] - p[1]), (b[0] - q[0], b[1] - q[1])
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-9:
        return None
    rx, ry = q[0] - p[0], q[1] - p[1]
    t, u = (rx * d2[1] - ry * d2[0]) / det, (rx * d1[1] - ry * d1[0]) / det
    if t < 1.0 or u < 1.0:
        return None
    return p[0] + t * d1[0], p[1] + t * d1[1]


def _commands(cfg: Config, start: Pose2D, waypoints, goal_yaw: float | None, speed: int):
    """Turn-then-drive commands through the waypoints, each with the nominal
    pose it ends at. Headings are tracked as the rounded turns leave them, so
    rounding never accumulates."""
    units, lim = cfg.rover.distance_units_per_m, cfg.limits
    x, y, heading = start.x, start.y, start.yaw_deg
    out, turned = [], 0.0

    def turn(to_deg: float) -> None:
        nonlocal heading, turned
        total = round(wrap_deg(to_deg - heading))
        parts = math.ceil(abs(total) / lim.max_turn_deg) if total else 0
        for k in range(parts):
            deg = abs(total) // parts + (1 if k < abs(total) % parts else 0)
            heading = wrap_deg(heading + math.copysign(deg, total))
            out.append(({"command": "LEFT" if total > 0 else "RIGHT", "speed": speed, "distance": deg},
                        Pose2D(x, y, heading)))
        turned += abs(total)

    for wx, wy in waypoints[1:]:
        total = round(math.hypot(wx - x, wy - y) * units)
        if total == 0:
            continue
        turn(math.degrees(math.atan2(wy - y, wx - x)))
        parts = math.ceil(total / lim.max_distance)
        for k in range(parts):
            d = total // parts + (1 if k < total % parts else 0)
            x += d / units * math.cos(math.radians(heading))
            y += d / units * math.sin(math.radians(heading))
            out.append(({"command": "FWD", "speed": speed, "distance": d}, Pose2D(x, y, heading)))
    if goal_yaw is not None:
        turn(goal_yaw)
    return out, turned


def _plans(cfg: Config, start: Pose2D, commands, request: RouteRequest) -> list[dict]:
    """Commands in plans of at most limits.max_commands, each ending on its
    nominal pose; the last ends on the requested goal."""
    size = cfg.limits.max_commands
    plans = []
    for k in range(0, len(commands), size):
        chunk = commands[k:k + size]
        end = chunk[-1][1]
        if k + size >= len(commands):
            end = Pose2D(request.goal_x, request.goal_y,
                         end.yaw_deg if request.goal_yaw_deg is None else request.goal_yaw_deg)
        plans.append({"commands": [c for c, _ in chunk], "goal": _pose(end)})
    return plans


def _footprint(half_length: float, half_width: float, yaw_deg: float, cell: float) -> np.ndarray:
    """The cells under the footprint turned to ``yaw_deg``, as a symmetric kernel."""
    r = math.ceil(math.hypot(half_length, half_width) / cell)
    u, v = np.meshgrid(np.arange(-r, r + 1) * cell, np.arange(-r, r + 1) * cell, indexing="ij")
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    along, across = c * u + s * v, -s * u + c * v
    return ((np.abs(along) <= half_length) & (np.abs(across) <= half_width)).astype(float)


def _any(mask: np.ndarray, k: int, shape) -> np.ndarray:
    """Whether any fine cell in each k x k block is set."""
    padded = np.zeros((shape[0] * k, shape[1] * k), bool)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded.reshape(shape[0], k, shape[1], k).any(axis=(1, 3))


def _bearing(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _pose(p: Pose2D) -> dict:
    return {"x": round(p.x, 4), "y": round(p.y, 4), "yaw_deg": round(p.yaw_deg, 2)}


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _integer(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer, got {value!r}")
    return value
