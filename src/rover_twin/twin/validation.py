"""Command validation: a plan is rolled forward on the live terrain before the
rover may run it.

The footprint is swept over the live elevation grid before any sim time is
spent; then the rover drives in the simulation, closing the loop on measured
progress, while watchdogs catch the stalls and deflections that contact
causes. docs/twin-api.md is the request and result format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from rover_twin.config import Config
from rover_twin.frames import Pose2D, wrap_deg
from rover_twin.mapping.elevation import UNKNOWN
from rover_twin.twin.terrain import TerrainMap

COAST_MARGIN_M = 0.005        # stop this short; the coast after STOP covers it
STALL_WINDOW_S = 0.5          # a drive slower than STALL_RATIO of the rolling model
STALL_RATIO = 0.6             #   for this long is pushing against something
DEFLECTION_LIMIT_DEG = 15.0   # a straight drive never turns this far on its own
TURN_STALL_DEG_S = 1.2        # a turn slower than this for TURN_STALL_S is stuck
TURN_STALL_S = 2.0
NO_OBSTACLE_M = 999.0
TRAJECTORY_DT_S = 0.05        # the trajectory a result reports, thinned from every physics step


class Command(str, Enum):
    STOP = "STOP"
    FWD = "FWD"
    REV = "REV"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class Step:
    command: Command
    speed: int
    distance: int          # linear units (Limits.distance_units_per_m) or degrees for turns


@dataclass(frozen=True)
class Request:
    start: Pose2D
    goal: Pose2D
    steps: tuple[Step, ...]

    @classmethod
    def from_dict(cls, data: dict) -> "Request":
        """docs/twin-api.md's schema; raises ValueError (KeyError, TypeError) on anything else."""
        def pose(d):
            return Pose2D(*(_number(d[k]) for k in ("x", "y", "yaw_deg")))

        steps = tuple(Step(Command(c["command"]), _integer(c["speed"]), _integer(c["distance"]))
                      for c in data["commands"])
        return cls(pose(data["start_pose"]), pose(data["goal"]), steps)


@dataclass(frozen=True)
class Result:
    approved: bool
    reason: str
    final_pose: Pose2D
    min_clearance_m: float
    failure_type: str | None = None
    failed_command_index: int | None = None
    goal_error_m: float | None = None
    goal_error_yaw_deg: float | None = None
    trajectory: tuple[Pose2D, ...] = ()
    step_results: tuple[dict, ...] = ()
    expected_step_end_poses: tuple[Pose2D, ...] = ()
    step_s: float = 0.0          # simulated time between trajectory samples

    @property
    def duration_s(self) -> float:
        return max(0, len(self.trajectory) - 1) * self.step_s

    def to_dict(self) -> dict:
        def pose(p):
            return {"x": p.x, "y": p.y, "yaw_deg": p.yaw_deg}

        stride = max(1, round(TRAJECTORY_DT_S / self.step_s)) if self.step_s else 1
        thinned = self.trajectory[::stride] + (self.trajectory[-1:] if (len(self.trajectory) - 1) % stride else ())
        data = {"approved": self.approved, "reason": self.reason, "final_pose": pose(self.final_pose),
                "min_clearance_m": self.min_clearance_m, "duration_s": round(self.duration_s, 3),
                "trajectory_dt_s": stride * self.step_s, "trajectory": [pose(p) for p in thinned],
                "step_results": list(self.step_results),
                "expected_step_end_poses": [pose(p) for p in self.expected_step_end_poses]}
        if not self.approved:      # always both, null where no single command is at fault
            data["failure_type"], data["failed_command_index"] = self.failure_type, self.failed_command_index
        for key in ("goal_error_m", "goal_error_yaw_deg"):
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)
        return data


class SimRover(Protocol):
    steps_per_second: float
    ride_height_m: float         # root body above the ground it rests on

    def step(self) -> None: ...

    def pose(self) -> Pose2D: ...

    def place(self, pose: Pose2D, z: float) -> None: ...

    def drive(self, left_rad_s: float, right_rad_s: float) -> None: ...

    def follow(self, pose: Pose2D, z: float) -> None: ...

    def render(self) -> None: ...

    def reload(self) -> None: ...


class Validator:
    def __init__(self, cfg: Config, rover: SimRover, terrain: TerrainMap) -> None:
        self.cfg, self.rover, self.terrain = cfg, rover, terrain
        self._obstacles = np.zeros((0, 2))

    def validate(self, request: Request) -> Result:
        refused = self._check_plan(request)
        if refused is not None:
            return refused
        self._obstacles = self.terrain.obstacle_cells(self.cfg.rover.max_step_m)
        unsafe = self._stand(request.start)       # also for a plan that never moves
        if unsafe is not None:
            return Result(False, unsafe[1], request.start, self._clearance([request.start]), unsafe[0], None)
        self.place(request.start)
        dt = 1.0 / self.rover.steps_per_second
        pose, trajectory, ends, steps_done = self.rover.pose(), [], [], []
        trajectory.append(pose)
        for index, step in enumerate(request.steps):
            samples, failure = self._run(pose, step)
            trajectory.extend(samples[1:])
            pose = samples[-1]
            failure = failure or self._out_of_bounds(samples)
            if failure is not None:
                return Result(False, failure[1], pose, self._clearance(trajectory), failure[0], index,
                              trajectory=tuple(trajectory), step_results=tuple(steps_done),
                              expected_step_end_poses=tuple(ends), step_s=dt)
            ends.append(pose)
            steps_done.append({"command_index": index, "status": "completed", "end_pose": {
                "x": pose.x, "y": pose.y, "yaw_deg": pose.yaw_deg},
                "duration_s": round((len(samples) - 1) * dt, 3),
                "command": {"command": step.command.value, "speed": step.speed, "distance": step.distance}})
        limits = self.cfg.limits
        error_m = math.hypot(pose.x - request.goal.x, pose.y - request.goal.y)
        error_deg = abs(wrap_deg(pose.yaw_deg - request.goal.yaw_deg))
        approved = error_m <= limits.goal_tolerance_m and error_deg <= limits.goal_tolerance_yaw_deg
        reason = ("Plan reached target with no collisions" if approved
                  else f"Final pose error {error_m:.3f} m, yaw error {error_deg:.1f} deg")
        return Result(approved, reason, pose, self._clearance(trajectory), None if approved else "goal_not_reached",
                      None, error_m, error_deg, tuple(trajectory), tuple(steps_done), tuple(ends), dt)

    def resting_z(self, pose: Pose2D) -> float:
        """Root height that puts the rover on the highest known ground under
        it: a rollout must never start inside a bump. Physics settles it from there."""
        heights, states = self._footprint(pose)
        known = heights[states != UNKNOWN]
        return (known.max() if known.size else self.cfg.start.z_m) + self.rover.ride_height_m

    def place(self, pose: Pose2D) -> None:
        """Put the rover at ``pose`` resting on the terrain and let it settle."""
        self.rover.place(pose, self.resting_z(pose))
        for _ in range(max(1, round(self.cfg.twin.settle_s * self.rover.steps_per_second))):
            self.rover.drive(0.0, 0.0)
            self.rover.step()

    def _check_plan(self, request: Request) -> Result | None:
        limits = self.cfg.limits
        refuse = None
        if len(request.steps) > limits.max_commands:
            refuse = ("too_many_commands", None, "Plan has too many commands")
        for index, step in enumerate(request.steps):
            if refuse:
                break
            if step.command != Command.STOP and not limits.min_speed <= step.speed <= limits.max_speed:
                refuse = ("speed_limit", index, "Command speed is outside configured limits")
            elif not 0 <= step.distance <= limits.max_distance:
                refuse = ("distance_limit", index, "Command distance is outside configured limits")
            elif step.command in (Command.LEFT, Command.RIGHT) and step.distance > limits.max_turn_deg:
                refuse = ("turn_distance_limit", index, f"Turn distance {step.distance} deg exceeds "
                          f"{limits.max_turn_deg} deg")
            elif step.command == Command.STOP and (step.speed or step.distance):
                refuse = ("invalid_stop", index, "STOP command must use speed 0 and distance 0")
        if refuse is None:
            return None
        return Result(False, refuse[2], request.start, 0.0, refuse[0], refuse[1])

    def _run(self, pose: Pose2D, step: Step) -> tuple[list[Pose2D], tuple[str, str] | None]:
        if step.command == Command.STOP:
            self.rover.drive(0.0, 0.0)
            self.rover.step()
            return [self.rover.pose()], None
        refused = self._sweep(pose, step)
        if refused is not None:
            return [pose], refused
        return self._turn(step) if step.command in (Command.LEFT, Command.RIGHT) else self._straight(step)

    def _straight(self, step: Step):
        rover, rc = self.rover, self.cfg.rover
        sps = rover.steps_per_second
        left, right = self._wheels(step)
        start = rover.pose()
        samples, anchor, failure = [start], start, None
        heading = math.radians(start.yaw_deg)
        target_m = step.distance / rc.distance_units_per_m
        speed = max(0.01, step.speed * rc.wheel_velocity_scale) * rc.wheel_diameter_m / 2.0
        window = max(1, round(STALL_WINDOW_S * sps))
        steps = max(1, min(round(15.0 * sps), round((2.0 * target_m / speed + 0.5) * sps)))
        for k in range(1, steps + 1):
            rover.drive(left, right)
            rover.step()
            now = rover.pose()
            samples.append(now)
            if abs((now.x - start.x) * math.cos(heading) + (now.y - start.y) * math.sin(heading)) \
                    >= target_m - COAST_MARGIN_M:
                break
            deflection = abs(wrap_deg(now.yaw_deg - start.yaw_deg))
            if deflection > DEFLECTION_LIMIT_DEG:
                failure = ("collision", f"Drive deflected {deflection:.1f} deg off heading during a straight "
                                        "command; glancing contact with terrain")
                break
            if k % window == 0:
                moved, expected = math.hypot(now.x - anchor.x, now.y - anchor.y), speed * window / sps
                if moved < STALL_RATIO * expected:
                    failure = ("collision", f"Drive stalled: moved {moved:.3f} m in {window / sps:.1f} s vs "
                                            f"{expected:.3f} m expected; likely contact with terrain")
                    break
                anchor = now
        samples += self._halt()
        return samples, failure

    def _turn(self, step: Step):
        rover = self.rover
        sps = rover.steps_per_second
        left, right = self._wheels(step)
        samples = [rover.pose()]
        achieved, still, last = 0.0, 0, samples[0].yaw_deg
        timeout = min(round(30.0 * sps), max(1, int(self.cfg.limits.command_timeout_s * sps)))
        for _ in range(timeout):
            rover.drive(left, right)
            rover.step()
            samples.append(rover.pose())
            delta = abs(wrap_deg(samples[-1].yaw_deg - last))
            achieved, last = achieved + delta, samples[-1].yaw_deg
            if achieved >= step.distance:
                break
            still = still + 1 if delta < TURN_STALL_DEG_S / sps else 0
            if still >= round(TURN_STALL_S * sps):
                break
        samples += self._halt()
        if achieved >= step.distance:
            return samples, None
        return samples, ("turn_not_completed", f"Turn reached {achieved:.1f} deg of requested {step.distance:.1f} deg")

    def _halt(self) -> list[Pose2D]:
        samples = []
        for _ in range(10):
            self.rover.drive(0.0, 0.0)
            self.rover.step()
            samples.append(self.rover.pose())
        return samples

    def _wheels(self, step: Step) -> tuple[float, float]:
        v = step.speed * self.cfg.rover.wheel_velocity_scale
        return {Command.FWD: (v, v), Command.REV: (-v, -v), Command.LEFT: (-v, v), Command.RIGHT: (v, -v)}[step.command]

    def _sweep(self, start: Pose2D, step: Step) -> tuple[str, str] | None:
        """Sweep the footprint along the commanded motion before simulating it:
        every pose must stand on observed terrain and cover no obstacle cell."""
        for probe in self._probes(start, step):
            unsafe = self._stand(probe)
            if unsafe is not None:
                return unsafe
        return None

    def _stand(self, pose: Pose2D) -> tuple[str, str] | None:
        """Why the rover cannot stand at ``pose``: unobserved ground or an obstacle under it."""
        _, states = self._footprint(pose)
        if states.size == 0 or np.mean(states != UNKNOWN) < self.cfg.limits.min_known_fraction:
            return "unknown_not_ready", "Planned path crosses terrain the map has not observed."
        inside = self._covered(pose, self._obstacles)
        if inside.any():
            x, y = self._obstacles[np.argmax(inside)]
            return "collision", f"Predicted collision: rover footprint would overlap terrain near ({x:.2f}, {y:.2f})"
        return None

    def _probes(self, start: Pose2D, step: Step) -> list[Pose2D]:
        """Poses along a command: every 5 cm of a drive, every 10 deg of a turn."""
        if step.command in (Command.FWD, Command.REV):
            sign = 1.0 if step.command == Command.FWD else -1.0
            total = sign * step.distance / self.cfg.rover.distance_units_per_m
            n = max(1, math.ceil(abs(total) / 0.05))
            yaw = math.radians(start.yaw_deg)
            return [Pose2D(start.x + math.cos(yaw) * total * i / n, start.y + math.sin(yaw) * total * i / n,
                           start.yaw_deg) for i in range(n + 1)]
        total = (1.0 if step.command == Command.LEFT else -1.0) * step.distance
        n = max(1, int(abs(total) / 10.0))
        return [Pose2D(start.x, start.y, start.yaw_deg + total * i / n) for i in range(n + 1)]

    def _covered(self, pose: Pose2D, xy: np.ndarray) -> np.ndarray:
        yaw = math.radians(pose.yaw_deg)
        d = xy - [pose.x, pose.y]
        u = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        v = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
        rc = self.cfg.rover
        return (np.abs(u) < rc.footprint_half_length_m) & (np.abs(v) < rc.footprint_half_width_m)

    def _footprint(self, pose: Pose2D):
        rc = self.cfg.rover
        return self.terrain.footprint(pose, rc.footprint_half_length_m, rc.footprint_half_width_m)

    def _out_of_bounds(self, samples: list[Pose2D]) -> tuple[str, str] | None:
        x0, x1, y0, y1 = self.cfg.limits.lab_bounds_m
        if any(not (x0 <= p.x <= x1 and y0 <= p.y <= y1) for p in samples):
            return "out_of_bounds", "Rover left configured lab bounds"
        return None

    def _clearance(self, trajectory: list[Pose2D]) -> float:
        """Closest approach of the rover centre to an obstacle, less the footprint's reach."""
        if len(self._obstacles) == 0:
            return NO_OBSTACLE_M
        reach = math.hypot(self.cfg.rover.footprint_half_length_m, self.cfg.rover.footprint_half_width_m)
        centres = np.array([[p.x, p.y] for p in trajectory[::10] + trajectory[-1:]])
        nearest = min(np.min(np.linalg.norm(self._obstacles - c, axis=1)) for c in centres)
        return max(0.0, float(nearest) - reach)


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _integer(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer, got {value!r}")
    return value
