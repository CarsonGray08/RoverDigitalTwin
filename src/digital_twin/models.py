from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationError(ValueError):
    """Raised when an external JSON-like object violates the strict schema."""


class CommandName(str, Enum):
    STOP = "STOP"
    FWD = "FWD"
    REV = "REV"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


COMMAND_TO_INT: dict[CommandName, int] = {
    CommandName.STOP: 0,
    CommandName.FWD: 1,
    CommandName.REV: 2,
    CommandName.LEFT: 3,
    CommandName.RIGHT: 4,
}

INT_TO_COMMAND: dict[int, CommandName] = {value: key for key, value in COMMAND_TO_INT.items()}


def _require_keys(data: dict[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ValidationError(f"{path} has unexpected fields: {sorted(extra)}")
    missing = required - set(data)
    if missing:
        raise ValidationError(f"{path} is missing required fields: {sorted(missing)}")


def _as_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path} must be a number")
    return float(value)


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{path} must be an integer")
    return value


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{path} must be a string")
    return value


def _as_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be an object")
    return value


def _as_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list")
    return value


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw_deg: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "pose") -> "Pose":
        _require_keys(data, {"x", "y", "yaw_deg"}, {"x", "y", "yaw_deg"}, path)
        return cls(
            x=_as_number(data["x"], f"{path}.x"),
            y=_as_number(data["y"], f"{path}.y"),
            yaw_deg=_as_number(data["yaw_deg"], f"{path}.yaw_deg"),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "yaw_deg": self.yaw_deg}


@dataclass(frozen=True)
class CommandStep:
    command: CommandName
    speed: int
    distance: int

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "command") -> "CommandStep":
        _require_keys(data, {"command", "speed", "distance"}, {"command", "speed", "distance"}, path)
        command_value = _as_str(data["command"], f"{path}.command")
        try:
            command = CommandName(command_value)
        except ValueError as exc:
            allowed = [item.value for item in CommandName]
            raise ValidationError(f"{path}.command must be one of {allowed}") from exc
        return cls(
            command=command,
            speed=_as_int(data["speed"], f"{path}.speed"),
            distance=_as_int(data["distance"], f"{path}.distance"),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {"command": self.command.value, "speed": self.speed, "distance": self.distance}

    def to_ros_payload(self) -> list[int]:
        return [COMMAND_TO_INT[self.command], self.speed, self.distance]


@dataclass(frozen=True)
class CommandPlan:
    goal: Pose
    commands: tuple[CommandStep, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "plan") -> "CommandPlan":
        _require_keys(data, {"goal", "commands"}, {"goal", "commands"}, path)
        commands = tuple(
            CommandStep.from_dict(_as_dict(command, f"{path}.commands[{index}]"), f"{path}.commands[{index}]")
            for index, command in enumerate(_as_list(data["commands"], f"{path}.commands"))
        )
        return cls(goal=Pose.from_dict(_as_dict(data["goal"], f"{path}.goal"), f"{path}.goal"), commands=commands)

    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal.to_dict(), "commands": [command.to_dict() for command in self.commands]}


@dataclass(frozen=True)
class LabBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @classmethod
    def default(cls) -> "LabBounds":
        return cls(min_x=-5.0, max_x=5.0, min_y=-5.0, max_y=5.0)

    def contains(self, pose: Pose) -> bool:
        return self.min_x <= pose.x <= self.max_x and self.min_y <= pose.y <= self.max_y

    def to_dict(self) -> dict[str, float]:
        return {"min_x": self.min_x, "max_x": self.max_x, "min_y": self.min_y, "max_y": self.max_y}


@dataclass(frozen=True)
class RectZone:
    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, pose: Pose) -> bool:
        return self.min_x <= pose.x <= self.max_x and self.min_y <= pose.y <= self.max_y

    def clearance(self, pose: Pose) -> float:
        if self.contains(pose):
            return 0.0
        dx = max(self.min_x - pose.x, 0.0, pose.x - self.max_x)
        dy = max(self.min_y - pose.y, 0.0, pose.y - self.max_y)
        return (dx * dx + dy * dy) ** 0.5

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
        }


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    x: float
    y: float
    radius_m: float

    def clearance(self, pose: Pose) -> float:
        center_distance = ((pose.x - self.x) ** 2 + (pose.y - self.y) ** 2) ** 0.5
        return max(0.0, center_distance - self.radius_m)

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "x": self.x, "y": self.y, "radius_m": self.radius_m}


@dataclass(frozen=True)
class SafetyLimits:
    min_speed: int = 0
    max_speed: int = 60
    max_distance: int = 30000
    min_clearance_m: float = 0.25
    goal_tolerance_m: float = 0.15
    goal_tolerance_yaw_deg: float = 8.0
    max_commands: int = 12
    command_timeout_s: float = 60.0
    distance_units_per_m: float = 23000.0
    turn_degrees_per_distance_unit: float = 90.0 / 24000.0
    max_turn_degrees: int = 360

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "max_distance": self.max_distance,
            "min_clearance_m": self.min_clearance_m,
            "goal_tolerance_m": self.goal_tolerance_m,
            "goal_tolerance_yaw_deg": self.goal_tolerance_yaw_deg,
            "max_commands": self.max_commands,
            "command_timeout_s": self.command_timeout_s,
            "distance_units_per_m": self.distance_units_per_m,
            "turn_degrees_per_distance_unit": self.turn_degrees_per_distance_unit,
            "max_turn_degrees": self.max_turn_degrees,
        }


@dataclass(frozen=True)
class WorldModel:
    lab_bounds: LabBounds = field(default_factory=LabBounds.default)
    obstacles: tuple[CircleObstacle, ...] = ()
    forbidden_zones: tuple[RectZone, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "lab_bounds": self.lab_bounds.to_dict(),
            "known_obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "forbidden_zones": [zone.to_dict() for zone in self.forbidden_zones],
        }


@dataclass(frozen=True)
class ValidationRequest:
    start_pose: Pose
    goal: Pose
    commands: tuple[CommandStep, ...]
    validation_profile: str = "safe_lab_default"

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "validation_request") -> "ValidationRequest":
        _require_keys(
            data,
            {"start_pose", "goal", "commands", "validation_profile"},
            {"start_pose", "goal", "commands"},
            path,
        )
        commands = tuple(
            CommandStep.from_dict(_as_dict(command, f"{path}.commands[{index}]"), f"{path}.commands[{index}]")
            for index, command in enumerate(_as_list(data["commands"], f"{path}.commands"))
        )
        return cls(
            start_pose=Pose.from_dict(_as_dict(data["start_pose"], f"{path}.start_pose"), f"{path}.start_pose"),
            goal=Pose.from_dict(_as_dict(data["goal"], f"{path}.goal"), f"{path}.goal"),
            commands=commands,
            validation_profile=_as_str(data.get("validation_profile", "safe_lab_default"), f"{path}.validation_profile"),
        )

    @classmethod
    def from_plan(
        cls,
        start_pose: Pose,
        plan: CommandPlan,
        validation_profile: str = "safe_lab_default",
    ) -> "ValidationRequest":
        return cls(
            start_pose=start_pose,
            goal=plan.goal,
            commands=plan.commands,
            validation_profile=validation_profile,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_pose": self.start_pose.to_dict(),
            "goal": self.goal.to_dict(),
            "commands": [command.to_dict() for command in self.commands],
            "validation_profile": self.validation_profile,
        }


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    reason: str
    final_pose: Pose
    min_clearance_m: float
    failure_type: str | None = None
    failed_command_index: int | None = None
    goal_error_m: float | None = None
    goal_error_yaw_deg: float | None = None
    trajectory: tuple[Pose, ...] = ()
    step_results: tuple[dict[str, Any], ...] = ()
    expected_step_end_poses: tuple[Pose, ...] = ()
    # Camera frames captured at the failure pose ({"vision": ..., "vision_chase": ...},
    # same encoding as /observation) so the planner can SEE what it hit.
    failure_frames: dict[str, Any] | None = None
    # Free-direction sweep at the failure pose: list of {heading_deg, free_m,
    # open, clear_point} so the planner knows which way out is clear.
    failure_clearance: tuple[dict[str, Any], ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str = "validation_result") -> "ValidationResult":
        allowed = {
            "approved",
            "reason",
            "final_pose",
            "min_clearance_m",
            "failure_type",
            "failed_command_index",
            "goal_error_m",
            "goal_error_yaw_deg",
            "trajectory",
            "step_results",
            "expected_step_end_poses",
            "failure_frames",
            "failure_clearance",
        }
        _require_keys(data, allowed, {"approved", "reason", "final_pose", "min_clearance_m"}, path)
        if not isinstance(data["approved"], bool):
            raise ValidationError(f"{path}.approved must be a boolean")
        failure_type = data.get("failure_type")
        if failure_type is not None:
            failure_type = _as_str(failure_type, f"{path}.failure_type")
        failed_index = data.get("failed_command_index")
        if failed_index is not None:
            failed_index = _as_int(failed_index, f"{path}.failed_command_index")
        return cls(
            approved=data["approved"],
            reason=_as_str(data["reason"], f"{path}.reason"),
            final_pose=Pose.from_dict(_as_dict(data["final_pose"], f"{path}.final_pose"), f"{path}.final_pose"),
            min_clearance_m=_as_number(data["min_clearance_m"], f"{path}.min_clearance_m"),
            failure_type=failure_type,
            failed_command_index=failed_index,
            goal_error_m=None if data.get("goal_error_m") is None else _as_number(data["goal_error_m"], f"{path}.goal_error_m"),
            goal_error_yaw_deg=None
            if data.get("goal_error_yaw_deg") is None
            else _as_number(data["goal_error_yaw_deg"], f"{path}.goal_error_yaw_deg"),
            trajectory=tuple(
                Pose.from_dict(_as_dict(pose, f"{path}.trajectory[{index}]"), f"{path}.trajectory[{index}]")
                for index, pose in enumerate(data.get("trajectory", []))
            ),
            step_results=tuple(_as_dict(item, f"{path}.step_results[]") for item in data.get("step_results", [])),
            expected_step_end_poses=tuple(
                Pose.from_dict(
                    _as_dict(pose, f"{path}.expected_step_end_poses[{index}]"),
                    f"{path}.expected_step_end_poses[{index}]",
                )
                for index, pose in enumerate(data.get("expected_step_end_poses", []))
            ),
            failure_frames=None
            if data.get("failure_frames") is None
            else _as_dict(data["failure_frames"], f"{path}.failure_frames"),
            failure_clearance=None
            if data.get("failure_clearance") is None
            else tuple(
                _as_dict(item, f"{path}.failure_clearance[{index}]")
                for index, item in enumerate(data["failure_clearance"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "approved": self.approved,
            "reason": self.reason,
            "final_pose": self.final_pose.to_dict(),
            "min_clearance_m": self.min_clearance_m,
            "trajectory": [pose.to_dict() for pose in self.trajectory],
            "step_results": list(self.step_results),
            "expected_step_end_poses": [pose.to_dict() for pose in self.expected_step_end_poses],
        }
        if self.failure_type is not None:
            data["failure_type"] = self.failure_type
        if self.failed_command_index is not None:
            data["failed_command_index"] = self.failed_command_index
        if self.goal_error_m is not None:
            data["goal_error_m"] = self.goal_error_m
        if self.goal_error_yaw_deg is not None:
            data["goal_error_yaw_deg"] = self.goal_error_yaw_deg
        if self.failure_frames is not None:
            data["failure_frames"] = self.failure_frames
        if self.failure_clearance is not None:
            data["failure_clearance"] = list(self.failure_clearance)
        return data
