from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from digital_twin.config import PlannerSettings
from digital_twin.geometry import bearing_deg, distance_m, signed_yaw_delta_deg, yaw_error_deg
from digital_twin.models import CommandName, CommandPlan, CommandStep, Pose, ValidationResult


# Lab-frame point quoted in the Isaac validator's collision reasons, e.g.
# "... would overlap scan geometry near (0.90, -2.43) after 0.30 m of FWD".
_COLLISION_POINT_RE = re.compile(r"near \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")

# min_clearance_m is a large sentinel when the validator has no indexed
# obstacles to measure against; passing it to the model is just noise.
_NO_CLEARANCE_DATA = 900.0


@dataclass(frozen=True)
class FailedAttempt:
    plan: CommandPlan
    validation: ValidationResult

    def to_dict(self) -> dict:
        validation = self.validation.to_dict()
        if "failure_frames" in validation:
            # base64 frames are for the vision planner, not for result dumps
            validation["failure_frames"] = "<failure-point camera frames omitted>"
        return {"plan": self.plan.to_dict(), "validation": validation}

    def collision_point(self) -> tuple[float, float] | None:
        """Lab-frame collision point parsed from the validator's reason, if any."""
        if self.validation.failure_type != "collision" or not self.validation.reason:
            return None
        match = _COLLISION_POINT_RE.search(self.validation.reason)
        if match is None:
            return None
        return (float(match.group(1)), float(match.group(2)))

    def to_summary(self) -> dict:
        """Concise validator-rejection feedback for the planning model."""
        summary = {
            "rejected_commands": [command.to_dict() for command in self.plan.commands],
            "failure_type": self.validation.failure_type,
            "reason": self.validation.reason,
            "failed_command_index": self.validation.failed_command_index,
            "stopped_at_pose": self.validation.final_pose.to_dict(),
        }
        if self.validation.min_clearance_m < _NO_CLEARANCE_DATA:
            summary["min_clearance_m"] = round(self.validation.min_clearance_m, 3)
        point = self.collision_point()
        if point is not None:
            summary["collision_near"] = {"x": point[0], "y": point[1]}
        if self.validation.failure_type == "goal_not_reached":
            if self.validation.goal_error_m is not None:
                summary["goal_error_m"] = round(self.validation.goal_error_m, 3)
            if self.validation.goal_error_yaw_deg is not None:
                summary["goal_error_yaw_deg"] = round(self.validation.goal_error_yaw_deg, 1)
            if self.validation.failed_command_index is None:
                summary["route_was_driveable"] = True
        if self.validation.failure_clearance:
            # Only the directions worth driving, most-open first; the full
            # blocked set is noise for the planner.
            summary["clear_directions_from_failure"] = [
                {"heading_deg": d["heading_deg"], "free_m": d["free_m"], "clear_point": d["clear_point"]}
                for d in self.validation.failure_clearance
                if d.get("open")
            ]
        return summary


@dataclass(frozen=True)
class ExecutionOutcome:
    """What actually happened after a command slice ran on the rover."""

    executed_commands: tuple[CommandStep, ...]
    expected_pose: Pose
    observed_pose: Pose
    aborted: bool
    note: str = ""

    def position_drift_m(self) -> float:
        return distance_m(self.observed_pose, self.expected_pose)

    def yaw_drift_deg(self) -> float:
        return yaw_error_deg(self.observed_pose.yaw_deg, self.expected_pose.yaw_deg)

    def to_summary(self) -> dict:
        """Concise execution feedback so the model can correct its next plan."""
        return {
            "executed_commands": [command.to_dict() for command in self.executed_commands],
            "expected_pose_after": self.expected_pose.to_dict(),
            "observed_pose_after": self.observed_pose.to_dict(),
            "position_drift_m": round(self.position_drift_m(), 3),
            "yaw_drift_deg": round(self.yaw_drift_deg(), 1),
            "aborted": self.aborted,
            "note": self.note,
        }


@dataclass(frozen=True)
class PlannerContext:
    current_simulated_pose: Pose
    target_pose: Pose
    allowed_commands: tuple[CommandName, ...]
    settings: PlannerSettings
    previous_failures: tuple[FailedAttempt, ...] = ()
    recent_executions: tuple[ExecutionOutcome, ...] = ()

    def goal_relative(self) -> dict:
        """Goal range and bearing relative to the rover's current heading.

        The model has no map, so it cannot localize the goal from the camera
        alone. We compute the easy geometry here and hand the model a heading it
        can act on, leaving the camera to do obstacle avoidance only.
        """
        range_m = distance_m(self.current_simulated_pose, self.target_pose)
        heading_to_goal = bearing_deg(self.current_simulated_pose, self.target_pose)
        relative_bearing = signed_yaw_delta_deg(self.current_simulated_pose.yaw_deg, heading_to_goal)
        final_yaw_delta = signed_yaw_delta_deg(self.current_simulated_pose.yaw_deg, self.target_pose.yaw_deg)
        # After turning to face the goal and driving there, the rover faces
        # heading_to_goal; this is the extra turn to then match the target yaw.
        final_turn_after_arrival = signed_yaw_delta_deg(heading_to_goal, self.target_pose.yaw_deg)
        if range_m < self.settings.limits.goal_tolerance_m:
            turn_to_face_goal = "NONE"
        elif relative_bearing > 1.0:
            turn_to_face_goal = "LEFT"
        elif relative_bearing < -1.0:
            turn_to_face_goal = "RIGHT"
        else:
            turn_to_face_goal = "NONE"
        return {
            "range_m": round(range_m, 3),
            "bearing_deg": round(relative_bearing, 1),
            "turn_to_face_goal": turn_to_face_goal,
            "final_yaw_delta_deg": round(final_yaw_delta, 1),
            "final_yaw_turn_after_arrival_deg": round(final_turn_after_arrival, 1),
        }

    def _command_units(self) -> dict:
        """Spell out what a command's distance field means.

        FWD/REV distance is scaled units (meters * distance_units_per_m), but
        LEFT/RIGHT distance is whole degrees. Two different scales in one field,
        so we state both explicitly rather than handing the model raw constants.
        """
        units_per_m = round(self.settings.limits.distance_units_per_m)
        max_step = self.settings.limits.max_distance
        return {
            "FWD_REV_distance": f"integer = meters * {units_per_m} (e.g. 1.0 m -> {units_per_m})",
            "LEFT_RIGHT_distance": "integer = whole degrees to turn (e.g. 90 -> quarter turn); NOT scaled",
            "STOP": "speed 0 and distance 0",
            "max_distance_per_step": max_step,
            "hint": (
                f"to drive goal_relative_to_rover.range_m meters, use FWD distance = "
                f"round(range_m * {units_per_m}), split into steps no larger than {max_step}"
            ),
        }

    def to_dict(self) -> dict:
        return {
            "current_simulated_rover_pose": self.current_simulated_pose.to_dict(),
            "target_pose": self.target_pose.to_dict(),
            "goal_relative_to_rover": self.goal_relative(),
            "allowed_commands": [command.value for command in self.allowed_commands],
            "speed_limits": {
                "min_speed": self.settings.limits.min_speed,
                "max_speed": self.settings.limits.max_speed,
            },
            "command_units": self._command_units(),
            "rejected_plan_feedback": [failure.to_summary() for failure in self.previous_failures[-4:]],
            "execution_feedback": [outcome.to_summary() for outcome in self.recent_executions[-4:]],
        }


class LowLevelPlanner(Protocol):
    def generate_plan(self, context: PlannerContext) -> CommandPlan:
        """Generate a strict low-level command plan."""


class HeuristicLowLevelPlanner:
    """Deterministic fallback planner for local demos and tests.

    This is intentionally simple: rotate toward the target, drive straight,
    rotate to the requested final yaw, and stop. It exercises the same schema
    and validator loop an AI planner will use.
    """

    _GOAL_REPAIR_FAILURE_TYPES = {"goal_not_reached", "yaw_not_reached"}
    _DOGLEG_REPAIR_FAILURE_TYPES = {"collision", "out_of_bounds", "forbidden_zone"}

    def generate_plan(self, context: PlannerContext) -> CommandPlan:
        start = context.current_simulated_pose
        goal = context.target_pose

        goal_repair = self._goal_repair_plan(context)
        if goal_repair is not None:
            return goal_repair

        route = [goal]
        safety_failure_count = self._safety_failure_count(context)
        if safety_failure_count > 0:
            repair_offsets = (35.0, -35.0, 70.0, -70.0)
            offset = repair_offsets[(safety_failure_count - 1) % len(repair_offsets)]
            route = [self._dogleg_waypoint(start, goal, offset), goal]

        commands = self._route_commands(start, goal, route, context)
        return CommandPlan(goal=goal, commands=tuple(self._with_final_stop(commands, context)))

    def generate_route_plan(self, context: PlannerContext, intermediate_waypoints: list[Pose]) -> CommandPlan:
        """Convert route waypoints into the strict low-level command schema."""
        route = [*intermediate_waypoints, context.target_pose]
        commands = self._route_commands(
            start=context.current_simulated_pose,
            goal=context.target_pose,
            route=route,
            context=context,
        )
        return CommandPlan(goal=context.target_pose, commands=tuple(self._with_final_stop(commands, context)))

    def _goal_repair_plan(self, context: PlannerContext) -> CommandPlan | None:
        if not context.previous_failures:
            return None

        last_failure = context.previous_failures[-1]
        if last_failure.validation.failure_type not in self._GOAL_REPAIR_FAILURE_TYPES:
            return None

        base_commands = [
            command for command in last_failure.plan.commands if command.command != CommandName.STOP
        ]
        repaired_commands = self._adjust_near_miss_linear_command(
            base_commands,
            last_failure.validation.final_pose,
            context,
        )
        if repaired_commands is None:
            correction_commands = self._near_miss_correction_commands(
                last_failure.validation.final_pose,
                context,
            )
            if correction_commands is not None:
                repaired_commands = base_commands + correction_commands
        if repaired_commands is None:
            correction_commands = self._route_commands(
                start=last_failure.validation.final_pose,
                goal=context.target_pose,
                route=[context.target_pose],
                context=context,
            )
            repaired_commands = base_commands + correction_commands
        return CommandPlan(goal=context.target_pose, commands=tuple(self._with_final_stop(repaired_commands, context)))

    def _adjust_near_miss_linear_command(
        self,
        commands: list[CommandStep],
        final_pose: Pose,
        context: PlannerContext,
    ) -> list[CommandStep] | None:
        linear_index = self._last_linear_command_index(commands)
        if linear_index is None:
            return None

        forward_error_units = self._near_miss_forward_error_units(final_pose, context)
        if forward_error_units is None:
            return None

        linear_command = commands[linear_index]
        signed_distance = linear_command.distance
        if linear_command.command == CommandName.REV:
            signed_distance *= -1
        adjusted_signed_distance = signed_distance + forward_error_units
        if adjusted_signed_distance == 0 or (adjusted_signed_distance > 0) != (signed_distance > 0):
            return None

        adjusted_command = CommandName.FWD if adjusted_signed_distance > 0 else CommandName.REV
        adjusted_steps = self._split_linear_steps(
            adjusted_command,
            linear_command.speed,
            abs(adjusted_signed_distance),
            context.settings.limits.max_distance,
        )
        return commands[:linear_index] + adjusted_steps + commands[linear_index + 1 :]

    def _near_miss_correction_commands(
        self,
        final_pose: Pose,
        context: PlannerContext,
    ) -> list[CommandStep] | None:
        forward_error_units = self._near_miss_forward_error_units(final_pose, context)
        if forward_error_units is None:
            return None

        command = CommandName.FWD if forward_error_units >= 0 else CommandName.REV
        return self._split_linear_steps(
            command,
            context.settings.default_forward_speed,
            abs(forward_error_units),
            context.settings.limits.max_distance,
        )

    @staticmethod
    def _last_linear_command_index(commands: list[CommandStep]) -> int | None:
        for index in range(len(commands) - 1, -1, -1):
            if commands[index].command in (CommandName.FWD, CommandName.REV):
                return index
        return None

    def _near_miss_forward_error_units(
        self,
        final_pose: Pose,
        context: PlannerContext,
    ) -> int | None:
        goal = context.target_pose
        if yaw_error_deg(final_pose.yaw_deg, goal.yaw_deg) > context.settings.limits.goal_tolerance_yaw_deg:
            return None

        yaw_rad = math.radians(final_pose.yaw_deg)
        dx = goal.x - final_pose.x
        dy = goal.y - final_pose.y
        forward_error_m = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
        lateral_error_m = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)
        if abs(lateral_error_m) > context.settings.limits.goal_tolerance_m:
            return None

        return round(forward_error_m * context.settings.limits.distance_units_per_m)

    def _route_commands(
        self,
        start: Pose,
        goal: Pose,
        route: list[Pose],
        context: PlannerContext,
    ) -> list[CommandStep]:
        commands: list[CommandStep] = []
        current = start
        current_yaw = start.yaw_deg
        for waypoint in route:
            segment_bearing = bearing_deg(current, waypoint)
            commands.extend(self._turn_steps(signed_yaw_delta_deg(current_yaw, segment_bearing), context))
            forward_units = round(distance_m(current, waypoint) * context.settings.limits.distance_units_per_m)
            commands.extend(
                self._split_linear_steps(
                    CommandName.FWD,
                    context.settings.default_forward_speed,
                    forward_units,
                    context.settings.limits.max_distance,
                )
            )
            current = Pose(waypoint.x, waypoint.y, segment_bearing)
            current_yaw = segment_bearing

        commands.extend(self._turn_steps(signed_yaw_delta_deg(current_yaw, goal.yaw_deg), context))
        return commands

    @classmethod
    def _safety_failure_count(cls, context: PlannerContext) -> int:
        return sum(
            1
            for failure in context.previous_failures
            if failure.validation.failure_type in cls._DOGLEG_REPAIR_FAILURE_TYPES
        )

    @staticmethod
    def _with_final_stop(commands: list[CommandStep], context: PlannerContext) -> list[CommandStep]:
        max_commands = context.settings.limits.max_commands
        if max_commands <= 0:
            return []
        trimmed = commands[: max_commands - 1]
        trimmed.append(CommandStep(CommandName.STOP, 0, 0))
        return trimmed

    @staticmethod
    def _dogleg_waypoint(start: Pose, goal: Pose, offset_deg: float) -> Pose:
        import math

        direct_bearing = bearing_deg(start, goal)
        direct_distance = distance_m(start, goal)
        midpoint_x = (start.x + goal.x) / 2.0
        midpoint_y = (start.y + goal.y) / 2.0
        side = 1.0 if offset_deg >= 0.0 else -1.0
        lateral_m = max(1.2, direct_distance * 0.6)
        perpendicular = math.radians(direct_bearing + side * 90.0)
        return Pose(
            x=midpoint_x + math.cos(perpendicular) * lateral_m,
            y=midpoint_y + math.sin(perpendicular) * lateral_m,
            yaw_deg=direct_bearing,
        )

    def _turn_steps(self, delta_yaw_deg: float, context: PlannerContext) -> list[CommandStep]:
        if yaw_error_deg(0.0, delta_yaw_deg) < 1.0:
            return []
        command = CommandName.LEFT if delta_yaw_deg >= 0.0 else CommandName.RIGHT
        distance = round(abs(delta_yaw_deg))
        return self._split_linear_steps(
            command,
            context.settings.default_turn_speed,
            distance,
            context.settings.limits.max_distance,
        )

    @staticmethod
    def _split_linear_steps(
        command: CommandName,
        speed: int,
        distance: int,
        max_distance: int,
    ) -> list[CommandStep]:
        if distance <= 0:
            return []
        steps: list[CommandStep] = []
        remaining = distance
        while remaining > 0:
            step_distance = min(remaining, max_distance)
            steps.append(CommandStep(command, speed, step_distance))
            remaining -= step_distance
        return steps
