from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from digital_twin.config import PlannerSettings
from digital_twin.geometry import advance_pose, distance_m, rotate_pose, yaw_error_deg
from digital_twin.models import CommandName, Pose, ValidationRequest, ValidationResult


class PlanValidator(Protocol):
    def validate(self, request: ValidationRequest) -> ValidationResult:
        """Validate a command sequence against a digital twin or compatible stub."""


@dataclass
class KinematicValidator:
    settings: PlannerSettings
    samples_per_meter: int = 20
    samples_per_turn: int = 12

    def validate(self, request: ValidationRequest) -> ValidationResult:
        pose = request.start_pose
        trajectory: list[Pose] = [pose]
        step_results: list[dict] = []
        expected_step_end_poses: list[Pose] = []
        min_clearance = float("inf")

        limits_result = self._validate_command_limits(request.commands)
        if limits_result is not None:
            return limits_result

        for index, command in enumerate(request.commands):
            samples = self._simulate_command(pose, command)
            for sample in samples[1:]:
                trajectory.append(sample)
                clear_result = self._check_safety(sample)
                min_clearance = min(min_clearance, clear_result["min_clearance_m"])
                if clear_result["failure_type"] is not None:
                    return ValidationResult(
                        approved=False,
                        failure_type=clear_result["failure_type"],
                        failed_command_index=index,
                        reason=clear_result["reason"],
                        final_pose=sample,
                        min_clearance_m=min_clearance,
                        trajectory=tuple(trajectory),
                        step_results=tuple(step_results),
                    )

            pose = samples[-1]
            expected_step_end_poses.append(pose)
            step_results.append(
                {
                    "command_index": index,
                    "command": command.to_dict(),
                    "end_pose": pose.to_dict(),
                    "status": "completed",
                }
            )

        goal_error = distance_m(pose, request.goal)
        yaw_error = yaw_error_deg(pose.yaw_deg, request.goal.yaw_deg)
        if goal_error > self.settings.limits.goal_tolerance_m:
            return ValidationResult(
                approved=False,
                failure_type="goal_not_reached",
                failed_command_index=None,
                reason=f"Final pose is {goal_error:.3f} m from target",
                final_pose=pose,
                min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
                goal_error_m=goal_error,
                goal_error_yaw_deg=yaw_error,
                trajectory=tuple(trajectory),
                step_results=tuple(step_results),
                expected_step_end_poses=tuple(expected_step_end_poses),
            )
        if yaw_error > self.settings.limits.goal_tolerance_yaw_deg:
            return ValidationResult(
                approved=False,
                failure_type="yaw_not_reached",
                failed_command_index=None,
                reason=f"Final yaw is {yaw_error:.1f} deg from target",
                final_pose=pose,
                min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
                goal_error_m=goal_error,
                goal_error_yaw_deg=yaw_error,
                trajectory=tuple(trajectory),
                step_results=tuple(step_results),
                expected_step_end_poses=tuple(expected_step_end_poses),
            )

        return ValidationResult(
            approved=True,
            reason="Plan reached target with no collisions",
            final_pose=pose,
            min_clearance_m=0.0 if min_clearance == float("inf") else min_clearance,
            goal_error_m=goal_error,
            goal_error_yaw_deg=yaw_error,
            trajectory=tuple(trajectory),
            step_results=tuple(step_results),
            expected_step_end_poses=tuple(expected_step_end_poses),
        )

    def _validate_command_limits(self, commands) -> ValidationResult | None:
        if len(commands) > self.settings.limits.max_commands:
            return ValidationResult(
                approved=False,
                failure_type="too_many_commands",
                failed_command_index=None,
                reason=f"Plan has {len(commands)} commands; max is {self.settings.limits.max_commands}",
                final_pose=Pose(0.0, 0.0, 0.0),
                min_clearance_m=0.0,
            )

        for index, command in enumerate(commands):
            if command.speed < self.settings.limits.min_speed or command.speed > self.settings.limits.max_speed:
                return ValidationResult(
                    approved=False,
                    failure_type="speed_limit",
                    failed_command_index=index,
                    reason=f"Command {index} speed {command.speed} is outside configured limits",
                    final_pose=Pose(0.0, 0.0, 0.0),
                    min_clearance_m=0.0,
                )
            if command.distance < 0 or command.distance > self.settings.limits.max_distance:
                return ValidationResult(
                    approved=False,
                    failure_type="distance_limit",
                    failed_command_index=index,
                    reason=f"Command {index} distance {command.distance} is outside configured limits",
                    final_pose=Pose(0.0, 0.0, 0.0),
                    min_clearance_m=0.0,
                )
            if command.command in (CommandName.LEFT, CommandName.RIGHT) and command.distance > self.settings.limits.max_turn_degrees:
                return ValidationResult(
                    approved=False,
                    failure_type="turn_distance_limit",
                    failed_command_index=index,
                    reason=(
                        f"Command {index} turn distance {command.distance} deg exceeds "
                        f"{self.settings.limits.max_turn_degrees} deg"
                    ),
                    final_pose=Pose(0.0, 0.0, 0.0),
                    min_clearance_m=0.0,
                )
            if command.command == CommandName.STOP and (command.speed != 0 or command.distance != 0):
                return ValidationResult(
                    approved=False,
                    failure_type="invalid_stop",
                    failed_command_index=index,
                    reason="STOP command must use speed 0 and distance 0",
                    final_pose=Pose(0.0, 0.0, 0.0),
                    min_clearance_m=0.0,
                )
        return None

    def _simulate_command(self, pose: Pose, command) -> list[Pose]:
        if command.command == CommandName.STOP:
            return [pose]

        if command.command in (CommandName.FWD, CommandName.REV):
            direction = 1.0 if command.command == CommandName.FWD else -1.0
            total_m = direction * command.distance / self.settings.limits.distance_units_per_m
            sample_count = max(1, round(abs(total_m) * self.samples_per_meter))
            return [advance_pose(pose, total_m * step / sample_count) for step in range(sample_count + 1)]

        turn_sign = 1.0 if command.command == CommandName.LEFT else -1.0
        total_deg = turn_sign * command.distance
        sample_count = max(1, round(abs(total_deg) / 30.0))
        return [rotate_pose(pose, total_deg * step / sample_count) for step in range(sample_count + 1)]

    def _check_safety(self, pose: Pose) -> dict:
        if not self.settings.world.lab_bounds.contains(pose):
            return {
                "failure_type": "out_of_bounds",
                "reason": "Rover left configured lab bounds",
                "min_clearance_m": 0.0,
            }

        for zone in self.settings.world.forbidden_zones:
            if zone.contains(pose):
                return {
                    "failure_type": "forbidden_zone",
                    "reason": f"Rover entered forbidden zone {zone.name}",
                    "min_clearance_m": 0.0,
                }

        clearances = [obstacle.clearance(pose) for obstacle in self.settings.world.obstacles]
        clearances.extend(zone.clearance(pose) for zone in self.settings.world.forbidden_zones)
        min_clearance = min(clearances, default=999.0)
        if min_clearance < self.settings.limits.min_clearance_m:
            return {
                "failure_type": "collision",
                "reason": f"Minimum clearance {min_clearance:.3f} m is below threshold",
                "min_clearance_m": min_clearance,
            }

        return {"failure_type": None, "reason": "", "min_clearance_m": min_clearance}
