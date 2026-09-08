from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Protocol

from digital_twin.bootstrap import navigation_blocked_by_bootstrap
from digital_twin.config import GateMode, PlannerSettings
from digital_twin.realtime_safety import (
    ExecutionErrorLog,
    TrackingDecision,
    TrackingGate,
    is_pose_jump,
)
from digital_twin.executor import StepwiseExecutor
from digital_twin.geometry import distance_m, signed_yaw_delta_deg, yaw_error_deg
from digital_twin.models import CommandName, CommandPlan, CommandStep, Pose, ValidationRequest, ValidationResult
from digital_twin.observation_client import RoverObservation
from digital_twin.planner import (
    ExecutionOutcome,
    FailedAttempt,
    HeuristicLowLevelPlanner,
    LowLevelPlanner,
    PlannerContext,
)
from digital_twin.validator import PlanValidator


class ObservationClient(Protocol):
    def observe(self) -> RoverObservation:
        """Return the latest rover pose, camera frame, and scene context."""


@dataclass(frozen=True)
class ClosedLoopResult:
    reached_goal: bool
    cycles_completed: int
    attempts: tuple[FailedAttempt, ...]
    final_pose: Pose


@dataclass
class ClosedLoopController:
    planner: LowLevelPlanner
    validator: PlanValidator
    executor: StepwiseExecutor
    settings: PlannerSettings
    observation_client: ObservationClient | None = None
    commands_per_cycle: int = 1
    max_cycles: int = 8
    max_consecutive_execution_failures: int = 3
    # A plan that drives clean but stops short of the goal by no more than
    # this is still worth executing: only a slice runs before the next cycle
    # replans from the observed pose, so the residual shrinks every cycle.
    # Rejecting such plans makes the model abandon working routes.
    near_miss_accept_m: float = 0.5
    # When set, refuse to navigate until the startup-360 coverage gate reports
    # navigation is enabled. Off by default so existing flows are
    # unaffected; the mounted-rover flow turns it on.
    require_bootstrap: bool = False
    # mounted-rover hooks (all opt-in; None/False preserves existing
    # behavior). tracking_gate pauses/halts on phone tracking quality;
    # execution_log records predicted-vs-measured error; use_phone_pose makes the
    # loop localize from the phone-derived rover pose; pose-jump thresholds halt
    # the loop on an ARKit jump.
    tracking_gate: TrackingGate | None = None
    execution_log: ExecutionErrorLog | None = None
    use_phone_pose: bool = False
    pose_jump_m: float = 1.0
    pose_jump_deg: float = 45.0

    def run(self, start_pose: Pose, goal: Pose) -> ClosedLoopResult:
        current_pose = start_pose
        attempts: list[FailedAttempt] = []
        executions: list[ExecutionOutcome] = []
        consecutive_failures = 0

        # Bootstrap gate: do not drive until the startup 360 has mapped enough of
        # the surroundings. Checked against the live observation so the gate's
        # state (not a local guess) decides.
        if self.require_bootstrap and self.observation_client is not None:
            observation = self._observe()
            if observation is not None and navigation_blocked_by_bootstrap(observation.payload):
                return ClosedLoopResult(False, 0, tuple(attempts), current_pose)

        # Place the visual goal marker before the first observation so the
        # planner's camera frames show where the rover needs to go.
        set_goal_marker = getattr(self.validator, "set_goal_marker", None)
        if callable(set_goal_marker):
            set_goal_marker(goal)

        if self.observation_client is not None:
            prime_result = self._prime_validator_at_pose(start_pose)
            if not prime_result.approved:
                attempts.append(
                    FailedAttempt(
                        plan=CommandPlan(
                            goal=start_pose,
                            commands=(CommandStep(CommandName.STOP, 0, 0),),
                        ),
                        validation=prime_result,
                    )
                )
                return ClosedLoopResult(False, 0, tuple(attempts), prime_result.final_pose)

        for cycle in range(self.max_cycles):
            observation = self._observe()
            measured = self._measured_pose(observation)
            # Localize from the measured pose. With phone localization, do this on
            # the FIRST cycle too: the rover's real pose, not the CLI start_pose,
            # is what the first command must be planned/validated from. Without
            # phone localization, keep the legacy behavior of trusting the caller
            # start_pose on cycle 0.
            if measured is not None and (cycle > 0 or self.use_phone_pose):
                current_pose = measured

            # Tracking gate: pause (don't command) while phone tracking is not
            # usable; halt the loop on an unsafe state. Active only when set.
            if self.tracking_gate is not None:
                decision = self.tracking_gate.evaluate(self._phone_pose_block(observation))
                if decision is TrackingDecision.HALT:
                    return ClosedLoopResult(False, cycle, tuple(attempts), current_pose)
                if decision is TrackingDecision.WAIT:
                    continue

            set_observation = getattr(self.planner, "set_observation", None)
            if callable(set_observation):
                set_observation(observation)

            if self._goal_reached(current_pose, goal):
                return ClosedLoopResult(True, cycle, tuple(attempts), current_pose)

            # If the straight line to the goal is clear, drive it
            # deterministically (accurate motion lands it). Only when something
            # blocks that line do we consult the vision planner to route around
            # it — the navigator steers by position waypoints and would orbit a
            # clear goal without converging.
            plan = validation = None
            direct = self._direct_drive_plan(current_pose, goal)
            if direct is not None:
                plan, validation = direct
            if plan is None or validation is None:
                plan, validation = self._approved_plan(current_pose, goal, attempts, executions)
            if plan is None or validation is None:
                return ClosedLoopResult(False, cycle, tuple(attempts), current_pose)

            execution_plan, execution_validation = self._execution_slice(plan, validation)
            expected_pose = execution_validation.expected_step_end_poses[-1]
            executed_ok = self.executor.execute(execution_plan, execution_validation)
            self._sync_dry_run_observation_pose(expected_pose)

            observation = self._observe()
            measured_after = self._measured_pose(observation)
            if measured_after is not None:
                current_pose = measured_after
            else:
                current_pose = expected_pose

            # Log predicted-vs-measured execution error, and halt on a pose jump
            # (divergence too large for normal drift => ARKit jump). Active only
            # when the mounted-rover hooks are configured.
            if self.execution_log is not None:
                block = self._phone_pose_block(observation) or {}
                self.execution_log.record(
                    cycle,
                    expected_pose,
                    current_pose,
                    phone_timestamp_s=block.get("phone_timestamp_s"),
                    tracking_normal=block.get("tracking_normal"),
                )
            if self.tracking_gate is not None and is_pose_jump(
                expected_pose, current_pose, self.pose_jump_m, self.pose_jump_deg
            ):
                return ClosedLoopResult(False, cycle + 1, tuple(attempts), current_pose)

            self._record_execution(executions, execution_plan, expected_pose, current_pose, executed_ok)

            if not executed_ok:
                # In enforce mode, feed the abort back to the planner and let it
                # self-correct on the next cycle instead of giving up, but bail if
                # the rover keeps failing to make progress.
                consecutive_failures += 1
                if not self._execution_feedback_enabled() or consecutive_failures >= self.max_consecutive_execution_failures:
                    return ClosedLoopResult(False, cycle + 1, tuple(attempts), current_pose)
            else:
                consecutive_failures = 0

        return ClosedLoopResult(self._goal_reached(current_pose, goal), self.max_cycles, tuple(attempts), current_pose)

    def _execution_feedback_enabled(self) -> bool:
        # Only enforce mode actually moves the rover, so only then is the
        # observed-vs-expected pose a real execution outcome. In dry_run the
        # validator's own simulation drives the reported pose, which would look
        # like spurious drift.
        return self.settings.mode == GateMode.ENFORCE

    def _record_execution(
        self,
        executions: list[ExecutionOutcome],
        plan: CommandPlan,
        expected_pose: Pose,
        observed_pose: Pose,
        executed_ok: bool,
    ) -> None:
        if not self._execution_feedback_enabled():
            return
        if executed_ok:
            note = "Commands ran. Compare observed_pose_after to expected_pose_after; correct any drift."
        else:
            note = (
                "Execution aborted and the rover was stopped (command-status failure or pose drift "
                "beyond tolerance). Re-plan from observed_pose_after and avoid repeating the failed move."
            )
        executions.append(
            ExecutionOutcome(
                executed_commands=tuple(plan.commands),
                expected_pose=expected_pose,
                observed_pose=observed_pose,
                aborted=not executed_ok,
                note=note,
            )
        )

    def _approved_plan(
        self,
        current_pose: Pose,
        goal: Pose,
        attempts: list[FailedAttempt],
        executions: list[ExecutionOutcome],
    ) -> tuple[CommandPlan | None, ValidationResult | None]:
        for _ in range(self.settings.max_attempts):
            context = PlannerContext(
                current_simulated_pose=current_pose,
                target_pose=goal,
                allowed_commands=(
                    CommandName.STOP,
                    CommandName.FWD,
                    CommandName.REV,
                    CommandName.LEFT,
                    CommandName.RIGHT,
                ),
                settings=self.settings,
                previous_failures=tuple(attempts),
                recent_executions=tuple(executions),
            )
            plan = self.planner.generate_plan(context)
            validation = self.validator.validate(
                ValidationRequest.from_plan(
                    start_pose=current_pose,
                    plan=plan,
                    validation_profile=self.settings.validation_profile,
                )
            )
            if validation.approved:
                return plan, validation
            if self._near_miss(validation):
                print(
                    f"[closed-loop] accepting near-miss plan: clean run ended "
                    f"{validation.goal_error_m:.3f} m from goal (within "
                    f"near_miss_accept_m={self.near_miss_accept_m}); later cycles re-plan "
                    "from the observed pose and close the gap.",
                    file=sys.stderr,
                )
                return plan, validation
            attempts.append(FailedAttempt(plan=plan, validation=validation))
        return None, None

    def _near_miss(self, validation: ValidationResult) -> bool:
        """A collision-free run that merely stopped short of the goal."""
        return (
            validation.failure_type == "goal_not_reached"
            and validation.failed_command_index is None
            and validation.goal_error_m is not None
            and validation.goal_error_m <= self.near_miss_accept_m
        )

    def _execution_slice(
        self,
        plan: CommandPlan,
        validation: ValidationResult,
    ) -> tuple[CommandPlan, ValidationResult]:
        # commands_per_cycle counts DRIVE moves (FWD/REV); the turns that aim
        # the rover at each leg ride along for free. Counting every command
        # (turns included) means a [turn, FWD] route executes only the turn at
        # commands_per_cycle=1, so the rover keeps re-aiming and never drives —
        # it dithers in place. A pure-rotation plan (no drive, e.g. the final
        # heading fix at the goal) falls back to counting the turns.
        plan_has_drive = any(
            command.command in (CommandName.FWD, CommandName.REV) for command in plan.commands
        )
        commands: list[CommandStep] = []
        progress = 0
        for command in plan.commands:
            commands.append(command)
            if command.command == CommandName.STOP:
                break
            if plan_has_drive:
                if command.command in (CommandName.FWD, CommandName.REV):
                    progress += 1
            else:
                progress += 1
            if progress >= self.commands_per_cycle:
                break

        expected_poses = list(validation.expected_step_end_poses[: len(commands)])
        if not expected_poses:
            expected_poses = [validation.final_pose]

        if commands[-1].command != CommandName.STOP:
            commands.append(CommandStep(CommandName.STOP, 0, 0))
            expected_poses.append(expected_poses[-1])

        execution_plan = CommandPlan(goal=plan.goal, commands=tuple(commands))
        execution_validation = replace(
            validation,
            final_pose=expected_poses[-1],
            expected_step_end_poses=tuple(expected_poses),
            step_results=validation.step_results[: len(commands)],
        )
        return execution_plan, execution_validation

    def _observe(self) -> RoverObservation | None:
        if self.observation_client is None:
            return None
        return self.observation_client.observe()

    @staticmethod
    def _phone_pose_block(observation: RoverObservation | None) -> dict | None:
        if observation is None or not isinstance(observation.payload, dict):
            return None
        block = observation.payload.get("phone_pose")
        return block if isinstance(block, dict) else None

    def _measured_pose(self, observation: RoverObservation | None) -> Pose | None:
        """The pose to localize from. With ``use_phone_pose`` and an aligned
        twin, prefer the phone-derived rover pose; otherwise the observation
        pose (preserving existing behavior)."""
        if self.use_phone_pose:
            block = self._phone_pose_block(observation)
            if block and block.get("rover_pose_aligned") and isinstance(block.get("rover_pose"), dict):
                return Pose.from_dict(block["rover_pose"], "phone_pose.rover_pose")
        return observation.pose if observation is not None else None

    def _prime_validator_at_pose(self, pose: Pose) -> ValidationResult:
        """Reset the digital twin before taking the first camera observation."""
        return self.validator.validate(
            ValidationRequest(
                start_pose=pose,
                goal=pose,
                commands=(CommandStep(CommandName.STOP, 0, 0),),
                validation_profile=self.settings.validation_profile,
            )
        )

    def _sync_dry_run_observation_pose(self, pose: Pose) -> None:
        if self.observation_client is None or self.settings.mode != GateMode.DRY_RUN:
            return
        self._prime_validator_at_pose(pose)

    def _goal_reached(self, current_pose: Pose, goal: Pose) -> bool:
        return (
            distance_m(current_pose, goal) <= self.settings.limits.goal_tolerance_m
            and yaw_error_deg(current_pose.yaw_deg, goal.yaw_deg) <= self.settings.limits.goal_tolerance_yaw_deg
        )

    def _position_reached(self, current_pose: Pose, goal: Pose) -> bool:
        return distance_m(current_pose, goal) <= self.settings.limits.goal_tolerance_m

    def _direct_drive_plan(
        self, current_pose: Pose, goal: Pose
    ) -> tuple[CommandPlan, ValidationResult] | None:
        """A deterministic plan straight at the goal, if the line is clear.

        Builds the heuristic turn-drive-turn plan to the goal and validates it.
        If it drives without hitting anything (the direct line is open), use it
        and skip the vision planner entirely. If it collides, the line is
        blocked: return None so the caller routes around with the navigator.
        At the goal position it degenerates to a final heading turn.
        """
        if self._position_reached(current_pose, goal):
            return self._final_yaw_plan(current_pose, goal)
        context = PlannerContext(
            current_simulated_pose=current_pose,
            target_pose=goal,
            allowed_commands=(
                CommandName.STOP,
                CommandName.FWD,
                CommandName.REV,
                CommandName.LEFT,
                CommandName.RIGHT,
            ),
            settings=self.settings,
        )
        plan = HeuristicLowLevelPlanner().generate_plan(context)
        validation = self.validator.validate(
            ValidationRequest.from_plan(
                start_pose=current_pose,
                plan=plan,
                validation_profile=self.settings.validation_profile,
            )
        )
        if self._line_is_clear(validation):
            return plan, validation
        return None

    @staticmethod
    def _line_is_clear(validation: ValidationResult) -> bool:
        """True when a plan drove its whole route without hitting anything.

        approved = reached the goal; goal_not_reached with no failed command =
        drove clean but stopped short (still a clear line, keep closing in).
        Any collision/bounds/stall failure means the direct line is blocked.
        """
        return validation.approved or (
            validation.failure_type == "goal_not_reached"
            and validation.failed_command_index is None
        )

    def _final_yaw_plan(
        self, current_pose: Pose, goal: Pose
    ) -> tuple[CommandPlan, ValidationResult] | None:
        """A deterministic in-place turn to the goal heading, if it validates.

        Returns None when already aligned, or when the turn cannot be validated
        cleanly (e.g. it would sweep into geometry) so the caller falls back to
        the planner.
        """
        if yaw_error_deg(current_pose.yaw_deg, goal.yaw_deg) <= self.settings.limits.goal_tolerance_yaw_deg:
            return None
        delta = signed_yaw_delta_deg(current_pose.yaw_deg, goal.yaw_deg)
        command = CommandName.LEFT if delta >= 0 else CommandName.RIGHT
        distance = min(round(abs(delta)), self.settings.limits.max_turn_degrees)
        if distance <= 0:
            return None
        plan = CommandPlan(
            goal=goal,
            commands=(
                CommandStep(command, self.settings.default_turn_speed, distance),
                CommandStep(CommandName.STOP, 0, 0),
            ),
        )
        validation = self.validator.validate(
            ValidationRequest.from_plan(
                start_pose=current_pose,
                plan=plan,
                validation_profile=self.settings.validation_profile,
            )
        )
        # The whole point is the heading; accept a clean turn even if it ends a
        # hair off the goal yaw, but not one that hit geometry mid-pivot.
        if validation.approved or self._near_miss(validation):
            return plan, validation
        return None
