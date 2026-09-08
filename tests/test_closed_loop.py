import contextlib
import io
import json
import unittest

from digital_twin.closed_loop import ClosedLoopController
from digital_twin.config import PlannerSettings
from digital_twin.executor import DryRunPublisher, StepwiseExecutor
from digital_twin.llm_client import LLMClientError, LLMReply
from digital_twin.llm_planner import LLMRouteAdvisorPlanner
from digital_twin.models import (
    CircleObstacle,
    CommandName,
    CommandPlan,
    CommandStep,
    Pose,
    ValidationRequest,
    WorldModel,
)
from digital_twin.observation_client import RoverObservation
from digital_twin.realtime_safety import ExecutionErrorLog, TrackingGate
from digital_twin.validator import KinematicValidator


class FakeLLMClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self, text: str | Exception):
        self.text = text
        self.calls: list[dict] = []

    def complete_json(self, system, user_text, json_schema, images_png_base64=None):
        self.calls.append({"user_text": user_text, "images": images_png_base64})
        if isinstance(self.text, Exception):
            raise self.text
        return LLMReply(text=self.text, latency_s=0.01, provider=self.provider, model=self.model)


def _direct_advice() -> str:
    return json.dumps({"path_blocked": False, "rationale": "", "intermediate_waypoints": []})


def _advisor(client: FakeLLMClient) -> LLMRouteAdvisorPlanner:
    return LLMRouteAdvisorPlanner(client=client)


class FakeObservationClient:
    def __init__(self):
        self.calls = 0

    def observe(self):
        self.calls += 1
        return RoverObservation(
            pose=Pose(0, 0, 0),
            payload={"rover_pose": Pose(0, 0, 0).to_dict(), "vision": {"status": "ok", "rgb_base64": "abc123"}},
            vision_rgb_base64="abc123",
        )


class BootstrapObservationClient:
    """Reports the startup-360 gate state; tracks how many times it was polled."""

    def __init__(self, navigation_enabled: bool):
        self.navigation_enabled = navigation_enabled
        self.calls = 0

    def observe(self):
        self.calls += 1
        return RoverObservation(
            pose=Pose(0, 0, 0),
            payload={
                "rover_pose": Pose(0, 0, 0).to_dict(),
                "vision": {"status": "ok", "rgb_base64": "abc123"},
                "bootstrap": {"enabled": True, "navigation_enabled": self.navigation_enabled},
            },
            vision_rgb_base64="abc123",
        )


class StaleObservationClient:
    def observe(self):
        return RoverObservation(
            pose=Pose(0, 0, 0),
            payload={"rover_pose": Pose(0, 0, 0).to_dict(), "vision": {"status": "ok", "rgb_base64": "abc123"}},
            vision_rgb_base64="abc123",
        )


class RecordingValidator:
    def __init__(self, settings):
        self.inner = KinematicValidator(settings)
        self.requests = []

    def validate(self, request):
        self.requests.append(request)
        return self.inner.validate(request)


class ClosedLoopTests(unittest.TestCase):
    def test_closed_loop_attaches_camera_image_to_advisor_request(self):
        # An obstacle on the direct line forces the vision planner to be
        # consulted (a clear line would be driven deterministically instead).
        settings = PlannerSettings(
            world=WorldModel(obstacles=(CircleObstacle(name="rock", x=0.5, y=0.0, radius_m=0.2),))
        )
        client = FakeLLMClient(_direct_advice())
        controller = ClosedLoopController(
            planner=_advisor(client),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=FakeObservationClient(),
            commands_per_cycle=1,
            max_cycles=1,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertEqual(client.calls[0]["images"], ["abc123"])
        request = json.loads(client.calls[0]["user_text"])
        self.assertIn("onboard forward camera", request["cameras"][0])

    def test_clear_line_drives_deterministically_without_calling_advisor(self):
        # No obstacle: the direct line is clear, so the deterministic planner
        # handles it and the vision model is never consulted. No observation
        # client, so the validated expected pose threads forward each cycle.
        settings = PlannerSettings()
        client = FakeLLMClient(_direct_advice())
        controller = ClosedLoopController(
            planner=_advisor(client),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            commands_per_cycle=1,
            max_cycles=4,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertEqual(len(client.calls), 0)
        self.assertTrue(result.reached_goal)

    def test_closed_loop_executes_prefix_then_reobserves(self):
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=FakeObservationClient(),
            commands_per_cycle=1,
            max_cycles=1,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertFalse(result.reached_goal)
        self.assertEqual(result.cycles_completed, 1)

    def test_closed_loop_primes_start_pose_before_first_planning_observation(self):
        # Obstacle on the direct line so the vision planner is consulted (and we
        # can assert what pose it saw).
        settings = PlannerSettings(
            world=WorldModel(obstacles=(CircleObstacle(name="rock", x=2.5, y=0.0, radius_m=0.2),))
        )
        client = FakeLLMClient(_direct_advice())
        validator = RecordingValidator(settings)
        controller = ClosedLoopController(
            planner=_advisor(client),
            validator=validator,
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=StaleObservationClient(),
            commands_per_cycle=1,
            max_cycles=1,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            controller.run(start_pose=Pose(2, 0, 0), goal=Pose(3, 0, 0))

        # The planner sees the primed start pose, not the stale (0,0,0) observation.
        request = json.loads(client.calls[0]["user_text"])
        self.assertEqual(request["rover_pose"], {"x": 2, "y": 0, "yaw_deg": 0})
        # First validator request is the STOP prime at the start pose.
        self.assertEqual(validator.requests[0].start_pose, Pose(2, 0, 0))
        self.assertEqual([command.command for command in validator.requests[0].commands], [CommandName.STOP])

    def test_execution_slice_repeats_expected_pose_for_added_stop(self):
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
        )
        plan = CommandPlan(
            goal=Pose(1, 0, 0),
            commands=(
                CommandStep(CommandName.FWD, 40, 23000),
                CommandStep(CommandName.STOP, 0, 0),
            ),
        )
        validation = KinematicValidator(settings).validate(ValidationRequest.from_plan(Pose(0, 0, 0), plan))

        prefix_plan, prefix_validation = controller._execution_slice(plan, validation)

        self.assertEqual([command.command for command in prefix_plan.commands], [CommandName.FWD, CommandName.STOP])
        self.assertEqual(prefix_validation.expected_step_end_poses[0], prefix_validation.expected_step_end_poses[1])

    def test_execution_slice_drives_through_leading_turn(self):
        # commands_per_cycle counts DRIVE moves: a [turn, FWD] route at
        # commands_per_cycle=1 must execute the FWD too, not stop after the
        # turn (which would leave the rover dithering in place).
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            commands_per_cycle=1,
        )
        plan = CommandPlan(
            goal=Pose(0, 1, 90),
            commands=(
                CommandStep(CommandName.LEFT, 56, 90),
                CommandStep(CommandName.FWD, 40, 23000),
                CommandStep(CommandName.STOP, 0, 0),
            ),
        )
        validation = KinematicValidator(settings).validate(ValidationRequest.from_plan(Pose(0, 0, 0), plan))

        prefix_plan, _ = controller._execution_slice(plan, validation)

        self.assertEqual(
            [command.command for command in prefix_plan.commands],
            [CommandName.LEFT, CommandName.FWD, CommandName.STOP],
        )

    def test_execution_slice_pure_rotation_plan_still_executes(self):
        # A plan with no drive move (e.g. final heading fix) falls back to
        # counting turns so it still executes instead of stalling.
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            commands_per_cycle=1,
        )
        plan = CommandPlan(
            goal=Pose(0, 0, 45),
            commands=(
                CommandStep(CommandName.LEFT, 56, 45),
                CommandStep(CommandName.STOP, 0, 0),
            ),
        )
        validation = KinematicValidator(settings).validate(ValidationRequest.from_plan(Pose(0, 0, 0), plan))

        prefix_plan, _ = controller._execution_slice(plan, validation)

        self.assertEqual(
            [command.command for command in prefix_plan.commands],
            [CommandName.LEFT, CommandName.STOP],
        )

    def test_final_yaw_uses_deterministic_turn_not_the_planner(self):
        # At the goal position but off-heading, the loop should turn in place
        # to the goal yaw deterministically and NOT consult the vision planner.
        settings = PlannerSettings()
        client = FakeLLMClient(_direct_advice())
        controller = ClosedLoopController(
            planner=_advisor(client),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            commands_per_cycle=1,
        )
        # Already at goal x/y, 40 deg off the goal heading.
        at_goal_off_yaw = Pose(1.0, 0.0, 0.0)
        goal = Pose(1.0, 0.0, 40.0)

        yaw_plan = controller._final_yaw_plan(at_goal_off_yaw, goal)

        self.assertIsNotNone(yaw_plan)
        plan, _ = yaw_plan
        self.assertEqual(plan.commands[0].command, CommandName.LEFT)  # +40 deg -> LEFT
        self.assertEqual(plan.commands[0].distance, 40)
        self.assertEqual(len(client.calls), 0)  # planner never consulted

    def test_final_yaw_plan_none_when_already_aligned(self):
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
        )
        # Within the yaw tolerance already -> no turn plan.
        self.assertIsNone(controller._final_yaw_plan(Pose(1.0, 0.0, 2.0), Pose(1.0, 0.0, 0.0)))

    def test_closed_loop_reaches_goal_when_advisor_unreachable(self):
        settings = PlannerSettings()
        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(LLMClientError("unreachable"))),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=FakeObservationClient(),
            commands_per_cycle=4,
            max_cycles=4,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(0, 0, 0))

        self.assertTrue(result.reached_goal)


if __name__ == "__main__":
    unittest.main()


class NearMissValidator:
    """Always returns a clean goal_not_reached just outside tolerance."""

    def __init__(self):
        self.requests = []

    def validate(self, request):
        from digital_twin.models import ValidationResult

        self.requests.append(request)
        end_pose = request.goal if len(request.commands) == 1 else Pose(
            request.goal.x - 0.21, request.goal.y, request.goal.yaw_deg
        )
        return ValidationResult(
            approved=len(request.commands) == 1,  # STOP-only prime passes
            failure_type=None if len(request.commands) == 1 else "goal_not_reached",
            reason="Final pose error 0.210 m, yaw error 1.8 deg",
            final_pose=end_pose,
            min_clearance_m=999.0,
            goal_error_m=None if len(request.commands) == 1 else 0.21,
            goal_error_yaw_deg=1.8,
            expected_step_end_poses=tuple(end_pose for _ in request.commands),
        )


class NearMissAcceptanceTests(unittest.TestCase):
    def test_clean_near_miss_plan_is_executed_not_rejected(self):
        settings = PlannerSettings()
        client = FakeLLMClient(_direct_advice())
        controller = ClosedLoopController(
            planner=_advisor(client),
            validator=NearMissValidator(),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            commands_per_cycle=1,
            max_cycles=1,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        # One planning attempt sufficed: the near-miss was accepted and executed
        # instead of burning all attempts re-planning a route that worked.
        self.assertEqual(len(result.attempts), 0)
        self.assertEqual(result.cycles_completed, 1)

    def test_far_miss_is_still_rejected(self):
        from digital_twin.models import ValidationResult

        controller = ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=NearMissValidator(),
            executor=StepwiseExecutor(settings=PlannerSettings(), publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=PlannerSettings(),
        )
        far = ValidationResult(
            approved=False,
            failure_type="goal_not_reached",
            reason="Final pose error 1.4 m",
            final_pose=Pose(0, 0, 0),
            min_clearance_m=999.0,
            goal_error_m=1.4,
        )
        collision = ValidationResult(
            approved=False,
            failure_type="collision",
            reason="hit",
            final_pose=Pose(0, 0, 0),
            min_clearance_m=999.0,
            failed_command_index=1,
            goal_error_m=0.1,
        )
        self.assertFalse(controller._near_miss(far))
        self.assertFalse(controller._near_miss(collision))


class BootstrapGuardTests(unittest.TestCase):
    def _controller(self, client, require_bootstrap):
        settings = PlannerSettings()
        return ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=client,
            commands_per_cycle=1,
            max_cycles=3,
            require_bootstrap=require_bootstrap,
        )

    def test_navigation_refused_until_bootstrap_complete(self):
        client = BootstrapObservationClient(navigation_enabled=False)
        controller = self._controller(client, require_bootstrap=True)

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertFalse(result.reached_goal)
        self.assertEqual(result.cycles_completed, 0)  # never entered a nav cycle
        self.assertEqual(client.calls, 1)  # only the gate check ran

    def test_navigation_proceeds_once_bootstrap_enabled(self):
        # The fake observation reports a static pose, so the goal is never
        # "reached"; the point is that the guard does NOT block — navigation
        # cycles run (the observation is polled more than just the gate check).
        client = BootstrapObservationClient(navigation_enabled=True)
        controller = self._controller(client, require_bootstrap=True)

        with contextlib.redirect_stderr(io.StringIO()):
            controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertGreater(client.calls, 1)  # gate check + navigation cycles

    def test_guard_off_by_default_ignores_bootstrap_block(self):
        # Without require_bootstrap, a blocking bootstrap payload is ignored.
        client = BootstrapObservationClient(navigation_enabled=False)
        controller = self._controller(client, require_bootstrap=False)

        with contextlib.redirect_stderr(io.StringIO()):
            controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertGreater(client.calls, 1)  # navigation ran despite the block


class PhonePoseObservationClient:
    def __init__(self, rover_pose, *, tracking_normal=True, aligned=True, enabled=True):
        self.rover_pose = rover_pose
        self.tracking_normal = tracking_normal
        self.aligned = aligned
        self.enabled = enabled
        self.calls = 0

    def observe(self):
        self.calls += 1
        return RoverObservation(
            pose=Pose(0, 0, 0),
            payload={
                "rover_pose": Pose(0, 0, 0).to_dict(),
                "vision": {"status": "ok", "rgb_base64": "abc123"},
                "phone_pose": {
                    "enabled": self.enabled,
                    "have_pose": True,
                    "tracking_normal": self.tracking_normal,
                    "rover_pose_aligned": self.aligned,
                    "rover_pose": self.rover_pose.to_dict(),
                    "phone_timestamp_s": 5.0,
                },
            },
            vision_rgb_base64="abc123",
        )


class SequencePhonePoseObservationClient(PhonePoseObservationClient):
    def __init__(self, rover_poses, *, tracking_normal=True, aligned=True, enabled=True):
        super().__init__(
            rover_poses[0],
            tracking_normal=tracking_normal,
            aligned=aligned,
            enabled=enabled,
        )
        self.rover_poses = list(rover_poses)

    def observe(self):
        index = min(self.calls, len(self.rover_poses) - 1)
        self.rover_pose = self.rover_poses[index]
        return super().observe()


class Phase7HookTests(unittest.TestCase):
    def _controller(self, client, **kw):
        settings = PlannerSettings()
        return ClosedLoopController(
            planner=_advisor(FakeLLMClient(_direct_advice())),
            validator=KinematicValidator(settings),
            executor=StepwiseExecutor(settings=settings, publisher=DryRunPublisher(), inter_command_delay_s=0),
            settings=settings,
            observation_client=client,
            commands_per_cycle=1,
            max_cycles=3,
            **kw,
        )

    def test_tracking_not_normal_pauses_execution(self):
        # Bad tracking -> the gate WAITs every cycle -> nothing is ever executed,
        # so the predicted-vs-measured log stays empty.
        log = ExecutionErrorLog()
        client = PhonePoseObservationClient(Pose(0, 0, 0), tracking_normal=False)
        controller = self._controller(client, tracking_gate=TrackingGate(), execution_log=log)

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertFalse(result.reached_goal)
        self.assertEqual(log.records, [])

    def test_normal_tracking_executes_and_logs_error(self):
        log = ExecutionErrorLog()
        client = PhonePoseObservationClient(Pose(0, 0, 0), tracking_normal=True, aligned=True)
        controller = self._controller(
            client, tracking_gate=TrackingGate(), execution_log=log,
            use_phone_pose=True, pose_jump_m=100.0, pose_jump_deg=720.0,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertGreaterEqual(len(log.records), 1)  # predicted-vs-measured logged

    def test_first_cycle_localizes_from_phone_pose(self):
        # The phone reports the rover is already at the goal. With phone
        # localization the loop must recognize that on cycle 0 -- i.e. it plans
        # from the measured phone pose, not the CLI start_pose.
        client = PhonePoseObservationClient(Pose(1, 0, 0), tracking_normal=True, aligned=True)
        controller = self._controller(client, tracking_gate=TrackingGate(), use_phone_pose=True)

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertTrue(result.reached_goal)
        self.assertEqual(result.cycles_completed, 0)

    def test_pose_jump_halts_loop(self):
        # The phone starts at a usable measured seed pose, then reports a pose far
        # from where the command should have landed -> a jump -> the loop halts
        # after one cycle, with that cycle logged.
        log = ExecutionErrorLog()
        client = SequencePhonePoseObservationClient(
            [Pose(0, 0, 0), Pose(9, 9, 0)],
            tracking_normal=True,
            aligned=True,
        )
        controller = self._controller(
            client, tracking_gate=TrackingGate(), execution_log=log,
            use_phone_pose=True, pose_jump_m=0.5, pose_jump_deg=30.0,
        )

        with contextlib.redirect_stderr(io.StringIO()):
            result = controller.run(start_pose=Pose(0, 0, 0), goal=Pose(1, 0, 0))

        self.assertFalse(result.reached_goal)
        self.assertEqual(result.cycles_completed, 1)
        self.assertEqual(len(log.records), 1)
