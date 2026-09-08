import json
import unittest
from unittest.mock import patch

from digital_twin.config import GateMode, PlannerSettings
from digital_twin.executor import (
    CommandCompletion,
    DurationCompletionWaiter,
    HttpWheelCommandPublisher,
    PoseTelemetryCompletionWaiter,
    StepwiseExecutor,
    rover_http_payload,
    rover_name_from_ip,
)
from digital_twin.models import CommandName, CommandPlan, CommandStep, Pose, ValidationResult
from digital_twin.rover_command_model import RoverCommandCalibration


class RecordingPublisher:
    def __init__(self):
        self.commands: list[CommandStep] = []

    def publish(self, command: CommandStep) -> None:
        self.commands.append(command)


class FixedCompletionWaiter:
    def __init__(self, completion: CommandCompletion):
        self.completion = completion
        self.waited_commands: list[CommandStep] = []

    def wait_for_completion(self, command: CommandStep, command_index: int, timeout_s: float) -> CommandCompletion:
        self.waited_commands.append(command)
        return self.completion


class SequencePoseProvider:
    def __init__(self, poses: list[Pose]):
        self.poses = poses
        self.index = 0

    def current_pose(self) -> Pose | None:
        if not self.poses:
            return None
        return self.poses[min(self.index, len(self.poses) - 1)]

    def advance(self, _seconds: float) -> None:
        self.index += 1


class ExecutorTests(unittest.TestCase):
    def test_enforce_waits_for_command_status(self):
        publisher = RecordingPublisher()
        waiter = FixedCompletionWaiter(CommandCompletion(completed=True, status="completed"))
        executor = StepwiseExecutor(
            settings=PlannerSettings(mode=GateMode.ENFORCE),
            publisher=publisher,
            completion_waiter=waiter,
        )
        plan = CommandPlan(
            goal=Pose(1.0, 0.0, 0.0),
            commands=(CommandStep(CommandName.FWD, 40, 1000),),
        )
        validation = ValidationResult(
            approved=True,
            reason="ok",
            final_pose=Pose(1.0, 0.0, 0.0),
            min_clearance_m=999.0,
        )

        self.assertTrue(executor.execute(plan, validation))
        self.assertEqual(publisher.commands, list(plan.commands))
        self.assertEqual(waiter.waited_commands, list(plan.commands))

    def test_enforce_stops_after_failed_command_status(self):
        publisher = RecordingPublisher()
        waiter = FixedCompletionWaiter(CommandCompletion(completed=False, failed=True, status="failed"))
        executor = StepwiseExecutor(
            settings=PlannerSettings(mode=GateMode.ENFORCE),
            publisher=publisher,
            completion_waiter=waiter,
        )
        plan = CommandPlan(
            goal=Pose(1.0, 0.0, 0.0),
            commands=(CommandStep(CommandName.FWD, 40, 1000),),
        )
        validation = ValidationResult(
            approved=True,
            reason="ok",
            final_pose=Pose(1.0, 0.0, 0.0),
            min_clearance_m=999.0,
        )

        self.assertFalse(executor.execute(plan, validation))
        self.assertEqual(
            [command.command for command in publisher.commands],
            [CommandName.FWD, CommandName.STOP],
        )

    def test_enforce_can_override_real_command_speed(self):
        publisher = RecordingPublisher()
        waiter = FixedCompletionWaiter(CommandCompletion(completed=True, status="completed"))
        executor = StepwiseExecutor(
            settings=PlannerSettings(mode=GateMode.ENFORCE),
            publisher=publisher,
            completion_waiter=waiter,
            execution_speed=2,
            inter_command_delay_s=0.0,
        )
        plan = CommandPlan(
            goal=Pose(1.0, 0.0, 0.0),
            commands=(
                CommandStep(CommandName.RIGHT, 30, 30),
                CommandStep(CommandName.FWD, 40, 8856),
                CommandStep(CommandName.STOP, 0, 0),
            ),
        )
        validation = ValidationResult(
            approved=True,
            reason="ok",
            final_pose=Pose(1.0, 0.0, 0.0),
            min_clearance_m=999.0,
        )

        self.assertTrue(executor.execute(plan, validation))
        self.assertEqual(
            [command.to_ros_payload() for command in publisher.commands],
            [[4, 2, 8000], [1, 2, 8856], [0, 0, 0]],
        )
        self.assertEqual(
            [command.to_ros_payload() for command in waiter.waited_commands],
            [[4, 2, 8000], [1, 2, 8856], [0, 0, 0]],
        )

    def test_enforce_converts_turn_degrees_to_real_steps_without_speed_override(self):
        publisher = RecordingPublisher()
        executor = StepwiseExecutor(
            settings=PlannerSettings(mode=GateMode.ENFORCE),
            publisher=publisher,
        )
        plan = CommandPlan(
            goal=Pose(0.0, 0.0, 90.0),
            commands=(CommandStep(CommandName.LEFT, 30, 90),),
        )
        validation = ValidationResult(
            approved=True,
            reason="ok",
            final_pose=Pose(0.0, 0.0, 90.0),
            min_clearance_m=999.0,
        )

        self.assertTrue(executor.execute(plan, validation))
        self.assertEqual(
            [command.to_ros_payload() for command in publisher.commands],
            [[3, 30, 24000]],
        )

    def test_enforce_can_oversize_real_command_for_telemetry_stop(self):
        publisher = RecordingPublisher()
        executor = StepwiseExecutor(
            settings=PlannerSettings(mode=GateMode.ENFORCE),
            publisher=publisher,
            execution_distance_scale=1.25,
        )
        plan = CommandPlan(
            goal=Pose(0.0, 0.0, 90.0),
            commands=(CommandStep(CommandName.LEFT, 30, 90),),
        )
        validation = ValidationResult(
            approved=True,
            reason="ok",
            final_pose=Pose(0.0, 0.0, 90.0),
            min_clearance_m=999.0,
        )

        self.assertTrue(executor.execute(plan, validation))
        self.assertEqual(
            [command.to_ros_payload() for command in publisher.commands],
            [[3, 30, 30000]],
        )


class RoverNameTests(unittest.TestCase):
    def test_rover_name_from_ip_matches_bridge_format(self):
        self.assertEqual(rover_name_from_ip("192.168.1.5"), "ip_192_168_1_5")
        self.assertEqual(rover_name_from_ip(" 10.0.0.42 "), "ip_10_0_0_42")


class RoverHttpPublisherTests(unittest.TestCase):
    class Response:
        status = 200

        def __init__(self, body=b'{"status": 200}'):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def test_continual_payload_matches_rover_controller_contract(self):
        command = CommandStep(CommandName.LEFT, 1, 24_000)
        self.assertEqual(
            rover_http_payload(command, "continual"),
            {"wheel_action": {"linear_x": 0.0, "angular_z": 1.0}},
        )

    def test_step_payload_matches_rover_controller_contract(self):
        command = CommandStep(CommandName.FWD, 2, 8_000)
        self.assertEqual(
            rover_http_payload(command, "step"),
            {"wheel_action": {"command": "FWD", "speed": 2, "distance": 8_000}},
        )

    @patch("digital_twin.executor.urllib.request.urlopen")
    def test_publisher_posts_json_to_rover_port_5000(self, urlopen):
        urlopen.return_value = self.Response()
        publisher = HttpWheelCommandPublisher("192.168.0.157", command_mode="continual")

        publisher.publish(CommandStep(CommandName.STOP, 0, 0))

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://192.168.0.157:5000/")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["Content-type"], "application/json")
        self.assertEqual(
            json.loads(request.data),
            {"wheel_action": {"linear_x": 0.0, "angular_z": 0.0}},
        )

    @patch("digital_twin.executor.urllib.request.urlopen")
    def test_status_is_read_through_rover_http_controller(self, urlopen):
        urlopen.return_value = self.Response(b'{"status": "Success7"}')
        publisher = HttpWheelCommandPublisher("192.168.0.157")

        self.assertEqual(publisher.latest_status(), "Success7")
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://192.168.0.157:5000/ip_192_168_0_157/command_status",
        )


class DurationCompletionWaiterTests(unittest.TestCase):
    def setUp(self):
        self.cal = RoverCommandCalibration()
        self.slept: list[float] = []

    def _waiter(self, status_provider=None):
        return DurationCompletionWaiter(
            calibration=self.cal,
            status_provider=status_provider,
            sleep=self.slept.append,
            poll_interval_s=0.05,
        )

    def test_completes_after_deterministic_duration(self):
        # 11500 steps (0.5 m) at ~389 steps/s ~= 29.6 s + 0.2 latency.
        command = self.cal.drive_command(0.5)
        completion = self._waiter().wait_for_completion(command, 0, timeout_s=120.0)
        self.assertTrue(completion.completed)
        # It slept roughly the predicted duration in total.
        self.assertAlmostEqual(sum(self.slept), self.cal.duration_s(command.distance), delta=0.1)

    def test_obstacle_mode_fails_the_command(self):
        # Status reports the bridge's obstacle mode partway through.
        calls = {"n": 0}

        def status():
            calls["n"] += 1
            return "Ignored wheel command: obstacle mode" if calls["n"] > 3 else "Success"

        completion = self._waiter(status_provider=status).wait_for_completion(
            self.cal.drive_command(1.0), 0, timeout_s=120.0
        )
        self.assertFalse(completion.completed)
        self.assertTrue(completion.failed)
        self.assertIn("obstacle", completion.status)

    def test_timeout_caps_the_wait(self):
        command = self.cal.drive_command(1.0)  # long motion
        self._waiter().wait_for_completion(command, 0, timeout_s=1.0)
        self.assertAlmostEqual(sum(self.slept), 1.0, delta=0.06)


class PoseTelemetryCompletionWaiterTests(unittest.TestCase):
    def setUp(self):
        self.cal = RoverCommandCalibration()
        self.publisher = RecordingPublisher()

    def test_stops_forward_command_at_expected_pose(self):
        provider = SequencePoseProvider([
            Pose(0.0, 0.0, 0.0),
            Pose(0.4, 0.0, 0.0),
            Pose(0.96, 0.0, 0.0),
        ])
        waiter = PoseTelemetryCompletionWaiter(
            pose_provider=provider,
            calibration=self.cal,
            stop_publisher=self.publisher,
            sleep=provider.advance,
            linear_tolerance_m=0.05,
        )
        waiter.set_expected_pose(provider.current_pose(), Pose(1.0, 0.0, 0.0))
        result = waiter.wait_for_completion(self.cal.drive_command(1.25), 0, timeout_s=5.0)

        self.assertTrue(result.completed)
        self.assertEqual([c.command for c in self.publisher.commands], [CommandName.STOP])

    def test_stops_turn_command_at_expected_yaw(self):
        provider = SequencePoseProvider([
            Pose(0.0, 0.0, 0.0),
            Pose(0.0, 0.0, 45.0),
            Pose(0.0, 0.0, 88.0),
        ])
        waiter = PoseTelemetryCompletionWaiter(
            pose_provider=provider,
            calibration=self.cal,
            stop_publisher=self.publisher,
            sleep=provider.advance,
            yaw_tolerance_deg=3.0,
        )
        waiter.set_expected_pose(provider.current_pose(), Pose(0.0, 0.0, 90.0))
        result = waiter.wait_for_completion(self.cal.turn_command(120.0), 0, timeout_s=5.0)

        self.assertTrue(result.completed)
        self.assertEqual([c.command for c in self.publisher.commands], [CommandName.STOP])

    def test_timeout_stops_and_fails_instead_of_continuing(self):
        provider = SequencePoseProvider([Pose(0.0, 0.0, 0.0)])
        waiter = PoseTelemetryCompletionWaiter(
            pose_provider=provider,
            calibration=self.cal,
            stop_publisher=self.publisher,
            sleep=provider.advance,
        )

        result = waiter.wait_for_completion(
            self.cal.drive_command(1.0), 0, timeout_s=0.01
        )

        self.assertFalse(result.completed)
        self.assertTrue(result.failed)
        self.assertTrue(result.timed_out)
        self.assertEqual([c.command for c in self.publisher.commands], [CommandName.STOP])


if __name__ == "__main__":
    unittest.main()
