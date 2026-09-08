from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from digital_twin.config import GateMode, PlannerSettings
from digital_twin.geometry import distance_m, normalize_yaw_deg, signed_yaw_delta_deg, yaw_error_deg
from digital_twin.models import CommandName, CommandPlan, CommandStep, Pose, ValidationResult
from digital_twin.rover_command_model import RoverCommandCalibration

DEFAULT_COMMAND_MODE = "continual"


class PoseProvider(Protocol):
    def current_pose(self) -> Pose | None:
        """Return the latest real rover pose if available."""


class CommandPublisher(Protocol):
    def publish(self, command: CommandStep) -> None:
        """Publish one low-level command to the rover."""


@dataclass(frozen=True)
class CommandCompletion:
    completed: bool
    timed_out: bool = False
    failed: bool = False
    status: str | None = None


class CommandCompletionWaiter(Protocol):
    def wait_for_completion(self, command: CommandStep, command_index: int, timeout_s: float) -> CommandCompletion:
        """Wait for the rover to report that one published command finished."""


class TargetAwareCompletionWaiter(Protocol):
    def set_expected_pose(self, start_pose: Pose | None, expected_pose: Pose | None) -> None:
        """Provide the expected command endpoint before wait_for_completion."""


def rover_name_from_ip(ip_address: str) -> str:
    """Return the rover's name as used by its HTTP command-status route."""
    return "ip_" + ip_address.strip().replace(".", "_")


class DryRunPublisher:
    def publish(self, command: CommandStep) -> None:
        print(f"DRY_RUN publish {command.to_ros_payload()} {command.to_dict()}")


class DriveAborted(RuntimeError):
    """Raised when the operator declines a command at the confirmation prompt."""


@dataclass
class ConfirmingPublisher:
    """Gate every motion command behind an interactive confirm prompt.

    The command has already passed twin validation by the time it reaches here;
    this is the human-in-the-loop check before anything is sent to the real
    rover. STOP is a safety command and is always forwarded without a prompt, so
    the rover can still be halted instantly. Declining a move raises
    :class:`DriveAborted`, which the driver turns into a clean stop + exit.
    ``auto_yes`` forwards everything unprompted (for non-interactive runs/tests).
    """

    inner: CommandPublisher
    auto_yes: bool = False
    prompt: Callable[[str], str] = input
    out: Callable[[str], None] = print

    def publish(self, command: CommandStep) -> None:
        if command.command is CommandName.STOP:
            self.inner.publish(command)
            return
        if not self.auto_yes:
            label = f"{command.command.value} speed={command.speed} distance={command.distance}"
            answer = self.prompt(f"  validated in twin. Send to rover: {label}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                raise DriveAborted(f"operator declined command: {label}")
        self.inner.publish(command)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def continual_twist_components(command: CommandStep) -> tuple[float, float]:
    """Translate a rover command into the continual Mode-A Twist controls."""
    if command.command is CommandName.FWD:
        return 1.0, 0.0
    if command.command is CommandName.REV:
        return -1.0, 0.0
    if command.command is CommandName.LEFT:
        return 0.0, 1.0
    if command.command is CommandName.RIGHT:
        return 0.0, -1.0
    return 0.0, 0.0


def rover_http_payload(command: CommandStep, command_mode: str) -> dict:
    """Serialize a wheel command for ``ezrassor_controller_server``."""
    if command_mode == "continual":
        linear_x, angular_z = continual_twist_components(command)
        wheel_action: dict[str, str | int | float] = {
            "linear_x": linear_x,
            "angular_z": angular_z,
        }
    elif command_mode == "step":
        wheel_action = command.to_dict()
    else:
        raise ValueError(f"unsupported rover command mode: {command_mode}")
    return {"wheel_action": wheel_action}


@dataclass
class HttpWheelCommandPublisher:
    """Send wheel commands through the rover's HTTP-to-ROS controller."""

    rover_ip: str
    command_mode: str = DEFAULT_COMMAND_MODE
    subscribe_status: bool = True
    timeout_s: float = 5.0
    controller_port: int = 5000

    def __post_init__(self) -> None:
        self.rover_ip = self.rover_ip.strip()
        if not self.rover_ip:
            raise ValueError("rover_ip must not be empty")
        if self.command_mode not in ("continual", "step"):
            raise ValueError(f"unsupported rover command mode: {self.command_mode}")
        self._command_url = f"http://{self.rover_ip}:{self.controller_port}/"
        rover_name = rover_name_from_ip(self.rover_ip)
        self._status_url = f"{self._command_url}{rover_name}/command_status"
        self._latest_status: str | None = None

    def publish(self, command: CommandStep) -> None:
        payload = rover_http_payload(command, self.command_mode)
        request = urllib.request.Request(
            self._command_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
                status_code = response.status
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"rover controller rejected command at {self._command_url}: HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"could not reach rover controller at {self._command_url}: {exc}"
            ) from exc
        if not 200 <= status_code < 300:
            raise RuntimeError(f"rover controller returned HTTP {status_code}")
        if body:
            try:
                reply = json.loads(body)
            except (TypeError, json.JSONDecodeError):
                reply = None
            if isinstance(reply, dict) and reply.get("status") not in (None, 200, "200"):
                raise RuntimeError(f"rover controller rejected command: {reply}")
        print(f"Sent {self.command_mode} wheel command by HTTP to {self._command_url}: {payload}")

    def latest_status(self) -> str | None:
        if not self.subscribe_status:
            return None
        request = urllib.request.Request(self._status_url, method="GET")
        try:
            with urllib.request.urlopen(
                request, timeout=min(self.timeout_s, 1.0)
            ) as response:
                reply = json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError):
            return None
        status = reply.get("status") if isinstance(reply, dict) else None
        self._latest_status = str(status) if status is not None else None
        return self._latest_status


_OBSTACLE_STATUS_TOKENS = ("obstacle", "ignored")


@dataclass
class DurationCompletionWaiter:
    """Completion via deterministic stepper timing.

    The real bridge publishes command_status="Success" the instant it writes
    serial, NOT when motion ends, so it cannot be used as a motion-complete
    signal. Steppers move an exact step count at a fixed rate, so the duration is
    deterministic: wait ``calibration.duration_s(distance)``. While waiting, an
    optional ``status_provider`` is polled for the bridge's obstacle-mode (it
    halts + reverses and ignores commands), which fails the command.
    """

    calibration: RoverCommandCalibration
    status_provider: Callable[[], str | None] | None = None
    sleep: Callable[[float], None] = time.sleep
    poll_interval_s: float = 0.05

    def wait_for_completion(self, command: CommandStep, command_index: int, timeout_s: float) -> CommandCompletion:
        duration = min(self.calibration.duration_s(command.distance), timeout_s)
        waited = 0.0
        while waited < duration:
            step = min(self.poll_interval_s, duration - waited)
            self.sleep(step)
            waited += step
            if self._obstacle_active():
                return CommandCompletion(completed=False, failed=True, status="obstacle mode active")
        return CommandCompletion(completed=True, status=f"duration {duration:.2f}s elapsed")

    def _obstacle_active(self) -> bool:
        if self.status_provider is None:
            return False
        status = (self.status_provider() or "").lower()
        return any(token in status for token in _OBSTACLE_STATUS_TOKENS)


@dataclass
class PoseTelemetryCompletionWaiter:
    """Stop motion from live camera-derived rover pose instead of raw step time.

    The rover firmware still only accepts wheel commands with a finite step
    count. This waiter sends a deliberately bounded command, then polls live pose
    telemetry and publishes STOP as soon as the measured distance/yaw target is
    reached. If telemetry never arrives, the command fails instead of silently
    trusting step counts.
    """

    pose_provider: PoseProvider
    calibration: RoverCommandCalibration
    stop_publisher: CommandPublisher
    status_provider: Callable[[], str | None] | None = None
    sleep: Callable[[float], None] = time.sleep
    poll_interval_s: float = 0.1
    linear_tolerance_m: float = 0.04
    yaw_tolerance_deg: float = 2.5
    telemetry_timeout_s: float = 3.0
    _expected_start_pose: Pose | None = None
    _expected_end_pose: Pose | None = None

    def set_expected_pose(self, start_pose: Pose | None, expected_pose: Pose | None) -> None:
        self._expected_start_pose = start_pose
        self._expected_end_pose = expected_pose

    def wait_for_completion(self, command: CommandStep, command_index: int, timeout_s: float) -> CommandCompletion:
        if command.command is CommandName.STOP or command.distance <= 0:
            return CommandCompletion(completed=True, status="stop/no-motion")

        start = self._expected_start_pose or self._wait_for_pose(timeout_s=min(self.telemetry_timeout_s, timeout_s))
        if start is None:
            self._publish_stop()
            return CommandCompletion(completed=False, failed=True, status="no live pose before command")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._obstacle_active():
                self._publish_stop()
                return CommandCompletion(completed=False, failed=True, status="obstacle mode active")
            pose = self.pose_provider.current_pose()
            if pose is None:
                self.sleep(self.poll_interval_s)
                continue
            if self._target_reached(command, start, pose):
                self._publish_stop()
                return CommandCompletion(completed=True, status=self._status(command, start, pose))
            self.sleep(self.poll_interval_s)

        self._publish_stop()
        return CommandCompletion(
            completed=False,
            timed_out=True,
            failed=True,
            status="telemetry target timeout",
        )

    def _wait_for_pose(self, timeout_s: float) -> Pose | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pose = self.pose_provider.current_pose()
            if pose is not None:
                return pose
            self.sleep(self.poll_interval_s)
        return None

    def _target_reached(self, command: CommandStep, start: Pose, current: Pose) -> bool:
        expected = self._expected_end_pose
        if expected is not None:
            if command.command in (CommandName.LEFT, CommandName.RIGHT):
                return yaw_error_deg(current.yaw_deg, expected.yaw_deg) <= self.yaw_tolerance_deg
            if command.command in (CommandName.FWD, CommandName.REV):
                return distance_m(current, expected) <= self.linear_tolerance_m

        if command.command in (CommandName.LEFT, CommandName.RIGHT):
            target_deg = self.calibration.steps_to_yaw(command.distance)
            signed_progress = signed_yaw_delta_deg(start.yaw_deg, current.yaw_deg)
            if command.command is CommandName.RIGHT:
                signed_progress *= -1.0
            return signed_progress >= max(0.0, target_deg - self.yaw_tolerance_deg)

        if command.command in (CommandName.FWD, CommandName.REV):
            target_m = self.calibration.steps_to_distance(command.distance)
            yaw_rad = math.radians(start.yaw_deg)
            dx = current.x - start.x
            dy = current.y - start.y
            signed_progress = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
            if command.command is CommandName.REV:
                signed_progress *= -1.0
            return signed_progress >= max(0.0, target_m - self.linear_tolerance_m)

        return True

    def _status(self, command: CommandStep, start: Pose, current: Pose) -> str:
        expected = self._expected_end_pose
        if expected is not None:
            if command.command in (CommandName.LEFT, CommandName.RIGHT):
                return (
                    f"telemetry yaw target error "
                    f"{yaw_error_deg(current.yaw_deg, expected.yaw_deg):.1f} deg"
                )
            return f"telemetry target distance {distance_m(current, expected):.3f} m"
        if command.command in (CommandName.LEFT, CommandName.RIGHT):
            target_deg = self.calibration.steps_to_yaw(command.distance)
            progress = abs(normalize_yaw_deg(current.yaw_deg - start.yaw_deg))
            return f"telemetry yaw {progress:.1f}/{target_deg:.1f} deg"
        target_m = self.calibration.steps_to_distance(command.distance)
        progress = distance_m(start, current)
        return f"telemetry distance {progress:.3f}/{target_m:.3f} m"

    def _obstacle_active(self) -> bool:
        if self.status_provider is None:
            return False
        status = (self.status_provider() or "").lower()
        return any(token in status for token in _OBSTACLE_STATUS_TOKENS)

    def _publish_stop(self) -> None:
        self.stop_publisher.publish(CommandStep(CommandName.STOP, 0, 0))


@dataclass
class StepwiseExecutor:
    settings: PlannerSettings
    publisher: CommandPublisher
    completion_waiter: CommandCompletionWaiter | None = None
    pose_provider: PoseProvider | None = None
    execution_speed: int | None = None
    execution_distance_scale: float = 1.0
    inter_command_delay_s: float = 15.0

    def execute(self, plan: CommandPlan, validation: ValidationResult) -> bool:
        if self.settings.mode == GateMode.DRY_RUN:
            for index, command in enumerate(plan.commands):
                print(f"DRY_RUN step {index}: {command.to_dict()}")
            return True

        if self.settings.mode != GateMode.ENFORCE:
            print(f"Mode {self.settings.mode.value}: validated plan logged, rover execution skipped")
            return True

        if not validation.approved:
            raise ValueError("Refusing to execute an unapproved plan")

        expected_poses = list(validation.expected_step_end_poses)
        command_count = len(plan.commands)
        for index, command in enumerate(plan.commands):
            execution_command = self._execution_command(command)
            print(f"ENFORCE step {index}: {execution_command.to_dict()}")
            start_pose = self.pose_provider.current_pose() if self.pose_provider is not None else None
            expected_pose = expected_poses[index] if index < len(expected_poses) else None
            set_expected = getattr(self.completion_waiter, "set_expected_pose", None)
            if callable(set_expected):
                set_expected(start_pose, expected_pose)
            self.publisher.publish(execution_command)
            if not self._wait_for_command(execution_command, index):
                self.publisher.publish(CommandStep.from_dict({"command": "STOP", "speed": 0, "distance": 0}))
                return False
            if self.pose_provider is not None and index < len(expected_poses):
                real_pose = self.pose_provider.current_pose()
                if real_pose is not None and not self._drift_ok(real_pose, expected_poses[index]):
                    self.publisher.publish(CommandStep.from_dict({"command": "STOP", "speed": 0, "distance": 0}))
                    print(f"Stopped after command {index}: real pose diverged from simulated expectation")
                    return False
            next_command = plan.commands[index + 1] if index + 1 < command_count else None
            self._wait_between_commands(index, command_count, next_command)
        return True

    def _execution_command(self, command: CommandStep) -> CommandStep:
        if command.speed <= 0:
            return command
        speed = self.execution_speed if self.execution_speed is not None else command.speed
        distance = command.distance
        if command.command in (CommandName.LEFT, CommandName.RIGHT):
            distance = round(command.distance / self.settings.limits.turn_degrees_per_distance_unit)
        if command.command in (CommandName.FWD, CommandName.REV, CommandName.LEFT, CommandName.RIGHT):
            distance = min(
                self.settings.limits.max_distance,
                max(0, round(distance * max(1.0, self.execution_distance_scale))),
            )
        return CommandStep(command=command.command, speed=speed, distance=distance)

    def _wait_for_command(self, command: CommandStep, command_index: int) -> bool:
        if self.completion_waiter is None:
            return True

        completion = self.completion_waiter.wait_for_completion(
            command,
            command_index,
            self._status_wait_timeout_s(command),
        )
        if completion.completed:
            print(f"Command {command_index} completed from status: {completion.status}")
            return True
        if completion.failed:
            print(f"Command {command_index} failed from status: {completion.status}")
            return False

        print(f"Command {command_index} status wait timed out; continuing without estimated wait")
        return True

    def _status_wait_timeout_s(self, command: CommandStep) -> float:
        return self.settings.limits.command_timeout_s

    def _wait_between_commands(
        self,
        command_index: int,
        command_count: int,
        next_command: CommandStep | None = None,
    ) -> None:
        if command_index >= command_count - 1 or self.inter_command_delay_s <= 0.0:
            return
        if next_command is not None and next_command.command is CommandName.STOP:
            return
        print(f"Waiting {self.inter_command_delay_s:.1f}s before next command")
        time.sleep(self.inter_command_delay_s)

    def _drift_ok(self, actual: Pose, expected: Pose) -> bool:
        return (
            distance_m(actual, expected) <= self.settings.acceptable_drift_m
            and yaw_error_deg(actual.yaw_deg, expected.yaw_deg) <= self.settings.acceptable_drift_yaw_deg
        )
