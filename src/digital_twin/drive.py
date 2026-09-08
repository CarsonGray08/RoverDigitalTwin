#!/usr/bin/env python3
"""Drive the rover to a goal a fixed distance ahead of its current pose.

Prerequisite: the live twin is already up (mapper + Isaac validator)::

    ./scripts/live_twin.sh

and a baseline scan exists (``python -m digital_twin.scan``). This command
pauses map integration and resumes the phone stream for localization, keeping
the baseline map fixed during driving by default.

What it does each run:

  1. Pause map integration, then resume the phone stream (``POST /scan/start``).
  2. Read the live rover pose (``rover_pose.json``) the mapper publishes.
  3. Compute a goal ``--distance-m`` straight ahead of that pose (same heading).
  4. Run the closed loop: each command is built and validated against the live
     twin first; only an approved command is offered to the rover.

By default this is a rehearsal: it validates the drive against the live twin and
prints the plan, but sends nothing. Real motion needs both ``--rover-ip`` and
``--arm``; then every approved command pauses for an explicit ``y`` confirmation
before it is published to the rover (STOP is always sent immediately).

    # rehearse the 1 m drive against the live twin (no motion)
    python -m digital_twin.drive

    # actually drive, confirming each command
    python -m digital_twin.drive --rover-ip 192.168.0.157 --arm --telemetry-stop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from digital_twin.closed_loop import ClosedLoopController
from digital_twin.config import GateMode, PlannerSettings
from digital_twin.executor import (
    ConfirmingPublisher,
    DEFAULT_COMMAND_MODE,
    DriveAborted,
    DryRunPublisher,
    DurationCompletionWaiter,
    HttpWheelCommandPublisher,
    PoseTelemetryCompletionWaiter,
    StepwiseExecutor,
)
from digital_twin.geometry import advance_pose
from digital_twin.live_pose import LivePoseFileProvider, pose_from_payload, read_pose
from digital_twin.live_twin_session import resolve_session_dir
from digital_twin.models import CommandName, CommandStep, Pose, SafetyLimits, WorldModel
from digital_twin.observation_client import HttpObservationClient, RoverObservation
from digital_twin.planner import HeuristicLowLevelPlanner
from digital_twin.rover_command_model import RoverCommandCalibration
from digital_twin.validator_client import HttpPlanValidator


FIRMWARE_SPEED_CHOICES = (1, 2, 3)
DEFAULT_REAL_COMMAND_SPEED = 1


class LivePoseObservationClient:
    """Observation client that uses mapper pose as truth, not validator playback.

    The Isaac observation endpoint reports the visible articulation pose. During
    `/validate_plan` that articulation is intentionally driven through a proposed
    command, so it can briefly or persistently differ from the real rover. The
    mapper's `rover_pose.json` is the real localization stream used for driving.
    """

    def __init__(self, inner: HttpObservationClient, pose_provider: LivePoseFileProvider):
        self.inner = inner
        self.pose_provider = pose_provider

    def observe(self) -> RoverObservation:
        observation = self.inner.observe()
        live_pose = self.pose_provider.current_pose()
        if live_pose is None:
            return observation
        payload = dict(observation.payload)
        payload["rover_pose"] = live_pose.to_dict()
        live_twin = payload.get("live_twin")
        if isinstance(live_twin, dict):
            live_twin = dict(live_twin)
            live_twin["pose_source"] = "mapper_rover_pose_json"
            payload["live_twin"] = live_twin
        return RoverObservation(
            pose=live_pose,
            payload=payload,
            vision_rgb_base64=observation.vision_rgb_base64,
            chase_rgb_base64=observation.chase_rgb_base64,
        )


def post_control(base_url: str, path: str) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            # The mapper is up but no phone is connected to it.
            raise RuntimeError(
                "the phone is not connected to the mapper, so the scan can't be "
                "resumed. Open the LiveScanner app and tap Connect, then re-run "
                "(or pass --no-auto-scan if the scan is already running).") from exc
        raise RuntimeError(f"mapper control request failed for /{path}: {exc}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"mapper control request failed for /{path}: {exc}") from exc


def scan_control(base_url: str, action: str) -> None:
    post_control(base_url, f"scan/{action}")


def integration_control(base_url: str, action: str) -> None:
    post_control(base_url, f"map/integration/{action}")


def wait_for_live_pose(map_dir: Path, timeout_s: float) -> Pose:
    """Poll ``rover_pose.json`` until a normal-tracking pose appears.

    The mapper only publishes a twin-frame rover pose once alignment is
    established and tracking is normal, so this also confirms the live twin is
    actually localizing the rover before we plan a drive from it."""
    deadline = time.monotonic() + timeout_s
    last_reason = "no rover_pose.json yet"
    while time.monotonic() < deadline:
        payload = read_pose(map_dir)
        if payload is None:
            last_reason = "no rover_pose.json yet"
        elif not payload.get("tracking_normal", False):
            last_reason = "rover pose present but tracking not normal"
        else:
            return pose_from_payload(payload)
        time.sleep(0.25)
    raise TimeoutError(
        f"no live rover pose within {timeout_s:.0f}s ({last_reason}). Is the scan "
        f"running and aligned? Start/resume the scan and make sure tracking is normal.")


def build_settings(args: argparse.Namespace, mode: GateMode) -> PlannerSettings:
    limits = SafetyLimits(
        max_speed=args.max_speed,
        max_distance=args.max_distance,
        min_clearance_m=args.min_clearance_m,
        goal_tolerance_m=args.goal_tolerance_m,
        goal_tolerance_yaw_deg=args.goal_tolerance_yaw_deg,
        command_timeout_s=args.command_timeout_s,
    )
    world = WorldModel(obstacles=())
    return PlannerSettings(mode=mode, max_attempts=args.max_attempts, limits=limits, world=world)


def build_executor(args: argparse.Namespace, settings: PlannerSettings) -> StepwiseExecutor:
    completion_waiter = None
    pose_provider = None
    if settings.mode == GateMode.ENFORCE:
        inner = HttpWheelCommandPublisher(
            rover_ip=args.rover_ip,
            command_mode=args.command_mode,
            subscribe_status=not args.no_command_status_wait,
            timeout_s=args.rover_http_timeout_s,
        )
        # Confirm before each real send; STOP always passes through.
        publisher = ConfirmingPublisher(inner, auto_yes=args.yes)
        calibration = RoverCommandCalibration.from_limits(settings.limits)
        if args.telemetry_stop:
            pose_provider = LivePoseFileProvider(
                args.map_dir,
                max_age_s=args.telemetry_max_pose_age_s,
            )
            completion_waiter = PoseTelemetryCompletionWaiter(
                pose_provider=pose_provider,
                calibration=calibration,
                stop_publisher=publisher,
                status_provider=None if args.no_command_status_wait else inner.latest_status,
                linear_tolerance_m=args.telemetry_distance_tolerance_m,
                yaw_tolerance_deg=args.telemetry_yaw_tolerance_deg,
                telemetry_timeout_s=args.telemetry_pose_timeout_s,
            )
        else:
            completion_waiter = DurationCompletionWaiter(
                calibration=calibration,
                status_provider=None if args.no_command_status_wait else inner.latest_status,
            )
    else:
        publisher = DryRunPublisher()
    return StepwiseExecutor(
        settings=settings,
        publisher=publisher,
        completion_waiter=completion_waiter,
        pose_provider=pose_provider,
        execution_speed=args.real_command_speed,
        execution_distance_scale=args.telemetry_command_scale if args.telemetry_stop else 1.0,
        inter_command_delay_s=args.inter_command_delay_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rover-ip", help="rover IP; required with --arm for HTTP commands")
    parser.add_argument("--arm", action="store_true", help="allow real HTTP wheel commands")
    parser.add_argument("--yes", action="store_true",
                        help="with --arm, skip the per-command confirm prompt (send approved "
                             "commands automatically). Default is to confirm each one.")
    parser.add_argument("--distance-m", type=float, default=1.0,
                        help="how far straight ahead of the current pose to set the goal")
    parser.add_argument("--map-dir", type=Path, default=Path("/tmp/nvblox_map"),
                        help="map root or session dir; defaults to the active live_twin.sh "
                             "session via its 'latest' pointer, so normally omit it")
    parser.add_argument("--control-url", default="http://127.0.0.1:9100",
                        help="mapper control URL (resumes the scan)")
    parser.add_argument("--validator-url", default="http://127.0.0.1:8765/validate_plan",
                        help="Isaac live-twin /validate_plan URL")
    parser.add_argument("--observation-url", default=None,
                        help="live-twin /observation URL; defaults from --validator-url")
    parser.add_argument("--no-auto-scan", action="store_true",
                        help="do not resume the scan; assume it is already running")
    parser.add_argument("--map-integration-during-drive", action="store_true",
                        help="keep fusing incoming RGB-D/depth frames into the nvblox map "
                             "while driving. By default the stream is localization-only "
                             "during drive commands so validation/replan frames cannot skew "
                             "the baseline 360 map.")
    parser.add_argument("--resume-map-integration-after-drive", action="store_true",
                        help="after the drive script exits, resume map integration. The "
                             "default leaves it paused so post-drive localization does not "
                             "rewrite the map until you explicitly resume/scan again.")
    parser.add_argument("--pose-timeout-s", type=float, default=20.0,
                        help="how long to wait for a live rover pose before giving up")
    parser.add_argument("--commands-per-cycle", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=4)
    # Physical command timing and safety settings.
    parser.add_argument(
        "--real-command-speed",
        type=int,
        choices=FIRMWARE_SPEED_CHOICES,
        default=DEFAULT_REAL_COMMAND_SPEED,
        help=(
            "firmware-compatible speed sent to the real rover for non-STOP "
            "commands. The live twin still validates with planner speeds."
        ),
    )
    parser.add_argument(
        "--command-mode",
        choices=("continual", "step"),
        default=DEFAULT_COMMAND_MODE,
        help="continual sends Mode-A HTTP commands until odometry sends STOP; "
             "step sends bounded Mode-B HTTP commands",
    )
    parser.add_argument("--no-command-status-wait", action="store_true")
    parser.add_argument("--rover-http-timeout-s", type=float, default=5.0,
                        help="timeout for the rover HTTP controller on port 5000")
    parser.add_argument("--inter-command-delay-s", type=float, default=15.0)
    parser.add_argument("--telemetry-stop", action="store_true",
                        help="while armed, stop each real command from live camera pose "
                             "telemetry instead of only waiting estimated step duration")
    parser.add_argument("--telemetry-command-scale", type=float, default=1.25,
                        help="with --telemetry-stop, multiply the real command's step "
                             "distance by this factor so camera telemetry, not the "
                             "firmware step count, determines the stop point")
    parser.add_argument("--telemetry-distance-tolerance-m", type=float, default=0.05)
    parser.add_argument("--telemetry-yaw-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--telemetry-pose-timeout-s", type=float, default=3.0,
                        help="how long to wait for pose telemetry when a command starts")
    parser.add_argument("--telemetry-max-pose-age-s", type=float, default=2.0,
                        help="ignore rover_pose.json samples older than this")
    # World / limits.
    parser.add_argument("--max-speed", type=int, default=60)
    parser.add_argument("--max-distance", type=int, default=30000)
    parser.add_argument("--min-clearance-m", type=float, default=0.25)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.15)
    parser.add_argument("--goal-tolerance-yaw-deg", type=float, default=8.0)
    parser.add_argument("--command-timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    if args.arm and args.command_mode == "continual":
        args.telemetry_stop = True

    if args.arm != bool(args.rover_ip):
        parser.error("real motion requires both --rover-ip and --arm; omit both for a twin-only rehearsal")
    if args.distance_m <= 0:
        parser.error("--distance-m must be positive")

    args.map_dir = resolve_session_dir(args.map_dir)
    print(f"Using map dir: {args.map_dir}")

    if not args.no_auto_scan:
        try:
            scan_control(args.control_url, "start")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Scan resumed through mapper control (continuing the existing map).")
    if not args.map_integration_during_drive:
        try:
            integration_control(args.control_url, "pause")
        except RuntimeError as exc:
            print(
                f"{exc}\n"
                "Could not pause map integration. Restart scripts/live_twin.sh so the mapper "
                "has the /map/integration/pause endpoint, or pass "
                "--map-integration-during-drive to allow mapping during motion.",
                file=sys.stderr,
            )
            return 2
        print("Map integration paused; using incoming frames for localization only.")

    try:
        start_pose = wait_for_live_pose(args.map_dir, args.pose_timeout_s)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    goal = advance_pose(start_pose, args.distance_m)
    print(f"Live rover pose: x={start_pose.x:.3f} y={start_pose.y:.3f} yaw={start_pose.yaw_deg:.1f} deg")
    print(f"Goal ({args.distance_m:.2f} m ahead): x={goal.x:.3f} y={goal.y:.3f} yaw={goal.yaw_deg:.1f} deg")

    mode = GateMode.ENFORCE if args.arm else GateMode.DRY_RUN
    settings = build_settings(args, mode)
    validator = HttpPlanValidator(args.validator_url)
    observation_url = args.observation_url or (
        args.validator_url[: -len("/validate_plan")] + "/observation"
        if args.validator_url.endswith("/validate_plan") else None)
    observation_client = None
    if observation_url:
        observation_client = LivePoseObservationClient(
            HttpObservationClient(observation_url),
            LivePoseFileProvider(args.map_dir, max_age_s=args.telemetry_max_pose_age_s),
        )
    executor = build_executor(args, settings)
    controller = ClosedLoopController(
        planner=HeuristicLowLevelPlanner(),
        validator=validator,
        executor=executor,
        settings=settings,
        observation_client=observation_client,
        commands_per_cycle=max(1, args.commands_per_cycle),
        max_cycles=max(1, args.max_cycles),
    )

    mode_label = "ARMED REAL ROVER" if args.arm else "TWIN-ONLY REHEARSAL"
    print(f"{mode_label}: driving to the goal, validating each command in the live twin.")
    exit_code = 0
    try:
        result = controller.run(start_pose=start_pose, goal=goal)
    except DriveAborted as exc:
        # Operator declined a command: stop the rover and exit cleanly.
        print(f"Aborted: {exc}", file=sys.stderr)
        executor.publisher.publish(CommandStep(CommandName.STOP, 0, 0))
        exit_code = 130
        result = None
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        result = None
        exit_code = 1
    finally:
        close = getattr(executor.publisher, "close", None)
        if callable(close):
            close()
        if (
            not args.map_integration_during_drive
            and args.resume_map_integration_after_drive
        ):
            try:
                integration_control(args.control_url, "resume")
                print("Map integration resumed.")
            except RuntimeError as exc:
                print(f"Could not resume map integration: {exc}", file=sys.stderr)

    if result is not None:
        print(json.dumps({
            "reached_goal": result.reached_goal,
            "cycles_completed": result.cycles_completed,
            "final_pose": result.final_pose.to_dict(),
        }, indent=2))
        exit_code = 0 if result.reached_goal else 1
    if args.map_integration_during_drive or args.resume_map_integration_after_drive:
        print("Scan is still running (resumable): issue another goal, or stop it on the phone / Ctrl-C live_twin.sh.")
    else:
        print("Scan stream is still running for localization, with map integration paused.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
