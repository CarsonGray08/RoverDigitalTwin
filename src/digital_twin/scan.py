#!/usr/bin/env python3
"""Run the real-phone nvblox 360-degree scan sweep.

Start the mapper and Isaac validator first:

    ./scripts/live_twin.sh

Connect the phone, then run a dry motion rehearsal:

    python -m digital_twin.scan

The rehearsal prints the same four 90-degree commands but never publishes. A
physical turn requires both the rover IP and an explicit arm flag:

    python -m digital_twin.scan --rover-ip 192.168.0.157 --arm --telemetry-stop

The command starts and stops the phone scan automatically, including during
rehearsal; use ``--no-auto-scan`` to leave the phone stream unchanged. After the
turns it waits for the mapper's tile manifest and reports the final map revision.
Ctrl-C publishes STOP when the real publisher has already been armed.
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

from digital_twin.executor import (
    DEFAULT_COMMAND_MODE,
    DryRunPublisher,
    HttpWheelCommandPublisher,
)
from digital_twin.geometry import normalize_yaw_deg, yaw_error_deg
from digital_twin.live_pose import LivePoseFileProvider
from digital_twin.live_twin_session import resolve_session_dir
from digital_twin.models import CommandName, CommandStep
from digital_twin.rover_command_model import SCHEMA_MAX_DISTANCE, RoverCommandCalibration


def turn_segments(calibration: RoverCommandCalibration, degrees: float, segment_deg: float) -> list[CommandStep]:
    if degrees == 0:
        raise ValueError("--degrees must be non-zero")
    max_segment = SCHEMA_MAX_DISTANCE / calibration.steps_per_deg
    if not 0 < segment_deg <= max_segment:
        raise ValueError(f"--segment-deg must be in (0, {max_segment:.1f}]")
    sign = 1.0 if degrees > 0 else -1.0
    remaining = abs(degrees)
    result: list[CommandStep] = []
    while remaining > 1e-6:
        result.append(calibration.turn_command(sign * min(segment_deg, remaining)))
        remaining -= min(segment_deg, remaining)
    return result


def wait_for_final_manifest(map_dir: Path, after_s: float, timeout_s: float) -> dict:
    path = map_dir / "manifest.json"
    final_mesh = map_dir / "nvblox_mesh.npz"
    deadline = time.monotonic() + timeout_s
    last_mtime = -1.0
    stable_since = None
    while time.monotonic() < deadline:
        if path.exists() and final_mesh.exists() and final_mesh.stat().st_mtime >= after_s:
            mtime = path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                stable_since = time.monotonic()
            elif stable_since is not None and time.monotonic() - stable_since >= 1.0:
                return json.loads(path.read_text())
        time.sleep(0.25)
    raise TimeoutError(f"no stable manifest at {path} within {timeout_s:.0f}s")


def scan_control(base_url: str, action: str) -> None:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/scan/{action}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not {action} scan through mapper control: {exc}") from exc


def pause_map_integration(base_url: str) -> None:
    """Tell the mapper to use incoming frames for localization only.

    Paused BEFORE the tracking resume so no frame can integrate (and skew the
    finalized 360 map) in the gap between /scan/start and the pause."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/map/integration/pause", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not pause map integration: {exc}") from exc


def request_under_rover_fill(base_url: str) -> None:
    """Ask the mapper to close the rover's occlusion disc into observed terrain.

    The mapper runs the fill on its current, settled map and rewrites the tiles +
    manifest; the caller then waits for that fresh manifest to confirm it landed."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/map/fill_under_rover", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            if response.status not in (200, 202):
                raise RuntimeError(f"HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not request under-rover fill: {exc}") from exc


def publish_stop_best_effort(publisher, timeout_s: float = 3.0) -> bool:
    """Send STOP without letting error cleanup hang on an HTTP request."""
    old_timeout = getattr(publisher, "timeout_s", None)
    if old_timeout is not None:
        publisher.timeout_s = min(float(old_timeout), float(timeout_s))
    try:
        publisher.publish(CommandStep(CommandName.STOP, 0, 0))
        return True
    except RuntimeError as exc:
        print(f"Could not publish STOP: {exc}", file=sys.stderr)
        return False
    finally:
        if old_timeout is not None:
            publisher.timeout_s = old_timeout


def execute_turn_segment_with_telemetry(
    *,
    publisher,
    pose_provider: LivePoseFileProvider,
    command: CommandStep,
    calibration: RoverCommandCalibration,
    command_scale: float,
    yaw_tolerance_deg: float,
    pose_timeout_s: float,
    poll_interval_s: float,
    target_yaw: float | None = None,
    republish_interval_s: float = 0.0,
) -> bool:
    start = pose_provider.current_pose()
    if start is None:
        print("No fresh live rover pose before turn segment.", file=sys.stderr)
        publish_stop_best_effort(publisher)
        return False
    sign = 1.0 if command.command is CommandName.LEFT else -1.0
    target_delta = sign * calibration.steps_to_yaw(command.distance)
    target_yaw = normalize_yaw_deg(start.yaw_deg + target_delta) if target_yaw is None else target_yaw
    execution_distance = min(
        SCHEMA_MAX_DISTANCE,
        max(command.distance, round(command.distance * max(1.0, command_scale))),
    )
    execution_command = CommandStep(command.command, command.speed, execution_distance)
    print(
        f"  telemetry target yaw={target_yaw:.1f} deg "
        f"(command max {calibration.steps_to_yaw(execution_distance):.1f} deg)",
        flush=True,
    )
    publisher.publish(execution_command)
    effective_timeout_s = max(pose_timeout_s, calibration.duration_s(execution_distance) + 5.0)
    deadline = time.monotonic() + effective_timeout_s
    next_republish = (
        time.monotonic() + republish_interval_s
        if republish_interval_s > 0.0 else float("inf")
    )
    last_pose = None
    last_err = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_republish:
            publisher.publish(execution_command)
            next_republish = now + republish_interval_s
        pose = pose_provider.current_pose()
        if pose is not None:
            err = yaw_error_deg(pose.yaw_deg, target_yaw)
            last_pose = pose
            last_err = err
            if err <= yaw_tolerance_deg:
                publish_stop_best_effort(publisher)
                print(f"  telemetry yaw reached: current={pose.yaw_deg:.1f} target={target_yaw:.1f} err={err:.1f}")
                return True
        time.sleep(poll_interval_s)
    if last_pose is None:
        print("Telemetry turn target timed out with no fresh pose; publishing STOP.", file=sys.stderr)
    else:
        print(
            "Telemetry turn target timed out; "
            f"last_yaw={last_pose.yaw_deg:.1f} target={target_yaw:.1f} "
            f"err={last_err:.1f}; publishing STOP.",
            file=sys.stderr,
        )
    publish_stop_best_effort(publisher)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rover-ip", help="rover IP; required with --arm for HTTP commands")
    parser.add_argument("--arm", action="store_true", help="allow real HTTP wheel commands")
    parser.add_argument("--degrees", type=float, default=360.0, help="positive=LEFT/CCW, negative=RIGHT/CW")
    parser.add_argument("--segment-deg", type=float, default=90.0, help="turn segment size; 90 degrees is the tested default")
    parser.add_argument("--speed", type=int, default=1, help="firmware-compatible speed field")
    parser.add_argument("--turn-command-mode", choices=("continual", "step"), default=DEFAULT_COMMAND_MODE,
                        help="continual sends linear/angular HTTP commands until STOP; "
                             "step sends bounded command/speed/distance HTTP commands")
    parser.add_argument("--map-dir", type=Path, default=Path("/tmp/nvblox_map"),
                        help="map root or session dir; defaults to the active live_twin.sh "
                             "session via its 'latest' pointer, so normally omit it")
    parser.add_argument("--finalize-timeout-s", type=float, default=30.0)
    parser.add_argument("--control-url", default="http://127.0.0.1:9100",
                        help="mapper control URL; run_nvblox_live.sh starts this by default")
    parser.add_argument("--no-auto-scan", action="store_true",
                        help="leave start/stop scan under manual phone control")
    parser.add_argument("--rover-http-timeout-s", type=float, default=5.0,
                        help="timeout for the rover HTTP controller on port 5000")
    parser.add_argument("--stop-scan-on-error", action="store_true",
                        help="send mapper /scan/stop even when rover "
                             "publishing fails. By default, publish failures leave "
                             "the scan running for recovery/debugging.")
    parser.add_argument("--no-fill", action="store_true",
                        help="skip closing the empty circle under the rover after the "
                             "scan. By default the driver asks the mapper to fill that "
                             "occlusion disc flush with the floor so the rover sits on "
                             "real terrain (and its tracked Z registers on it).")
    parser.add_argument("--telemetry-stop", action="store_true",
                        help="with --arm, stop each turn segment from live camera yaw "
                             "instead of waiting only estimated step duration")
    parser.add_argument("--telemetry-command-scale", type=float, default=1.25,
                        help="with --telemetry-stop, oversize the real turn command by "
                             "this factor so camera yaw determines the stop point")
    parser.add_argument("--telemetry-yaw-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--telemetry-pose-timeout-s", type=float, default=45.0)
    parser.add_argument("--telemetry-max-pose-age-s", type=float, default=2.0)
    parser.add_argument("--telemetry-poll-s", type=float, default=0.1)
    parser.add_argument("--continual-republish-s", type=float, default=0.5,
                        help="with --turn-command-mode continual, resend the turn "
                             "command at this cadence. Repeated continual commands "
                             "ramp some firmware builds to a faster turn tier.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the interactive 'press Enter to begin' prompt, "
                             "for driving this script from another program (e.g. the "
                             "dashboard) where the launch action is itself the confirm")
    parser.add_argument("--track-after", action="store_true",
                        help="after the map is finalized, resume the scan with map "
                             "integration paused so the twin keeps localizing the rover "
                             "(rover_pose.json stays live) without rewriting the 360 map. "
                             "Without this the phone stops streaming and the twin's pose "
                             "freezes until the scan is resumed.")
    args = parser.parse_args()

    if args.arm != bool(args.rover_ip):
        parser.error("real motion requires both --rover-ip and --arm; omit both for a dry rehearsal")
    if args.arm and args.turn_command_mode == "continual":
        args.telemetry_stop = True

    args.map_dir = resolve_session_dir(args.map_dir)
    print(f"Using map dir: {args.map_dir}")

    calibration = RoverCommandCalibration()
    try:
        commands = turn_segments(calibration, args.degrees, args.segment_deg)
    except ValueError as exc:
        parser.error(str(exc))
    commands = [CommandStep(command.command, args.speed, command.distance) for command in commands]
    total_duration = sum(calibration.duration_s(command.distance) for command in commands)
    mode = "ARMED REAL ROVER" if args.arm else "DRY RUN"
    print(f"{mode}: {args.degrees:.0f} degrees as {len(commands)} turn segments; expected motion {total_duration:.1f}s")
    for index, command in enumerate(commands, 1):
        print(f"  {index}: {command.command.value} {calibration.steps_to_yaw(command.distance):.1f} degrees / {command.distance} steps")
    if not args.yes:
        input("Confirm the phone is connected, the rover area is clear, then press Enter to begin: ")

    if not args.no_auto_scan:
        try:
            scan_control(args.control_url, "start")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("Scan started through mapper control.")
    scan_started_at = time.time()
    if args.arm:
        publisher = HttpWheelCommandPublisher(
            rover_ip=args.rover_ip,
            command_mode=args.turn_command_mode,
            timeout_s=args.rover_http_timeout_s,
        )
    else:
        publisher = DryRunPublisher()
    pose_provider = (
        LivePoseFileProvider(args.map_dir, max_age_s=args.telemetry_max_pose_age_s)
        if args.arm and args.telemetry_stop else None
    )
    telemetry_start_pose = None
    if pose_provider is not None:
        deadline = time.monotonic() + args.telemetry_pose_timeout_s
        while time.monotonic() < deadline:
            telemetry_start_pose = pose_provider.current_pose()
            if telemetry_start_pose is not None:
                break
            time.sleep(args.telemetry_poll_s)
        if telemetry_start_pose is None:
            print("No fresh live rover pose before telemetry 360.", file=sys.stderr)
            publish_stop_best_effort(publisher)
            return 2
    motion_failed = False
    interrupted = False
    try:
        for index, command in enumerate(commands, 1):
            if pose_provider is not None:
                print(f"[{index}/{len(commands)}] telemetry-controlled turn")
                completed_deg = sum(
                    calibration.steps_to_yaw(prior.distance)
                    * (1.0 if prior.command is CommandName.LEFT else -1.0)
                    for prior in commands[:index]
                )
                target_yaw = normalize_yaw_deg(telemetry_start_pose.yaw_deg + completed_deg)
                ok = execute_turn_segment_with_telemetry(
                    publisher=publisher,
                    pose_provider=pose_provider,
                    command=command,
                    calibration=calibration,
                    command_scale=args.telemetry_command_scale,
                    yaw_tolerance_deg=args.telemetry_yaw_tolerance_deg,
                    pose_timeout_s=args.telemetry_pose_timeout_s,
                    poll_interval_s=args.telemetry_poll_s,
                    target_yaw=target_yaw,
                    republish_interval_s=(
                        args.continual_republish_s
                        if args.turn_command_mode == "continual" else 0.0
                    ),
                )
                if not ok:
                    raise RuntimeError("telemetry turn target was not reached")
            else:
                publisher.publish(command)
                duration = calibration.duration_s(command.distance)
                print(f"[{index}/{len(commands)}] wait {duration:.1f}s for physical completion")
                time.sleep(duration if args.arm else 0.0)
    except KeyboardInterrupt:
        interrupted = True
        if args.arm and publish_stop_best_effort(publisher):
            print("Interrupted: STOP published.")
        return 130
    except RuntimeError as exc:
        motion_failed = True
        print(f"Motion command failed: {exc}", file=sys.stderr)
        if args.arm and publish_stop_best_effort(publisher):
            print("STOP published after motion-command failure.")
    finally:
        close = getattr(publisher, "close", None)
        if callable(close):
            close()
        if not args.no_auto_scan:
            should_stop_scan = not motion_failed or args.stop_scan_on_error or interrupted
            if should_stop_scan:
                try:
                    scan_control(args.control_url, "stop")
                    print("Scan stop requested through mapper control.")
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
            else:
                print(
                    "Leaving scan running because rover publishing failed. Fix/restart "
                    "the rover HTTP controller, then either re-run this script or stop "
                    "the scan manually.",
                    file=sys.stderr,
                )

    if motion_failed:
        return 1

    try:
        manifest = wait_for_final_manifest(args.map_dir, scan_started_at, args.finalize_timeout_s)
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Map verification failed: {exc}", file=sys.stderr)
        return 2
    print(f"Map ready: revision={manifest.get('rev')} tiles={len(manifest.get('tiles', {}))} at {args.map_dir}")

    if not args.no_fill and not args.no_auto_scan:
        print("Closing the empty circle under the rover...")
        fill_requested_at = time.time()
        try:
            request_under_rover_fill(args.control_url)
            manifest = wait_for_final_manifest(
                args.map_dir, fill_requested_at, args.finalize_timeout_s)
        except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
            print(f"Under-rover fill failed: {exc}", file=sys.stderr)
            return 2
        print(f"Dead zone closed: revision={manifest.get('rev')} "
              f"tiles={len(manifest.get('tiles', {}))}. The rover now sits on the "
              f"filled floor; Track mode re-registers its Z on the next pose.")

    if args.track_after and not args.no_auto_scan:
        try:
            pause_map_integration(args.control_url)
            scan_control(args.control_url, "start")
        except RuntimeError as exc:
            # The 360 map itself is done and good -- report the tracking failure
            # but do not fail the scan run over it.
            print(
                f"Could not enter tracking mode: {exc}\n"
                "The map is finished, but the twin will not follow the rover until "
                "the scan is resumed (POST /scan/start on the mapper control port).",
                file=sys.stderr,
            )
            return 0
        print("Tracking mode: scan resumed with map integration paused -- the twin "
              "now follows the rover live without rewriting the 360 map.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
