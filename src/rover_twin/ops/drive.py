"""Fly a plan on the real rover, one command at a time, rehearsing first.

    python -m rover_twin.ops.drive plan.json                      # rehearse from the live pose
    python -m rover_twin.ops.drive plan.json --start 0 0 0        # rehearse from a pose (offline twin)
    python -m rover_twin.ops.drive plan.json --arm --rover-ip 192.168.0.157

``plan.json`` is ``{"commands": [...], "goal": {...}, "rehearse": true}`` in
the twin's schema (docs/twin-api.md); without a goal it is where the rest of the commands end
on flat ground from the live pose at each rehearsal, so the drift of one
command (a turn in place wanders) does not refuse the next; give a goal when
where the rover ends up matters. Before every command, the rest of the plan is
rehearsed from the rover's live pose on the latest map, and a refusal stops
the drive there: that is the moment for the software that wrote the plan to
replan. A command is sent as a bounded step count (ops/rover.py) and stopped
when the live pose says it is done; the rover is stopped on finish, on any
failure (including a fault the rover reports, or a live pose that is late or
lost), and on Ctrl-C. Without ``--arm`` nothing moves: the plan is only
rehearsed. A plan with ``"rehearse": false`` is flown without asking the twin
(the dashboard's 360 scan, turns in place the twin cannot judge well); the
live pose still stops every command. While the mapper is connected but has
not sent a live pose yet (a scan just starting), the drive waits for one.

Each command flown is logged to ``--log`` (JSON lines): start and end pose,
how long it really took, and how long the twin predicted.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from rover_twin.client import TwinClient
from rover_twin.frames import wrap_deg
from rover_twin.ops.rover import Rover, steps

POLL_S = 0.1
POSE_MAX_AGE_S = 1.0                  # an older live pose is not driven on
STOP_SHORT_M, STOP_SHORT_DEG = 0.01, 1.0     # the rover coasts a little after STOP
STALL_S = 3.0                         # no progress for this long while moving: stuck
TIMEOUT_S = 120.0                     # one command, however slow the rover
SETTLE_S = 0.5                        # after STOP, before the next live pose
POSE_WAIT_S = 15.0                    # for the first live pose of a scan just starting


@dataclass(frozen=True)
class Outcome:
    completed: bool
    reason: str
    commands_done: int


class Driver:
    def __init__(self, twin: TwinClient, rover: Rover, log: Path | None = None, pose=None,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        self.twin, self.rover, self.log = twin, rover, log
        self.pose = pose or (lambda: live_pose(twin))
        self.clock, self.sleep = clock, sleep

    def run(self, commands: list[dict], goal=None, rehearse: bool = True) -> Outcome:
        try:
            start = self.pose()
            for i, command in enumerate(commands):
                predicted_s = None
                if rehearse:
                    check = self.twin.rehearse(start, commands[i:], goal)
                    if not check["approved"]:
                        return Outcome(False, f"command {i}: the twin refused the rest of the plan: "
                                              f"{check.get('failure_type')}: {check['reason']}", i)
                    predicted_s = check["step_results"][0]["duration_s"]
                status, end, seconds = self._fly(command, start)
                self._record(i, command, start, end, status, seconds, predicted_s)
                if status != "completed":
                    return Outcome(False, f"command {i}: {status}", i)
                self.sleep(SETTLE_S)
                start = self.pose()
            return Outcome(True, "plan flown", len(commands))
        finally:
            self.rover.stop()

    def _fly(self, command: dict, start) -> tuple[str, tuple, float]:
        """Send one command and stop it when the live pose has done it:
        (status, end pose, seconds it took)."""
        kind, t0 = command["command"], self.clock()
        if kind == "STOP":
            return "completed", start, 0.0
        linear = kind in ("FWD", "REV")
        size = command["distance"] / self.twin.rover["distance_units_per_m"] if linear else command["distance"]
        target, moved = (size - STOP_SHORT_M, 0.005) if linear else (size - STOP_SHORT_DEG, 0.5)
        progress, best, best_t, pose = 0.0, 0.0, t0, start
        self.rover.send(kind, steps(kind, size))
        try:
            while progress < target:
                self.sleep(POLL_S)
                try:
                    last, pose = pose, self.pose()
                except (OSError, RuntimeError) as exc:      # the twin is unreachable, or its pose stale
                    return f"lost the live pose ({exc})", pose, self.clock() - t0
                progress = _along(kind, start, pose) if linear else progress + _turned(kind, last, pose)
                now = self.clock()
                if progress > best + moved:
                    best, best_t = progress, now
                if (fault := self.rover.fault()) is not None:
                    return f"the rover reported a fault ({fault})", pose, now - t0
                if now - best_t > STALL_S:
                    return f"no progress for {STALL_S:.0f} s ({progress:.3f} of {target:.3f})", pose, now - t0
                if now - t0 > TIMEOUT_S:
                    return f"timed out after {TIMEOUT_S:.0f} s", pose, now - t0
        finally:
            self.rover.stop()
        return "completed", pose, self.clock() - t0

    def _record(self, index, command, start, end, status, seconds, predicted_s) -> None:
        entry = {"t": time.time(), "index": index, "command": command, "status": status,
                 "start": list(start), "end": list(end), "seconds": round(seconds, 3),
                 "twin_predicted_s": predicted_s}
        twin = "not rehearsed" if predicted_s is None else f"twin {predicted_s:.1f} s"
        print(f"  {index}: {command['command']} {command['distance']} -> {status} in {entry['seconds']:.1f} s "
              f"({twin}), at ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.1f})")
        if self.log is not None:
            with open(self.log, "a") as fh:
                fh.write(json.dumps(entry) + "\n")


def live_pose(twin: TwinClient) -> tuple[float, float, float]:
    """The real rover's pose from the twin, refusing a stale one."""
    live = twin.observation()["live_pose"]
    if live is None or live["age_s"] > POSE_MAX_AGE_S:
        raise RuntimeError(f"no fresh live pose from the twin ({live}): is the phone mapping?")
    return live["x"], live["y"], live["yaw_deg"]


def wait_for_pose(twin: TwinClient, timeout_s: float = POSE_WAIT_S, poll_s: float = 0.5) -> tuple[float, float, float]:
    """The live pose; while the mapper is connected but has not anchored and
    sent one yet, waiting up to ``timeout_s``. Without a mapper, at once."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return live_pose(twin)
        except RuntimeError:
            if time.monotonic() >= deadline or not twin.observation()["mapper_connected"]:
                raise
        time.sleep(poll_s)


def _along(kind: str, start, now) -> float:
    """How far a drive has gone along its start heading, in its direction."""
    yaw = math.radians(start[2])
    along = (now[0] - start[0]) * math.cos(yaw) + (now[1] - start[1]) * math.sin(yaw)
    return along if kind == "FWD" else -along


def _turned(kind: str, last, now) -> float:
    """Yaw gained in a turn's direction between two poses."""
    turned = wrap_deg(now[2] - last[2])
    return turned if kind == "LEFT" else -turned


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--twin", default="http://127.0.0.1:8765")
    parser.add_argument("--rover-ip", default="")
    parser.add_argument("--rover-name", default="", help="the rover's ROVER_NAME, if it sets one")
    parser.add_argument("--level", type=int, default=1, choices=(1, 2, 3),
                        help="wheel speed level: 60, 100 or 140 motor RPM")
    parser.add_argument("--arm", action="store_true", help="move the real rover")
    parser.add_argument("--log", type=Path, default=Path("drive_log.jsonl"))
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                        help="rehearse from this pose instead of the live one (not with --arm)")
    args = parser.parse_args(argv)
    if args.arm and args.start:
        parser.error("--start is for rehearsing: an armed drive starts where the rover is")
    plan = json.loads(args.plan.read_text())
    goal = plan.get("goal")
    goal = (goal["x"], goal["y"], goal["yaw_deg"]) if goal else None
    rehearse = plan.get("rehearse", True)
    twin = TwinClient(args.twin)
    try:
        start = tuple(args.start) if args.start else wait_for_pose(twin)
    except RuntimeError as exc:
        raise SystemExit(f"{exc} (rehearsing on the offline twin? pass --start X Y YAW_DEG)")
    if not args.arm and not rehearse:
        print("not armed and not rehearsed: nothing was sent to the rover")
        return
    if not args.arm:
        result = twin.rehearse(start, plan["commands"], goal)
        print(json.dumps({k: result.get(k) for k in ("approved", "reason", "failure_type", "duration_s",
                                                       "final_pose")}, indent=1))
        print("not armed: nothing was sent to the rover" if result["approved"] else
              f"not armed, and the twin refused it: {result.get('failure_type')}: {result['reason']}")
        return
    if not args.rover_ip:
        parser.error("--arm needs --rover-ip")
    rover = Rover(args.rover_ip, args.level, name=args.rover_name)
    outcome = Driver(twin, rover, args.log).run(plan["commands"], goal, rehearse=rehearse)
    print(("done: " if outcome.completed else "stopped: ") + outcome.reason)


if __name__ == "__main__":
    main()
