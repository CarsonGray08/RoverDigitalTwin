"""Fly a scheduled mission on the real rover, replanning as it goes.

    python -m rover_twin.ops.mission mission.json --start 0 0 0          # plan and rehearse (offline twin)
    python -m rover_twin.ops.mission mission.json                        # plan and rehearse from the live pose
    python -m rover_twin.ops.mission mission.json --arm --rover-ip 192.168.0.157

``mission.json`` is a ``POST /plan_mission`` body without ``start_pose``:
``{"jobs": [...], "sol_limit_min": 25, ...}`` (docs/twin-api.md). Without
``--arm`` the mission is only planned and rehearsed, and ``--out`` saves the
plan. Armed, the twin plans the mission from the rover's live pose, the
operator confirms it, and the rover flies the schedule step by step:

- drive: to the next stop on a route the twin plans and rehearses
  (``Driver.go_to``, replanning the route from the live pose as it goes);
- dig: the rover waits at the job while the operator runs the drums, then
  answers Enter (done) or ``skip``;
- unload and charge at base: the operator does them and answers Enter.

The mission keeps its own account: elapsed time from the clock, cargo and
delivered kg from the operator's answers, and battery estimated from the
scheduler's rover model (there is no battery telemetry). The rest of the
mission is planned again from that state and the live pose when:

- the twin refuses a route (the map changed): at once;
- a drive fails for any other reason (a fault, a lost pose, a stall): the
  rover stops, and the operator decides whether to replan or end;
- the operator skips a job;
- the mission runs behind its plan by more than ``DRIFT_MIN`` minutes and
  ``DRIFT_FRACTION`` of the plan's time so far.

Base stays where the mission started (or the file's ``base``). Every step,
plan and replan is logged to ``--log`` (JSON lines); every command to
``--drive-log``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from rover_twin.client import TwinClient
from rover_twin.ops.drive import Driver, wait_for_pose
from rover_twin.ops.rover import Rover

DRIFT_MIN = 2.0               # behind plan by more than this many minutes ...
DRIFT_FRACTION = 0.2          # ... and this fraction of the plan's time so far: replan
MAX_REPLANS = 10
REFUSED_ROUTE = "the twin refused a route"     # how Driver.go_to says the map changed


@dataclass
class Account:
    """What the mission has done so far, as the scheduler's ``state`` needs it."""

    battery_wh: float
    elapsed_min: float = 0.0
    load_kg: float = 0.0
    delivered_kg: float = 0.0
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def state(self) -> dict:
        return {"battery_wh": max(0.0, self.battery_wh), "elapsed_min": self.elapsed_min,
                "load_kg": self.load_kg, "delivered_kg": self.delivered_kg}


@dataclass(frozen=True)
class MissionOutcome:
    completed: bool
    reason: str
    account: Account
    replans: int


class MissionRunner:
    def __init__(self, twin: TwinClient, driver: Driver, mission: dict, ask=input, say=print,
                 clock=time.monotonic, log: Path | None = None) -> None:
        if not mission.get("jobs"):
            raise ValueError("a mission needs jobs")
        self.twin, self.driver, self.mission = twin, driver, dict(mission)
        self.ask, self.say, self.clock, self.log = ask, say, clock, log
        self.jobs = {j["id"]: j for j in mission["jobs"]}

    def run(self) -> MissionOutcome:
        pose = self.driver.pose()
        self.mission.setdefault("base", {"x": pose[0], "y": pose[1]})
        plan = self._plan(pose, None)
        self.rover = plan.get("assumptions", {}).get("rover", {})
        account = Account(battery_wh=self.rover.get("battery_wh", 0.0))
        if not plan["approved"]:
            return MissionOutcome(False, f"the twin refused the mission: {plan['reason']}", account, 0)
        self.say(summary(plan))
        if not _yes(self.ask("Start the mission? [y/N] ")):
            return MissionOutcome(False, "not started", account, 0)
        self.t0, self.elapsed0 = self.clock(), 0.0
        replans = 0
        while True:
            outcome, why = self._fly(plan, account)
            if outcome == "done":
                return MissionOutcome(True, f"mission done: {account.delivered_kg:g} kg delivered from "
                                            f"{', '.join(account.done) or 'no jobs'} in {account.elapsed_min:.1f} min",
                                      account, replans)
            if outcome == "stop" or replans >= MAX_REPLANS:
                why = why if outcome == "stop" else f"{why}; gave up after {MAX_REPLANS} replans"
                return MissionOutcome(False, why, account, replans)
            replans += 1
            self.say(f"replanning: {why}")
            try:
                plan = self._replan(account)
            except _Stop as stop:
                return MissionOutcome(False, str(stop), account, replans)
            if plan is None:
                return MissionOutcome(True, f"mission done: {account.delivered_kg:g} kg delivered from "
                                            f"{', '.join(account.done) or 'no jobs'} in {account.elapsed_min:.1f} min",
                                      account, replans)
            self._record("replan", why=why, plan=_brief(plan))
            if not plan["approved"]:
                return MissionOutcome(False, f"the twin refused the replanned mission: {plan['reason']}",
                                      account, replans)
            self.say(summary(plan))

    def _plan(self, pose, account: Account | None) -> dict:
        body = {k: v for k, v in self.mission.items() if k != "jobs"}
        left = [j for j in self.mission["jobs"] if account is None
                or j["id"] not in account.done + account.skipped]
        if account is not None:
            body["state"] = account.state()
        plan = self.twin.plan_mission(pose, left, **body)
        self._record("plan", pose=list(pose), plan=_brief(plan))
        return plan

    def _replan(self, account: Account) -> dict | None:
        """The rest of the mission from the live pose; None once nothing is left
        to do (any cargo is brought home first)."""
        pose = self.driver.pose()
        if any(j not in account.done + account.skipped for j in self.jobs):
            return self._plan(pose, account)
        base = self.mission["base"]
        if math.hypot(pose[0] - base["x"], pose[1] - base["y"]) > self.twin.rover["goal_tolerance"][0]:
            failed = self._drive("BASE", (base["x"], base["y"]), account)
            if failed is not None:
                raise _Stop(f"every job is done or skipped, but the drive back to base failed: {failed}")
        self._unload(account)
        return None

    def _fly(self, plan: dict, account: Account) -> tuple[str, str]:
        """Fly the plan's steps: ("done", ""), ("replan", why) or ("stop", why)."""
        base = self.mission["base"]
        planned0 = plan["events"][0]["elapsed_min"]
        for event in plan["events"][1:]:
            action = event["action"]
            if action == "travel":
                stop = event["location"]
                job = self.jobs.get(stop)
                goal = (base["x"], base["y"]) if job is None else \
                    (job["x"], job["y"], *([job["yaw_deg"]] if job.get("yaw_deg") is not None else []))
                failed = self._drive(stop, goal, account)
                if failed is not None:
                    if failed.startswith(REFUSED_ROUTE):
                        return "replan", failed
                    if _yes(self.ask(f"The rover stopped: {failed}. Replan from here and carry on? [y/N] ")):
                        return "replan", failed
                    return "stop", failed
            elif action == "excavate":
                if not self._dig(event["task_id"], event, account):
                    return "replan", f"job {event['task_id']} skipped"
            elif action == "unload":
                self._unload(account)
            elif action == "charge":
                self.ask("At base: charge the rover to full, then press Enter. ")
                account.battery_wh = self.rover.get("battery_capacity_wh", account.battery_wh)
                self._tick(account)
                self._record("charge", account=account.state())
            elif action == "finish":
                return "done", ""
            behind = account.elapsed_min - event["elapsed_min"]
            if behind > DRIFT_MIN and behind > DRIFT_FRACTION * (event["elapsed_min"] - planned0):
                return "replan", f"{behind:.1f} min behind the plan"
        return "done", ""

    def _drive(self, stop: str, goal, account: Account) -> str | None:
        """Drive to a stop; None when there, else why not."""
        self.say(f"driving to {stop}")
        t = self.clock()
        outcome = self.driver.go_to(goal)
        minutes = (self.clock() - t) / 60.0
        account.battery_wh -= self.rover.get("travel_power_w", 0.0) * minutes / 60.0
        self._tick(account)
        self._record("drive", stop=stop, goal=list(goal), completed=outcome.completed, reason=outcome.reason,
                     minutes=round(minutes, 3), pose=list(self.driver.pose()), account=account.state())
        return None if outcome.completed else outcome.reason

    def _dig(self, job_id: str, event: dict, account: Account) -> bool:
        job = self.jobs[job_id]
        answer = self.ask(f"At job {job_id}: excavate {job['material_kg']:g} kg (about {job['task_min']:g} min). "
                          "Press Enter when done, or type skip. ")
        self._tick(account)
        if answer.strip().lower() == "skip":
            account.skipped.append(job_id)
            self._record("skip", job=job_id, account=account.state())
            return False
        account.load_kg += event["collected_kg"]
        account.battery_wh -= event["task_energy_wh"]
        account.done.append(job_id)
        self._record("dig", job=job_id, account=account.state())
        return True

    def _unload(self, account: Account) -> None:
        if account.load_kg <= 0:
            return
        self.ask(f"At base: unload {account.load_kg:g} kg, then press Enter. ")
        account.delivered_kg += account.load_kg
        account.load_kg = 0.0
        self._tick(account)
        self._record("unload", account=account.state())

    def _tick(self, account: Account) -> None:
        account.elapsed_min = self.elapsed0 + (self.clock() - self.t0) / 60.0

    def _record(self, kind: str, **entry) -> None:
        if self.log is not None:
            with open(self.log, "a") as fh:
                fh.write(json.dumps({"t": time.time(), "kind": kind, **entry}) + "\n")


class _Stop(Exception):
    pass


def summary(plan: dict) -> str:
    """The plan for the operator: what it delivers, the stops and each leg."""
    if not plan.get("approved"):
        return f"no mission: {plan.get('failure_type')}: {plan['reason']}"
    lines = [f"mission: {plan['reason']}", f"  stops: {' -> '.join(plan['route'])}"]
    for leg in plan["legs"]:
        lines.append(f"  {leg['from']:>5} -> {leg['to']:<5} {leg['length_m']:5.2f} m, ~{leg['time_min']:.1f} min, "
                     f"{len(leg['plans'])} plan(s)")
    for job in plan["jobs"]:
        if job["status"] != "delivered":
            why = job.get("unreachable") or ", ".join(job["reasons"]) or job["status"]
            lines.append(f"  not done: {job['id']} ({why})")
    return "\n".join(lines)


def _brief(plan: dict) -> dict:
    return {k: plan.get(k) for k in ("approved", "reason", "failure_type", "route", "order", "objective_kg",
                                     "total_time_min", "refused_legs")}


def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mission", type=Path, help="a /plan_mission body without start_pose")
    parser.add_argument("--twin", default="http://127.0.0.1:8765")
    parser.add_argument("--rover-ip", default="")
    parser.add_argument("--rover-name", default="", help="the rover's ROVER_NAME, if it sets one")
    parser.add_argument("--level", type=int, default=1, choices=(1, 2, 3),
                        help="wheel speed level: 60, 100 or 140 motor RPM")
    parser.add_argument("--arm", action="store_true", help="move the real rover")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                        help="plan from this pose instead of the live one (not with --arm)")
    parser.add_argument("--out", type=Path, help="save the plan (not armed)")
    parser.add_argument("--log", type=Path, default=Path("mission_log.jsonl"))
    parser.add_argument("--drive-log", type=Path, default=Path("drive_log.jsonl"))
    args = parser.parse_args(argv)
    if args.arm and args.start:
        parser.error("--start is for planning: an armed mission starts where the rover is")
    if args.arm and not args.rover_ip:
        parser.error("--arm needs --rover-ip")
    mission = json.loads(args.mission.read_text())
    twin = TwinClient(args.twin)
    if not args.arm:
        try:
            start = tuple(args.start) if args.start else wait_for_pose(twin)
        except RuntimeError as exc:
            raise SystemExit(f"{exc} (planning on the offline twin? pass --start X Y YAW_DEG)")
        plan = twin.plan_mission(start, mission["jobs"], **{k: v for k, v in mission.items() if k != "jobs"})
        print(summary(plan))
        if args.out:
            args.out.write_text(json.dumps(plan, indent=1))
            print(f"plan saved to {args.out}")
        print("not armed: nothing was sent to the rover")
        return
    rover = Rover(args.rover_ip, args.level, name=args.rover_name)
    driver = Driver(twin, rover, args.drive_log)
    try:
        outcome = MissionRunner(twin, driver, mission, log=args.log).run()
    except KeyboardInterrupt:
        rover.stop()
        raise SystemExit("stopped by Ctrl-C")
    print(("done: " if outcome.completed else "stopped: ") + outcome.reason)


if __name__ == "__main__":
    main()
