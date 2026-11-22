"""Common mission simulator / schedule evaluator.

This module owns the *physics and rules* of a mission. Any scheduling method
(the nearest-feasible-job baseline today; GA and SA later) calls into it and
never re-implements energy, time, cargo or sol-limit rules.

Two public entry points:

* ``MissionSimulator(mission)`` - parse once, then call ``evaluate(sequence)``
  as many times as you like (cheap; no file I/O, no mutation of the mission).
* ``evaluate_sequence(mission, sequence)`` - one-shot convenience wrapper.

The evaluator takes a proposed order of job IDs and *decodes* it into a
physically valid schedule:

    for each job in the proposed order:
        if it fits now                         -> drive there and do it
        elif it fits after returning + unload  -> return, unload, do it
        elif it fits after a full recharge     -> return, unload, charge, do it
        else                                   -> skip it (recorded with reasons)
    finally                                    -> return to base and unload

A job "fits" when cargo capacity, the battery floor (reserve) on the whole
trip including the direct return, and the sol time limit (job + direct
return + unload) are all satisfied. Returns, unloads and charges are inserted
automatically, so a sequence never needs explicit BASE tokens.

Primary objective: maximise ``objective_kg`` = regolith delivered to base
within the sol. Secondary measures (time, energy, jobs) are reported for
tie-breaking or multi-objective work later.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Iterable

from .scheduler_core import (BASE_ID, EPS, START_ID, MissionError, Point, Rover, Task, distance, parse_mission,
                             parse_start, parse_travel)

REASON_CAPACITY = "carrying capacity"
REASON_BATTERY = "battery including return reserve"
REASON_SOL = "sol time limit including return and unload"
REASON_EXCLUDED = "not in proposed sequence"
REASON_NO_ROUTE = "no route on the twin's map"   # [twin] only with a travel table

# job_status values: every job in the mission gets exactly one of these.
STATUS_DELIVERED = "delivered"                 # completed and regolith unloaded at base
STATUS_SKIPPED = "skipped_infeasible"          # in the sequence, but blocked by a hard constraint
STATUS_EXCLUDED = "excluded_from_sequence"     # candidate sequence deliberately left it out


@dataclass
class MissionState:
    """Everything that changes while the mission runs. Cheap to copy."""
    position: Point
    location: str
    battery_wh: float
    elapsed_min: float
    load_kg: float = 0.0
    delivered_kg: float = 0.0
    distance_m: float = 0.0
    charged_wh: float = 0.0
    charging_stops: int = 0
    charging_time_min: float = 0.0
    unload_count: int = 0
    completed: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def copy(self) -> "MissionState":
        return replace(self, completed=list(self.completed), events=list(self.events))


class MissionSimulator:
    """Holds the parsed mission and applies the rules to a MissionState."""

    def __init__(self, mission: dict):
        self.rover, self.base, self.tasks, self.sol_limit_min = parse_mission(mission)
        self.by_id: dict[str, Task] = {t.id: t for t in self.tasks}
        self._start_battery = self.rover.battery_wh
        # [twin] optional: the twin's routes in place of straight lines, and a mid-mission start
        self.travel = parse_travel(mission, {BASE_ID, START_ID, *self.by_id})
        self.start = parse_start(mission, self.rover, self.base)
        if self.start is not None:
            self._start_battery = self.start.battery_wh

    # ------------------------------------------------------------------ state
    def initial_state(self) -> MissionState:
        if self.start is not None:      # [twin]
            st = self.start
            s = MissionState(position=Point(st.x_m, st.y_m), location=st.location, battery_wh=st.battery_wh,
                             elapsed_min=st.elapsed_min, load_kg=st.load_kg, delivered_kg=st.delivered_kg)
            self._record(s, "start")
            return s
        s = MissionState(position=self.base, location=BASE_ID,
                         battery_wh=self.rover.battery_wh, elapsed_min=0.0)
        self._record(s, "start")
        return s

    def _record(self, s: MissionState, action: str, **details) -> None:
        s.events.append({
            "action": action,
            "location": s.location,
            "elapsed_min": s.elapsed_min,
            "battery_wh": s.battery_wh,
            "battery_pct": 100.0 * s.battery_wh / self.rover.battery_capacity_wh,
            "load_kg": s.load_kg,
            "total_delivered_kg": s.delivered_kg,
            **details,
        })

    # -------------------------------------------------------------- rule check
    def why_not_feasible(self, task: Task, s: MissionState, *,
                         from_base_full: bool = False) -> list[str]:
        """Empty list = the job can be done from state ``s`` right now.

        ``from_base_full`` instead asks: "if the rover went home now, unloaded
        and charged to full, could it do the job?" The hypothetical is produced
        by *running those actions on a copy of the state*, so it can never
        disagree with what the real actions would do (charge time, in
        particular, depends on the battery after the return drive). This is the
        test for "worth going home / charging for" and for "unserviceable".
        """
        if from_base_full:
            s = self.after_return_and_charge(s)
        if self.travel is not None:     # [twin]
            return self._why_not_feasible_on_routes(task, s)
        r = self.rover
        out_m = distance(s.position, task)
        home_m = distance(task, self.base)
        reasons = []
        if s.load_kg + task.material_kg > r.capacity_kg + EPS:
            reasons.append(REASON_CAPACITY)
        need_wh = r.travel_energy(out_m + home_m) + task.energy_wh + r.reserve_wh
        if s.battery_wh + EPS < need_wh:
            reasons.append(REASON_BATTERY)
        if self.sol_limit_min is not None:
            finish = (s.elapsed_min + r.travel_time(out_m) + task.task_min
                      + r.travel_time(home_m) + r.unload_min)
            if finish > self.sol_limit_min + EPS:
                reasons.append(REASON_SOL)
        return reasons

    def _why_not_feasible_on_routes(self, task: Task, s: MissionState) -> list[str]:
        """[twin] The same rules, with each leg's distance and time from the travel table."""
        r = self.rover
        out, home = self.travel.get((s.location, task.id)), self.travel.get((task.id, BASE_ID))
        if out is None or home is None:
            return [REASON_NO_ROUTE]
        reasons = []
        if s.load_kg + task.material_kg > r.capacity_kg + EPS:
            reasons.append(REASON_CAPACITY)
        need_wh = r.travel_power_w * (out.time_min + home.time_min) / 60.0 + task.energy_wh + r.reserve_wh
        if s.battery_wh + EPS < need_wh:
            reasons.append(REASON_BATTERY)
        if self.sol_limit_min is not None:
            finish = s.elapsed_min + out.time_min + task.task_min + home.time_min + r.unload_min
            if finish > self.sol_limit_min + EPS:
                reasons.append(REASON_SOL)
        return reasons

    def distance_to(self, s: MissionState, task: Task) -> float:
        """Distance from the current position to a job: straight-line, or the twin's route."""
        if self.travel is None:
            return distance(s.position, task)
        leg = self.travel.get((s.location, task.id))
        return math.inf if leg is None else leg.distance_m

    def feasible_now(self, task: Task, s: MissionState) -> bool:
        return not self.why_not_feasible(task, s)

    def after_return_and_unload(self, s: MissionState) -> MissionState:
        """Copy of ``s`` after driving home (if away) and unloading."""
        c = s.copy()
        self.return_and_unload(c)
        return c

    def after_return_and_charge(self, s: MissionState) -> MissionState:
        """Copy of ``s`` after driving home, unloading and charging to full."""
        c = self.after_return_and_unload(s)
        if c.battery_wh < self.rover.battery_capacity_wh - EPS:
            self.charge(c)
        return c

    # --------------------------------------------------------------- actions
    def travel_to(self, s: MissionState, target, name: str) -> None:
        if self.travel is not None:     # [twin]
            leg = self.travel.get((s.location, name))
            if leg is None:
                raise MissionError(f"no route from {s.location} to {name} in the travel table")
            s.battery_wh -= self.rover.travel_power_w * leg.time_min / 60.0
            s.elapsed_min += leg.time_min
            s.distance_m += leg.distance_m
            s.position, s.location = Point(target.x_m, target.y_m), name
            self._record(s, "travel", distance_m=leg.distance_m)
            return
        d = distance(s.position, target)
        s.battery_wh -= self.rover.travel_energy(d)
        s.elapsed_min += self.rover.travel_time(d)
        s.distance_m += d
        s.position, s.location = Point(target.x_m, target.y_m), name
        self._record(s, "travel", distance_m=d)

    def excavate(self, s: MissionState, task: Task) -> None:
        s.battery_wh -= task.energy_wh
        s.elapsed_min += task.task_min
        s.load_kg += task.material_kg
        s.completed.append(task.id)
        self._record(s, "excavate", task_id=task.id, collected_kg=task.material_kg,
                     task_energy_wh=task.energy_wh)

    def unload(self, s: MissionState) -> None:
        if s.load_kg <= EPS:
            return
        amount = s.load_kg
        s.elapsed_min += self.rover.unload_min
        s.delivered_kg += amount
        s.load_kg = 0.0
        s.unload_count += 1
        self._record(s, "unload", delivered_kg=amount)

    def charge(self, s: MissionState) -> None:
        r = self.rover
        before = s.battery_wh
        added = r.battery_capacity_wh - s.battery_wh
        duration = added / r.charging_power_w * 60.0
        started = s.elapsed_min
        s.elapsed_min += duration
        s.battery_wh = r.battery_capacity_wh
        s.charged_wh += added
        s.charging_time_min += duration
        s.charging_stops += 1
        self._record(s, "charge", battery_before_wh=before, added_wh=added,
                     charge_start_min=started, duration_min=duration)

    def do_job(self, s: MissionState, task: Task) -> None:
        """Travel to the job and complete it. Caller must have checked feasibility."""
        self.travel_to(s, task, task.id)
        self.excavate(s, task)

    def return_and_unload(self, s: MissionState) -> None:
        if s.location != BASE_ID:
            self.travel_to(s, self.base, BASE_ID)
        self.unload(s)

    # -------------------------------------------------------------- finish
    def finish(self, s: MissionState) -> None:
        self.return_and_unload(s)
        self._record(s, "finish")

    def result(self, s: MissionState, *, skipped: list[dict], algorithm: str,
               sequence: list[str] | None = None, runtime_s: float | None = None) -> dict:
        r = self.rover
        all_done = len(s.completed) == len(self.tasks)
        skipped_by_id = {d["id"]: d["reasons"] for d in skipped}
        job_status = []
        for t in self.tasks:
            if t.id in s.completed:
                job_status.append({"id": t.id, "status": STATUS_DELIVERED, "reasons": [],
                                   "material_kg": t.material_kg})
            elif skipped_by_id.get(t.id) == [REASON_EXCLUDED]:
                job_status.append({"id": t.id, "status": STATUS_EXCLUDED, "reasons": [],
                                   "material_kg": t.material_kg})
            else:
                job_status.append({"id": t.id, "status": STATUS_SKIPPED,
                                   "reasons": skipped_by_id.get(t.id, []),
                                   "material_kg": t.material_kg})
        counts = {k: sum(1 for j in job_status if j["status"] == k)
                  for k in (STATUS_DELIVERED, STATUS_SKIPPED, STATUS_EXCLUDED)}
        return {
            "algorithm": algorithm,
            "proposed_sequence": sequence,
            "runtime_s": runtime_s,
            "job_status": job_status,
            "jobs_delivered": counts[STATUS_DELIVERED],
            "jobs_skipped_infeasible": counts[STATUS_SKIPPED],
            "jobs_excluded_from_sequence": counts[STATUS_EXCLUDED],
            "route": [s.events[0]["location"]] + [e["location"] for e in s.events if e["action"] == "travel"],
            "completed_tasks": list(s.completed),
            "unserviceable_tasks": skipped,          # with reasons
            "objective_kg": s.delivered_kg,           # primary objective (maximise)
            "feasible": True,                         # evaluator never breaks a rule
            "all_jobs_completed": all_done,
            "sol_limit_min": self.sol_limit_min,
            "sol_time_remaining_min": (None if self.sol_limit_min is None
                                       else self.sol_limit_min - s.elapsed_min),
            "material_delivered_kg": s.delivered_kg,
            "total_time_min": s.elapsed_min,
            "total_distance_m": s.distance_m,
            "energy_used_wh": self._start_battery + s.charged_wh - s.battery_wh,
            "charged_energy_wh": s.charged_wh,
            "charging_stops": s.charging_stops,
            "charging_time_min": s.charging_time_min,
            "unload_count": s.unload_count,
            "battery_remaining_wh": s.battery_wh,
            "battery_remaining_pct": 100.0 * s.battery_wh / r.battery_capacity_wh,
            "events": list(s.events),
        }

    # ------------------------------------------------------------ evaluator
    def evaluate(self, sequence: Iterable[str]) -> dict:
        """Decode a proposed job order into a valid schedule and score it."""
        t_start = time.perf_counter()
        seq = list(sequence)
        unknown = [j for j in seq if j not in self.by_id]
        if unknown:
            raise MissionError(f"sequence contains unknown job id(s): {', '.join(map(str, unknown))}")
        if len(set(seq)) != len(seq):
            raise MissionError("sequence contains a repeated job id")

        s = self.initial_state()
        skipped: list[dict] = []
        for job_id in seq:
            task = self.by_id[job_id]
            # Stage 1: fits from where we are?
            if self.feasible_now(task, s):
                self.do_job(s, task)
                continue
            # Stage 2: fits after going home and unloading (same charge)?
            home = self.after_return_and_unload(s)
            if self.feasible_now(task, home):
                s = home
                self.do_job(s, task)
                continue
            # Stage 3: fits after a full recharge at base?
            charged = self.after_return_and_charge(s)
            if self.feasible_now(task, charged):
                s = charged
                self.do_job(s, task)
                continue
            # Otherwise it is unserviceable from here: skip without moving.
            skipped.append({"id": job_id, "reasons": self.why_not_feasible(task, charged)})

        # Jobs not in the sequence at all are reported too, so the result is
        # a complete account of the mission.
        for t in self.tasks:
            if t.id not in seq:
                skipped.append({"id": t.id, "reasons": [REASON_EXCLUDED]})
        self.finish(s)
        return self.result(s, skipped=skipped, algorithm="sequence_evaluation", sequence=seq,
                           runtime_s=time.perf_counter() - t_start)


def evaluate_sequence(mission: dict, sequence: Iterable[str]) -> dict:
    """One-shot: parse the mission, evaluate one sequence, return the result."""
    return MissionSimulator(mission).evaluate(sequence)
