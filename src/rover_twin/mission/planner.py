"""Missions on the twin: the students' scheduler orders the jobs, the twin
routes and rehearses every leg.

    job sites --/route_costs--> travel table --scheduler--> job order
              --/plan_route per leg--> rehearsed plans

The scheduler is the students' code, vendored in ``mission/scheduler/`` (see
VENDORED.md there). It sees the twin only through the mission dict: a
``travel`` table of each leg's metres and minutes, and a ``start`` when the
mission is replanned from where the rover is. A leg's minutes are its route
length at the rover's speed plus its turning at ``turn_s_per_90deg``: the
corners on the route, and ``END_TURN_DEG`` to set off along it, an average,
since that turn depends on the leg before. A leg the twin refuses to
rehearse is taken out of the table and the jobs are scheduled again.
docs/twin-api.md is the request and answer format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from rover_twin.frames import Pose2D
from rover_twin.mission.scheduler import ga, sa
from rover_twin.mission.scheduler.calibration import load_calibration, rover_values
from rover_twin.mission.scheduler.scheduler import schedule
from rover_twin.mission.scheduler.scheduler_core import BASE_ID, START_ID, MissionError, parse_mission
from rover_twin.twin.route import RouteRequest, _integer, _number

CALIBRATION = Path(__file__).parent / "scheduler" / "rover_calibration_preliminary_v2.json"
ALGORITHMS = ("simulated_annealing", "genetic_algorithm", "baseline")
END_TURN_DEG = 90.0           # turning to set off along a leg, on average
TURN_S_PER_90DEG = 40.0       # ASSUMED until drive logs measure it (30-45 s seen at level 1)
MAX_JOBS = 30                 # base and start take the other two of /route_costs's 32 points
MAX_SCHEDULES = 4             # the first, and one more for each leg the twin refuses
STATE_KEYS = ("battery_wh", "elapsed_min", "load_kg", "delivered_kg")
_UNSET = object()


@dataclass(frozen=True)
class MissionRequest:
    start: Pose2D
    jobs: tuple[dict, ...]
    base: tuple[float, float] | None = None
    rover: dict = field(default_factory=dict)
    sol_limit_min: object = _UNSET            # a number, None for no limit, or the calibration's
    state: dict = field(default_factory=dict)
    turn_s_per_90deg: float = TURN_S_PER_90DEG
    algorithm: str = "simulated_annealing"
    seed: int = 0
    rehearse: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "MissionRequest":
        """docs/twin-api.md's schema; raises ValueError (KeyError, TypeError) on anything else."""
        s = data["start_pose"]
        jobs = data["jobs"]
        if not isinstance(jobs, list) or not 1 <= len(jobs) <= MAX_JOBS:
            raise ValueError(f"jobs must be a list of 1 to {MAX_JOBS} jobs")
        clean = []
        for i, j in enumerate(jobs):
            if not isinstance(j.get("id"), str) or j["id"] in (BASE_ID, START_ID):
                raise ValueError(f"jobs[{i}].id must be a string other than {BASE_ID} or {START_ID}")
            job = {"id": j["id"], "x": _number(j["x"]), "y": _number(j["y"]),
                   "material_kg": _number(j["material_kg"]), "task_min": _number(j["task_min"])}
            for key in ("excavation_power_w", "yaw_deg"):
                if j.get(key) is not None:
                    job[key] = _number(j[key])
            clean.append(job)
        base = data.get("base")
        rover = data.get("rover") or {}
        state = data.get("state") or {}
        if not isinstance(rover, dict) or not isinstance(state, dict) or set(state) - set(STATE_KEYS):
            raise ValueError(f"rover must be an object; state an object of {', '.join(STATE_KEYS)}")
        algorithm = data.get("algorithm", "simulated_annealing")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
        rehearse = data.get("rehearse", True)
        if not isinstance(rehearse, bool):
            raise ValueError("rehearse must be true or false")
        sol = data["sol_limit_min"] if "sol_limit_min" in data else _UNSET
        return cls(Pose2D(_number(s["x"]), _number(s["y"]), _number(s["yaw_deg"])), tuple(clean),
                   None if base is None else (_number(base["x"]), _number(base["y"])),
                   {k: _number(v) for k, v in rover.items()},
                   sol if sol is _UNSET or sol is None else _number(sol),
                   {k: _number(v) for k, v in state.items()},
                   _number(data.get("turn_s_per_90deg", TURN_S_PER_90DEG)), algorithm,
                   _integer(data.get("seed", 0)), rehearse)


def plan_mission(request: MissionRequest, costs, route, at_base_m: float) -> dict:
    """Schedule the jobs on the twin's routes and plan every leg.

    ``costs(points)`` answers as ``/route_costs`` and ``route(RouteRequest)``
    as ``/plan_route``. The rover is at base when the start pose is within
    ``at_base_m`` of it. Raises MissionError for a mission the scheduler rejects.
    """
    start = request.start
    base = request.base or (start.x, start.y)
    at_base = math.hypot(start.x - base[0], start.y - base[1]) <= at_base_m
    where = {BASE_ID: base, **({} if at_base else {START_ID: (start.x, start.y)}),
             **{j["id"]: (j["x"], j["y"]) for j in request.jobs}}
    ids = list(where)
    answer = costs([where[i] for i in ids])
    blocked = dict(zip(ids, answer["blocked"]))
    for place, failure in ((BASE_ID, "base_blocked"), (START_ID, "start_blocked")):
        if blocked.get(place):
            return _refused(failure, f"{place.capitalize()} {blocked[place]}")
    if not at_base and answer["length_m"][ids.index(START_ID)][ids.index(BASE_ID)] is None:
        return _refused("no_route_home", "No route from the start back to base")

    mission = _mission(request, base, at_base, ids, answer)
    parse_mission(mission)                             # a bad rover or job fails here, with the field
    speed = mission["rover"]["speed_m_per_min"]
    refused: list[dict] = []
    for _ in range(MAX_SCHEDULES):
        try:
            result = _run(request, mission)
        except MissionError as exc:          # a refused leg left the rover no way home
            return {**_refused("no_route_home", str(exc)), "refused_legs": refused}
        legs, failed = _legs(request, result["route"], where, route, mission["travel"])
        if failed is None:
            break
        refused.append(failed)
        mission["travel"][failed["from"]][failed["to"]] = None
    approved = failed is None
    reason = (f"{len(result['completed_tasks'])} of {len(request.jobs)} jobs, "
              f"{result['objective_kg']:.2f} kg in {result['total_time_min']:.1f} min; "
              + ("every leg rehearsed" if request.rehearse else "legs planned, not rehearsed")
              if approved else f"leg {failed['from']} to {failed['to']}: {failed['reason']}")
    out = {"approved": approved, "reason": reason, "algorithm": result["algorithm"], "seed": request.seed,
           "order": result["completed_tasks"], "route": result["route"], "legs": legs, "refused_legs": refused,
           "jobs": [{**j, "unreachable": blocked.get(j["id"])} for j in result["job_status"]]}
    for key in ("objective_kg", "total_time_min", "total_distance_m", "battery_remaining_wh", "energy_used_wh",
                "sol_limit_min", "sol_time_remaining_min", "baseline_objective_kg", "improvement_kg", "events"):
        if key in result:
            out[key] = result[key]
    out["assumptions"] = {"rover": mission["rover"], "rover_source": "calibration preliminary v2 (16 Nov), "
                          "not physically validated, plus the request's rover overrides",
                          "speed_m_per_min": speed, "turn_s_per_90deg": request.turn_s_per_90deg,
                          "end_turn_deg": END_TURN_DEG}
    if not approved:
        out.update(failure_type="leg_refused", failed_leg=failed)
    return out


def _mission(request: MissionRequest, base, at_base: bool, ids: list[str], answer: dict) -> dict:
    """The scheduler's mission dict, with the twin's travel table and start."""
    calibration = load_calibration(CALIBRATION)
    rover = {**rover_values(calibration), **request.rover}
    if "battery_wh" not in request.rover:
        rover["battery_wh"] = rover["battery_capacity_wh"]
    sol = request.sol_limit_min
    if sol is _UNSET:
        sol = float(calibration["parameters"]["sol_limit_min"]["value"])
    speed = rover["speed_m_per_min"]
    if not isinstance(speed, (int, float)) or speed <= 0:
        raise MissionError("rover.speed_m_per_min must be positive")
    per_deg_min = request.turn_s_per_90deg / 90.0 / 60.0
    travel = {}
    for a, from_id in enumerate(ids):
        travel[from_id] = {}
        for b, to_id in enumerate(ids):
            length, turn = answer["length_m"][a][b], answer["turn_deg"][a][b]
            if a == b:
                continue
            travel[from_id][to_id] = None if length is None else {
                "distance_m": length, "time_min": length / speed + (turn + END_TURN_DEG) * per_deg_min}
    mission = {"rover": rover, "base": {"x_m": base[0], "y_m": base[1]}, "sol_limit_min": sol, "travel": travel,
               "tasks": [{"id": j["id"], "x_m": j["x"], "y_m": j["y"], "material_kg": j["material_kg"],
                          "task_min": j["task_min"],
                          **({"excavation_power_w": j["excavation_power_w"]} if "excavation_power_w" in j else {})}
                         for j in request.jobs]}
    if not at_base or request.state:
        mission["start"] = {"location": BASE_ID if at_base else START_ID, **request.state,
                            **({} if at_base else {"x_m": request.start.x, "y_m": request.start.y})}
    return mission


def _run(request: MissionRequest, mission: dict) -> dict:
    if request.algorithm == "baseline":
        return schedule(mission)
    if request.algorithm == "genetic_algorithm":
        return ga.optimise(mission, ga.GAConfig(seed=request.seed))
    return sa.optimise(mission, sa.SAConfig(seed=request.seed))


def _legs(request: MissionRequest, stops: list[str], where: dict, route, travel: dict):
    """Plan (and rehearse) each leg of the schedule in order, from where the
    last one ended: (legs, None), or (legs so far, the refused leg)."""
    yaw = {j["id"]: j.get("yaw_deg") for j in request.jobs}
    pose, legs = request.start, []
    for a, b in zip(stops, stops[1:]):
        goal = where[b] if yaw.get(b) is None else (*where[b], yaw[b])
        x, y, *heading = goal
        answer = route(RouteRequest(pose, x, y, heading[0] if heading else None, None, request.rehearse))
        leg = travel[a][b]
        legs.append({"from": a, "to": b, "time_min": leg["time_min"] if leg else None, **answer})
        if not answer["approved"]:
            return legs, {"from": a, "to": b, "failure_type": answer.get("failure_type"), "reason": answer["reason"]}
        if answer["rehearsals"]:
            end = answer["rehearsals"][-1]["final_pose"]
        elif answer["plans"]:
            end = answer["plans"][-1]["goal"]
        else:
            continue
        pose = Pose2D(end["x"], end["y"], end["yaw_deg"])
    return legs, None


def _refused(failure: str, reason: str) -> dict:
    return {"approved": False, "reason": reason, "failure_type": failure, "order": [], "route": [], "legs": [],
            "refused_legs": [], "jobs": []}
