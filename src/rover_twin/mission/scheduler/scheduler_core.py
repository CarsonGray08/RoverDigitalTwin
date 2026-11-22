"""Shared data models and input validation for the rover mission.

Python 3.10+, standard library only. Used by simulator.py (rules) and
scheduler.py (baseline policy). No scheduling logic lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-9           # tolerance for feasibility comparisons (floating point)
TIE_DECIMALS = 9     # distances equal to this many decimals count as a tie

BASE_ID = "BASE"
START_ID = "START"   # [twin] a mid-mission start away from base (see parse_start)


# ----------------------------------------------------------------------------
# Input models
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Rover:
    battery_capacity_wh: float   # full usable battery energy
    battery_wh: float            # energy at mission start (may be below full)
    reserve_wh: float            # minimum energy on every return to base
    capacity_kg: float           # cargo limit
    speed_m_per_min: float       # constant driving speed
    travel_power_w: float        # constant driving power
    unload_min: float            # time per nonempty unloading
    charging_power_w: float      # constant effective charging power

    @property
    def travel_wh_per_m(self) -> float:
        """Energy per metre: power (W) x time (min) / 60, with time = 1/speed."""
        return self.travel_power_w / (60.0 * self.speed_m_per_min)

    def travel_energy(self, distance_m: float) -> float:
        return distance_m * self.travel_wh_per_m

    def travel_time(self, distance_m: float) -> float:
        return distance_m / self.speed_m_per_min


@dataclass(frozen=True)
class Task:
    id: str
    x_m: float
    y_m: float
    material_kg: float
    task_min: float
    excavation_power_w: float
    job_type: str = ""           # descriptive only; never used for scheduling

    @property
    def energy_wh(self) -> float:
        """Excavation energy = power (W) x duration (min) / 60."""
        return self.excavation_power_w * self.task_min / 60.0


@dataclass(frozen=True)
class Point:
    x_m: float
    y_m: float


def distance(a, b) -> float:
    """Straight-line distance between two objects that have x_m and y_m."""
    return math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

class MissionError(ValueError):
    """Raised for a mission file that cannot be scheduled meaningfully."""


def _require(mapping, keys, where):
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise MissionError(f"{where} is missing required field(s): {', '.join(missing)}")


def _number(value, name, *, minimum=0.0, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MissionError(f"{name} must be a finite number")
    if value < minimum or (positive and value <= 0):
        raise MissionError(f"{name} is outside its allowed range (got {value})")
    return float(value)


def parse_mission(mission: dict) -> tuple[Rover, Point, list[Task], float | None]:
    """Turn the mission dictionary into typed objects, rejecting bad input early.

    Returns (rover, base, tasks, sol_limit_min). ``sol_limit_min`` is the fixed
    mission duration (a scaled lunar sol); ``None`` or a missing key means no
    time limit, which reproduces the Phase 1 behaviour exactly.
    """
    if not isinstance(mission, dict):
        raise MissionError("mission must be a JSON object")
    _require(mission, ("rover", "base", "tasks"), "mission")

    r = mission["rover"]
    _require(r, ("battery_capacity_wh", "battery_wh", "reserve_wh", "capacity_kg",
                 "speed_m_per_min", "travel_power_w", "unload_min", "charging_power_w"), "rover")
    default_power = r.get("default_excavation_power_w")
    if default_power is not None:
        default_power = _number(default_power, "default_excavation_power_w")
    rover = Rover(
        battery_capacity_wh=_number(r["battery_capacity_wh"], "battery_capacity_wh", positive=True),
        battery_wh=_number(r["battery_wh"], "battery_wh"),
        reserve_wh=_number(r["reserve_wh"], "reserve_wh"),
        capacity_kg=_number(r["capacity_kg"], "capacity_kg", positive=True),
        speed_m_per_min=_number(r["speed_m_per_min"], "speed_m_per_min", positive=True),
        travel_power_w=_number(r["travel_power_w"], "travel_power_w"),
        unload_min=_number(r["unload_min"], "unload_min"),
        charging_power_w=_number(r["charging_power_w"], "charging_power_w", positive=True),
    )
    if rover.battery_wh > rover.battery_capacity_wh + EPS:
        raise MissionError("Starting energy cannot exceed full battery capacity")
    if rover.reserve_wh > rover.battery_capacity_wh + EPS:
        raise MissionError("Reserve cannot exceed full battery capacity")
    # Note: a starting battery below the reserve is allowed. The rover is at
    # base, so the scheduler simply charges first if any job fits a full charge.

    b = mission["base"]
    _require(b, ("x_m", "y_m"), "base")
    base = Point(_number(b["x_m"], "base.x_m", minimum=-math.inf),
                 _number(b["y_m"], "base.y_m", minimum=-math.inf))

    if not isinstance(mission["tasks"], list):
        raise MissionError("tasks must be a list")
    tasks: list[Task] = []
    seen = {BASE_ID}
    for i, t in enumerate(mission["tasks"]):
        where = f"tasks[{i}]"
        _require(t, ("id", "x_m", "y_m", "material_kg", "task_min"), where)
        if "excavation_power_w" not in t:
            if default_power is None:
                raise MissionError(f"{where} has no excavation_power_w and rover has no "
                                   "default_excavation_power_w")
            t = {**t, "excavation_power_w": default_power}
        tid = t["id"]
        if not isinstance(tid, str) or not tid.strip() or tid in seen:
            raise MissionError(f"{where}: id must be a unique nonempty string other than {BASE_ID}")
        seen.add(tid)
        tasks.append(Task(
            id=tid,
            x_m=_number(t["x_m"], f"{where}.x_m", minimum=-math.inf),
            y_m=_number(t["y_m"], f"{where}.y_m", minimum=-math.inf),
            material_kg=_number(t["material_kg"], f"{where}.material_kg", positive=True),
            task_min=_number(t["task_min"], f"{where}.task_min", positive=True),
            excavation_power_w=_number(t["excavation_power_w"], f"{where}.excavation_power_w"),
            job_type=str(t.get("job_type", "")),
        ))
    sol = mission.get("sol_limit_min")
    if sol is not None:
        sol = _number(sol, "sol_limit_min", positive=True)
    return rover, base, tasks, sol




# ----------------------------------------------------------------------------
# [twin] Optional inputs added by the digital twin. Both are absent from every
# mission the scheduler's own tools write, and then nothing above changes.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One drive between two locations, as the twin routes it."""
    distance_m: float
    time_min: float


@dataclass(frozen=True)
class Start:
    """Where and in what state the mission starts, if not at base, full and empty."""
    location: str
    x_m: float
    y_m: float
    battery_wh: float
    elapsed_min: float
    load_kg: float
    delivered_kg: float


def parse_travel(mission: dict, locations: set[str]) -> dict[tuple[str, str], Leg | None] | None:
    """``mission["travel"]``: ``{from_id: {to_id: {"distance_m", "time_min"} | null}}``.

    Replaces straight-line distance and constant speed with the twin's routes.
    A missing or null leg means there is no route. None when absent.
    """
    travel = mission.get("travel")
    if travel is None:
        return None
    if not isinstance(travel, dict):
        raise MissionError("travel must be an object of objects")
    table: dict[tuple[str, str], Leg | None] = {}
    for a, row in travel.items():
        if a not in locations or not isinstance(row, dict):
            raise MissionError(f"travel: {a!r} is not a location of this mission, or its row is not an object")
        for b, leg in row.items():
            if b not in locations:
                raise MissionError(f"travel[{a}]: {b!r} is not a location of this mission")
            if leg is None:
                table[(a, b)] = None
                continue
            _require(leg, ("distance_m", "time_min"), f"travel[{a}][{b}]")
            table[(a, b)] = Leg(_number(leg["distance_m"], f"travel[{a}][{b}].distance_m"),
                                _number(leg["time_min"], f"travel[{a}][{b}].time_min"))
    return table


def parse_start(mission: dict, rover: Rover, base: Point) -> Start | None:
    """``mission["start"]``: where a replanned mission starts. ``location`` is
    START_ID (away from base, at ``x_m``, ``y_m``) or BASE_ID; ``battery_wh``,
    ``elapsed_min``, ``load_kg`` and ``delivered_kg`` default to a fresh start.
    Jobs already done are left out of ``tasks``. None when absent."""
    s = mission.get("start")
    if s is None:
        return None
    if not isinstance(s, dict):
        raise MissionError("start must be an object")
    location = s.get("location", START_ID)
    if location not in (START_ID, BASE_ID):
        raise MissionError(f"start.location must be {START_ID} or {BASE_ID}")
    if location == START_ID:
        _require(s, ("x_m", "y_m"), "start")
        x, y = _number(s["x_m"], "start.x_m", minimum=-math.inf), _number(s["y_m"], "start.y_m", minimum=-math.inf)
    else:
        x, y = base.x_m, base.y_m
    start = Start(location, x, y,
                  _number(s.get("battery_wh", rover.battery_wh), "start.battery_wh"),
                  _number(s.get("elapsed_min", 0.0), "start.elapsed_min"),
                  _number(s.get("load_kg", 0.0), "start.load_kg"),
                  _number(s.get("delivered_kg", 0.0), "start.delivered_kg"))
    if start.battery_wh > rover.battery_capacity_wh + EPS:
        raise MissionError("start.battery_wh cannot exceed full battery capacity")
    if start.load_kg > rover.capacity_kg + EPS:
        raise MissionError("start.load_kg cannot exceed cargo capacity")
    return start
