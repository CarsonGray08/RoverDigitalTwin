import copy
import json
import math
import random
from pathlib import Path

import pytest
from support import TEST_CONFIG, saved_map

from rover_twin.client import TwinClient
from rover_twin.mission import planner
from rover_twin.mission.scheduler import ga, sa
from rover_twin.mission.scheduler.scheduler import schedule
from rover_twin.mission.scheduler.scheduler_core import distance, parse_mission
from rover_twin.mission.scheduler.simulator import REASON_NO_ROUTE, MissionSimulator
from rover_twin.twin import offline

STANDARD = json.loads((Path(__file__).parent / "data" / "scheduler_mission.json").read_text())
FLOOR = (-1.0, 3.0, -2.0, 2.0, 0.0)
BOX = (1.0, 1.4, -0.4, 0.4, 0.25)
JOBS = [{"id": "A", "x": 2.4, "y": 0.0, "material_kg": 4, "task_min": 4},     # behind the box
        {"id": "B", "x": 0.0, "y": 1.2, "material_kg": 3, "task_min": 3},
        {"id": "C", "x": 2.2, "y": -1.2, "material_kg": 5, "task_min": 5},
        {"id": "D", "x": 1.2, "y": 0.0, "material_kg": 2, "task_min": 2}]    # on the box


def straight_table(mission):
    """A travel table that is exactly the straight lines the scheduler assumes."""
    rover, base, tasks, _ = parse_mission(mission)
    places = {"BASE": base, **{t.id: t for t in tasks}}
    return {a: {b: {"distance_m": distance(p, q), "time_min": rover.travel_time(distance(p, q))}
                for b, q in places.items() if b != a} for a, p in places.items()}


def test_the_vendored_scheduler_gives_the_students_published_results():
    base = schedule(STANDARD)
    assert base["objective_kg"] == 24.0 and base["total_time_min"] == pytest.approx(58.61, abs=0.005)
    assert base["route"] == ["BASE", "B", "D", "E", "BASE", "C", "BASE", "F", "BASE"]
    unlimited = schedule({**STANDARD, "sol_limit_min": None})
    assert unlimited["objective_kg"] == 47.0 and unlimited["total_time_min"] == pytest.approx(168.22, abs=0.005)
    assert sa.optimise(STANDARD, sa.SAConfig(seed=0))["objective_kg"] == 35.0
    assert ga.optimise(STANDARD, ga.GAConfig(seed=0))["objective_kg"] == 35.0


def test_a_travel_table_of_straight_lines_schedules_exactly_as_without_one():
    routed = {**STANDARD, "travel": straight_table(STANDARD)}
    for run in (schedule, lambda m: sa.optimise(m, sa.SAConfig(seed=3))):
        plain, table = run(STANDARD), run(routed)
        assert table["route"] == plain["route"] and table["objective_kg"] == plain["objective_kg"]
        assert table["total_time_min"] == pytest.approx(plain["total_time_min"], abs=1e-9)
    rng = random.Random(0)
    plain_sim, table_sim = MissionSimulator(STANDARD), MissionSimulator(routed)
    for _ in range(200):
        order = [t["id"] for t in STANDARD["tasks"]]
        rng.shuffle(order)
        a, b = plain_sim.evaluate(order), table_sim.evaluate(order)
        assert (a["route"], a["objective_kg"]) == (b["route"], b["objective_kg"])
        assert a["energy_used_wh"] == pytest.approx(b["energy_used_wh"], abs=1e-9)


def test_a_job_with_no_route_is_skipped_with_that_reason():
    routed = {**STANDARD, "travel": straight_table(STANDARD)}
    for a in routed["travel"]:
        if "B" in routed["travel"][a]:
            routed["travel"][a]["B"] = None
    result = schedule(routed)
    assert "B" not in result["completed_tasks"]
    assert {"id": "B", "reasons": [REASON_NO_ROUTE]} in result["unserviceable_tasks"]


def test_a_mission_can_start_mid_way_with_cargo_and_time_spent():
    mission = copy.deepcopy(STANDARD)
    mission["travel"] = straight_table(mission)
    for row in mission["travel"].values():
        row["START"] = None
    mission["travel"]["START"] = {b: {"distance_m": 5.0, "time_min": 1.0} for b in mission["travel"]}
    mission["start"] = {"x_m": 3.0, "y_m": 4.0, "elapsed_min": 100.0, "load_kg": 6.0, "battery_wh": 40.0}
    result = schedule(mission)
    first = result["events"][0]
    assert (first["location"], first["elapsed_min"], first["load_kg"]) == ("START", 100.0, 6.0)
    assert result["route"][0] == "START" and result["route"][-1] == "BASE"
    assert result["objective_kg"] >= 6.0                      # what it carried is delivered too
    assert result["total_time_min"] <= STANDARD["sol_limit_min"]


@pytest.fixture
def twin(tmp_path):
    path = saved_map(tmp_path / "map.npz", FLOOR, BOX)
    _, server = offline.start(path, TEST_CONFIG, "127.0.0.1", 0)
    yield TwinClient(f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()


def test_a_mission_is_scheduled_on_the_twins_routes_and_every_leg_rehearsed(twin):
    mission = twin.plan_mission((0.0, 0.0, 0.0), JOBS, sol_limit_min=25)
    assert mission["approved"], mission["reason"]
    assert mission["route"][0] == "BASE" and mission["route"][-1] == "BASE"
    assert mission["total_time_min"] <= 25
    legs = mission["legs"]
    assert [(leg["from"], leg["to"]) for leg in legs] == list(zip(mission["route"], mission["route"][1:]))
    assert all(leg["approved"] and leg["rehearsed"] for leg in legs)
    places = {"BASE": (0.0, 0.0), **{j["id"]: (j["x"], j["y"]) for j in JOBS}}
    for leg in legs:                                          # each leg ends at its stop, within tolerance
        end = leg["rehearsals"][-1]["final_pose"]
        assert math.dist((end["x"], end["y"]), places[leg["to"]]) <= twin.rover["goal_tolerance"][0]
    status = {j["id"]: j for j in mission["jobs"]}
    assert status["D"]["status"] == "skipped_infeasible" and "obstacle" in status["D"]["unreachable"]
    assert mission["objective_kg"] >= mission["baseline_objective_kg"]


def test_a_mission_replanned_from_where_the_rover_is(twin):
    mission = twin.plan_mission((2.4, 0.0, 0.0), JOBS[1:3], base={"x": 0.0, "y": 0.0}, sol_limit_min=25,
                                state={"elapsed_min": 10.0, "load_kg": 4.0, "battery_wh": 150.0})
    assert mission["approved"], mission["reason"]
    assert mission["route"][0] == "START" and mission["events"][0]["load_kg"] == 4.0
    assert mission["objective_kg"] >= 4.0


def test_a_leg_the_twin_refuses_is_scheduled_around(twin):
    request = planner.MissionRequest.from_dict({"start_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
                                                "jobs": JOBS[:3], "sol_limit_min": None, "rehearse": False})
    places = {"BASE": (0.0, 0.0), **{j["id"]: (j["x"], j["y"]) for j in JOBS}}

    def name(x, y):
        return next(k for k, p in places.items() if math.dist(p, (x, y)) < 1e-3)

    def refusing(banned):
        def route(r):
            if (name(r.start.x, r.start.y), name(r.goal_x, r.goal_y)) == banned:
                return {"approved": False, "reason": "Predicted collision", "failure_type": "collision",
                        "plans": [], "rehearsals": []}
            return twin.plan_route((r.start.x, r.start.y, r.start.yaw_deg), (r.goal_x, r.goal_y), rehearse=False)
        return route

    first = planner.plan_mission(request, twin.route_costs, refusing(None), 0.15)
    banned = tuple(first["route"][:2])
    planned = planner.plan_mission(request, twin.route_costs, refusing(banned), 0.15)
    assert planned["approved"], planned["reason"]
    assert [(leg["from"], leg["to"]) for leg in planned["refused_legs"]] == [banned]
    assert banned not in list(zip(planned["route"], planned["route"][1:]))


def test_missions_the_twin_cannot_start_or_the_scheduler_rejects_are_refused(twin):
    boxed = twin.plan_mission((0.8, 0.0, 0.0), JOBS)
    assert not boxed["approved"] and boxed["failure_type"] == "base_blocked"
    away = twin.plan_mission((0.8, 0.0, 0.0), JOBS, base={"x": 0.0, "y": 0.0})
    assert away["failure_type"] == "start_blocked"
    with pytest.raises(Exception) as bad:
        twin.plan_mission((0.0, 0.0, 0.0), JOBS, rover={"capacity_kg": -1})
    assert bad.value.code == 400 and "capacity_kg" in json.loads(bad.value.read())["reason"]
    with pytest.raises(Exception) as unknown:
        twin.plan_mission((0.0, 0.0, 0.0), JOBS, algorithm="greedy")
    assert unknown.value.code == 400


def test_a_mission_request_uses_the_calibration_unless_told_otherwise():
    request = planner.MissionRequest.from_dict({"start_pose": {"x": 0, "y": 0, "yaw_deg": 0}, "jobs": JOBS[:1]})
    costs = {"length_m": [[0.0, 2.0], [2.0, 0.0]], "turn_deg": [[0.0, 0.0], [0.0, 0.0]], "blocked": [None, None]}
    out = planner.plan_mission(request, lambda points: costs,
                               lambda r: {"approved": True, "reason": "", "plans": [], "rehearsals": []}, 0.15)
    rover = out["assumptions"]["rover"]
    assert rover["speed_m_per_min"] == 2.1 and rover["battery_wh"] == rover["battery_capacity_wh"]
    assert out["sol_limit_min"] == 120.0
    leg = out["legs"][0]["time_min"]                    # 2 m at 2.1 m/min, and setting off: 90 deg at 40 s
    assert leg == pytest.approx(2.0 / 2.1 + 40.0 / 60.0)
