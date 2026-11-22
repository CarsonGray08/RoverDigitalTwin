import json
import math

import pytest
from support import TEST_CONFIG, saved_map
from test_drive import FakeRover

from rover_twin.client import TwinClient
from rover_twin.ops.drive import Driver
from rover_twin.ops.mission import MissionRunner

FLOOR = (-1.0, 3.0, -2.0, 2.0, 0.0)
BOX = (1.0, 1.4, -0.4, 0.4, 0.25)
JOBS = [{"id": "A", "x": 2.4, "y": 0.0, "material_kg": 4, "task_min": 4},     # behind the box
        {"id": "B", "x": 0.0, "y": 1.2, "material_kg": 3, "task_min": 3},
        {"id": "C", "x": 2.2, "y": -1.2, "material_kg": 5, "task_min": 5}]
REACH = math.hypot(TEST_CONFIG.rover.footprint_half_length_m, TEST_CONFIG.rover.footprint_half_width_m)


@pytest.fixture
def url(tmp_path):
    from rover_twin.twin import offline
    path = saved_map(tmp_path / "map.npz", FLOOR, BOX)
    _, server = offline.start(path, TEST_CONFIG, "127.0.0.1", 0)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class Operator:
    """Answers the runner's prompts from a script (default: Enter), and keeps them."""

    def __init__(self, rover, *answers, on=None):
        self.rover, self.answers, self.prompts, self.on = rover, list(answers), [], on or {}

    def __call__(self, prompt):
        self.prompts.append(prompt)
        for words, seconds in self.on.items():      # time the operator takes, on the rover's clock
            if words in prompt:
                self.rover.t += seconds
        return self.answers.pop(0) if self.answers else ""


def runner(twin, rover, operator, tmp_path, jobs=JOBS, **mission):
    path = []
    driver = Driver(twin, rover, tmp_path / "drive.jsonl", pose=lambda: path.append(rover.pose) or rover.pose,
                    clock=lambda: rover.t, sleep=rover.sleep)
    run = MissionRunner(twin, driver, {"jobs": jobs, "sol_limit_min": None, **mission}, ask=operator,
                        say=lambda line: None, clock=lambda: rover.t, log=tmp_path / "mission.jsonl")
    return run, path


def log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "mission.jsonl").read_text().splitlines()]


def distance_to_box(x, y):
    x0, x1, y0, y1, _ = BOX
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def test_a_mission_is_flown_job_by_job_with_the_operator(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover()
    operator = Operator(rover, "y")
    run, path = runner(twin, rover, operator, tmp_path)
    outcome = run.run()
    assert outcome.completed, outcome.reason
    assert sorted(outcome.account.done) == ["A", "B", "C"] and outcome.account.delivered_kg == 12
    assert outcome.account.load_kg == 0 and outcome.replans == 0
    assert math.hypot(rover.pose[0], rover.pose[1]) <= twin.rover["goal_tolerance"][0]      # home
    digs = [p for p in operator.prompts if "excavate" in p]
    assert [p.split()[2].rstrip(":") for p in digs] == outcome.account.done
    assert any("unload 12 kg" in p or "unload" in p for p in operator.prompts)
    assert min(distance_to_box(x, y) for x, y, _ in path) > REACH - 0.05                     # around the box
    kinds = [e["kind"] for e in log(tmp_path)]
    assert kinds[0] == "plan" and kinds.count("dig") == 3 and "unload" in kinds
    assert outcome.account.battery_wh < run.rover["battery_capacity_wh"]                     # driving and digging cost


def test_nothing_moves_until_the_operator_confirms(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover()
    run, _ = runner(twin, rover, Operator(rover, "n"), tmp_path)
    outcome = run.run()
    assert not outcome.completed and outcome.reason == "not started" and rover.sent == []


def test_a_skipped_job_is_left_out_and_the_rest_replanned(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover()
    run, _ = runner(twin, rover, Operator(rover, "y", "skip"), tmp_path)
    outcome = run.run()
    assert outcome.completed, outcome.reason
    assert len(outcome.account.skipped) == 1 and outcome.replans == 1
    assert sorted(outcome.account.done + outcome.account.skipped) == ["A", "B", "C"]
    assert outcome.account.delivered_kg == sum(j["material_kg"] for j in JOBS if j["id"] in outcome.account.done)


class RefusesOnce(TwinClient):
    """A twin whose map changes under the first drive: it refuses that route once."""

    refused = False

    def plan_route(self, start, goal, speed=None, rehearse=True):
        if not self.refused and math.dist(start[:2], (0.0, 0.0)) > 0.3:
            self.refused = True
            return {"approved": False, "reason": "Planned path crosses terrain the map has not observed.",
                    "failure_type": "unknown_not_ready", "plans": []}
        return super().plan_route(start, goal, speed, rehearse)


def test_a_route_the_twin_refuses_on_the_way_is_replanned_at_once(url, tmp_path):
    twin, rover = RefusesOnce(url), FakeRover()
    operator = Operator(rover, "y")
    run, _ = runner(twin, rover, operator, tmp_path)
    outcome = run.run()
    assert twin.refused and outcome.completed, outcome.reason
    assert outcome.replans == 1 and not any("Replan from here" in p for p in operator.prompts)
    replan = next(e for e in log(tmp_path) if e["kind"] == "replan")
    assert "the twin refused a route" in replan["why"]


def test_a_rover_fault_stops_the_rover_and_the_operator_decides(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover(fault_after_s=20.0)
    operator = Operator(rover, "y", "n")
    run, _ = runner(twin, rover, operator, tmp_path)
    outcome = run.run()
    assert not outcome.completed and "fault" in outcome.reason
    assert rover.sent[-1] == "STOP" and any("Replan from here" in p for p in operator.prompts)


def test_a_mission_running_behind_its_plan_is_replanned(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover()
    operator = Operator(rover, "y", on={"excavate": 30 * 60.0})     # every dig takes half an hour
    run, _ = runner(twin, rover, operator, tmp_path)
    outcome = run.run()
    assert outcome.completed, outcome.reason
    assert outcome.replans >= 1
    assert any("behind the plan" in e["why"] for e in log(tmp_path) if e["kind"] == "replan")
    assert outcome.account.elapsed_min > 90


def test_a_time_limit_ends_the_mission_with_the_cargo_home(url, tmp_path):
    twin, rover = TwinClient(url), FakeRover()
    operator = Operator(rover, "y", on={"excavate": 20 * 60.0})
    run, _ = runner(twin, rover, operator, tmp_path, sol_limit_min=40)
    outcome = run.run()
    assert outcome.completed, outcome.reason
    assert 0 < len(outcome.account.done) < 3 and outcome.account.load_kg == 0
    assert math.hypot(rover.pose[0], rover.pose[1]) <= twin.rover["goal_tolerance"][0]
