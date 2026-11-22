import math
from dataclasses import replace

import numpy as np
import pytest
from support import TEST_CONFIG, saved_map
from test_drive import FakeRover

from rover_twin.client import TwinClient
from rover_twin.ops.drive import Driver
from rover_twin.twin import offline

FLOOR = (-1.0, 3.0, -2.0, 2.0, 0.0)
BOX = (1.0, 1.4, -0.4, 0.4, 0.25)            # across the straight line from the start to (2.4, 0)
REACH = math.hypot(TEST_CONFIG.rover.footprint_half_length_m, TEST_CONFIG.rover.footprint_half_width_m)


@pytest.fixture
def lab(tmp_path):
    """``lab(*surfaces, cfg=...)``: a client on an offline twin of those surfaces."""
    servers = []

    def start(*surfaces, cfg=TEST_CONFIG):
        path = saved_map(tmp_path / f"map{len(servers)}.npz", *surfaces)
        _, server = offline.start(path, cfg, "127.0.0.1", 0)
        servers.append(server)
        return TwinClient(f"http://127.0.0.1:{server.server_address[1]}")

    yield start
    for server in servers:
        server.shutdown()


def distance_to_box(x, y):
    x0, x1, y0, y1, _ = BOX
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def along(waypoints, step=0.02):
    for a, b in zip(waypoints, waypoints[1:]):
        n = max(1, math.ceil(math.dist((a["x"], a["y"]), (b["x"], b["y"])) / step))
        for t in np.linspace(0, 1, n + 1):
            yield a["x"] + t * (b["x"] - a["x"]), a["y"] + t * (b["y"] - a["y"])


def test_a_route_goes_around_a_box_and_is_rehearsed_to_the_goal(lab):
    twin = lab(FLOOR, BOX)
    route = twin.plan_route((0.0, 0.0, 0.0), (2.4, 0.0))
    assert route["approved"] and route["rehearsed"], route["reason"]
    assert len(route["waypoints"]) >= 3                         # at least one corner
    assert route["length_m"] > 2.4 + 0.5                        # not through the box
    # The rover could turn anywhere on the way without touching the box.
    assert min(distance_to_box(x, y) for x, y in along(route["waypoints"])) > REACH
    for plan in route["plans"]:
        assert len(plan["commands"]) <= twin.rover["max_commands"]
        assert all(c["distance"] <= twin.rover["max_distance_units"] for c in plan["commands"])
    assert [r["approved"] for r in route["rehearsals"]] == [True] * len(route["plans"])
    end = route["rehearsals"][-1]["final_pose"]
    assert math.hypot(end["x"] - 2.4, end["y"]) <= twin.rover["goal_tolerance"][0]


def test_a_route_arrives_facing_the_goal_heading_when_one_is_given(lab):
    twin = lab(FLOOR, BOX)
    route = twin.plan_route((0.0, 0.0, 0.0), (2.4, 0.0, 90.0))
    assert route["approved"], route["reason"]
    assert route["plans"][-1]["goal"]["yaw_deg"] == 90.0
    assert route["rehearsals"][-1]["final_pose"]["yaw_deg"] == pytest.approx(90.0, abs=8.0)


def test_a_long_straight_is_split_into_drives_the_twin_accepts(lab):
    twin = lab((-1.0, 4.5, -1.0, 1.0, 0.0))
    route = twin.plan_route((0.0, 0.0, 0.0), (3.5, 0.0))
    assert route["approved"], route["reason"]
    commands = [c for p in route["plans"] for c in p["commands"]]
    assert [c["command"] for c in commands] == ["FWD"] * 3       # 3.5 m in drives of at most 1.30 m
    assert sum(c["distance"] for c in commands) == round(3.5 * twin.rover["distance_units_per_m"])


def test_a_route_longer_than_one_plan_comes_as_several_each_rehearsed(lab):
    cfg = replace(TEST_CONFIG, limits=replace(TEST_CONFIG.limits, max_commands=2))
    twin = lab(FLOOR, BOX, cfg=cfg)
    route = twin.plan_route((0.0, 0.0, 0.0), (2.4, 0.0, 0.0))
    assert route["approved"], route["reason"]
    assert len(route["plans"]) >= 3 and all(len(p["commands"]) <= 2 for p in route["plans"])
    assert len(route["rehearsals"]) == len(route["plans"])


@pytest.mark.parametrize("goal, failure, words", [
    ((1.2, 0.0), "goal_blocked", "obstacle"),                  # in the box
    ((1.2, 0.7), "goal_blocked", "obstacle"),                  # beside it, too close to turn
    ((0.0, 1.9), "goal_blocked", "not observed"),              # at the edge of what was mapped
    ((8.0, 0.0), "goal_blocked", "mapped area"),
])
def test_goals_the_rover_cannot_stand_at_are_refused_with_the_reason(lab, goal, failure, words):
    route = lab(FLOOR, BOX).plan_route((0.0, 0.0, 0.0), goal)
    assert not route["approved"] and route["failure_type"] == failure and words in route["reason"]
    assert route["plans"] == [] and route["failed_plan_index"] is None


def test_no_route_through_a_wall_or_from_beside_one_or_outside_the_lab_bounds(lab):
    walled = lab(FLOOR, (1.0, 1.4, -2.0, 2.0, 0.25))
    assert walled.plan_route((0.0, 0.0, 0.0), (2.4, 0.0))["failure_type"] == "no_route"
    assert walled.plan_route((0.8, 0.0, 0.0), (-0.5, 0.0))["failure_type"] == "start_blocked"
    cfg = replace(TEST_CONFIG, limits=replace(TEST_CONFIG.limits, lab_bounds_m=(-1.0, 2.0, -2.0, 2.0)))
    bounded = lab(FLOOR, cfg=cfg).plan_route((0.0, 0.0, 0.0), (2.5, 0.0))
    assert bounded["failure_type"] == "goal_blocked" and "lab bounds" in bounded["reason"]


def test_route_costs_give_every_leg_between_points_for_a_scheduler(lab):
    twin = lab(FLOOR, BOX)
    costs = twin.route_costs([(0.0, 0.0), (2.4, 0.0), (0.0, 1.0), (1.2, 0.0)])
    length, turn = costs["length_m"], costs["turn_deg"]
    assert length[0][0] == 0.0 and length[0][2] == pytest.approx(1.0, abs=0.05)   # in sight: straight
    assert length[0][1] > 2.9 and turn[0][1] > 0                                    # around the box
    assert length[0][1] == pytest.approx(length[1][0], rel=0.05)
    assert costs["blocked"][3] and all(row[3] is None for row in length) and length[3] == [None] * 4


def test_the_rover_drives_to_a_goal_replanning_from_its_live_pose(lab, tmp_path):
    twin = lab(FLOOR, BOX)
    rover = FakeRover()
    path = []
    driver = Driver(twin, rover, tmp_path / "log.jsonl", pose=lambda: path.append(rover.pose) or rover.pose,
                    clock=lambda: rover.t, sleep=rover.sleep)
    outcome = driver.go_to((2.4, 0.0))
    assert outcome.completed, outcome.reason
    x, y, _ = rover.pose
    assert math.hypot(x - 2.4, y) <= twin.rover["goal_tolerance"][0]
    assert min(distance_to_box(px, py) for px, py, _ in path) > REACH - 0.05    # it went around


def test_a_rover_is_not_sent_toward_a_goal_the_twin_refuses(lab, tmp_path):
    twin = lab(FLOOR, (1.0, 1.4, -2.0, 2.0, 0.25))
    rover = FakeRover()
    driver = Driver(twin, rover, tmp_path / "log.jsonl", pose=lambda: rover.pose,
                    clock=lambda: rover.t, sleep=rover.sleep)
    outcome = driver.go_to((2.4, 0.0))
    assert not outcome.completed and "no_route" in outcome.reason
    assert outcome.commands_done == 0 and rover.sent in ([], ["STOP"])
