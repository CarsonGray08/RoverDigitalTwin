import json
import math

import pytest
from support import TEST_CONFIG, saved_map

from rover_twin.client import TwinClient
from rover_twin.ops.drive import Driver
from rover_twin.twin import offline


class FakeRover:
    """The real rover as the drive loop sees it: commands in, a live pose that
    moves at the real rover's pace, and a controller that can report a fault."""

    def __init__(self, fault_after_s=math.inf):
        self.t, self.pose = 0.0, (0.0, 0.0, 0.0)
        self.twist, self.sent, self.fault_after_s = (0.0, 0.0), [], fault_after_s

    def send(self, command, step_count=0):
        self.sent.append(command)
        self.twist = {"FWD": (1, 0), "REV": (-1, 0), "LEFT": (0, 1), "RIGHT": (0, -1), "STOP": (0, 0)}[command]

    def stop(self):
        self.send("STOP")

    def fault(self):
        return "Timeout7" if self.t > self.fault_after_s and self.twist != (0, 0) else None

    def sleep(self, s):        # time passes: the rover moves 0.035 m/s or turns 3 deg/s
        x, y, yaw = self.pose
        if self.fault() is None:
            v, w = 0.035 * self.twist[0], 3.0 * self.twist[1]
            x, y = x + v * s * math.cos(math.radians(yaw)), y + v * s * math.sin(math.radians(yaw))
            yaw += w * s
        self.pose, self.t = (x, y, yaw), self.t + s


@pytest.fixture
def twin(tmp_path):
    path = saved_map(tmp_path / "map.npz", (-1.0, 3.0, -1.5, 1.5, 0.0), (1.5, 1.8, -0.15, 0.15, 0.25))
    _, server = offline.start(path, TEST_CONFIG, "127.0.0.1", 0)
    yield TwinClient(f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()


def drive(twin, rover, tmp_path, plan):
    driver = Driver(twin, rover, tmp_path / "log.jsonl", pose=lambda: rover.pose,
                    clock=lambda: rover.t, sleep=rover.sleep)
    return driver.run(plan)


def test_a_plan_is_flown_command_by_command_on_the_live_pose(twin, tmp_path):
    rover = FakeRover()
    outcome = drive(twin, rover, tmp_path, [twin.forward(0.5), twin.left(90), twin.forward(0.3)])
    assert outcome.completed, outcome.reason
    x, y, yaw = rover.pose
    assert (x, y) == pytest.approx((0.5, 0.3), abs=0.02)
    assert yaw == pytest.approx(90.0, abs=1.5)      # stopped short for the coast this fake does not have
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [e["status"] for e in log] == ["completed"] * 3
    assert log[0]["seconds"] == pytest.approx(0.49 / 0.035, abs=0.2)       # the rover's pace, not the twin's
    assert rover.sent[-1] == "STOP"


def test_without_a_goal_a_turn_that_drifts_does_not_refuse_the_next(twin, tmp_path):
    class Wandering(FakeRover):          # a turn in place that walks the rover sideways, as the real one does
        def sleep(self, s):
            super().sleep(s)
            if self.twist[1]:
                x, y, yaw = self.pose
                self.pose = (x - 0.006 * s, y, yaw)

    rover = Wandering()
    outcome = drive(twin, rover, tmp_path, [twin.left(90)] * 4)
    assert outcome.completed, outcome.reason
    assert rover.pose[0] < -0.5           # well past the goal tolerance from where it started


def test_unrehearsed_turns_fly_where_the_twin_would_refuse(twin, tmp_path):
    rover = FakeRover()
    rover.pose = (-0.9, 0.0, 0.0)               # at the edge of the map: half the footprint unseen
    assert not drive(twin, rover, tmp_path, [twin.left(90)]).completed
    driver = Driver(twin, rover, tmp_path / "log.jsonl", pose=lambda: rover.pose, clock=lambda: rover.t,
                    sleep=rover.sleep)
    outcome = driver.run([twin.left(90)] * 4, rehearse=False)
    assert outcome.completed, outcome.reason
    assert rover.pose[2] == pytest.approx(360.0, abs=6.0) and rover.sent[-1] == "STOP"
    assert json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])["twin_predicted_s"] is None


def test_a_drive_stops_where_the_twin_or_the_rover_says_so(twin, tmp_path):
    rover = FakeRover()
    refused = drive(twin, rover, tmp_path, [twin.forward(1.3)])          # into the box ahead
    assert not refused.completed and "collision" in refused.reason and rover.sent == ["STOP"]

    rover = FakeRover(fault_after_s=2.0)
    stopped = drive(twin, rover, tmp_path, [twin.forward(0.5)])
    assert not stopped.completed and "fault" in stopped.reason and rover.sent[-1] == "STOP"

    rover, reads = FakeRover(), iter(range(20))

    def pose():                                  # the twin answers, then stops answering
        if next(reads) >= 10:
            raise OSError("timed out")
        return rover.pose

    lost = Driver(twin, rover, tmp_path / "log.jsonl", pose=pose, clock=lambda: rover.t,
                  sleep=rover.sleep).run([twin.forward(0.5)])
    assert not lost.completed and "lost the live pose" in lost.reason and rover.sent[-1] == "STOP"
