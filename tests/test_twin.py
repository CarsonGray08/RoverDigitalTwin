import json
import math
import queue
import threading
import time
import urllib.request
from dataclasses import replace

import numpy as np
import pytest
from support import TEST_CONFIG, map_scan

from rover_twin.config import Elevation
from rover_twin.frames import Pose2D, wrap_deg
from rover_twin.link.client import LinkClient, Update
from rover_twin.link.messages import Patch
from rover_twin.link.server import LinkServer
from rover_twin.mapping.elevation import ElevationGrid
from rover_twin.twin.main import Twin, serve
from rover_twin.twin.terrain import StageTerrain, TerrainMap
from rover_twin.twin.validation import Command, Request, Step, Validator

CFG = replace(TEST_CONFIG, twin=replace(TEST_CONFIG.twin, settle_s=0.1, port=0))
UNITS = CFG.rover.distance_units_per_m


class FakeRover:
    """Kinematic skid-steer: wheel speeds in, pose out; blocked past ``wall_x``."""

    steps_per_second = 60.0
    ride_height_m = 0.17

    def __init__(self, wall_x: float = math.inf):
        self.x = self.y = self.yaw = 0.0
        self.z, self.wheels, self.wall_x = 0.0, (0.0, 0.0), wall_x

    def step(self):
        radius, dt = CFG.rover.wheel_diameter_m / 2, 1.0 / self.steps_per_second
        left, right = self.wheels
        v, w = radius * (left + right) / 2, radius * (right - left) / 0.5
        self.yaw += math.degrees(w * dt)
        x = self.x + v * math.cos(math.radians(self.yaw)) * dt
        if x < self.wall_x:
            self.x, self.y = x, self.y + v * math.sin(math.radians(self.yaw)) * dt

    def pose(self):
        return Pose2D(self.x, self.y, wrap_deg(self.yaw))

    def place(self, pose, z):
        self.x, self.y, self.yaw, self.z = pose.x, pose.y, pose.yaw_deg, z

    def drive(self, left, right):
        self.wheels = (left, right)

    def follow(self, pose, min_z):
        self.place(pose, max(self.z, min_z))

    def render(self):
        pass

    def reload(self):
        pass


def tiles(*surfaces) -> Update:
    """The tiles of flat (x0, x1, y0, y1, height) patches, seen from above."""
    grid = ElevationGrid(Elevation())
    for x0, x1, y0, y1, h in surfaces:
        xs, ys = np.meshgrid(np.arange(x0, x1, 0.01) + 0.005, np.arange(y0, y1, 0.01) + 0.005)
        grid.integrate(np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, h)]), np.array([0.0, 0.0, 2.0]))
    return Update(patches={k: Patch(k, *grid.patch(k)) for k in grid.tiles}, cell_m=grid.cfg.cell_m, n=grid.n)


def terrain(*surfaces) -> TerrainMap:
    m = TerrainMap()
    m.apply(tiles(*surfaces))
    return m


def plan(goal, *steps):
    return Request(Pose2D(0.0, 0.0, 0.0), goal, tuple(Step(Command(c), s, d) for c, s, d in steps))


FLOOR = (-1.0, 3.0, -1.5, 1.5, 0.0)
BOX = (0.8, 1.0, -0.15, 0.15, 0.25)


def test_a_drive_and_a_turn_on_mapped_flat_ground_reach_their_goals():
    rover = FakeRover()
    validator = Validator(CFG, rover, terrain(FLOOR))
    result = validator.validate(plan(Pose2D(1.0, 0.0, 90.0), ("FWD", 40, int(UNITS)), ("LEFT", 40, 90)))
    assert result.approved, result.reason
    assert result.final_pose.x == pytest.approx(1.0, abs=0.02) and result.final_pose.yaw_deg == pytest.approx(90, abs=3)
    assert len(result.expected_step_end_poses) == 2
    assert rover.z == pytest.approx(0.0 + FakeRover.ride_height_m)         # placed on the ground it reset on


def test_plans_over_unseen_ground_or_into_a_mapped_box_are_refused_before_moving():
    rover = FakeRover()
    validator = Validator(CFG, rover, terrain((-1.0, 0.8, -1.5, 1.5, 0.0)))
    far = validator.validate(plan(Pose2D(1.2, 0.0, 0.0), ("FWD", 40, int(1.2 * UNITS))))
    assert far.failure_type == "unknown_not_ready"
    # Out and back ends where it started: the commands' sweep, not the start-goal line, is checked.
    back = validator.validate(plan(Pose2D(0.0, 0.0, 0.0), ("FWD", 40, int(1.2 * UNITS)), ("REV", 40, int(1.2 * UNITS))))
    assert back.failure_type == "unknown_not_ready" and back.failed_command_index == 0

    validator = Validator(CFG, rover, terrain(FLOOR, BOX))
    boxed = validator.validate(plan(Pose2D(1.2, 0.0, 0.0), ("FWD", 40, int(1.2 * UNITS))))
    assert boxed.failure_type == "collision" and "Predicted" in boxed.reason
    assert rover.pose().x == pytest.approx(0.0)       # refused by the sweep, before any sim time
    on_box = validator.validate(Request(Pose2D(0.9, 0.0, 0.0), Pose2D(0.9, 0.0, 0.0), ()))
    assert on_box.failure_type == "collision"          # a plan that never moves still stands somewhere

    # The rover fits under nothing, and the grid cannot tell a table from a
    # cabinet: anything taller is solid. A drop is refused like a step up.
    for ahead in ((0.8, 1.0, -1.5, 1.5, 0.7), (0.8, 3.0, -1.5, 1.5, -0.3)):
        validator = Validator(CFG, rover, terrain((-1.0, 0.8, -1.5, 1.5, 0.0), ahead))
        result = validator.validate(plan(Pose2D(1.2, 0.0, 0.0), ("FWD", 40, int(1.2 * UNITS))))
        assert result.failure_type == "collision", ahead


def test_contact_the_map_does_not_show_stalls_the_drive():
    validator = Validator(CFG, FakeRover(wall_x=0.4), terrain(FLOOR))
    result = validator.validate(plan(Pose2D(1.0, 0.0, 0.0), ("FWD", 40, int(UNITS))))
    assert result.failure_type == "collision" and "stalled" in result.reason
    assert result.final_pose.x == pytest.approx(0.4, abs=0.01)


def test_validation_runs_on_terrain_that_arrived_before_it():
    class PendingLink:
        def __init__(self, *updates):
            self.updates, self.connected = list(updates), threading.Event()

        def take(self):
            return self.updates.pop(0) if self.updates else None

    class NoStage:
        def apply(self, update):
            pass

        def show(self, mesh):
            pass

    twin = Twin(CFG, FakeRover(), NoStage(), PendingLink(tiles(FLOOR), tiles(FLOOR, BOX)))
    twin.tick()                                   # the floor is applied
    reply = queue.Queue()
    twin.jobs.put(("validate", plan(Pose2D(1.2, 0.0, 0.0), ("FWD", 40, int(1.2 * UNITS))), reply))
    # While a job waits (or runs), the live pose the drive loop stops on is still answered.
    http = serve(twin, "127.0.0.1", 0)
    try:
        url = f"http://127.0.0.1:{http.server_address[1]}/observation"
        assert "live_pose" in json.loads(urllib.request.urlopen(url, timeout=1.0).read())
    finally:
        http.shutdown()
    twin.tick()                                   # a box arrived with the request
    status, result = reply.get_nowait()
    assert status == 200 and result["failure_type"] == "collision"


def test_twin_tracks_the_mapper_and_validates_on_its_terrain(scan, tmp_path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    server = LinkServer("127.0.0.1", 0)
    link = LinkClient("127.0.0.1", server.port)
    twin = Twin(CFG, FakeRover(), StageTerrain(stage), link)
    http = serve(twin, "127.0.0.1", 0)
    stop = threading.Event()

    def run_loop():
        while not stop.is_set():
            twin.tick()

    loop = threading.Thread(target=run_loop)
    try:
        deadline = time.monotonic() + 10.0
        while not server.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        mapper = map_scan(scan("drive"), tmp_path, link=server)
        loop.start()
        last = mapper.track[-1].t_phone
        while (twin.live is None or twin.live.t_phone != last) and time.monotonic() < deadline:
            time.sleep(0.01)

        tiles = [p for p in stage.Traverse() if p.GetName().startswith("tile_")]
        assert len(tiles) == len(mapper.grid.tiles) == len(twin.map.patches)
        n = mapper.grid.n
        for prim in tiles:
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            assert points.shape == ((n + 1) ** 2, 3) and np.isfinite(points).all()
        assert twin.rover.pose().x == pytest.approx(1.6, abs=1e-6)          # following the real rover

        body = json.dumps({"start_pose": {"x": 1.6, "y": 0.0, "yaw_deg": 0.0},
                           "goal": {"x": 1.6, "y": 0.0, "yaw_deg": 90.0},
                           "commands": [{"command": "LEFT", "speed": 40, "distance": 90}]}).encode()
        url = f"http://127.0.0.1:{http.server_address[1]}/validate_plan"
        result = json.loads(urllib.request.urlopen(urllib.request.Request(url, body), timeout=10).read())
        assert result["approved"], result["reason"]
        assert twin.rover.pose().yaw_deg == pytest.approx(0.0, abs=1e-6)    # back on the live pose after
    finally:
        stop.set()
        if loop.is_alive():
            loop.join()
        http.shutdown()
        link.close()
        server.close()
    twin.write_metrics(tmp_path / "twin_metrics.json")
    metrics = json.loads((tmp_path / "twin_metrics.json").read_text())
    assert metrics["latency_ms"]["collider_live"]["n"] > 0 and metrics["values"]["real_time_factor"] > 0
