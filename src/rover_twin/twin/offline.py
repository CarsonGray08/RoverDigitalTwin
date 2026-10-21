"""The twin without Isaac Sim: the same HTTP API on a saved map, for building
software that calls the twin on any laptop.

    python -m rover_twin.twin.offline assets/lab_map.npz [--port 8765]

The map is a mapper run's ``elevation.npz``. Validation runs the same checks
(limits, unknown terrain, the footprint sweep against obstacles, goal
tolerance) with a kinematic rover in place of Newton, so it predicts no
stalls, slips or tipping: rehearse on the lab twin before the real rover.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

from rover_twin import config
from rover_twin.frames import Pose2D, wrap_deg
from rover_twin.link.client import Update
from rover_twin.link.messages import Patch
from rover_twin.mapping.elevation import ElevationGrid
from rover_twin.twin.main import DEFAULT_CONFIG, Twin, serve

STEPS_PER_SECOND = 100.0
# Skid-steer slips in a turn: the Newton rover turns at 19.5 deg/s and drives
# at 0.186 m/s at speed 30, which a kinematic model matches with this track.
EFFECTIVE_TRACK_M = 1.10


class KinematicRover:
    """Wheel speeds in, pose out, at the Newton rover's measured rates."""

    steps_per_second = STEPS_PER_SECOND
    ride_height_m = 0.0

    def __init__(self, cfg: config.Config) -> None:
        self._radius = cfg.rover.wheel_diameter_m / 2.0
        self._pose, self._wheels = Pose2D(cfg.start.x_m, cfg.start.y_m, cfg.start.yaw_deg), (0.0, 0.0)

    def step(self) -> None:
        left, right = self._wheels
        dt, p = 1.0 / self.steps_per_second, self._pose
        v, w = self._radius * (left + right) / 2.0, self._radius * (right - left) / EFFECTIVE_TRACK_M
        yaw = math.radians(p.yaw_deg)
        self._pose = Pose2D(p.x + v * math.cos(yaw) * dt, p.y + v * math.sin(yaw) * dt,
                            wrap_deg(p.yaw_deg + math.degrees(w * dt)))

    def pose(self) -> Pose2D:
        return self._pose

    def place(self, pose: Pose2D, z: float) -> None:
        self._pose, self._wheels = pose, (0.0, 0.0)

    def drive(self, left_rad_s: float, right_rad_s: float) -> None:
        self._wheels = (left_rad_s, right_rad_s)

    def follow(self, pose: Pose2D, min_z: float) -> None:
        self._pose = pose

    def render(self) -> None:
        pass

    def reload(self) -> None:
        pass


class SavedMap:
    """Stands in for the mapper link: hands the twin one map, once."""

    def __init__(self, path: Path) -> None:
        grid = ElevationGrid.load(path)
        self._update: Update | None = Update(patches={k: Patch(k, *grid.patch(k)) for k in grid.tiles},
                                             cell_m=grid.cfg.cell_m, n=grid.n)
        self.connected = threading.Event()

    def take(self) -> Update | None:
        update, self._update = self._update, None
        return update

    def close(self) -> None:
        pass


class NoStage:
    def apply(self, update: Update) -> None:
        pass

    def show(self, mesh) -> None:
        pass


def start(map_path: Path, cfg: config.Config, host: str, port: int) -> tuple[Twin, object]:
    """A running offline twin and its HTTP server (port 0 picks a free one)."""
    twin = Twin(cfg, KinematicRover(cfg), NoStage(), SavedMap(map_path))
    twin.tick()                                   # take the map before the first request

    def loop():
        while True:
            twin.tick()
            time.sleep(1.0 / STEPS_PER_SECOND)

    threading.Thread(target=loop, name="offline-twin", daemon=True).start()
    return twin, serve(twin, host, port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", type=Path, help="a mapper run's elevation.npz")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    cfg = config.load(args.config)
    _, server = start(args.map, cfg, args.host, args.port)
    print(f"offline twin on {args.map}: http://{args.host}:{server.server_address[1]}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
