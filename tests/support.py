"""Fakes and helpers shared by the tests."""

from pathlib import Path

import numpy as np

from rover_twin import config
from rover_twin.config import Elevation
from rover_twin.mapper.pipeline import Mapper
from rover_twin.mapping.elevation import ElevationGrid
from rover_twin.mapping.tsdf import Mesh
from rover_twin.phone import recording

LAB = config.load(Path(__file__).parents[1] / "config" / "rover_twin.toml")
TEST_CONFIG = LAB


class FakeTsdf:
    """Stands in for nvblox: counts what it is fed, has no mesh."""

    def __init__(self):
        self.integrated = 0

    def integrate(self, depth, intrinsics, T_W_C, jpeg, color_intrinsics):
        self.integrated += 1

    def mesh(self):
        return Mesh.empty()

    def clear(self):
        self.integrated = 0


class CollectingLink:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def snapshot_requested(self):
        return False


def map_scan(scanlog: Path, run_dir: Path, link=None, cfg=TEST_CONFIG) -> Mapper:
    mapper = Mapper(cfg, FakeTsdf(), link or CollectingLink(), run_dir)
    mapper.run(recording.replay(scanlog, speed=0))
    return mapper


def truth_of(scanlog: Path) -> Path:
    return scanlog.with_name(scanlog.stem + ".truth.toml")


def saved_map(path, *surfaces):
    """A map file of flat (x0, x1, y0, y1, height) patches, seen from above."""
    grid = ElevationGrid(Elevation())
    for x0, x1, y0, y1, h in surfaces:
        xs, ys = np.meshgrid(np.arange(x0, x1, 0.01) + 0.005, np.arange(y0, y1, 0.01) + 0.005)
        grid.integrate(np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, h)]), np.array([0.0, 0.0, 2.0]))
    grid.save(path)
    return path
