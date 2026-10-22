import numpy as np
import pytest

from rover_twin.config import Elevation
from rover_twin.mapping.elevation import INFERRED, OBSERVED, UNKNOWN, ElevationGrid

CAMERA = np.array([0.0, 0.0, 0.25])
CFG = Elevation(cell_m=0.02, tile_cells=8)


def patch_points(x0, x1, y0, y1, z, step=0.005):
    xs, ys = np.meshgrid(x0 + step * (np.arange(round((x1 - x0) / step)) + 0.5),
                         y0 + step * (np.arange(round((y1 - y0) / step)) + 0.5))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


def test_fusion_averages_a_surface_takes_an_obstacle_top_and_ignores_what_is_under_it():
    grid = ElevationGrid(CFG)
    rng = np.random.default_rng(0)
    floor = patch_points(1.0, 1.16, 0.0, 0.16, 0.0)
    for _ in range(20):
        grid.integrate(floor + [0, 0, 1] * rng.normal(0, 0.003, (len(floor), 1)), CAMERA)
    h = grid.points()[:, 2]
    assert np.abs(h.mean()) < 5e-4 and h.std() < 2e-3

    box = patch_points(1.04, 1.08, 0.04, 0.08, 0.30)          # a box top lands on four cells
    grid.integrate(box, CAMERA)
    grid.integrate(floor, CAMERA)                             # the floor under it, seen again
    points = grid.points()
    on_box = (np.abs(points[:, 0] - 1.06) < 0.02) & (np.abs(points[:, 1] - 0.06) < 0.02)
    assert on_box.sum() == 4 and np.allclose(points[on_box, 2], 0.30)
    assert np.all(np.abs(points[~on_box, 2]) < 0.01)


def test_patches_share_edges_and_a_change_reaches_the_neighbour_patch():
    grid = ElevationGrid(CFG)
    tile = CFG.cell_m * CFG.tile_cells
    dirty = grid.integrate(patch_points(0.0, 2 * tile, 0.0, tile, 0.01), CAMERA)
    assert dirty == {(0, 0), (1, 0)}
    a, sa = grid.patch((0, 0))
    b, _ = grid.patch((1, 0))
    np.testing.assert_array_equal(a[-1, :], b[0, :])
    assert np.isnan(a[:, -1]).all() and (sa[:, -1] == UNKNOWN).all()      # nothing seen at +y yet

    raised = grid.integrate(patch_points(tile, tile + 2 * CFG.cell_m, 0.0, CFG.cell_m, 0.2), CAMERA)   # 2 cells: not a lone spike
    assert raised == {(0, 0), (1, 0)}        # the first column of tile 1 is the apron of tile 0
    assert grid.patch((0, 0))[0][-1, 0] == pytest.approx(0.2)


def test_the_unseen_start_footprint_is_inferred_and_yields_to_observation():
    cfg = Elevation(cell_m=0.02, tile_cells=16, footprint_radius_m=0.2, footprint_ring_m=0.1)
    grid = ElevationGrid(cfg)
    ring = patch_points(-0.32, 0.32, -0.32, 0.32, 0.015)
    ring = ring[np.hypot(ring[:, 0], ring[:, 1]) > 0.2]
    grid.integrate(ring, CAMERA)
    assert grid.infer_footprint((0.0, 0.0))
    inferred = grid.points(INFERRED)
    assert len(inferred) > 250 and np.allclose(inferred[:, 2], 0.015, atol=1e-6)
    assert np.hypot(inferred[:, 0], inferred[:, 1]).max() <= 0.2

    grid.integrate(patch_points(-0.02, 0.02, -0.02, 0.02, 0.03), CAMERA)   # driving off reveals it
    assert len(grid.points(INFERRED)) == len(inferred) - 4
    assert grid.infer_footprint((0.0, 0.0)) == set()                         # nothing left to change
    assert len(grid.points(OBSERVED)) == len(np.unique(np.floor(ring[:, :2] / 0.02), axis=0)) + 4


def test_the_start_footprint_falls_back_to_the_start_ground_until_its_ring_is_seen():
    cfg = Elevation(cell_m=0.02, tile_cells=16, footprint_radius_m=0.2, footprint_ring_m=0.1)
    grid = ElevationGrid(cfg)
    grid.integrate(patch_points(0.5, 1.0, -0.3, 0.3, 0.0), CAMERA)    # the camera sees only far ahead
    assert grid.infer_footprint((0.0, 0.0)) == set()
    assert grid.infer_footprint((0.0, 0.0), fallback_z=0.0)
    inferred = grid.points(INFERRED)
    assert len(inferred) > 250 and np.allclose(inferred[:, 2], 0.0)
    assert grid.infer_footprint((0.0, 0.0), fallback_z=0.004) == set()   # the fallback never rewrites

    ring = patch_points(-0.32, 0.32, -0.32, 0.32, 0.015)                 # turning in place reveals the ring
    grid.integrate(ring[np.hypot(ring[:, 0], ring[:, 1]) > 0.2], CAMERA)
    assert grid.infer_footprint((0.0, 0.0), fallback_z=0.0)
    assert np.allclose(grid.points(INFERRED)[:, 2], 0.015, atol=1e-6)


def test_a_lone_spike_is_lowered_and_anything_wider_kept():
    grid = ElevationGrid(CFG)
    grid.integrate(patch_points(0.0, 0.32, 0.0, 0.32, 0.0), CAMERA)            # floor
    grid.integrate(patch_points(0.10, 0.12, 0.10, 0.12, 0.12), CAMERA)         # one cell, 12 cm up
    grid.integrate(patch_points(0.20, 0.24, 0.20, 0.22, 0.10), CAMERA)         # two cells, 10 cm up
    h = np.vstack([np.hstack([grid.patch((i, j))[0][:-1, :-1] for j in range(2)]) for i in range(2)])
    assert grid.points()[:, 2].max() == pytest.approx(0.12)                   # the map itself keeps it
    assert h[5, 5] == pytest.approx(0.0, abs=1e-6)                             # the patch does not
    assert h[10, 10] == pytest.approx(0.10) and h[11, 10] == pytest.approx(0.10)
