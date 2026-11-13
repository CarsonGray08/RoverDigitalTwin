import numpy as np
import pytest

from rover_twin.config import Elevation
from rover_twin.frames import Intrinsics, backproject, transform
from rover_twin.mapping.elevation import CLEAR_FRAMES, INFERRED, OBSERVED, UNKNOWN, ElevationGrid, View

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



INTR = Intrinsics(180.7, 180.7, 128.0, 96.0, 256, 192)       # the phone's depth image
WALL = (3.0, 3.2, -2.0, 2.0, 0.0, 1.0)


def camera_at(x, z=0.355, tilt_deg=25.0):
    """Looking along +x from (x, 0, z), pitched down (CV axes: x right, y down, z ahead)."""
    t = np.radians(tilt_deg)
    ahead, right = np.array([np.cos(t), 0.0, -np.sin(t)]), np.array([0.0, -1.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, np.cross(ahead, right), ahead])
    T[:3, 3] = (x, 0.0, z)
    return T


def look(grid, T_W_C, *boxes, drop=None, patchy=False, dark=None, floor_at=None, t=None):
    """Render the floor and ``boxes`` (x0, x1, y0, y1, z0, z1) as the phone's depth
    from ``T_W_C`` and fuse it; ``drop`` ((x0, x1, y0, y1)) removes those points
    from what is fused, as the depth filters drop a surface's pixels. ``patchy``:
    ARKit's on grazing carpet, a third medium confidence and a tenth missing.
    ``dark``: a box whose own pixels have no depth (and so no points).
    ``floor_at`` ((x0, x1, y0, y1)): floor points fused there too, as a frame
    measures ground at a cell's edge beside something standing in it.
    ``t``: the frame's phone time, for the veto."""
    v, u = np.mgrid[0:INTR.height, 0:INTR.width] + 0.5
    rays = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u)], -1) @ T_W_C[:3, :3].T
    o = T_W_C[:3, 3]
    depth = np.where(rays[..., 2] < 0, -o[2] / np.minimum(rays[..., 2], -1e-9), np.inf)
    for box in boxes:
        lo, hi = np.array(box[0::2]), np.array(box[1::2])
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - o) / rays, (hi - o) / rays
        near, far = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        hit = (near <= far) & (near > 0) & (near < depth)
        depth = np.where(hit, near, depth)
        if box == dark:
            unlit = hit
    depth = np.where(depth < 5.0, depth, 0.0).astype(np.float32)
    if dark is not None:
        depth[unlit] = 0.0
    confidence = np.full(depth.shape, 2, np.uint8)
    if patchy:
        rng = np.random.default_rng(0)
        confidence[rng.random(depth.shape) < 0.33] = 1
        depth[rng.random(depth.shape) < 0.1] = 0.0
    points = transform(T_W_C, backproject(np.where(confidence >= 2, depth, 0.0), INTR))
    if drop is not None:
        x0, x1, y0, y1 = drop
        points = points[~((points[:, 0] > x0) & (points[:, 0] < x1) & (points[:, 1] > y0) & (points[:, 1] < y1)
                          & (points[:, 2] > 0.02))]
    if floor_at is not None:
        points = np.concatenate([points, patch_points(*floor_at, 0.0)])
    grid.integrate(points, T_W_C[:3, 3], View(depth, confidence, INTR, T_W_C), t)


def height_at(grid, x, y):
    points = grid.points()
    at = (np.abs(points[:, 0] - x) <= 0.01) & (np.abs(points[:, 1] - y) <= 0.01)
    return points[at, 2].max()


def test_someone_who_walks_away_is_cleared_after_three_frames_that_see_past_where_they_stood():
    grid = ElevationGrid(CFG)
    person = (1.40, 1.50, -0.10, 0.10, 0.0, 0.9)
    for _ in range(5):
        look(grid, camera_at(0.0), WALL, person)
    assert height_at(grid, 1.41, 0.01) > 0.3
    for _ in range(CLEAR_FRAMES - 1):                        # gone, but not yet seen gone long enough
        look(grid, camera_at(0.0), WALL)
    assert height_at(grid, 1.41, 0.01) > 0.3
    look(grid, camera_at(0.0), WALL)
    assert abs(height_at(grid, 1.41, 0.01)) < 0.01
    look(grid, camera_at(0.0), WALL, person)                # back again: mapped at once
    assert height_at(grid, 1.41, 0.01) > 0.3


def obstacle_near(grid, x0, x1, y0, y1):
    points = grid.points()
    at = (points[:, 0] > x0) & (points[:, 0] < x1) & (points[:, 1] > y0) & (points[:, 1] < y1)
    return int((points[at, 2] > CFG.step_m).sum())


def test_a_rebuild_does_not_bring_back_someone_seen_gone_but_maps_them_if_they_come_back():
    grid = ElevationGrid(CFG)
    person = (1.40, 1.50, -0.10, 0.10, 0.0, 0.9)
    where = (1.38, 1.52, -0.12, 0.12)
    look(grid, camera_at(0.0), WALL, person, t=1.0)          # a keyframe with them in it
    gone = []
    for i in range(CLEAR_FRAMES):
        look(grid, camera_at(0.0), WALL, t=2.0 + i)
        gone.append(grid.last_cleared)
    cells, ground = (np.concatenate(g) for g in zip(*gone))
    assert len(cells) and not obstacle_near(grid, *where)
    unvetoed = ElevationGrid(CFG)
    look(unvetoed, camera_at(0.0), WALL, person, t=1.0)
    wall = obstacle_near(unvetoed, 2.9, 3.3, -2.1, 2.1)
    # A rebuild: the grid starts over and re-fuses the keyframe from before they left.
    grid.clear()
    grid.set_veto(cells, np.full(len(cells), 4.0), ground)
    look(grid, camera_at(0.0), WALL, person, t=1.0)
    assert not obstacle_near(grid, *where)
    assert obstacle_near(grid, 2.9, 3.3, -2.1, 2.1) == wall     # the rest of that frame is fused as ever
    look(grid, camera_at(0.0), WALL, person, t=5.0)          # back after it was seen gone: mapped at once
    assert height_at(grid, 1.41, 0.01) > 0.3


def test_a_dark_seat_on_an_open_base_is_kept_when_the_floor_is_seen_under_it():
    # An office chair, as a phone tilted 10 deg down sees it in the lab: the
    # seat's underside once, then no depth of its own (dark fabric, at an
    # angle), while the floor is seen through the open base below it.
    grid = ElevationGrid(CFG)
    seat = (1.30, 1.70, -0.20, 0.20, 0.42, 0.50)
    camera = camera_at(0.0, tilt_deg=10.0)
    where = (1.28, 1.72, -0.22, 0.22)
    for _ in range(2):
        look(grid, camera, WALL, seat)
    mapped = obstacle_near(grid, *where)
    assert mapped >= 10
    for _ in range(CLEAR_FRAMES + 2):
        look(grid, camera, WALL, seat, dark=seat)
    assert obstacle_near(grid, *where) == mapped


def test_patchy_depth_past_where_someone_stood_still_clears_them():
    # Steeper on its mount, the phone sees carpet past them at a grazing angle:
    # a third of it medium confidence and some of it missing.
    grid = ElevationGrid(CFG)
    person = (1.40, 1.50, -0.10, 0.10, 0.0, 0.9)
    for _ in range(5):
        look(grid, camera_at(0.0, tilt_deg=34.0), person, patchy=True)
    assert height_at(grid, 1.41, 0.01) > 0.15
    for _ in range(CLEAR_FRAMES):
        look(grid, camera_at(0.0, tilt_deg=34.0), patchy=True)
    assert abs(height_at(grid, 1.41, 0.01)) < 0.01


def test_a_dark_thing_whose_own_pixels_have_no_depth_is_kept():
    # A black bin: mapped while lit, then its pixels go without depth while the
    # ground at its cells is measured. Past its top the wall is seen, but its
    # foot is not seen past.
    grid = ElevationGrid(CFG)
    bin_ = (1.40, 1.70, -0.15, 0.15, 0.0, 0.4)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL, bin_)
    top = height_at(grid, 1.41, 0.01)
    for _ in range(10):
        look(grid, camera_at(0.0), WALL, bin_, dark=bin_, floor_at=(1.40, 1.44, -0.04, 0.04))
    assert height_at(grid, 1.41, 0.01) == pytest.approx(top)


def test_a_low_overhang_the_floor_is_seen_under_is_kept():
    grid = ElevationGrid(CFG)
    shelf = (1.40, 1.60, -0.30, 0.30, 0.30, 0.40)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL, shelf)
    top = height_at(grid, 1.41, 0.01)
    assert top > 0.3
    for _ in range(10):        # its face filtered out, only the floor under it fused: the depth still meets it
        look(grid, camera_at(0.0), WALL, shelf, drop=(1.38, 1.62, -0.32, 0.32))
    assert height_at(grid, 1.41, 0.01) == pytest.approx(top)


def test_a_wall_seen_only_at_its_foot_up_close_keeps_its_top():
    grid = ElevationGrid(CFG)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL)                   # from 3 m: up to ~0.5 m of it
    top = height_at(grid, 3.01, 0.01)
    assert top > 0.45
    for _ in range(10):
        look(grid, camera_at(2.6), WALL)                   # from 0.4 m: only its foot
    assert height_at(grid, 3.01, 0.01) == pytest.approx(top)
