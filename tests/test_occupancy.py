import numpy as np
import pytest
from test_elevation import INTR, WALL, camera_at

from rover_twin import frames
from rover_twin.config import Occupancy
from rover_twin.frames import backproject, transform
from rover_twin.mapping.occupancy import OccupancyGrid

CFG = Occupancy(enabled=True)


def render(T_W_C, *boxes, dark=None):
    """Z-depth of the floor and ``boxes`` (x0, x1, y0, y1, z0, z1); ``dark``: a
    box whose own pixels have no depth."""
    v, u = np.mgrid[0:INTR.height, 0:INTR.width] + 0.5
    rays = np.stack([(u - INTR.cx) / INTR.fx, (v - INTR.cy) / INTR.fy, np.ones_like(u)], -1) @ T_W_C[:3, :3].T
    o = T_W_C[:3, 3]
    depth = np.where(rays[..., 2] < 0, -o[2] / np.minimum(rays[..., 2], -1e-9), np.inf)
    unlit = np.zeros(depth.shape, bool)
    for box in boxes:
        lo, hi = np.array(box[0::2]), np.array(box[1::2])
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (lo - o) / rays, (hi - o) / rays
        near, far = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
        hit = (near <= far) & (near > 0) & (near < depth)
        depth = np.where(hit, near, depth)
        unlit = np.where(hit, box == dark, unlit & ~hit)
    depth = np.where(depth < 5.0, depth, 0.0)
    depth[unlit] = 0.0
    # rays above the horizon see nothing in range; z-depth is depth along the optical axis
    return depth.astype(np.float32)


CLOCK = iter(range(10 ** 9))      # phone time: every look is later than the last


def look(grid, T_W_C, *boxes, dark=None, owner=0):
    """One frame, its ARKit world the twin's (W_T_A identity) for its keyframe ``owner``."""
    depth = render(T_W_C, *boxes, dark=dark)
    grid.integrate(transform(T_W_C, backproject(depth, INTR)), depth, INTR, T_W_C, owner, np.eye(4),
                   float(next(CLOCK)))


def occupied_in(grid, x0, x1, y0, y1):
    return grid.count_in(x0, x1, y0, y1)


PERSON = (1.40, 1.50, -0.10, 0.10, 0.0, 0.9)
WHERE = (1.36, 1.54, -0.14, 0.14)


def test_someone_who_leaves_is_freed_by_the_frames_that_see_past_where_they_stood():
    grid = OccupancyGrid(CFG)                              # the phone as mounted: 25 deg down
    for _ in range(5):
        look(grid, camera_at(0.0), WALL, PERSON)
    assert occupied_in(grid, *WHERE) >= 3
    frames_to_clear = 0
    while occupied_in(grid, *WHERE) and frames_to_clear < 30:
        look(grid, camera_at(0.0), WALL)
        frames_to_clear += 1
    assert frames_to_clear <= 10           # a second at 10 Hz
    look(grid, camera_at(0.0), WALL, PERSON)     # back again: there at once
    assert occupied_in(grid, *WHERE) >= 3


def test_a_dark_seat_fades_over_seconds_while_the_space_under_it_is_freed_at_once():
    grid = OccupancyGrid(CFG)
    seat = (1.30, 1.70, -0.20, 0.20, 0.42, 0.50)
    camera = camera_at(0.0, tilt_deg=10.0)
    under = (1.30, 1.70, -0.20, 0.20)       # columns whose centre is under the seat
    for _ in range(2):
        look(grid, camera, WALL, seat)
    held = occupied_in(grid, *under)
    assert held >= 10
    for _ in range(10):                     # a second with no depth from it: still there
        look(grid, camera, WALL, seat, dark=seat)
    assert occupied_in(grid, *under) == held
    below = grid.occupied()                 # ... and nothing below it: the floor seen under it freed that
    at = (below[:, 0] > 1.32) & (below[:, 0] < 1.68) & (np.abs(below[:, 1]) < 0.18)
    assert (below[at, 2] < 0.40).sum() == 0
    for _ in range(30):                     # a few seconds in view, never confirmed: forgotten
        look(grid, camera, WALL, seat, dark=seat)
    assert occupied_in(grid, *under) == 0


def test_what_is_out_of_view_stays_as_it_was():
    grid = OccupancyGrid(CFG)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL, PERSON)
    held = occupied_in(grid, *WHERE)
    for _ in range(50):                     # the rover drives on past them: behind the camera
        look(grid, camera_at(1.8), WALL)
    assert occupied_in(grid, *WHERE) == held


def test_what_is_hidden_behind_something_is_not_freed():
    grid = OccupancyGrid(CFG)
    behind = (2.00, 2.10, -0.10, 0.10, 0.0, 0.4)
    look(grid, camera_at(0.0), WALL, behind)
    held = occupied_in(grid, 1.96, 2.14, -0.14, 0.14)
    assert held >= 3
    screen = (1.00, 1.05, -0.40, 0.40, 0.0, 0.6)          # now in front of it, and it has gone
    for _ in range(20):
        look(grid, camera_at(0.0), WALL, screen)
    assert occupied_in(grid, 1.96, 2.14, -0.14, 0.14) == held


def test_a_rebuild_moves_each_voxel_with_its_keyframe_seen_gone_or_not():
    # The complete person, including its occupied voxel extents, is in view.
    # A downward view clips upper voxels; a short wall leaves no returns
    # behind them. Use a level view and a full-height background wall.
    grid = OccupancyGrid(CFG)
    for _ in range(3):
        look(grid, camera_at(0.0, tilt_deg=0.0), (*WALL[:5], 2.4), PERSON, owner=0)
    box = (1.40, 1.50, 0.60, 0.70, 0.0, 0.5)
    for _ in range(10):
        look(grid, camera_at(0.0, tilt_deg=0.0), (*WALL[:5], 2.4), box, owner=1)        # the person left; a box is seen
    assert occupied_in(grid, *WHERE) == 0
    i, j, k = np.nonzero(grid.logodds > CFG.occupied)
    moved = grid._centres(i, j, k)[grid.owner[i, j, k] == 1]
    grid.move([np.eye(4), frames.rot_z(0.0, (0.0, 0.2, 0.0))])   # keyframe 1 moved 0.2 m along +y
    after = {tuple(np.round(p, 3)) for p in grid.occupied()}
    assert {tuple(np.round(p + [0.0, 0.2, 0.0], 3)) for p in moved} <= after
    assert occupied_in(grid, *WHERE) == 0                      # what it saw gone stays gone


def test_small_corrections_add_up_instead_of_rounding_away():
    grid = OccupancyGrid(CFG)
    box = (1.40, 1.50, -0.10, 0.10, 0.0, 0.5)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL, box)

    def centre_x():
        p = grid.occupied()
        return p[(p[:, 0] > 1.2) & (p[:, 0] < 1.8) & (np.abs(p[:, 1]) < 0.14), 0].mean()

    before = centre_x()
    for n in range(1, 11):                                     # ten closures, 1 cm each, along +x
        grid.move([frames.rot_z(0.0, (0.01 * n, 0.0, 0.0))])
    assert centre_x() - before == pytest.approx(0.10, abs=0.02)


def test_an_old_obstacle_moved_onto_space_seen_empty_since_does_not_cover_it():
    grid = OccupancyGrid(CFG)
    for _ in range(3):
        look(grid, camera_at(0.0), WALL, PERSON, owner=0)      # seen at keyframe 0
    assert occupied_in(grid, *WHERE) >= 3
    for _ in range(3):
        look(grid, camera_at(1.55), WALL, owner=1)             # later, from past them: empty there
    assert occupied_in(grid, *WHERE) >= 3                      # (behind keyframe 1: not seen again)
    i, j, k = np.nonzero((grid.owner == 1) & (grid.logodds < 0))
    empty = {tuple(np.round(c, 3)) for c in grid._centres(i, j, k)}  # what keyframe 1 saw empty
    grid.move([frames.rot_z(0.0, (0.6, 0.0, 0.0)), np.eye(4)])  # keyframe 0's person lands there
    landed = {tuple(np.round(c, 3)) for c in grid.occupied()}
    assert len(empty & {tuple(np.round(c + [0.6, 0, 0], 3)) for c in grid._centres(*np.nonzero(grid.owner == 0))})
    assert not empty & landed


def test_someone_walking_into_space_seen_empty_shows_at_once():
    grid = OccupancyGrid(CFG)
    for _ in range(20):
        look(grid, camera_at(0.0), WALL)
    look(grid, camera_at(0.0), WALL, PERSON)
    assert occupied_in(grid, *WHERE) >= 3


@pytest.mark.parametrize('height', [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52])
@pytest.mark.parametrize('offset', [0.0, 0.013])
def test_thin_dark_surfaces_are_not_dropped_at_once_at_any_voxel_alignment(height, offset):
    grid = OccupancyGrid(CFG)
    seat = (1.30 + offset, 1.70 + offset, -0.20, 0.20, height, height + 0.01)
    camera = camera_at(0.0, tilt_deg=10.0)
    for _ in range(2):
        look(grid, camera, WALL, seat)
    before = grid.occupied()
    before = before[(before[:, 0] > 1.25) & (before[:, 0] < 1.75) & (np.abs(before[:, 1]) < 0.25)]
    assert len(before)
    for _ in range(10):
        look(grid, camera, WALL, seat, dark=seat)
    after = {tuple(p) for p in grid.occupied()}
    assert all(tuple(p) in after for p in before)
    # Once actually removed, a complete background view must clear it.
    for _ in range(10):
        look(grid, camera, (*WALL[:5], 2.4))
    after = {tuple(p) for p in grid.occupied()}
    assert not any(tuple(p) in after for p in before)


def test_column_count_does_not_merge_adjacent_half_integer_centres():
    grid = OccupancyGrid(CFG)
    grid._cover(np.array([0., 0.]), np.array([1., 1.]))
    idx = grid._index(np.array([[0.06, 0.06, 0.3], [0.10, 0.06, 0.3]]))
    grid.logodds[tuple(idx.T)] = 1.0
    assert grid.count_in(0, 1, 0, 1) == 2


def test_a_current_hit_wins_over_free_evidence_in_the_same_voxel():
    from rover_twin.frames import Intrinsics
    intr = Intrinsics(100, 100, 50, 50, 100, 100)
    grid = OccupancyGrid(CFG)
    camera = np.eye(4)
    depth = np.full((100, 100), 3.0)
    # Independent hit/free sampling can disagree at depth edges. A real hit
    # must not be cancelled by the free projection in that frame.
    for t in range(3):
        grid.integrate(np.empty((0, 3)), depth, intr, camera, 0, camera, t)
    hit = np.array([[0.02, 0.02, 0.32]])
    grid.integrate(hit, depth, intr, camera, 0, camera, 4)
    assert grid.logodds[tuple(grid._index(hit)[0])] > CFG.occupied
