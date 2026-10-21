import math

import numpy as np
import pytest

from rover_twin import frames
from rover_twin.metrics import fidelity


def grid_points(x0, x1, y0, y1, z, step=0.02):
    xs, ys = np.meshgrid(x0 + step * (np.arange(round((x1 - x0) / step)) + 0.5),
                         y0 + step * (np.arange(round((y1 - y0) / step)) + 0.5))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


def test_floor_step_sees_a_duplicated_sheet_but_not_a_few_noisy_cells():
    floor = grid_points(0.0, 3.0, 0.0, 3.0, 0.0, step=0.01)
    noisy = floor.copy()
    noisy[::5000, 2] = 0.04
    sheet = grid_points(1.0, 1.6, 1.0, 1.6, 0.03, step=0.01)        # a second sheet 3 cm up
    for points, expected in ((noisy, 0.0), (np.vstack([floor, sheet]), 0.03)):
        floor_fit, mask = fidelity.fit_floor(points)
        assert fidelity.floor_step(points, floor_fit, mask) == pytest.approx(expected, abs=0.005)


def test_box_size_measures_a_rotated_box_seen_from_two_sides():
    # Only the two faces toward the camera are mapped, one cell thick: the
    # minimum-area rectangle of that L is ambiguous, the rectangle fit is not.
    length, width, height, yaw = 0.40, 0.30, 0.25, math.radians(20.0)
    R = frames.rot_z(yaw)[:2, :2]
    s = np.arange(0.0, 1.0, 0.02)
    faces = np.vstack([np.column_stack([s * length, np.zeros_like(s)]),
                       np.column_stack([np.zeros_like(s), s * width])]) @ R.T + [1.4, -1.2]
    points = np.vstack([grid_points(0.5, 2.5, -2.5, 0.0, 0.0),
                        np.column_stack([faces, np.full(len(faces), height)])])
    floor, _ = fidelity.fit_floor(points)
    measured = fidelity.box_size(points, floor, (1.5, -1.05), search_m=0.4)
    assert measured == pytest.approx((length, width, height), abs=0.02)
