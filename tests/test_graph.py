import math

import numpy as np
import pytest

from rover_twin.pose import graph


def wall(x0, y0, x1, y1, step=0.02):
    """Points on a vertical wall from (x0, y0) to (x1, y1), 0.1-1.0 m high."""
    n = int(math.hypot(x1 - x0, y1 - y0) / step)
    t, z = np.meshgrid(np.linspace(0, 1, n), np.arange(0.1, 1.0, step))
    return np.column_stack([x0 + (x1 - x0) * t.ravel(), y0 + (y1 - y0) * t.ravel(), z.ravel()])


def test_registration_recovers_a_planar_offset_and_knows_what_it_cannot_see():
    corner = np.vstack([wall(2.0, -1.5, 2.0, 1.5), wall(-1.0, 1.5, 2.0, 1.5)])
    truth = np.array([0.04, -0.03, math.radians(2.0)])
    moving = graph.apply(graph.between(truth, np.zeros(3)), corner)    # seen from a pose off by truth
    D, info, inliers, residual = graph.register(corner, moving)
    assert D == pytest.approx(truth, abs=1e-4) and inliers > 0.99 and residual < 1e-3

    # One wall along y: sliding along it changes nothing, and the information says so.
    _, info, _, _ = graph.register(wall(2.0, -1.5, 2.0, 1.5), wall(2.0, -1.5, 2.0, 1.5) + [0.03, 0.0, 0.0])
    lam, vec = np.linalg.eigh(info)
    assert lam[0] < 1e-6 * lam[1] and abs(vec[1, 0]) == pytest.approx(1.0)
