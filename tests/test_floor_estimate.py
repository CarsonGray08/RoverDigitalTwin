import math
import random
import unittest

from digital_twin.floor_estimate import FloorEstimator, estimate_floor_height


def floor_patch(z=0.0, n=200, radius=1.5, rover_xy=(0.0, 0.0), jitter=0.0, seed=1):
    """A dense ring of floor points at height z around the rover."""
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.3, radius)
        zz = z + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        pts.append((rover_xy[0] + r * math.cos(ang), rover_xy[1] + r * math.sin(ang), zz))
    return pts


class EstimateFloorHeightTests(unittest.TestCase):
    def test_finds_floor_among_table_and_glitch(self):
        pts = floor_patch(z=0.0, n=300)
        # A dense table surface 0.7 m up (must be ignored: floor is the lowest).
        pts += floor_patch(z=0.7, n=300, seed=2)
        # A handful of sub-floor depth glitches (sparse -> must be rejected).
        pts += [(0.2, 0.1, -0.5), (-0.3, 0.2, -0.45), (0.1, -0.25, -0.6)]
        m = estimate_floor_height(pts, (0.0, 0.0))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.height_m, 0.0, delta=0.02)
        self.assertEqual(m.rejected_below, 3)

    def test_noisy_floor_median_is_accurate(self):
        pts = floor_patch(z=0.12, n=400, jitter=0.015, seed=7)
        m = estimate_floor_height(pts, (0.0, 0.0))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.height_m, 0.12, delta=0.01)

    def test_returns_none_without_enough_evidence(self):
        pts = [(0.5, 0.5, 0.0), (0.6, 0.4, 0.0)]
        self.assertIsNone(estimate_floor_height(pts, (0.0, 0.0), min_cluster_points=20))

    def test_excludes_rover_body_points(self):
        # Rover body/wheels: a dense low-ish blob right at the rover origin, plus
        # the real floor in the annulus. The body blob is inside exclude_radius
        # so it must not be picked even though it is dense.
        body = [(0.02 * math.cos(t), 0.02 * math.sin(t), 0.25) for t in range(100)]
        pts = body + floor_patch(z=0.0, n=300)
        m = estimate_floor_height(pts, (0.0, 0.0), exclude_radius_m=0.15)
        self.assertAlmostEqual(m.height_m, 0.0, delta=0.02)

    def test_respects_outer_radius(self):
        near = floor_patch(z=0.0, n=300, radius=1.0)
        far = floor_patch(z=-1.0, n=300, radius=5.0, seed=3)
        far = [(x, y, z) for (x, y, z) in far if math.hypot(x, y) > 2.5]
        m = estimate_floor_height(near + far, (0.0, 0.0), radius_m=2.0)
        self.assertAlmostEqual(m.height_m, 0.0, delta=0.02)

    def test_rejects_invalid_radii(self):
        with self.assertRaises(ValueError):
            estimate_floor_height([], (0.0, 0.0), radius_m=0.1, exclude_radius_m=0.2)


class FloorEstimatorTests(unittest.TestCase):
    def test_ema_moves_slowly_toward_new_height(self):
        est = FloorEstimator(smoothing=0.25)
        est.update(floor_patch(z=0.0, n=300), (0.0, 0.0))
        self.assertAlmostEqual(est.height_m, 0.0, delta=0.02)
        # Floor jumps to 0.2; one update should move only ~25% of the way.
        est.update(floor_patch(z=0.2, n=300, seed=9), (0.0, 0.0))
        self.assertGreater(est.height_m, 0.02)
        self.assertLess(est.height_m, 0.12)

    def test_becomes_stable_after_repeated_consistent_measurements(self):
        est = FloorEstimator(smoothing=0.5, stable_window=5, stable_tol_m=0.01)
        self.assertFalse(est.is_stable)
        for seed in range(8):
            est.update(floor_patch(z=0.05, n=300, jitter=0.005, seed=seed), (0.0, 0.0))
        self.assertTrue(est.is_stable)
        self.assertAlmostEqual(est.height_m, 0.05, delta=0.01)

    def test_holds_estimate_when_no_evidence(self):
        est = FloorEstimator()
        est.update(floor_patch(z=0.0, n=300), (0.0, 0.0))
        held = est.height_m
        # An empty observation must not wipe the estimate.
        self.assertEqual(est.update([], (0.0, 0.0)), held)
        self.assertEqual(est.samples, 1)

    def test_status_shape(self):
        est = FloorEstimator()
        self.assertFalse(est.status()["has_estimate"])
        est.update(floor_patch(z=0.0, n=300), (0.0, 0.0))
        status = est.status()
        self.assertTrue(status["has_estimate"])
        self.assertIsNotNone(status["height_m"])
        self.assertEqual(status["samples"], 1)


if __name__ == "__main__":
    unittest.main()
