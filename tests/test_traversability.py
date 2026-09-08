import unittest

from digital_twin.traversability import (
    INSUFFICIENT_CLEARANCE,
    PITCH_EXCEEDED,
    ROLL_EXCEEDED,
    SLOPE_EXCEEDED,
    TerrainSample,
    TraversabilityLimits,
    classify_sample,
    evaluate,
)


def _ok_sample(**kw):
    base = dict(slope_deg=5.0, roll_deg=2.0, pitch_deg=3.0, ground_clearance_m=0.2)
    base.update(kw)
    return TerrainSample(**base)


class ClassifySampleTest(unittest.TestCase):
    def setUp(self):
        self.limits = TraversabilityLimits()

    def test_traversable_sample_returns_none(self):
        self.assertIsNone(classify_sample(_ok_sample(), self.limits))

    def test_slope_limit(self):
        self.assertEqual(classify_sample(_ok_sample(slope_deg=25.0), self.limits), SLOPE_EXCEEDED)

    def test_roll_limit_either_sign(self):
        self.assertEqual(classify_sample(_ok_sample(roll_deg=20.0), self.limits), ROLL_EXCEEDED)
        self.assertEqual(classify_sample(_ok_sample(roll_deg=-20.0), self.limits), ROLL_EXCEEDED)

    def test_pitch_limit(self):
        self.assertEqual(classify_sample(_ok_sample(pitch_deg=-30.0), self.limits), PITCH_EXCEEDED)

    def test_clearance_limit_including_negative_belly_out(self):
        self.assertEqual(classify_sample(_ok_sample(ground_clearance_m=0.0), self.limits),
                         INSUFFICIENT_CLEARANCE)
        self.assertEqual(classify_sample(_ok_sample(ground_clearance_m=-0.1), self.limits),
                         INSUFFICIENT_CLEARANCE)

    def test_tipover_ranks_above_slope(self):
        # Over-steep AND rolling past the tip limit -> report the tip-over.
        s = _ok_sample(slope_deg=40.0, roll_deg=30.0)
        self.assertEqual(classify_sample(s, self.limits), ROLL_EXCEEDED)


class EvaluateRolloutTest(unittest.TestCase):
    def setUp(self):
        self.limits = TraversabilityLimits()

    def test_all_clear(self):
        verdict = evaluate([_ok_sample(), _ok_sample(slope_deg=10.0)], self.limits)
        self.assertTrue(verdict.traversable)
        self.assertIsNone(verdict.reason)
        self.assertEqual(verdict.max_slope_deg, 10.0)
        self.assertAlmostEqual(verdict.min_ground_clearance_m, 0.2)

    def test_fails_at_first_violation_with_worst_values_so_far(self):
        samples = [
            _ok_sample(slope_deg=8.0, ground_clearance_m=0.3),
            _ok_sample(slope_deg=12.0, roll_deg=-5.0, ground_clearance_m=0.15),
            _ok_sample(pitch_deg=45.0),   # violation here
            _ok_sample(roll_deg=80.0),    # never reached
        ]
        verdict = evaluate(samples, self.limits)
        self.assertFalse(verdict.traversable)
        self.assertEqual(verdict.failed_index, 2)
        self.assertEqual(verdict.reason, PITCH_EXCEEDED)
        self.assertEqual(verdict.max_slope_deg, 12.0)
        self.assertAlmostEqual(verdict.min_ground_clearance_m, 0.15)

    def test_empty_rollout_is_trivially_traversable(self):
        verdict = evaluate([], self.limits)
        self.assertTrue(verdict.traversable)

    def test_limits_validation(self):
        with self.assertRaises(ValueError):
            TraversabilityLimits(max_slope_deg=0.0)
        with self.assertRaises(ValueError):
            TraversabilityLimits(min_ground_clearance_m=-1.0)


if __name__ == "__main__":
    unittest.main()
