import unittest

from digital_twin.models import Pose
from digital_twin.realtime_safety import (
    ExecutionErrorLog,
    TrackingDecision,
    TrackingGate,
    is_pose_jump,
    pose_error,
)


def phone_block(**over):
    base = {
        "enabled": True,
        "have_pose": True,
        "tracking_normal": True,
        "rover_pose_aligned": True,
    }
    base.update(over)
    return base


class TrackingGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = TrackingGate()

    def test_execute_when_all_good(self):
        self.assertIs(self.gate.evaluate(phone_block()), TrackingDecision.EXECUTE)

    def test_wait_when_tracking_not_normal(self):
        self.assertIs(self.gate.evaluate(phone_block(tracking_normal=False)), TrackingDecision.WAIT)

    def test_wait_when_no_pose_or_disabled_or_unaligned(self):
        self.assertIs(self.gate.evaluate(phone_block(have_pose=False)), TrackingDecision.WAIT)
        self.assertIs(self.gate.evaluate(phone_block(enabled=False)), TrackingDecision.WAIT)
        self.assertIs(self.gate.evaluate(phone_block(rover_pose_aligned=False)), TrackingDecision.WAIT)

    def test_wait_on_missing_block(self):
        self.assertIs(self.gate.evaluate(None), TrackingDecision.WAIT)
        self.assertIs(self.gate.evaluate({}), TrackingDecision.WAIT)

    def test_alignment_not_required_when_disabled(self):
        gate = TrackingGate(require_aligned=False)
        self.assertIs(gate.evaluate(phone_block(rover_pose_aligned=False)), TrackingDecision.EXECUTE)


class PoseErrorTests(unittest.TestCase):
    def test_pose_error_components(self):
        d, yaw = pose_error(Pose(0, 0, 0), Pose(0.3, 0.4, 10))
        self.assertAlmostEqual(d, 0.5, places=6)
        self.assertAlmostEqual(yaw, 10.0, places=6)

    def test_is_pose_jump_thresholds(self):
        predicted = Pose(0, 0, 0)
        # Small drift: not a jump.
        self.assertFalse(is_pose_jump(predicted, Pose(0.1, 0.0, 5), 1.0, 45.0))
        # Big translation: jump.
        self.assertTrue(is_pose_jump(predicted, Pose(2.0, 0.0, 0), 1.0, 45.0))
        # Big rotation: jump.
        self.assertTrue(is_pose_jump(predicted, Pose(0.0, 0.0, 90), 1.0, 45.0))


class ExecutionErrorLogTests(unittest.TestCase):
    def test_record_computes_error_and_keeps_meta(self):
        log = ExecutionErrorLog()
        rec = log.record(
            2, Pose(0, 0, 0), Pose(0.3, 0.4, 8),
            phone_timestamp_s=12.5, tracking_normal=True,
        )
        self.assertAlmostEqual(rec.distance_error_m, 0.5, places=6)
        self.assertAlmostEqual(rec.yaw_error_deg, 8.0, places=6)
        self.assertEqual(rec.cycle, 2)
        self.assertEqual(rec.phone_timestamp_s, 12.5)
        self.assertTrue(rec.tracking_normal)

    def test_summary_empty_and_populated(self):
        log = ExecutionErrorLog()
        self.assertEqual(log.summary()["count"], 0)
        log.record(0, Pose(0, 0, 0), Pose(0.1, 0.0, 2))
        log.record(1, Pose(0, 0, 0), Pose(0.3, 0.0, 6))
        summary = log.summary()
        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["max_distance_error_m"], 0.3, places=6)
        self.assertAlmostEqual(summary["mean_yaw_error_deg"], 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
