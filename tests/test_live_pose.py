import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from digital_twin.live_pose import (
    POSE_NAME,
    pose_from_payload,
    pose_path,
    read_pose,
    write_pose,
)
from digital_twin.models import Pose


class LivePoseTransportTest(unittest.TestCase):
    def test_read_missing_returns_none(self):
        with TemporaryDirectory() as d:
            self.assertIsNone(read_pose(d))

    def test_round_trip_preserves_pose_and_metadata(self):
        with TemporaryDirectory() as d:
            write_pose(d, Pose(1.5, -2.25, 90.0), seq=7, timestamp_s=1234.5,
                       tracking_normal=True, z=0.3)
            payload = read_pose(d)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["seq"], 7)
            self.assertEqual(payload["timestamp_s"], 1234.5)
            self.assertTrue(payload["tracking_normal"])
            self.assertAlmostEqual(payload["z"], 0.3)
            pose = pose_from_payload(payload)
            self.assertAlmostEqual(pose.x, 1.5)
            self.assertAlmostEqual(pose.y, -2.25)
            self.assertAlmostEqual(pose.yaw_deg, 90.0)

    def test_write_is_atomic_no_temp_left_behind(self):
        with TemporaryDirectory() as d:
            write_pose(d, Pose(0.0, 0.0, 0.0), seq=1, timestamp_s=0.0)
            files = {p.name for p in Path(d).iterdir()}
            self.assertEqual(files, {POSE_NAME})  # no .tmp residue

    def test_latest_write_wins(self):
        with TemporaryDirectory() as d:
            write_pose(d, Pose(1.0, 1.0, 0.0), seq=1, timestamp_s=1.0)
            write_pose(d, Pose(2.0, 2.0, 45.0), seq=2, timestamp_s=2.0)
            payload = read_pose(d)
            self.assertEqual(payload["seq"], 2)
            self.assertAlmostEqual(payload["x"], 2.0)
            self.assertAlmostEqual(payload["yaw_deg"], 45.0)

    def test_creates_missing_directory(self):
        with TemporaryDirectory() as d:
            nested = Path(d) / "map" / "out"
            write_pose(nested, Pose(3.0, 4.0, 10.0), seq=1, timestamp_s=0.0)
            self.assertTrue(pose_path(nested).exists())

    def test_read_tolerates_corrupt_file(self):
        with TemporaryDirectory() as d:
            pose_path(d).write_text("{ not valid json")
            self.assertIsNone(read_pose(d))

    def test_tracking_not_normal_flag_round_trips(self):
        with TemporaryDirectory() as d:
            write_pose(d, Pose(0.0, 0.0, 0.0), seq=5, timestamp_s=9.0,
                       tracking_normal=False)
            self.assertFalse(read_pose(d)["tracking_normal"])


if __name__ == "__main__":
    unittest.main()
