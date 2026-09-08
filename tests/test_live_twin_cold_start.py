"""Cold-start state, annular coverage, and provisional support are pure Python.

These tests intentionally exercise the mapper/validator filesystem contract,
not Isaac, so they run on development machines without the hardware stack.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from digital_twin.bootstrap import BootstrapGate, cells_within_radius
from digital_twin.live_pose import read_pose, write_pose
from digital_twin.live_twin_session import (
    derive_live_twin_status,
    new_session,
    read_session,
    write_session,
)
from digital_twin.mesh_tiles import load_manifest
from digital_twin.models import Pose
from digital_twin.terrain_heightmap import TerrainHeightmap
from digital_twin.traversability import (
    INSUFFICIENT_CLEARANCE,
    TerrainSample,
    TraversabilityLimits,
    classify_sample,
)


class ColdStartTransportTest(unittest.TestCase):
    def test_new_session_directory_does_not_consume_parent_stale_transport(self):
        with TemporaryDirectory() as root:
            stale = Path(root) / "rover_pose.json"
            write_pose(root, Pose(9.0, 9.0, 0.0), seq=99, timestamp_s=1.0)
            (Path(root) / "manifest.json").write_text('{"rev": 99, "tiles": {}}')
            session_dir = Path(root) / "session-new"
            write_session(session_dir, new_session("new"))
            self.assertTrue(stale.exists())  # launch isolation is non-destructive
            self.assertIsNone(read_pose(session_dir))
            self.assertEqual(load_manifest(session_dir), {})
            self.assertEqual(read_session(session_dir)["session_id"], "new")

    def test_cold_start_reports_awaiting_scan(self):
        status = derive_live_twin_status(
            session=new_session("fresh"), map_dir="/tmp/session", pose=None,
            terrain_tiles=0, map_revision=0, baseline_min_terrain_tiles=1,
            coverage={"complete": False}, footprint_ground_source="unknown")
        self.assertEqual(status["phase"], "awaiting_scan")
        self.assertFalse(status["tracking_normal"])
        self.assertEqual(status["t_align"], "absent")
        self.assertEqual(status["terrain_tiles"], 0)
        self.assertFalse(status["baseline_ready"])

    def test_first_normal_pose_is_baseline_mapping_not_tracking(self):
        session = new_session("fresh")
        session["t_align"] = "present"
        status = derive_live_twin_status(
            session=session, map_dir="/tmp/session",
            pose={"tracking_normal": True, "x": 0, "y": 0}, terrain_tiles=0,
            map_revision=0, baseline_min_terrain_tiles=1,
            coverage={"complete": False}, footprint_ground_source="unknown")
        self.assertEqual(status["phase"], "baseline_mapping")
        self.assertFalse(status["baseline_ready"])


class AnnularCoverageAndSupportTest(unittest.TestCase):
    def test_annular_360_coverage_can_complete_while_center_stays_excluded(self):
        gate = BootstrapGate(radius_m=1.5, cell_m=0.25, coverage_threshold=0.95,
                             exclusion_radius_m=0.50)
        observed = set(cells_within_radius((0.0, 0.0), 1.5, 0.25))
        # Simulates a full manual rotation: every usable annular cell is seen,
        # and the body footprint remains intentionally unknown.
        self.assertTrue(gate.update(observed, (0.0, 0.0)))
        status = gate.status()
        self.assertGreater(status["coverage_fraction"], 0.95)
        self.assertEqual(status["coverage_annulus_inner_radius_m"], 0.50)

    def test_ring_support_is_inferred_not_observed_then_is_replaced_normally(self):
        axis = np.linspace(-1.0, 1.0, 21)
        xx, yy = np.meshgrid(axis, axis)
        rr = np.hypot(xx, yy)
        ring = (rr >= 0.45) & (rr <= 0.9)
        points = np.column_stack((xx[ring], yy[ring], np.full(ring.sum(), 0.02)))
        hm = TerrainHeightmap.from_points(points, cell_m=0.1)
        inferred = hm.ground_support(0.0, 0.0, 0.30, min_ring_cells=8)
        self.assertEqual(inferred.source, "inferred_support")
        self.assertAlmostEqual(inferred.height_m, 0.02, places=3)
        observed_points = np.vstack((points, [[0.0, 0.0, 0.02], [0.1, 0.0, 0.02]]))
        observed = TerrainHeightmap.from_points(observed_points, cell_m=0.1).ground_support(
            0.0, 0.0, 0.30)
        self.assertEqual(observed.source, "observed")
        self.assertAlmostEqual(observed.height_m, inferred.height_m, places=3)

    def test_unknown_terrain_never_becomes_infinite_or_clear(self):
        hm = TerrainHeightmap.from_points(np.zeros((0, 3)))
        support = hm.ground_support(0.0, 0.0, 0.3)
        self.assertEqual(support.source, "unknown")
        self.assertIsNone(support.height_m)
        self.assertEqual(
            classify_sample(TerrainSample(0, 0, 0, float("-inf")), TraversabilityLimits()),
            INSUFFICIENT_CLEARANCE,
        )

