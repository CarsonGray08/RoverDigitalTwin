import unittest

from digital_twin.bootstrap import (
    BootstrapGate,
    cells_within_radius,
    coverage_fraction,
    navigation_blocked_by_bootstrap,
    spin_in_place_plan,
)
from digital_twin.models import CommandName


class CoverageTests(unittest.TestCase):
    def test_disk_cell_count_is_reasonable(self):
        cells = list(cells_within_radius((0.0, 0.0), 1.5, 0.3))
        # A disk of radius 1.5 with 0.3 m cells: ~pi*r^2/cell^2 = ~78 cells.
        self.assertGreater(len(cells), 60)
        self.assertLess(len(cells), 100)
        # No duplicates.
        self.assertEqual(len(cells), len(set(cells)))

    def test_full_coverage(self):
        all_cells = set(cells_within_radius((0.0, 0.0), 1.5, 0.3))
        result = coverage_fraction(all_cells, (0.0, 0.0), 1.5, 0.3)
        self.assertEqual(result.fraction, 1.0)
        self.assertEqual(result.known_cells, result.total_cells)

    def test_zero_coverage(self):
        result = coverage_fraction(set(), (0.0, 0.0), 1.5, 0.3)
        self.assertEqual(result.fraction, 0.0)
        self.assertGreater(result.total_cells, 0)

    def test_partial_coverage_is_fraction(self):
        all_cells = list(cells_within_radius((0.0, 0.0), 1.5, 0.3))
        half = set(all_cells[: len(all_cells) // 2])
        result = coverage_fraction(half, (0.0, 0.0), 1.5, 0.3)
        self.assertAlmostEqual(result.fraction, len(half) / len(all_cells), places=6)

    def test_coverage_follows_rover_position(self):
        # Known cells clustered around origin; rover far away -> low coverage.
        near = set(cells_within_radius((0.0, 0.0), 1.5, 0.3))
        result = coverage_fraction(near, (10.0, 10.0), 1.5, 0.3)
        self.assertEqual(result.known_cells, 0)


class BootstrapGateTests(unittest.TestCase):
    def test_latches_complete_once_threshold_reached(self):
        gate = BootstrapGate(radius_m=1.5, cell_m=0.3, coverage_threshold=0.6)
        all_cells = list(cells_within_radius((0.0, 0.0), 1.5, 0.3))

        # Below threshold: not complete, navigation disabled.
        few = set(all_cells[: int(len(all_cells) * 0.3)])
        self.assertFalse(gate.update(few, (0.0, 0.0)))
        self.assertFalse(gate.navigation_enabled)

        # Cross threshold: complete.
        many = set(all_cells[: int(len(all_cells) * 0.7)])
        self.assertTrue(gate.update(many, (0.0, 0.0)))
        self.assertTrue(gate.navigation_enabled)

        # Latched: a later low-coverage observation does NOT re-disable.
        self.assertTrue(gate.update(set(), (0.0, 0.0)))
        self.assertTrue(gate.navigation_enabled)

    def test_status_shape(self):
        gate = BootstrapGate()
        gate.update(set(), (0.0, 0.0))
        status = gate.status()
        self.assertFalse(status["complete"])
        self.assertFalse(status["navigation_enabled"])
        self.assertIn("coverage_fraction", status)
        self.assertEqual(status["coverage_threshold"], gate.coverage_threshold)


class NavigationGuardTests(unittest.TestCase):
    def test_blocked_only_when_enabled_and_incomplete(self):
        self.assertFalse(navigation_blocked_by_bootstrap({}))  # no block
        self.assertFalse(
            navigation_blocked_by_bootstrap({"bootstrap": {"enabled": False, "navigation_enabled": False}})
        )
        self.assertTrue(
            navigation_blocked_by_bootstrap({"bootstrap": {"enabled": True, "navigation_enabled": False}})
        )
        self.assertFalse(
            navigation_blocked_by_bootstrap({"bootstrap": {"enabled": True, "navigation_enabled": True}})
        )


class SpinPlanTests(unittest.TestCase):
    def test_covers_full_turn_with_settle_stops(self):
        plan = spin_in_place_plan(total_deg=360.0, step_deg=30.0)
        turns = [s for s in plan if s.command == CommandName.LEFT]
        stops = [s for s in plan if s.command == CommandName.STOP]
        self.assertEqual(len(turns), 12)
        self.assertEqual(len(stops), 12)  # one settle after each turn
        # Identity degrees->distance: distances sum to ~360.
        self.assertAlmostEqual(sum(t.distance for t in turns), 360, delta=1)

    def test_calibration_maps_degrees_to_distance(self):
        plan = spin_in_place_plan(
            total_deg=180.0, step_deg=90.0, settle=False, degrees_to_distance=lambda d: d * 100
        )
        self.assertEqual([s.distance for s in plan], [9000, 9000])
        self.assertTrue(all(s.command == CommandName.LEFT for s in plan))

    def test_rejects_bad_args(self):
        with self.assertRaises(ValueError):
            spin_in_place_plan(total_deg=0.0)


if __name__ == "__main__":
    unittest.main()
