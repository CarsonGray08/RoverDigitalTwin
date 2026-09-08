import unittest

from digital_twin.occupancy import (
    CellState,
    FloorPlane,
    OccupancyConfig,
    OccupancyGrid,
)


def grid(floor_height=0.0, **cfg):
    return OccupancyGrid(floor=FloorPlane.level(floor_height), config=OccupancyConfig(**cfg))


def column(x, y, z, n):
    return [(x, y, z)] * n


class ClassificationTests(unittest.TestCase):
    def test_floor_obstacle_unknown_at_zero_floor(self):
        g = grid(0.0)
        g.ingest_points(column(0.0, 0.0, 0.0, 5))   # floor
        g.ingest_points(column(1.0, 0.0, 0.3, 5))   # obstacle (0.3 above)
        self.assertIs(g.state(g.cell_of(0.0, 0.0)), CellState.FLOOR)
        self.assertIs(g.state(g.cell_of(1.0, 0.0)), CellState.OCCUPIED)
        self.assertIs(g.state(g.cell_of(9.0, 9.0)), CellState.UNKNOWN)

    def test_plane_relative_for_raised_floor(self):
        # Floor raised to z=0.5. Points at the floor must read FLOOR (not
        # OCCUPIED), and points 0.3 m above it OCCUPIED. A fixed (0.12, 0.45)
        # world-Z band would mislabel both.
        g = grid(0.5)
        g.ingest_points(column(0.0, 0.0, 0.5, 5))   # on the raised floor
        g.ingest_points(column(1.0, 0.0, 0.8, 5))   # 0.3 m above it
        self.assertIs(g.state(g.cell_of(0.0, 0.0)), CellState.FLOOR)
        self.assertIs(g.state(g.cell_of(1.0, 0.0)), CellState.OCCUPIED)

    def test_sloped_floor_classified_as_floor(self):
        # Plane tilted ~5.7 deg (normal slightly off +Z). Points lying on the
        # slope should read FLOOR across a range of x.
        import math

        theta = math.radians(5.7)
        normal = (math.sin(theta), 0.0, math.cos(theta))
        floor = FloorPlane(normal=normal, offset=0.0)
        g = OccupancyGrid(floor=floor, config=OccupancyConfig())
        for x in (-1.0, 0.0, 1.0, 2.0):
            z = -math.tan(theta) * x  # point on the plane: n.p = 0
            g.ingest_points([(x, 5.0, z)] * 5)
            self.assertIs(g.state(g.cell_of(x, 5.0)), CellState.FLOOR)

    def test_single_stray_point_is_not_an_obstacle(self):
        g = grid(0.0, min_obstacle_points=3)
        g.ingest_points([(1.0, 1.0, 0.4)])  # one stray high point
        self.assertIs(g.state(g.cell_of(1.0, 1.0)), CellState.UNKNOWN)

    def test_obstacle_on_floor_reads_occupied(self):
        g = grid(0.0)
        cell_pts = column(2.0, 2.0, 0.0, 4) + column(2.0, 2.0, 0.3, 4)
        g.ingest_points(cell_pts)
        self.assertIs(g.state(g.cell_of(2.0, 2.0)), CellState.OCCUPIED)

    def test_gap_and_ceiling_points_ignored(self):
        g = grid(0.0, floor_band_m=0.05, obstacle_min_m=0.10, obstacle_max_m=2.0)
        g.ingest_points(column(0.0, 0.0, 0.07, 5))  # in the gap
        g.ingest_points(column(1.0, 0.0, 2.5, 5))   # above ceiling cutoff
        self.assertIs(g.state(g.cell_of(0.0, 0.0)), CellState.UNKNOWN)
        self.assertIs(g.state(g.cell_of(1.0, 0.0)), CellState.UNKNOWN)


class RayCarveTests(unittest.TestCase):
    def test_carve_marks_intervening_cells_free(self):
        g = grid(0.0)
        g.carve_ray((0.0, 0.0), (1.5, 0.0))
        self.assertIs(g.state(g.cell_of(0.45, 0.0)), CellState.FREE)

    def test_carve_does_not_override_obstacle(self):
        g = grid(0.0)
        g.ingest_points(column(0.45, 0.0, 0.3, 5))  # obstacle mid-ray
        g.carve_ray((0.0, 0.0), (1.5, 0.0))
        self.assertIs(g.state(g.cell_of(0.45, 0.0)), CellState.OCCUPIED)


class ProxyAndSummaryTests(unittest.TestCase):
    def test_row_of_occupied_cells_merges_into_one_box(self):
        g = grid(0.0)
        for i in range(3):  # three adjacent cells in a row, all obstacles
            g.ingest_points(column(i * 0.3 + 0.05, 0.05, 0.3, 5))
        proxies = g.obstacle_proxies()
        self.assertEqual(len(proxies), 1)
        box = proxies[0]
        self.assertAlmostEqual(box.min_x, 0.0, places=6)
        self.assertAlmostEqual(box.max_x, 0.9, places=6)  # 3 cells * 0.3
        self.assertGreater(box.max_z, box.min_z)

    def test_traversable_and_known_sets(self):
        g = grid(0.0)
        g.ingest_points(column(0.0, 0.0, 0.0, 5))   # floor
        g.ingest_points(column(1.0, 0.0, 0.3, 5))   # obstacle
        g.carve_ray((0.0, 0.0), (0.0, 1.5))         # free column
        floor_cell = g.cell_of(0.0, 0.0)
        obstacle_cell = g.cell_of(1.0, 0.0)
        self.assertIn(floor_cell, g.traversable_cells())
        self.assertNotIn(obstacle_cell, g.traversable_cells())
        self.assertEqual(g.known_cells(), set(g.classify()))

    def test_counts(self):
        g = grid(0.0)
        g.ingest_points(column(0.0, 0.0, 0.0, 5))
        g.ingest_points(column(1.0, 0.0, 0.3, 5))
        counts = g.counts()
        self.assertEqual(counts["floor"], 1)
        self.assertEqual(counts["occupied"], 1)


if __name__ == "__main__":
    unittest.main()
