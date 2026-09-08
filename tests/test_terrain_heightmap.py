import unittest

import numpy as np

from digital_twin.terrain_heightmap import TerrainHeightmap


class TerrainHeightmapTest(unittest.TestCase):
    def test_empty_points(self):
        hm = TerrainHeightmap.from_points(np.zeros((0, 3)))
        self.assertTrue(hm.is_empty)
        self.assertIsNone(hm.max_height_in_footprint(0.0, 0.0, 0.5))

    def test_returns_max_height_under_footprint(self):
        # flat z=0 sheet plus one tall point (a rock) at (1, 1, 0.3)
        xs, ys = np.meshgrid(np.linspace(0, 2, 21), np.linspace(0, 2, 21))
        pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
        pts = np.vstack([pts, [[1.0, 1.0, 0.3]]])
        hm = TerrainHeightmap.from_points(pts, cell_m=0.1)
        # footprint over the rock sees the rock
        self.assertAlmostEqual(hm.max_height_in_footprint(1.0, 1.0, 0.15), 0.3, places=5)
        # footprint away from the rock sees only the flat sheet
        self.assertAlmostEqual(hm.max_height_in_footprint(0.3, 0.3, 0.1), 0.0, places=5)

    def test_footprint_outside_map_returns_none(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]])
        hm = TerrainHeightmap.from_points(pts, cell_m=0.1)
        self.assertIsNone(hm.max_height_in_footprint(100.0, 100.0, 0.2))

    def test_radius_expands_the_query_window(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])  # bump 0.5 m away
        hm = TerrainHeightmap.from_points(pts, cell_m=0.1)
        # small footprint at origin misses the bump
        self.assertAlmostEqual(hm.max_height_in_footprint(0.0, 0.0, 0.05), 0.0, places=5)
        # large footprint reaches it
        self.assertAlmostEqual(hm.max_height_in_footprint(0.0, 0.0, 0.6), 0.5, places=5)

    def test_negative_terrain_handled(self):
        pts = np.array([[0.0, 0.0, -0.2], [0.1, 0.0, -0.1]])
        hm = TerrainHeightmap.from_points(pts, cell_m=0.1)
        self.assertAlmostEqual(hm.max_height_in_footprint(0.05, 0.0, 0.2), -0.1, places=5)


if __name__ == "__main__":
    unittest.main()
