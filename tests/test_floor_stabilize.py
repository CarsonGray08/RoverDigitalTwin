from __future__ import annotations

import unittest

import numpy as np

from digital_twin.floor_stabilize import stabilize_floor_mesh, stabilize_floor_vertices
from digital_twin.scan_loop_eval import evaluate_mesh


class FloorStabilizeTests(unittest.TestCase):
    def test_snaps_duplicate_floor_band_to_one_plane(self):
        xs = np.linspace(-1.0, 1.0, 11)
        ys = np.linspace(-1.0, 1.0, 11)
        xx, yy = np.meshgrid(xs, ys)
        base = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        duplicate = base.copy()
        duplicate[:, 2] = 0.06
        obstacle = np.asarray([[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.0, 0.1, 0.4]])
        verts = np.vstack([base, duplicate, obstacle])

        stabilized = stabilize_floor_vertices(verts, floor_band_m=0.08)
        metrics = evaluate_mesh(stabilized, np.zeros((0, 3), dtype=np.int64))

        self.assertLess(metrics.floor_height_p95_m, 1e-6)
        self.assertLess(metrics.largest_floor_height_discontinuity_m, 1e-6)
        np.testing.assert_allclose(stabilized[-3:], obstacle)

    def test_drops_tiny_disconnected_floor_island(self):
        main = np.asarray([
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.05, 0.05, 0.0],
            [0.10, 0.0, 0.0],
            [0.10, 0.05, 0.0],
        ], dtype=np.float32)
        island = np.asarray([
            [3.0, 3.0, 0.0],
            [3.05, 3.0, 0.0],
            [3.0, 3.05, 0.0],
        ], dtype=np.float32)
        verts = np.vstack([main, island])
        faces = np.asarray(
            [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3], [6, 7, 8]],
            dtype=np.int32,
        )
        colors = np.zeros((len(verts), 3), dtype=np.uint8)

        out_v, out_f, out_c = stabilize_floor_mesh(
            verts, faces, colors,
            grid_cell_m=0.05,
            min_floor_component_area_m2=0.01,
        )

        self.assertEqual(len(out_f), 4)
        self.assertEqual(len(out_v), 6)
        self.assertEqual(len(out_c), 6)


if __name__ == "__main__":
    unittest.main()
