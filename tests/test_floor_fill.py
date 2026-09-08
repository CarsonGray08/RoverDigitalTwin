from __future__ import annotations

import unittest

import numpy as np

from digital_twin.floor_fill import (
    apply_floor_disc_patch,
    build_floor_disc_patch,
    fill_floor_disc,
)
from digital_twin.terrain_heightmap import TerrainHeightmap


def _floor_with_hole(z_at=lambda x, y: 0.0, extent=1.5, step=0.05, hole_radius=0.45, center=(0.0, 0.0)):
    """A dense floor grid with a circular hole punched out around ``center``."""
    cx, cy = center
    xs = np.arange(-extent, extent + step, step)
    ys = np.arange(-extent, extent + step, step)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    pts[:, 2] = z_at(pts[:, 0], pts[:, 1])
    keep = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > hole_radius
    return pts[keep]


class FloorFillTests(unittest.TestCase):
    def test_fills_flat_floor_hole_flush(self):
        verts = _floor_with_hole()
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)
        faces = np.zeros((0, 3), dtype=np.int64)

        out_v, out_f, out_c = fill_floor_disc(
            verts, faces, colors, center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05)

        # New geometry was added inside the former hole, at floor height.
        self.assertGreater(len(out_v), len(verts))
        self.assertGreater(len(out_f), 0)
        new_v = out_v[len(verts):]
        # The patch stays local to the hole (flood-fill stops at the surrounding
        # floor); boundary corner nodes may overlap it by a cell or two, which is
        # what makes the seam flush.
        inside = np.hypot(new_v[:, 0], new_v[:, 1]) <= 0.45 + 2 * 0.05 + 1e-6
        self.assertTrue(inside.all())
        np.testing.assert_allclose(new_v[:, 2], 0.0, atol=1e-5)
        self.assertEqual(len(out_c), len(out_v))

        # The center is now observed terrain at the surrounding floor height.
        hm = TerrainHeightmap.from_points(out_v, cell_m=0.05)
        height = hm.max_height_in_footprint(0.0, 0.0, 0.1)
        self.assertIsNotNone(height)
        self.assertAlmostEqual(height, 0.0, places=4)

    def test_center_unobserved_before_fill(self):
        verts = _floor_with_hole()
        hm = TerrainHeightmap.from_points(verts, cell_m=0.05)
        # Sanity: the hole really is a hole before we fill it.
        self.assertIsNone(hm.max_height_in_footprint(0.0, 0.0, 0.1))

    def test_patch_follows_tilted_floor_plane(self):
        slope = 0.1  # 10 cm per metre in +x
        verts = _floor_with_hole(z_at=lambda x, y: slope * x)
        colors = np.full((len(verts), 3), 90, dtype=np.uint8)
        out_v, _out_f, _out_c = fill_floor_disc(
            verts, np.zeros((0, 3), np.int64), colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05)
        new_v = out_v[len(verts):]
        # Patch z should track the plane z = slope*x, not sit flat.
        np.testing.assert_allclose(new_v[:, 2], slope * new_v[:, 0], atol=2e-3)

    def test_patch_color_matches_surrounding_floor(self):
        verts = _floor_with_hole()
        colors = np.full((len(verts), 3), 0, dtype=np.uint8)
        colors[:, 0] = 200  # reddish floor
        out_v, _out_f, out_c = fill_floor_disc(
            verts, np.zeros((0, 3), np.int64), colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05)
        patch_colors = out_c[len(verts):]
        np.testing.assert_array_equal(
            patch_colors[0], np.array([200, 0, 0], dtype=np.uint8))

    def test_does_not_double_layer_existing_obstacle(self):
        # A floor hole, but with a small obstacle sitting inside the disc.
        verts = _floor_with_hole()
        obstacle = np.array([[0.0, 0.0, 0.3], [0.04, 0.0, 0.3], [0.0, 0.04, 0.3]])
        verts = np.vstack([verts, obstacle])
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)
        out_v, out_f, _out_c = fill_floor_disc(
            verts, np.zeros((0, 3), np.int64), colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05)
        new_f = out_f[(out_f >= len(verts)).all(axis=1)]
        # No patch *triangle* should cover the occupied obstacle cell [0,0.05).
        centroids = out_v[new_f].mean(axis=1)
        covers_obstacle = (
            (centroids[:, 0] >= 0.0) & (centroids[:, 0] < 0.05)
            & (centroids[:, 1] >= 0.0) & (centroids[:, 1] < 0.05)
        )
        self.assertFalse(covers_obstacle.any())

    def test_artifact_tolerant_fill_does_not_leave_fragment_shadow(self):
        verts = _floor_with_hole()
        # A vertical strip of bad reconstruction inside the rover footprint used
        # to act like a wall in the 2-D flood fill, leaving half the hole open.
        ys = np.arange(-0.35, 0.36, 0.05)
        zs = np.arange(0.18, 0.55, 0.08)
        yy, zz = np.meshgrid(ys, zs)
        fragments = np.column_stack([
            np.zeros(yy.size),
            yy.ravel(),
            zz.ravel(),
        ])
        verts = np.vstack([verts, fragments])
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)

        out_v, _out_f, _out_c = fill_floor_disc(
            verts, np.zeros((0, 3), np.int64), colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05,
            preserve_obstacles=False)

        hm = TerrainHeightmap.from_points(out_v, cell_m=0.05)
        for x in (-0.3, -0.15, 0.0, 0.15, 0.3):
            for y in (-0.3, -0.15, 0.0, 0.15, 0.3):
                height = hm.max_height_in_footprint(x, y, 0.025)
                self.assertIsNotNone(height, (x, y))
                self.assertAlmostEqual(height, 0.0, places=4)

    def test_saved_patch_reapplies_to_fresh_mesh_export(self):
        verts = _floor_with_hole()
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)
        faces = np.zeros((0, 3), np.int64)
        patch = build_floor_disc_patch(
            verts, faces, colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05,
            preserve_obstacles=False)
        self.assertIsNotNone(patch)

        # A later raw nvblox extraction can contain new reconstruction debris in
        # the saved footprint. Applying the saved overlay should drop it without
        # re-running the adaptive hole fill.
        fresh_v = np.vstack([verts, np.array([[0.0, 0.0, 0.35]], dtype=np.float32)])
        fresh_c = np.full((len(fresh_v), 3), 120, dtype=np.uint8)
        out_v, _out_f, _out_c = apply_floor_disc_patch(
            fresh_v, faces, fresh_c, patch, remove_original_hole=True)

        hm = TerrainHeightmap.from_points(out_v, cell_m=0.05)
        height = hm.max_height_in_footprint(0.0, 0.0, 0.025)
        self.assertIsNotNone(height)
        self.assertAlmostEqual(height, 0.0, places=4)

    def test_fill_can_overlap_floor_seam(self):
        verts = _floor_with_hole(hole_radius=0.45)
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)

        out_v, _out_f, _out_c = fill_floor_disc(
            verts, np.zeros((0, 3), np.int64), colors,
            center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05,
            seam_overlap_cells=2)

        new_v = out_v[len(verts):]
        radii = np.hypot(new_v[:, 0], new_v[:, 1])
        self.assertGreater(float(radii.max()), 0.45 + 0.05)

    def test_noop_without_surrounding_floor(self):
        # Only a few stray points: not enough ring to trust a fill height.
        verts = np.array([[2.0, 2.0, 0.0], [2.1, 2.0, 0.0], [2.0, 2.1, 0.0]])
        colors = np.full((len(verts), 3), 120, dtype=np.uint8)
        faces = np.zeros((0, 3), dtype=np.int64)
        out_v, out_f, out_c = fill_floor_disc(
            verts, faces, colors, center=(0.0, 0.0), max_radius_m=0.9, cell_m=0.05)
        self.assertEqual(len(out_v), len(verts))
        self.assertEqual(len(out_f), 0)

    def test_noop_on_empty_mesh(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)
        out_v, out_f, out_c = fill_floor_disc(
            empty, faces, empty, center=(0.0, 0.0), max_radius_m=0.9)
        self.assertEqual(len(out_v), 0)
        self.assertEqual(len(out_f), 0)


def _boundary_edge_count(vertices, faces, mask_fn=None):
    from collections import Counter
    und = Counter()
    for tri in np.asarray(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            und[(int(min(a, b)), int(max(a, b)))] += 1
    n = 0
    V = np.asarray(vertices, dtype=float)
    for (a, b), k in und.items():
        if k != 1:
            continue
        if mask_fn is None or mask_fn((V[a] + V[b]) / 2):
            n += 1
    return n


def _unwelded_floor_with_hole(hole_radius=0.4, extent=1.0, step=0.05, max_edge=0.12):
    """A triangulated floor with a hole, then vertices duplicated per face to
    mimic nvblox's unwelded output (every face has its own vertices)."""
    from scipy.spatial import Delaunay
    xs = np.arange(-extent, extent + step, step)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > hole_radius]
    tri = Delaunay(pts[:, :2])
    s = tri.simplices
    e = np.maximum.reduce([
        np.linalg.norm(pts[s[:, 0], :2] - pts[s[:, 1], :2], axis=1),
        np.linalg.norm(pts[s[:, 1], :2] - pts[s[:, 2], :2], axis=1),
        np.linalg.norm(pts[s[:, 2], :2] - pts[s[:, 0], :2], axis=1),
    ])
    faces = s[e <= max_edge]
    verts = pts[faces].reshape(-1, 3)            # unwelded: 3 fresh verts per face
    f = np.arange(len(verts)).reshape(-1, 3)
    return verts.astype(np.float32), f.astype(np.int64)


class WeldMeshTests(unittest.TestCase):
    def test_weld_merges_duplicates_and_remaps(self):
        from digital_twin.floor_fill import weld_mesh
        v, f = _unwelded_floor_with_hole()
        c = np.full((len(v), 3), 100, dtype=np.uint8)
        vw, fw, cw = weld_mesh(v, f, c, tol_m=0.0005)
        self.assertLess(len(vw), len(v))          # duplicates merged
        self.assertEqual(len(cw), len(vw))
        self.assertTrue((np.asarray(fw) < len(vw)).all())
        # welding turns shared edges from boundary into interior: far fewer.
        self.assertLess(_boundary_edge_count(vw, fw), _boundary_edge_count(v, f))


class DelaunayFillTests(unittest.TestCase):
    def test_remesh_fills_hole_continuously(self):
        from digital_twin.floor_fill import fill_floor_disc_delaunay
        v, f = _unwelded_floor_with_hole(hole_radius=0.4)
        c = np.full((len(v), 3), 100, dtype=np.uint8)
        ov, of, oc = fill_floor_disc_delaunay(
            v, f, c, center=(0.0, 0.0), max_radius_m=0.8, cell_m=0.05)
        self.assertLess(len(ov), len(v))           # welded
        self.assertEqual(len(oc), len(ov))
        # center now observed at floor height
        hm = TerrainHeightmap.from_points(ov, cell_m=0.05)
        h = hm.max_height_in_footprint(0.0, 0.0, 0.1)
        self.assertIsNotNone(h)
        self.assertAlmostEqual(h, 0.0, places=4)
        # interior of the filled disc is watertight (no boundary edges)
        interior = _boundary_edge_count(
            ov, of, mask_fn=lambda p: np.hypot(p[0], p[1]) < 0.5)
        self.assertEqual(interior, 0)

    def test_noop_returns_welded_when_no_hole(self):
        from digital_twin.floor_fill import fill_floor_disc_delaunay
        # solid floor, no hole, denser than the fill cell (as real nvblox floor
        # is) so every fill cell is occupied and nothing is fabricated.
        xs = np.arange(-0.8, 0.81, 0.02)
        xx, yy = np.meshgrid(xs, xs)
        v = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).astype(np.float32)
        f = np.zeros((0, 3), dtype=np.int64)
        c = np.full((len(v), 3), 100, dtype=np.uint8)
        ov, of, oc = fill_floor_disc_delaunay(
            v, f, c, center=(0.0, 0.0), max_radius_m=0.5, cell_m=0.05)
        # no fabricated geometry where the floor is already solid
        self.assertEqual(len(ov), len(v))


if __name__ == "__main__":
    unittest.main()
