"""End-to-end test of the mapper->validator filesystem contract the live twin
relies on: the mapper writes a per-tile mesh transport + a rover pose into one
map dir; the validator reads the manifest, loads the named tiles, and reads the
pose. No GPU/Isaac needed -- this is the pure data boundary between the two
processes (the parts that CAN be verified off-hardware)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from digital_twin.live_pose import read_pose, write_pose
from digital_twin.mesh_tiles import load_manifest, tile_npz_path, write_tiles
from digital_twin.models import Pose
from digital_twin.terrain_denoise import denoise_terrain_mesh


def _grid_mesh(side: int, spacing: float):
    """A flat triangulated grid spanning (side-1)*spacing metres in x and y."""
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1).astype(np.float32)
    faces = []
    for j in range(side - 1):
        for i in range(side - 1):
            v = j * side + i
            faces.append([v, v + 1, v + side])
            faces.append([v + 1, v + side + 1, v + side])
    faces = np.asarray(faces, dtype=np.int32)
    colors = np.full((len(verts), 3), 128, dtype=np.uint8)
    return verts, faces, colors


class LiveTwinTransportContractTest(unittest.TestCase):
    def test_consumer_can_reconstruct_tiles_from_manifest(self):
        verts, faces, colors = _grid_mesh(30, 0.1)  # ~2.9 m -> spans several 1 m tiles
        with TemporaryDirectory() as d:
            n, written, removed = write_tiles(d, verts, faces, colors,
                                              tile_size=1.0, state={}, rev=1)
            self.assertGreater(n, 1)        # actually tiled
            self.assertEqual(written, n)    # first write: all tiles new
            self.assertEqual(removed, 0)

            manifest = load_manifest(d)
            self.assertEqual(manifest["rev"], 1)
            self.assertEqual(set(manifest["tiles"]), set(self._tile_names(d)))

            for name in manifest["tiles"]:
                data = np.load(tile_npz_path(d, name))
                tv, tf, tc = data["vertices"], data["faces"], data["colors"]
                self.assertEqual(tv.shape[1], 3)
                self.assertEqual(tf.shape[1], 3)
                self.assertEqual(tc.shape, (len(tv), 3))
                # Faces must be LOCAL indices into this tile's vertices (what the
                # USD sink assumes when authoring a per-tile Mesh prim).
                if len(tf):
                    self.assertLess(int(tf.max()), len(tv))
                    self.assertGreaterEqual(int(tf.min()), 0)

    def test_incremental_rewrite_skips_unchanged_tiles(self):
        verts, faces, colors = _grid_mesh(20, 0.15)
        with TemporaryDirectory() as d:
            state: dict = {}
            write_tiles(d, verts, faces, colors, tile_size=1.0, state=state, rev=1)
            # Re-write the SAME mesh with the SAME state: nothing changed.
            n, written, removed = write_tiles(d, verts, faces, colors,
                                              tile_size=1.0, state=state, rev=2)
            self.assertEqual(written, 0)
            self.assertEqual(removed, 0)
            self.assertEqual(load_manifest(d)["rev"], 2)

    def test_manifest_fingerprints_are_stable_for_identical_geometry(self):
        # Stable fingerprints across independent writes are what lets the consumer
        # skip unchanged tiles (incremental authoring).
        verts, faces, colors = _grid_mesh(16, 0.2)
        with TemporaryDirectory() as d1, TemporaryDirectory() as d2:
            write_tiles(d1, verts, faces, colors, tile_size=1.0, state={}, rev=1)
            write_tiles(d2, verts, faces, colors, tile_size=1.0, state={}, rev=1)
            self.assertEqual(load_manifest(d1)["tiles"], load_manifest(d2)["tiles"])

    def test_pose_and_tiles_share_the_map_dir_without_clobbering(self):
        verts, faces, colors = _grid_mesh(12, 0.2)
        with TemporaryDirectory() as d:
            write_tiles(d, verts, faces, colors, tile_size=1.0, state={}, rev=1)
            write_pose(d, Pose(1.0, -2.0, 45.0), seq=3, timestamp_s=10.0)
            # Both consumers still see complete, correct data.
            self.assertTrue(load_manifest(d)["tiles"])
            payload = read_pose(d)
            self.assertEqual(payload["seq"], 3)
            self.assertAlmostEqual(payload["yaw_deg"], 45.0)

    def test_denoised_mesh_still_tiles_with_valid_indexing(self):
        # The mapper denoises producer-side BEFORE write_tiles; the transport must
        # stay coherent (faces/colors unchanged, local indices valid).
        verts, faces, colors = _grid_mesh(20, 0.1)
        noisy = verts.copy()
        noisy[:, 2] += np.random.default_rng(0).normal(0, 0.005, len(verts)).astype(np.float32)
        denoised = denoise_terrain_mesh(noisy, faces, iterations=10, max_displacement_m=0.02)
        self.assertEqual(denoised.shape, verts.shape)
        with TemporaryDirectory() as d:
            n, written, _ = write_tiles(d, denoised, faces, colors,
                                        tile_size=1.0, state={}, rev=1)
            self.assertGreater(written, 0)
            for name in load_manifest(d)["tiles"]:
                data = np.load(tile_npz_path(d, name))
                if len(data["faces"]):
                    self.assertLess(int(data["faces"].max()), len(data["vertices"]))

    @staticmethod
    def _tile_names(d):
        return [p.stem for p in (Path(d) / "tiles").glob("*.npz")]


if __name__ == "__main__":
    unittest.main()
