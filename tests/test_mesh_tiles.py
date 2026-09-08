import tempfile
import unittest
from pathlib import Path

import numpy as np

from digital_twin.mesh_tiles import (
    MANIFEST_NAME,
    fingerprint,
    load_manifest,
    split_into_tiles,
    tile_name,
    tile_npz_path,
    write_tiles,
)


def _two_tile_mesh():
    """Two triangles whose centroids fall in different 2 m tiles.

    Tri A centroid ~(0.5, 0.5, 0.0) -> tile (0,0,0).
    Tri B centroid ~(4.0, 0.5, 0.0) -> tile (2,0,0) at 2 m tiles.
    """
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.5, 0.0],   # A
        [3.5, 0.0, 0.0], [4.5, 0.0, 0.0], [4.0, 1.5, 0.0],   # B
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    colors = np.array([
        [10, 0, 0], [11, 0, 0], [12, 0, 0],
        [0, 20, 0], [0, 21, 0], [0, 22, 0],
    ], dtype=np.uint8)
    return verts, faces, colors


class TileNameTests(unittest.TestCase):
    def test_usd_safe_negatives(self):
        self.assertEqual(tile_name((3, -1, 0)), "tile_3_m1_0")
        self.assertEqual(tile_name((0, 0, 0)), "tile_0_0_0")


class SplitTests(unittest.TestCase):
    def test_empty(self):
        out = list(split_into_tiles(np.zeros((0, 3), np.float32),
                                    np.zeros((0, 3), np.int32),
                                    np.zeros((0, 3), np.uint8)))
        self.assertEqual(out, [])

    def test_faces_split_into_correct_tiles(self):
        verts, faces, colors = _two_tile_mesh()
        tiles = {name: (lv, lf, lc)
                 for name, lv, lf, lc in split_into_tiles(verts, faces, colors, 2.0)}
        self.assertEqual(set(tiles), {"tile_0_0_0", "tile_2_0_0"})
        for lv, lf, lc in tiles.values():
            # One triangle per tile, faces reindexed locally into its own verts.
            self.assertEqual(lf.shape, (1, 3))
            self.assertEqual(len(lv), 3)
            self.assertEqual(len(lc), 3)
            self.assertTrue((lf >= 0).all() and (lf < len(lv)).all())

    def test_reindex_preserves_triangle_geometry(self):
        verts, faces, colors = _two_tile_mesh()
        for _name, lv, lf, _lc in split_into_tiles(verts, faces, colors, 2.0):
            # The local triangle's vertex positions must match an original triangle.
            tri = lv[lf[0]]
            matches = [np.allclose(sorted(tri.tolist()), sorted(verts[f].tolist()))
                       for f in faces]
            self.assertTrue(any(matches))

    def test_single_tile_when_disabled(self):
        verts, faces, colors = _two_tile_mesh()
        tiles = list(split_into_tiles(verts, faces, colors, 0.0))
        self.assertEqual(len(tiles), 1)
        name, lv, lf, _lc = tiles[0]
        self.assertEqual(name, "tile_0_0_0")
        self.assertEqual(lf.shape, (2, 3))
        self.assertEqual(len(lv), 6)

    def test_negative_coordinates(self):
        verts = np.array([[-3.5, -0.1, 0.0], [-2.6, -0.1, 0.0], [-3.0, -1.5, 0.0]],
                         dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        colors = np.zeros((3, 3), dtype=np.uint8)
        names = [n for n, *_ in split_into_tiles(verts, faces, colors, 2.0)]
        self.assertEqual(names, ["tile_m2_m1_0"])


class FingerprintTests(unittest.TestCase):
    def test_invariant_to_vertex_reordering(self):
        v = np.random.default_rng(0).random((50, 3)).astype(np.float32)
        perm = np.random.default_rng(1).permutation(len(v))
        self.assertEqual(fingerprint(v, 12), fingerprint(v[perm], 12))

    def test_stable_under_subvoxel_jitter(self):
        # On-grid vertices (multiples of the 5 mm quantum) plus jitter under half
        # a quantum stay in the same bin -- the "settled tile nvblox keeps
        # re-emitting with tiny wobble" case we want to skip re-authoring for.
        grid = np.random.default_rng(2).integers(-50, 50, size=(50, 3)) * 0.005
        jittered = grid + np.random.default_rng(4).uniform(-0.002, 0.002, size=grid.shape)
        self.assertEqual(fingerprint(grid.astype(np.float32), 12),
                         fingerprint(jittered.astype(np.float32), 12))

    def test_sensitive_to_real_change(self):
        v = np.random.default_rng(3).random((50, 3)).astype(np.float32)
        moved = v.copy()
        moved[0] += 0.5  # a real 0.5 m move
        self.assertNotEqual(fingerprint(v, 12), fingerprint(moved, 12))
        self.assertNotEqual(fingerprint(v, 12), fingerprint(v, 13))  # face count too


class WriteTilesTests(unittest.TestCase):
    def test_manifest_and_per_tile_files(self):
        verts, faces, colors = _two_tile_mesh()
        with tempfile.TemporaryDirectory() as d:
            n, written, removed = write_tiles(d, verts, faces, colors, 2.0, state={}, rev=1)
            self.assertEqual((n, written, removed), (2, 2, 0))
            manifest = load_manifest(d)
            self.assertEqual(manifest["rev"], 1)
            self.assertEqual(set(manifest["tiles"]), {"tile_0_0_0", "tile_2_0_0"})
            for name in manifest["tiles"]:
                self.assertTrue(tile_npz_path(d, name).exists())

    def test_only_changed_tiles_rewritten(self):
        verts, faces, colors = _two_tile_mesh()
        with tempfile.TemporaryDirectory() as d:
            state: dict = {}
            write_tiles(d, verts, faces, colors, 2.0, state=state, rev=1)
            # Record mtimes, then re-write identical geometry: nothing should change.
            mtimes = {name: tile_npz_path(d, name).stat().st_mtime_ns
                      for name in load_manifest(d)["tiles"]}
            n, written, removed = write_tiles(d, verts, faces, colors, 2.0, state=state, rev=2)
            self.assertEqual((written, removed), (0, 0))
            self.assertEqual(n, 2)
            for name, mt in mtimes.items():
                self.assertEqual(tile_npz_path(d, name).stat().st_mtime_ns, mt)

    def test_moving_geometry_rewrites_one_tile(self):
        verts, faces, colors = _two_tile_mesh()
        with tempfile.TemporaryDirectory() as d:
            state: dict = {}
            write_tiles(d, verts, faces, colors, 2.0, state=state, rev=1)
            # Nudge tile B's geometry past the quantum; A stays put.
            verts2 = verts.copy()
            verts2[3:] += 0.3
            n, written, removed = write_tiles(d, verts2, faces, colors, 2.0, state=state, rev=2)
            self.assertEqual((written, removed), (1, 0))

    def test_vanished_tile_is_removed(self):
        verts, faces, colors = _two_tile_mesh()
        with tempfile.TemporaryDirectory() as d:
            state: dict = {}
            write_tiles(d, verts, faces, colors, 2.0, state=state, rev=1)
            self.assertTrue(tile_npz_path(d, "tile_2_0_0").exists())
            # Drop triangle B -> tile_2_0_0 should disappear from disk + manifest.
            n, written, removed = write_tiles(
                d, verts[:3], faces[:1], colors[:3], 2.0, state=state, rev=2)
            self.assertEqual((n, removed), (1, 1))
            self.assertFalse(tile_npz_path(d, "tile_2_0_0").exists())
            self.assertEqual(set(load_manifest(d)["tiles"]), {"tile_0_0_0"})

    def test_roundtrip_reconstructs_full_geometry(self):
        verts, faces, colors = _two_tile_mesh()
        with tempfile.TemporaryDirectory() as d:
            write_tiles(d, verts, faces, colors, 2.0, state={}, rev=1)
            total_faces = 0
            for name in load_manifest(d)["tiles"]:
                data = np.load(tile_npz_path(d, name))
                total_faces += len(data["faces"])
                self.assertEqual(len(data["vertices"]), len(data["colors"]))
            self.assertEqual(total_faces, len(faces))

    def test_load_manifest_missing_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_manifest(Path(d) / "nope"), {})


class ConsumerContractTests(unittest.TestCase):
    """Simulate the Isaac sink's incremental diff (author_nvblox_tiles, minus the
    USD calls) against the producer's write_tiles output. Locks in (a) the
    fingerprint comparison surviving the JSON tuple->list round-trip and (b) the
    key property: as the scan grows, the consumer only re-authors the frontier."""

    @staticmethod
    def _consumer_poll(map_dir, state):
        """Mirror author_nvblox_tiles' diff: returns the list of tile names that
        would be (re)authored, mutating ``state`` (name -> manifest fingerprint)
        and removing vanished tiles, exactly as the real sink does."""
        manifest = load_manifest(map_dir)
        tiles = manifest.get("tiles", {})
        authored = []
        for name, fp in tiles.items():
            if state.get(name) == fp:
                continue
            data = np.load(tile_npz_path(map_dir, name))  # the sink loads + authors
            assert len(data["vertices"]) == len(data["colors"])
            state[name] = fp
            authored.append(name)
        for name in [n for n in list(state) if n not in tiles]:
            del state[name]
        return authored

    def _grid_mesh(self, n_cols):
        """A row of ``n_cols`` triangles, one per 2 m tile along x (tiles
        (0,0,0), (1,0,0), ...)."""
        verts, faces, colors = [], [], []
        for i in range(n_cols):
            x = i * 2.0 + 0.5
            base = 3 * i
            verts += [[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 1.0, 0.0]]
            faces.append([base, base + 1, base + 2])
            colors += [[i % 256, 0, 0]] * 3
        return (np.array(verts, np.float32), np.array(faces, np.int32),
                np.array(colors, np.uint8))

    def test_growing_scan_authors_only_the_frontier(self):
        with tempfile.TemporaryDirectory() as d:
            prod_state: dict = {}
            cons_state: dict = {}

            # First scan tick: 3 tiles -> consumer authors all 3.
            v, f, c = self._grid_mesh(3)
            write_tiles(d, v, f, c, 2.0, state=prod_state, rev=1)
            self.assertEqual(sorted(self._consumer_poll(d, cons_state)),
                             ["tile_0_0_0", "tile_1_0_0", "tile_2_0_0"])

            # Scan extends by two tiles (frontier grows), the first 3 are unchanged.
            v, f, c = self._grid_mesh(5)
            n, written, _ = write_tiles(d, v, f, c, 2.0, state=prod_state, rev=2)
            self.assertEqual((n, written), (5, 2))  # producer rewrote only 2
            self.assertEqual(sorted(self._consumer_poll(d, cons_state)),
                             ["tile_3_0_0", "tile_4_0_0"])  # consumer authored only 2

            # No change at all: consumer authors nothing (constant work, not O(map)).
            write_tiles(d, v, f, c, 2.0, state=prod_state, rev=3)
            self.assertEqual(self._consumer_poll(d, cons_state), [])

    def test_consumer_removes_vanished_tiles(self):
        with tempfile.TemporaryDirectory() as d:
            prod_state: dict = {}
            cons_state: dict = {}
            v, f, c = self._grid_mesh(3)
            write_tiles(d, v, f, c, 2.0, state=prod_state, rev=1)
            self._consumer_poll(d, cons_state)
            # Map shrinks to one tile (e.g. a re-alignment cleared+rebuilt).
            v, f, c = self._grid_mesh(1)
            write_tiles(d, v, f, c, 2.0, state=prod_state, rev=2)
            self._consumer_poll(d, cons_state)
            self.assertEqual(set(cons_state), {"tile_0_0_0"})


if __name__ == "__main__":
    unittest.main()
