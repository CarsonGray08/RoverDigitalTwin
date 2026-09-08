import unittest

import numpy as np

from digital_twin.terrain_denoise import denoise_terrain_mesh


def _grid_mesh(side: int, spacing: float = 0.1):
    """A flat (z=0) triangulated grid of ``side`` x ``side`` vertices."""
    xs, ys = np.meshgrid(np.arange(side) * spacing, np.arange(side) * spacing)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1)
    faces = []
    for j in range(side - 1):
        for i in range(side - 1):
            v = j * side + i
            faces.append([v, v + 1, v + side])
            faces.append([v + 1, v + side + 1, v + side])
    return verts.astype(np.float64), np.asarray(faces, dtype=np.int32)


class TerrainDenoiseTest(unittest.TestCase):
    def test_empty_and_degenerate_inputs_return_unchanged(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        out = denoise_terrain_mesh(empty, np.zeros((0, 3), dtype=np.int32))
        self.assertEqual(out.shape, (0, 3))
        verts, _ = _grid_mesh(3)
        # no faces -> nothing to smooth
        same = denoise_terrain_mesh(verts, np.zeros((0, 3), dtype=np.int32))
        np.testing.assert_allclose(same, verts.astype(np.float32))

    def test_reduces_high_frequency_noise_on_flat_surface(self):
        rng = np.random.default_rng(0)
        verts, faces = _grid_mesh(12)
        noisy = verts.copy()
        noisy[:, 2] += rng.normal(0.0, 0.005, size=len(verts))  # 5 mm noise
        out = denoise_terrain_mesh(noisy, faces, iterations=20, step=0.5,
                                   max_displacement_m=0.02)
        # Interior noise (z about the true 0 plane) should shrink substantially.
        self.assertLess(np.std(out[:, 2]), 0.5 * np.std(noisy[:, 2]))

    def test_never_exceeds_max_displacement(self):
        rng = np.random.default_rng(1)
        verts, faces = _grid_mesh(10)
        noisy = verts.copy()
        noisy[:, 2] += rng.normal(0.0, 0.02, size=len(verts))
        cap = 0.01
        out = denoise_terrain_mesh(noisy, faces, iterations=50, step=0.8,
                                   max_displacement_m=cap)
        disp = np.linalg.norm(out.astype(np.float64) - noisy, axis=1)
        self.assertLessEqual(disp.max(), cap + 1e-6)

    def test_preserves_a_real_coherent_feature(self):
        # A tall, wide bump (a "rock"): a coherent feature far larger than the
        # displacement clamp must survive denoising rather than be flattened.
        verts, faces = _grid_mesh(15, spacing=0.1)
        side = 15
        zz = verts[:, 2].copy()
        # raise a 3x3 block in the middle by 0.10 m
        for j in range(6, 9):
            for i in range(6, 9):
                zz[j * side + i] = 0.10
        feature = verts.copy()
        feature[:, 2] = zz
        out = denoise_terrain_mesh(feature, faces, iterations=30, step=0.5,
                                   max_displacement_m=0.01)
        peak = out[(7 * side + 7), 2]  # center of the bump
        self.assertGreater(peak, 0.10 - 0.01 - 1e-6)  # within the clamp of original

    def test_shape_dtype_and_indexing_preserved(self):
        verts, faces = _grid_mesh(6)
        out = denoise_terrain_mesh(verts, faces, iterations=5)
        self.assertEqual(out.shape, verts.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_zero_iterations_is_identity(self):
        verts, faces = _grid_mesh(5)
        out = denoise_terrain_mesh(verts, faces, iterations=0)
        np.testing.assert_allclose(out, verts.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
