from __future__ import annotations

import unittest
import numpy as np

from app.stage1.geometry import normalize_mesh, pack_mask, row_rotation_matrix, unpack_mask


class GeometryTests(unittest.TestCase):
    def test_rotation_is_orthonormal(self):
        r = row_rotation_matrix(12.0, -37.0, 8.0)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=5)

    def test_yaw_roundtrip(self):
        a = row_rotation_matrix(0, 33, 0)
        b = row_rotation_matrix(0, -33, 0)
        np.testing.assert_allclose(a @ b, np.eye(3), atol=1e-6)

    def test_normalization(self):
        rng = np.random.default_rng(2)
        mesh = rng.normal(size=(100, 3)).astype(np.float32) + 12
        normalized, center, scale = normalize_mesh(mesh)
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=5e-6)
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.sum(normalized ** 2, axis=1)))), 1.0, places=6)
        np.testing.assert_allclose(normalized * scale + center, mesh, atol=2e-6)

    def test_packbits_roundtrip_non_multiple_of_eight(self):
        rng = np.random.default_rng(3)
        mask = rng.integers(0, 2, size=35709, dtype=np.uint8)
        np.testing.assert_array_equal(unpack_mask(pack_mask(mask), len(mask)), mask)
        self.assertEqual(pack_mask(mask).shape, (4464,))


if __name__ == "__main__": unittest.main()
