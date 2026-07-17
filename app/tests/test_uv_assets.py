from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from app.stage1.assets import _analysis_mesh_arrays, _complete_uv_texture, _confidence_preview


class UVAssetTests(unittest.TestCase):
    def test_completed_texture_has_no_hidden_side_hole(self):
        texture = np.zeros((32, 32, 3), np.uint8)
        texture[:, :16] = (80, 130, 190)
        observed = np.zeros((32, 32), bool)
        observed[:, :16] = True
        uv = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], np.int64)
        completed, method, domain = _complete_uv_texture(texture, observed, np.ones_like(observed), uv, triangles)
        np.testing.assert_array_equal(completed[observed], texture[observed])
        self.assertTrue(np.all(np.any(completed[domain], axis=1)))
        self.assertTrue(np.any(method[domain] > 0))

    def test_confidence_preview_is_continuous_not_binary(self):
        confidence = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (16, 1))
        preview = _confidence_preview(confidence, np.ones_like(confidence, bool))
        self.assertGreater(len(np.unique(preview.reshape(-1, 3), axis=0)), 20)
        self.assertFalse(np.array_equal(preview[:, 0], preview[:, -1]))

    def test_npz_analysis_mesh_is_visibility_cut(self):
        bundle = SimpleNamespace(
            triangles=np.asarray([[0, 1, 2], [0, 2, 3]], np.int64),
            combined_visible=np.asarray([True, True, True, False]),
            vertices_object=np.zeros((4, 3), np.float32),
            vertices_object_normalized=np.zeros((4, 3), np.float32),
            normals_object=np.zeros((4, 3), np.float32),
            uv_coords=np.zeros((4, 2), np.float32),
        )
        arrays = _analysis_mesh_arrays(bundle)
        self.assertEqual(arrays["analysis_mesh_vertices_object"].shape, (3, 3))
        self.assertEqual(arrays["analysis_mesh_triangles"].shape, (1, 3))
        np.testing.assert_array_equal(arrays["analysis_mesh_vertex_indices_full"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
