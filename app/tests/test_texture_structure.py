from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.stage2.texture_structure import compare_zone_structure, register_patches


class TextureStructureTests(unittest.TestCase):
    def _fixture(self):
        image = np.full((240, 240, 3), 140, np.uint8)
        for y in (70, 90, 112, 136):
            cv2.ellipse(image, (120, y), (58, 7), 0, 5, 175, (70, 70, 70), 2)
        mask = np.zeros((240, 240), bool)
        mask[35:190, 35:205] = True
        return image, mask

    def test_registered_structure_is_measured(self):
        a, mask = self._fixture()
        matrix = np.float32([[1, 0, 2], [0, 1, -1]])
        b = cv2.warpAffine(a, matrix, (240, 240), borderMode=cv2.BORDER_REFLECT101)
        result = compare_zone_structure(a, mask, b, mask)
        self.assertEqual(result["structure_status"], "measured")
        self.assertGreater(float(result["registered_ssim"]), 0.8)
        self.assertTrue(np.isfinite(float(result["ridge_map_delta"])))
        self.assertIn("skeleton_branchpoint_delta_abs", result)

    def test_excessive_shift_is_rejected(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(192, 192)).astype(np.float32)
        b = np.roll(a, 30, axis=1)
        mask = np.ones((192, 192), bool)
        result = register_patches(a, b, mask, mask)
        self.assertEqual(result["status"], "registration_unstable")


if __name__ == "__main__":
    unittest.main()
