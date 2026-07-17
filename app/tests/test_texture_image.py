from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.stage2.texture_image import LBP_BINS, _stats


class TextureImageTests(unittest.TestCase):
    def test_lbp_glcm_frequency_features_are_reported(self):
        rng = np.random.default_rng(42)
        img = np.zeros((72, 72, 3), np.uint8)
        img[:] = rng.normal(128, 22, img.shape).clip(0, 255).astype(np.uint8)
        # Add deterministic micro-texture so frequency / co-occurrence features are non-trivial.
        for y in range(8, 64, 4):
            cv2.line(img, (8, y), (63, y), (160, 160, 160), 1)
        mask = np.zeros((72, 72), bool)
        mask[10:62, 10:62] = True

        stats = _stats(img, mask)

        self.assertGreater(int(stats["texture_pixels"]), 1000)
        self.assertLess(int(stats["texture_pixels"]), int(mask.sum()))
        self.assertGreater(int(stats["roi_erosion_radius"]), 0)
        self.assertEqual(len(stats["lbp_histogram"]), LBP_BINS)
        self.assertAlmostEqual(float(sum(stats["lbp_histogram"])), 1.0, places=4)
        for key in (
            "glcm_contrast",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "high_frequency_ratio",
            "laplacian_var",
            "gradient_energy",
            "local_entropy",
        ):
            self.assertTrue(np.isfinite(float(stats[key])), key)
        self.assertGreaterEqual(float(stats["high_frequency_ratio"]), 0.0)
        self.assertEqual(len(stats["gabor_profile"]), 8)
        self.assertGreater(int(stats["usable_patch_count"]), 0)


if __name__ == "__main__":
    unittest.main()
