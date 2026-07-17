from __future__ import annotations

import unittest
from unittest.mock import patch
import sys
import types
import numpy as np

from app.stage1.masks import CHANNEL_NAMES, build_mask_bundle


class MaskTests(unittest.TestCase):
    def test_channel_contract(self):
        self.assertEqual(CHANNEL_NAMES[4], "nose")
        self.assertEqual(CHANNEL_NAMES[7], "skin")

    def test_skin_plus_nose_and_feature_exclusion(self):
        channels = np.zeros((224, 224, 8), np.float32)
        channels[20:100, 20:100, 7] = 1
        channels[100:130, 80:140, 4] = 1
        channels[40:60, 40:60, 0] = 1
        bundle = build_mask_bundle(channels, np.array([224, 224, 1, 112, 112], np.float32), (224, 224, 3))
        self.assertTrue(bundle.hard_224[110, 100])
        self.assertFalse(bundle.hard_224[50, 50])

    def test_projection_failure_never_resizes(self):
        channels = np.zeros((224, 224, 8), np.float32); channels[:, :, 7] = 1
        fake = types.ModuleType("util.io")
        fake.back_resize_crop_img = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        with patch.dict(sys.modules, {"util.io": fake}):
            bundle = build_mask_bundle(channels, np.ones(5), (1000, 800, 3))
        self.assertEqual(bundle.status, "projection_failed")
        self.assertIsNone(bundle.soft_original)
        self.assertFalse(bundle.metadata["fallback_used"])


if __name__ == "__main__": unittest.main()
