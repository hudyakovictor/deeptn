from __future__ import annotations

import unittest
import numpy as np

from app.stage2.core import robust_rigid_align
from app.stage2.corroboration import apply_cross_bin_corroboration, aggregate_events


class RobustAlignmentTests(unittest.TestCase):
    def test_trimmed_alignment_resists_local_outlier(self):
        rng = np.random.default_rng(7)
        target = rng.normal(size=(80, 3)).astype(np.float32)
        source = target + np.array([0.4, -0.2, 0.1], np.float32)
        source[:12] += 5.0
        aligned, _, _, meta = robust_rigid_align(source, target, trim_fraction=0.2)
        stable_error = np.median(np.linalg.norm(aligned[12:] - target[12:], axis=1))
        self.assertLess(float(stable_error), 1e-4)
        self.assertGreater(int(meta["trimmed_point_count"]), 0)
        self.assertEqual(meta["alignment_policy"], "iteratively_trimmed_kabsch_v1")


class CorroborationTests(unittest.TestCase):
    def test_independent_pose_support_is_secondary(self):
        rows = [
            {"pair_id": "a", "pair_type": "adjacent", "pose_bin": "frontal", "date_b": "2024-05-01", "status": "persistent_geometric_change", "quality_limited": False, "source_group_b": "news_a"},
            {"pair_id": "b", "pair_type": "adjacent", "pose_bin": "left_mid", "date_b": "2024-05-20", "status": "coherent_jump_candidate", "quality_limited": False, "source_group_b": "news_b"},
            {"pair_id": "c", "pair_type": "adjacent", "pose_bin": "right_mid", "date_b": "2024-05-22", "status": "coherent_jump_candidate", "quality_limited": False, "source_group_b": "news_c"},
        ]
        report = apply_cross_bin_corroboration(rows, window_days=45)
        self.assertEqual(rows[0]["cross_bin_corroboration_status"], "corroborated_multiple_pose_bins")
        self.assertEqual(rows[0]["status"], "persistent_geometric_change")
        self.assertGreaterEqual(report["event_count"], 1)
        events = aggregate_events(rows)
        self.assertEqual(len(events), 3)


if __name__ == "__main__":
    unittest.main()
