from __future__ import annotations

import unittest

from app.stage2.pose_leakage import pose_leakage_diagnostic


class PoseLeakageTests(unittest.TestCase):
    def test_pose_correlated_metric_is_flagged(self):
        rows = [
            {"pose_distance": i / 10.0, "ldm134_rmse": i / 100.0, "pair_type": "adjacent"}
            for i in range(1, 21)
        ]
        report = pose_leakage_diagnostic(rows)
        self.assertIn("ldm134_rmse", report["flagged_metrics"])
        self.assertEqual(report["metrics"]["ldm134_rmse"]["status"], "pose_leakage_candidate")

    def test_small_sample_is_not_overinterpreted(self):
        report = pose_leakage_diagnostic([{"pose_distance": 1.0, "ldm134_rmse": 0.1}])
        self.assertEqual(report["metrics"]["ldm134_rmse"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
