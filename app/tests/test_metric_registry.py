from __future__ import annotations

import re
import unittest

from app.stage2.evidence import packet_from_pair
from app.stage2.metric_registry import METRICS, NAMES, build_metric_catalog, metric_channel, validate_registry


class MetricRegistryTests(unittest.TestCase):
    def test_exactly_100_unique_canonical_names(self):
        self.assertEqual(len(METRICS), 100)
        self.assertEqual(len(set(NAMES)), 100)
        self.assertEqual(validate_registry(), [])
        self.assertTrue(all(re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in NAMES))

    def test_missing_and_profile_disabled_are_explicit(self):
        report = build_metric_catalog([{"ldm134_rmse": 0.01}], enabled={"ldm134_rmse": False})
        by_name = {item["name"]: item for item in report["metrics"]}
        self.assertEqual(by_name["ldm134_rmse"]["status"], "disabled_by_config")
        self.assertEqual(by_name["mesh_rmse"]["status"], "disabled_missing_data")
        self.assertIn("required_input_unavailable", by_name["mesh_rmse"]["reason"])

    def test_all_registered_metrics_reach_evidence_channel(self):
        row = {name: i for i, name in enumerate(NAMES)}
        row.update({"pair_id": "a__b", "status": "within_reconstruction_noise"})
        packet = packet_from_pair(row)
        self.assertEqual(packet["registered_metric_channel"], metric_channel(row))
        self.assertEqual(set(packet["registered_metric_channel"]), set(NAMES))


if __name__ == "__main__":
    unittest.main()
