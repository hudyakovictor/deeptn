from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.stage2.private_hypothesis import PrivateHypothesisConfig, PrivateHypothesisEngine


class PrivateHypothesisTests(unittest.TestCase):
    def test_lossless_import_and_current_retest_are_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); analysis = root / "analysis"; legacy = root / "legacy" / "final_inference"; output = root / "private"
            analysis.mkdir(); legacy.mkdir(parents=True)
            with (analysis / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["pair_id", "photo_a", "photo_b", "date_a", "date_b", "status", "evidence_state"])
                w.writeheader(); w.writerow({"pair_id": "a__b", "photo_a": "a", "photo_b": "b", "date_a": "2020-01-01", "date_b": "2020-01-02", "status": "within_reconstruction_noise", "evidence_state": "within_noise"})
            (analysis / "metric_catalog.json").write_text(json.dumps({"metrics": [{"name": "ldm134_rmse"}]}), encoding="utf-8")
            (legacy / "hypothesis_explanations.json").write_text(json.dumps({"entries": [{"photo_id": "b", "primary_hypothesis": "H2_DIFFERENT"}, {"photo_id": "missing", "primary_hypothesis": "H1_SYNTHETIC"}]}), encoding="utf-8")
            (legacy / "final_inference_manifest.json").write_text(json.dumps({"model_version": "old", "reference_use_mesh_alignment": False}), encoding="utf-8")
            manifest = PrivateHypothesisEngine(PrivateHypothesisConfig(analysis, root / "legacy", output, 0.95)).run()
            self.assertEqual(manifest["import_coverage_fraction"], 1.0)
            self.assertEqual(manifest["retested_with_current_alignment_count"], 1)
            self.assertEqual(manifest["pending_missing_current_data_count"], 1)
            lines = [json.loads(x) for x in (output / "hypothesis_retest_results.jsonl").read_text().splitlines()]
            self.assertEqual(lines[0]["status"], "retested_with_current_alignment")
            self.assertEqual(lines[1]["status"], "pending_missing_current_data")
            self.assertIn("re-estimated", manifest["range_policy"])


if __name__ == "__main__":
    unittest.main()
