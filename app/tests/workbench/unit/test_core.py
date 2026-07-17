from datetime import datetime, timezone
import unittest
from pydantic import ValidationError

from app.stage2.workbench.core.contracts import ApprovalGate, DatasetLock, DatasetStatus, MetricTrace, ResultStatus, StepConfig, TimelineItem, canonical_hash
from app.stage2.workbench.core.dag import ChangeClass, analyze_paths
from app.stage2.workbench.core.preview import ConfigPatch, DatasetImpactIndex, PreviewRequest, analyze_change


class ContractTests(unittest.TestCase):
    def test_hash_order_independent(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_frozen_dataset_rejects_missing(self):
        with self.assertRaises(ValidationError):
            DatasetLock(dataset_id="d", status=DatasetStatus.FROZEN, manifest_hash="h", photo_count=1, photo_ids_hash="p", source_hashes_hash="s", extraction_model={}, array_contracts={}, missing_artifacts=("mesh",))

    def test_config_hash_ignores_id_and_status(self):
        base = dict(step="quality_review", preset="balanced", parameters={"x": 1}, parameter_sources={"x": "preset"})
        a = StepConfig(config_id="a", status="draft", **base)
        b = StepConfig(config_id="b", status="approved", **base)
        self.assertEqual(a.config_hash, b.config_hash)

    def test_photo_item_requires_one_photo(self):
        with self.assertRaises(ValidationError):
            TimelineItem(item_id="i", item_type="photo_point", track_id="q", photo_ids=("a", "b"), start_time=datetime.now(timezone.utc), x_anchor="photo_time", value=1.0, status=ResultStatus.WITHIN_RANGE, quality_state="sufficient", trace_id="t")

    def test_changed_trace_requires_reason(self):
        with self.assertRaises(ValidationError):
            MetricTrace(trace_id="t2", metric_id="m", subject_ids=("p",), raw_value=1.0, normalized_value=2.0, calibration_snapshot_id="c", thresholds={}, quality_inputs={}, dependency_values={}, status=ResultStatus.CANDIDATE, status_reasons=("above",), config_id="cfg", previous_trace_id="t1")

    def test_approved_gate_rejects_blocker(self):
        with self.assertRaises(ValidationError):
            ApprovalGate(gate_id="g", step="quality_review", run_id="r", checks=(), blockers=({"code": "X"},), warnings=(), manual_review_complete=True, status="approved")


class DagTests(unittest.TestCase):
    def test_ui_change_is_local(self):
        result = analyze_paths(["ui.timeline.opacity"], ["geometry"])
        self.assertEqual(result.change_class, ChangeClass.UI_ONLY)
        self.assertEqual(result.affected_steps, ())

    def test_geometry_threshold_reclassifies_without_texture(self):
        result = analyze_paths(["geometry.decision.robust_z"], ["geometry", "texture", "fusion"])
        self.assertEqual(result.change_class, ChangeClass.RECLASSIFY)
        self.assertIn("geometry", result.affected_steps)
        self.assertIn("fusion", result.affected_steps)
        self.assertNotIn("texture", result.affected_steps)
        self.assertIn("geometry", result.approvals_to_mark_stale)

    def test_texture_compute_revokes_downstream_not_geometry(self):
        result = analyze_paths(["texture.compute.roi_size"], ["geometry", "texture", "fusion"])
        self.assertEqual(result.change_class, ChangeClass.RECOMPUTE)
        self.assertIn("texture", result.approvals_to_revoke)
        self.assertIn("fusion", result.approvals_to_revoke)
        self.assertNotIn("geometry", result.approvals_to_revoke)

    def test_stage1_change_invalidates_everything(self):
        result = analyze_paths(["stage1.model.version"], ["quality_review", "geometry"])
        self.assertEqual(result.change_class, ChangeClass.STAGE1_INVALIDATION)
        self.assertEqual(len(result.affected_steps), 10)


class PreviewTests(unittest.TestCase):
    def test_preview_reports_impacted_entities(self):
        request = PreviewRequest(request_id="1", idempotency_key="k", session_id="s", dataset_lock_id="d", base_config_id="c", step="geometry", mode="impact", config_patch=(ConfigPatch(op="set", path="geometry.decision.robust_z", value=4.0),), client_revision=1)
        index = DatasetImpactIndex(photo_count=3, pair_count=2, event_count=1, parameter_to_photos={}, parameter_to_pairs={"geometry.decision.robust_z": frozenset({"ab", "bc"})}, parameter_to_events={"geometry.decision.robust_z": frozenset({"e1"})})
        response = analyze_change(request, index, approved_steps=["geometry", "fusion"])
        self.assertEqual(response.change_class, "B")
        self.assertEqual(response.impact["affected_pair_count"], 2)
        self.assertEqual(response.cache["raw_value_misses"], 0)
        self.assertEqual(response.cache["status_reclassification_count"], 2)

    def test_stage1_patch_returns_blocker(self):
        request = PreviewRequest(request_id="2", idempotency_key="k2", session_id="s", dataset_lock_id="d", base_config_id="c", step="input_freeze", mode="impact", config_patch=(ConfigPatch(op="set", path="stage1.model.version", value="new"),), client_revision=1)
        response = analyze_change(request, DatasetImpactIndex(0, 0, 0, {}, {}, {}))
        self.assertEqual(response.change_class, "D")
        self.assertEqual(response.blockers[0]["code"], "STAGE1_CHANGE_REQUIRES_NEW_DATASET_LOCK")


if __name__ == "__main__":
    unittest.main()
