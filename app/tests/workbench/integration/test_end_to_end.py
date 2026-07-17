import unittest

from app.stage2.workbench.core.contracts import canonical_hash
from app.stage2.workbench.pipeline.release import (
    ApprovedSnapshot, PrivateHypothesisRecord, ReleaseBuildError,
    build_release_from_snapshots, private_retest_hash, run_private_retest,
)
from app.stage2.workbench.pipeline.quality_calibration import run_vertical_slice1
from app.stage2.workbench.pipeline.evidence_analysis import run_analysis_snapshot
from app.tests.workbench.integration.test_quality_calibration import photos, records
from app.tests.workbench.integration.test_evidence_analysis import ev, obs


REQUIRED = ("quality_review", "calibration", "geometry", "texture", "fusion", "cross_pose", "chronology")


def snapshots(dataset_hash="dataset-hash"):
    return tuple(ApprovedSnapshot(
        step=step, gate_id=f"gate:{step}", dataset_hash=dataset_hash,
        config_hash=canonical_hash({"step": step}),
        artifact_hashes={"snapshot": canonical_hash({"artifact": step})},
    ) for step in REQUIRED)


class ReleaseTests(unittest.TestCase):
    def test_release_builds_from_complete_chain(self):
        release = build_release_from_snapshots(
            release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
            dataset_hash="dataset-hash", snapshots=snapshots(), required_steps=REQUIRED,
            calibration_snapshot_ids=("cal:g", "cal:t"), metric_registry_hash="registry",
        )
        self.assertEqual(release.status, "approved")
        self.assertEqual(len(release.approved_gate_ids), len(REQUIRED))
        self.assertEqual(release.private_artifacts, ())

    def test_missing_step_blocks_release(self):
        with self.assertRaises(ReleaseBuildError):
            build_release_from_snapshots(
                release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
                dataset_hash="dataset-hash", snapshots=snapshots()[:-1], required_steps=REQUIRED,
                calibration_snapshot_ids=(), metric_registry_hash="registry",
            )

    def test_dataset_mismatch_blocks_release(self):
        broken = list(snapshots())
        broken[-1] = ApprovedSnapshot("chronology", "gate:chronology", "other", "cfg", {"snapshot": "h"})
        with self.assertRaises(ReleaseBuildError):
            build_release_from_snapshots(
                release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
                dataset_hash="dataset-hash", snapshots=broken, required_steps=REQUIRED,
                calibration_snapshot_ids=(), metric_registry_hash="registry",
            )


class PrivateRetestTests(unittest.TestCase):
    def release(self):
        return build_release_from_snapshots(
            release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
            dataset_hash="dataset-hash", snapshots=snapshots(), required_steps=REQUIRED,
            calibration_snapshot_ids=(), metric_registry_hash="registry",
        )

    def test_missing_current_data_is_explicit(self):
        result = run_private_retest(
            [PrivateHypothesisRecord("legacy:1", "aba", ("e1", "e2"))],
            public_release=self.release(), available_evidence_ids=("e1",),
        )[0]
        self.assertEqual(result.status, "pending_missing_current_data")

    def test_candidate_uses_current_evidence_only(self):
        result = run_private_retest(
            [PrivateHypothesisRecord("legacy:1", "aba", ("e1",), legacy_label="old", legacy_score=0.99)],
            public_release=self.release(), available_evidence_ids=("e1",), candidate_evidence_ids=("e1",),
        )[0]
        self.assertEqual(result.status, "technical_anomaly_candidate")

    def test_private_retest_does_not_change_public_release(self):
        release = self.release()
        before = release.reproducibility_hash
        results = run_private_retest(
            [PrivateHypothesisRecord("legacy:1", "aba", ("e1",))],
            public_release=release, available_evidence_ids=("e1",),
        )
        self.assertEqual(release.reproducibility_hash, before)
        self.assertTrue(private_retest_hash(results, release).startswith("sha256:"))


class FullPipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_to_release_is_deterministic(self):
        vertical1 = run_vertical_slice1(photos(), records(), manual_review_complete=True)
        analysis = run_analysis_snapshot(
            [ev("g", "pair", "geometry", 4), ev("t", "pair", "texture", 5)],
            [obs("a", 0, 0), obs("b", 10, 5), obs("a2", 20, 0.2)],
        )
        self.assertTrue(vertical1.quality_ready and vertical1.calibration_ready)
        self.assertTrue(any(item.combined_status == "channels_agree_candidate" for item in analysis.fusion))
        release_a = build_release_from_snapshots(
            release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
            dataset_hash="dataset-hash", snapshots=snapshots(), required_steps=REQUIRED,
            calibration_snapshot_ids=(vertical1.result_hash,), metric_registry_hash="registry",
        )
        release_b = build_release_from_snapshots(
            release_id="release:1", run_id="run:1", dataset_lock_id="lock:1",
            dataset_hash="dataset-hash", snapshots=reversed(snapshots()), required_steps=REQUIRED,
            calibration_snapshot_ids=(vertical1.result_hash,), metric_registry_hash="registry",
        )
        self.assertEqual(release_a.reproducibility_hash, release_b.reproducibility_hash)


if __name__ == "__main__":
    unittest.main()
