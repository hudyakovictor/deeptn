import unittest
from app.stage2.workbench.recommendation.engine import PhotoSummary, build_recommended
from app.stage2.workbench.pipeline.quality_calibration import CalibrationRecord, build_calibration_ranges, classify_photos, run_vertical_slice1


def photos(count=20):
    return tuple(PhotoSummary(photo_id=f"p{i:02d}", pose_bin="frontal", quality=0.4+i*0.02, visibility=0.55+i*0.015, abs_yaw=float(i), abs_pitch=float(i)/2, abs_roll=float(i)/3, alignment_residual=0.01+i*0.001, texture_roi_suitability=0.45+i*0.02, source_group=f"s{i//2}") for i in range(count))


def records(count=20, sources=2):
    return tuple(CalibrationRecord(record_id=f"r{i}", metric_id="geometry.ldm134.rmse", pose_bin="frontal", value=0.1+(i%5)*0.01, source_dataset=f"d{i%sources}", source_group=f"g{i}") for i in range(count))


class RecommendedTests(unittest.TestCase):
    def test_recommendation_is_deterministic(self):
        self.assertEqual(build_recommended(photos()).proposal_hash, build_recommended(reversed(photos())).proposal_hash)

    def test_strict_is_stricter(self):
        proposal = build_recommended(photos())
        strict = proposal.presets["strict"]["quality"]["admission"]
        balanced = proposal.presets["balanced"]["quality"]["admission"]
        self.assertGreaterEqual(strict["minimum_quality"], balanced["minimum_quality"])
        self.assertLessEqual(strict["maximum_abs_yaw"], balanced["maximum_abs_yaw"])

    def test_sensitive_is_more_permissive(self):
        proposal = build_recommended(photos())
        sensitive = proposal.presets["sensitive"]["quality"]["admission"]
        balanced = proposal.presets["balanced"]["quality"]["admission"]
        self.assertLessEqual(sensitive["minimum_quality"], balanced["minimum_quality"])
        self.assertGreaterEqual(sensitive["maximum_abs_yaw"], balanced["maximum_abs_yaw"])

    def test_low_sample_warning(self):
        self.assertIn("LOW_SAMPLE_COUNT", build_recommended(photos(5)).warnings)


class VerticalSliceTests(unittest.TestCase):
    def test_channel_suitability_is_separate(self):
        proposal = build_recommended(photos())
        modified = list(photos())
        p = modified[-1]
        modified[-1] = PhotoSummary(**{**p.__dict__, "texture_roi_suitability": None})
        result = classify_photos(modified, proposal.parameters)
        target = next(item for item in result if item.photo_id == p.photo_id)
        self.assertEqual(target.texture, "quality_limited")
        self.assertEqual(target.chronology, "included")

    def test_active_calibration_range(self):
        result = build_calibration_ranges(records(), minimum_records=8, minimum_sources=2)[0]
        self.assertEqual(result.status, "active")
        self.assertGreater(result.p95, result.median)

    def test_limited_calibration_range(self):
        self.assertEqual(build_calibration_ranges(records(5, 1), minimum_records=8, minimum_sources=2)[0].status, "calibration_limited")

    def test_full_slice_ready(self):
        result = run_vertical_slice1(photos(), records(), manual_review_complete=True)
        self.assertTrue(result.quality_ready)
        self.assertTrue(result.calibration_ready)
        self.assertTrue(result.result_hash.startswith("sha256:"))

    def test_manual_review_blocks_quality(self):
        result = run_vertical_slice1(photos(), records(), manual_review_complete=False)
        self.assertFalse(result.quality_ready)
        self.assertIn("MANUAL_REVIEW_INCOMPLETE", result.quality_blockers)

    def test_insufficient_sources_block_calibration(self):
        result = run_vertical_slice1(photos(), records(sources=1), manual_review_complete=True)
        self.assertFalse(result.calibration_ready)
        self.assertIn("CALIBRATION_SOURCE_COUNT_LOW", result.calibration_blockers)


if __name__ == "__main__":
    unittest.main()
