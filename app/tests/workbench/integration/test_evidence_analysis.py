from datetime import datetime, timedelta, timezone
import unittest

from app.stage2.workbench.pipeline.evidence_analysis import (
    Observation, PairEvidence, build_cross_pose_events, classify_evidence,
    detect_aba, detect_persistence, detect_rapid_changes, fuse_channels,
    run_analysis_snapshot,
)

T0 = datetime(2020, 1, 1, tzinfo=timezone.utc)


def ev(eid, pair, channel, value, *, pose="frontal", source="s1", zone="cheek", quality="sufficient", day=0):
    return PairEvidence(eid, pair, T0 + timedelta(days=day), pose, source, channel, f"{channel}_family", zone, value, 3.0, quality)


def obs(pid, day, value, *, pose="frontal", source="s1", quality="sufficient"):
    return Observation(pid, T0 + timedelta(days=day), pose, source, "metric", value, quality)


class EvidenceTests(unittest.TestCase):
    def test_missing_not_zero(self):
        result = classify_evidence([ev("e", "p", "geometry", None)])[0]
        self.assertEqual(result.status, "missing_input")

    def test_quality_limited_not_candidate(self):
        result = classify_evidence([ev("e", "p", "texture", 8.0, quality="limited")])[0]
        self.assertEqual(result.status, "quality_limited")

    def test_candidate_above_threshold(self):
        self.assertEqual(classify_evidence([ev("e", "p", "geometry", 4.0)])[0].status, "candidate_with_alternatives")


class FusionTests(unittest.TestCase):
    def test_channels_agree(self):
        result = fuse_channels(classify_evidence([ev("g", "p", "geometry", 4), ev("t", "p", "texture", 5)]))[0]
        self.assertEqual(result.combined_status, "channels_agree_candidate")

    def test_channels_conflict(self):
        result = fuse_channels(classify_evidence([ev("g", "p", "geometry", 4), ev("t", "p", "texture", 1)]))[0]
        self.assertEqual(result.combined_status, "channels_conflict")

    def test_missing_texture_preserved(self):
        result = fuse_channels(classify_evidence([ev("g", "p", "geometry", 4)]))[0]
        self.assertEqual(result.combined_status, "geometry_only_candidate")
        self.assertIn("texture_missing", result.limitations)


class CrossPoseTests(unittest.TestCase):
    def test_independent_cross_pose_support(self):
        items = classify_evidence([
            ev("a", "p1", "geometry", 4, pose="frontal", source="s1"),
            ev("b", "p2", "geometry", 5, pose="profile", source="s2"),
        ])
        self.assertEqual(build_cross_pose_events(items)[0].status, "cross_pose_support")

    def test_same_source_not_independent(self):
        items = classify_evidence([
            ev("a", "p1", "geometry", 4, pose="frontal", source="same"),
            ev("b", "p2", "geometry", 5, pose="profile", source="same"),
        ])
        self.assertEqual(build_cross_pose_events(items)[0].status, "insufficient_independence")


class ChronologyTests(unittest.TestCase):
    def test_rapid_respects_gap(self):
        observations = [obs("a", 0, 0), obs("b", 30, 5), obs("c", 400, 10)]
        result = detect_rapid_changes(observations, threshold=3, max_gap_days=90)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].photo_ids, ("a", "b"))

    def test_quality_limited_observation_excluded(self):
        observations = [obs("a", 0, 0), obs("b", 10, 8, quality="limited")]
        self.assertEqual(detect_rapid_changes(observations, threshold=3, max_gap_days=90), ())

    def test_persistence(self):
        observations = [obs("a", 0, 0), obs("b", 10, 0.2), obs("c", 20, 5), obs("d", 30, 5.5), obs("e", 40, 4.8)]
        result = detect_persistence(observations, threshold=3, minimum_followups=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_type, "persistent_change")

    def test_aba_return(self):
        observations = [obs("a", 0, 0), obs("b", 10, 5), obs("a2", 20, 0.4)]
        result = detect_aba(observations, forward_threshold=3, return_tolerance=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].photo_ids, ("a", "b", "a2"))

    def test_aba_requires_return(self):
        observations = [obs("a", 0, 0), obs("b", 10, 5), obs("c", 20, 4)]
        self.assertEqual(detect_aba(observations, forward_threshold=3, return_tolerance=1), ())


class IntegrationTests(unittest.TestCase):
    def test_snapshot_is_deterministic(self):
        evidence = [ev("g", "p", "geometry", 4), ev("t", "p", "texture", 5)]
        observations = [obs("a", 0, 0), obs("b", 10, 5), obs("a2", 20, 0.2)]
        a = run_analysis_snapshot(evidence, observations)
        b = run_analysis_snapshot(reversed(evidence), reversed(observations))
        self.assertEqual(a.result_hash, b.result_hash)
        self.assertEqual(a.fusion[0].combined_status, "channels_agree_candidate")
        self.assertTrue(any(event.event_type == "aba_return" for event in a.chronology))


if __name__ == "__main__":
    unittest.main()
