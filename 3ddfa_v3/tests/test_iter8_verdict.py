"""ITER8 verdict tests: no calendar prior, H0/H1/H2, fail-closed."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.verdict import (  # noqa: E402
    EvidenceBundle,
    ForensicStatus,
    FuzzyLabel,
    GeometryEvidenceMode,
    geometry_likelihoods,
    normalize_priors,
    render_verdict,
    texture_likelihoods,
    update_posteriors_log,
)


def test_priors_normalize_and_default_not_year_based():
    p = normalize_priors()
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["H1"] < p["H0"]


def test_same_person_low_snr():
    ev = EvidenceBundle(
        geometry_snr=0.3,
        texture_silicone=0.2,
        texture_reliability=0.8,
        shared_vertex_count=500,
        geometry_mode=GeometryEvidenceMode.CALIBRATED,
    )
    v = render_verdict(ev)
    assert v.status == ForensicStatus.SAME_PERSON
    assert v.probabilities["H0"] > v.probabilities["H2"]
    assert "no_calendar_prior" in v.reasoning


def test_different_person_high_snr():
    ev = EvidenceBundle(
        geometry_snr=3.5,
        texture_silicone=0.25,
        texture_reliability=0.7,
        shared_vertex_count=400,
        geometry_mode=GeometryEvidenceMode.CALIBRATED,
    )
    v = render_verdict(ev)
    assert v.status in (ForensicStatus.DIFFERENT_PERSON, ForensicStatus.UNCERTAIN, ForensicStatus.IDENTITY_SWAP)
    assert v.probabilities["H2"] > v.probabilities["H0"] or v.probabilities["H1"] > 0.3


def test_texture_high_with_match_not_forced_swap():
    # geometry match + high silicone → surface confound, not automatic H1 win
    ev = EvidenceBundle(
        geometry_snr=0.4,
        texture_silicone=0.8,
        texture_reliability=0.9,
        shared_vertex_count=500,
        geometry_mode=GeometryEvidenceMode.CALIBRATED,
    )
    v = render_verdict(ev)
    # should not be confident identity_swap solely from texture when geom matches
    assert not (v.status == ForensicStatus.IDENTITY_SWAP and v.confidence > 0.5)
    assert v.fuzzy_label in (
        FuzzyLabel.SUSPICIOUS_TEXTURE,
        FuzzyLabel.CONSISTENT,
        FuzzyLabel.STRONGLY_MATCHING,
        FuzzyLabel.WEAK_EVIDENCE,
        FuzzyLabel.IDENTITY_ANOMALY,
    )


def test_calendar_does_not_force_original():
    # large delta_years only posthoc flag; posteriors driven by evidence
    strong_diff = EvidenceBundle(
        geometry_snr=4.0,
        texture_silicone=0.1,
        texture_reliability=0.8,
        shared_vertex_count=500,
        geometry_mode=GeometryEvidenceMode.CALIBRATED,
        delta_years=50.0,
    )
    v = render_verdict(strong_diff)
    assert "large_time_gap_posthoc" in v.flags
    assert "large_time_gap_not_used_as_prior" in v.reasoning
    # still not forced to same_person
    assert v.status != ForensicStatus.SAME_PERSON

    # year gap alone cannot make match if evidence is mismatch
    v2 = render_verdict(strong_diff, citations=["pub:example"])
    assert "pub:example" in v2.citations
    assert v2.probabilities["H0"] < 0.5


def test_insufficient_data():
    ev = EvidenceBundle(
        geometry_mode=GeometryEvidenceMode.UNAVAILABLE,
        shared_vertex_count=10,
    )
    v = render_verdict(ev)
    assert v.status == ForensicStatus.INSUFFICIENT_DATA
    assert v.confidence == 0.0
    assert v.fuzzy_label == FuzzyLabel.INSUFFICIENT_DATA


def test_log_update_sanity():
    pr = normalize_priors({"H0": 0.5, "H1": 0.05, "H2": 0.45})
    post = update_posteriors_log(pr, [{"H0": 2.0, "H1": 1.0, "H2": 0.5}])
    assert post["H0"] > pr["H0"]
    assert abs(sum(post.values()) - 1.0) < 1e-9


def test_geometry_likelihood_shapes():
    low = geometry_likelihoods(0.2)
    high = geometry_likelihoods(4.0)
    assert low["H0"] > high["H0"]
    assert high["H2"] > low["H2"]


def test_modules_parse():
    ast.parse((ROOT / "util" / "verdict.py").read_text())


if __name__ == "__main__":
    test_priors_normalize_and_default_not_year_based()
    test_same_person_low_snr()
    test_different_person_high_snr()
    test_texture_high_with_match_not_forced_swap()
    test_calendar_does_not_force_original()
    test_insufficient_data()
    test_log_update_sanity()
    test_geometry_likelihood_shapes()
    test_modules_parse()
    print("ALL ITER8 UNIT TESTS PASSED")
