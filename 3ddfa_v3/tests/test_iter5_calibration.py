"""ITER5 unit tests: calibration noise floor, health, effective N."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.calibration import (  # noqa: E402
    NoiseModel,
    NoiseObservation,
    apply_person_baseline,
    block_bootstrap_ci,
    bucket_metric_health,
    build_calibration_summary,
    build_person_baselines,
    effective_sample_size,
    linear_snr,
    mad,
    metric_health_status,
    quality_band_from_score,
    same_person_residual_stats,
)


def test_mad_and_effective_n():
    assert mad([1, 2, 3, 100]) < 10
    n_eff = effective_sample_size(500, ar1_rho=0.95)
    assert 10 < n_eff < 20  # ~12.8
    assert effective_sample_size(0) == 0


def test_linear_snr():
    assert linear_snr(0.01, 0.02) == 0.0
    assert linear_snr(0.05, 0.01) > 1.0


def test_metric_health_status():
    assert metric_health_status(0.1, 0.01, 20) == "stable"
    assert metric_health_status(0.4, 0.1, 20) == "marginal"
    assert metric_health_status(0.9, 0.2, 20) == "replace"
    assert metric_health_status(0.1, 0.01, 2) == "insufficient"


def test_person_baselines():
    samples = [
        ("p1", "frontal", 0.10),
        ("p1", "frontal", 0.12),
        ("p1", "left_profile", 0.20),
        ("p2", "frontal", 0.30),
    ]
    base = build_person_baselines(samples)
    assert abs(base["p1"]["frontal"] - 0.11) < 1e-9
    adj = apply_person_baseline(0.15, "p1", "p1", "frontal", base)
    assert abs(adj - 0.04) < 1e-9
    # different persons: no subtract
    assert apply_person_baseline(0.15, "p1", "p2", "frontal", base) == 0.15


def test_noise_model_predict():
    model = NoiseModel()
    for i in range(12):
        model.add_observation(
            NoiseObservation(
                photo_id_a=f"a{i}",
                photo_id_b=f"b{i}",
                person_id="person_01",
                bucket="frontal",
                pose_delta_mag=5.0,
                zone_errors={"chin": 0.02 + 0.001 * (i % 3), "orbit_L": 0.03},
                quality_score=0.8,
                quality_band="high",
            )
        )
    n = model.predict_noise("chin", pose_delta_mag=10.0, bucket="frontal", quality_band="high")
    assert n > 0
    assert model.get_reliability("chin") > 0.05
    d = model.to_dict()
    assert d["n_observations"] == 12
    assert "chin" in d["zone_profiles"]


def test_build_calibration_summary_multi_person():
    records = []
    rng = np.random.default_rng(0)
    for person in ("person_01", "person_02", "person_03", "person_04", "person_05"):
        for bucket, yaw in [("frontal", 0), ("left_threequarter_mid", -30), ("right_threequarter_mid", 30)]:
            # person_03 right-only: skip left for p03
            if person == "person_03" and bucket.startswith("left"):
                continue
            # person_02 low-res → noisier metrics
            noise = 0.05 if person == "person_02" else 0.01
            for i in range(10):
                records.append({
                    "status": "ready",
                    "person_id": person,
                    "photo_id": f"{person}_{bucket}_{i}",
                    "bucket": bucket,
                    "pose": {"yaw": yaw, "pitch": 0},
                    "quality": {"overall_score": 0.5 if person == "person_02" else 0.9},
                    "metrics": {
                        "chin": float(0.1 + rng.normal(0, noise)),
                        "orbit_L": float(0.12 + rng.normal(0, noise)),
                    },
                })
    summary = build_calibration_summary(records)
    d = summary.to_dict()
    assert d["observation_count"] > 0
    assert d["bucket_coverage"]["frontal"] > 0
    assert "person_01" in d["person_coverage"]
    health = bucket_metric_health(summary, "frontal")
    assert len(health) >= 1
    # person_02 mid quality still included if above threshold
    assert summary.person_coverage.get("person_02", 0) > 0


def test_block_bootstrap_and_residuals():
    rng = np.random.default_rng(1)
    # AR-ish series
    x = [0.1]
    for _ in range(40):
        x.append(0.9 * x[-1] + 0.01 * rng.normal())
    ci = block_bootstrap_ci(x, block_size=4, n_boot=50)
    assert ci["n"] == 41
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    stats = same_person_residual_stats(x, block_size=4)
    assert stats["n"] == 41 and stats["mad"] >= 0


def test_quality_band():
    assert quality_band_from_score(0.9) == "high"
    assert quality_band_from_score(0.5) == "mid"
    assert quality_band_from_score(0.2) == "reject"


def test_modules_parse():
    ast.parse((ROOT / "util" / "calibration.py").read_text())


if __name__ == "__main__":
    test_mad_and_effective_n()
    test_linear_snr()
    test_metric_health_status()
    test_person_baselines()
    test_noise_model_predict()
    test_build_calibration_summary_multi_person()
    test_block_bootstrap_and_residuals()
    test_quality_band()
    test_modules_parse()
    print("ALL ITER5 UNIT TESTS PASSED")
