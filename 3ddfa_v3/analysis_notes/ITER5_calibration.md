# ITER5 — Calibration / S2 (completed)

Date: 2026-07-14
Scope: library calibration utilities (noise floor, health, effective N).

## Goals

1. Multi-person pose×quality noise model
2. Metric health buckets: stable / marginal / replace / insufficient
3. Effective N + block bootstrap CI
4. Person baselines (same-person residual subtraction)
5. Notes for weak cells (right-only / low-res)

## Files

| Path | Role |
|------|------|
| `util/calibration.py` | NoiseModel, summary, SNR, baselines |
| `tests/test_iter5_calibration.py` | unit tests |

## Usage

```python
from util.calibration import (
    NoiseModel, NoiseObservation, build_calibration_summary,
    build_person_baselines, apply_person_baseline, linear_snr,
)

model = NoiseModel()
model.add_observation(NoiseObservation(
    photo_id_a="a", photo_id_b="b", person_id="person_01",
    bucket="frontal", pose_delta_mag=8.0,
    zone_errors={"chin": 0.02}, quality_band="high",
))
noise = model.predict_noise("chin", 10.0, bucket="frontal", quality_band="high")
snr = linear_snr(measured_error, noise)

summary = build_calibration_summary(extraction_records)
baselines = build_person_baselines([(pid, bucket, err), ...])
adj = apply_person_baseline(err, pid, pid, bucket, baselines)
```

## Gate status

| Gate | Status |
|------|--------|
| effective N AR(1) | PASS |
| health statuses | PASS |
| multi-person summary | PASS |
| person baseline | PASS |
| noise model predict | PASS |
| real person_01..05 HPE run | SKIP (no full extract in sandbox) |

## Not in ITER5
- S3 pair compare engine (ITER6)
- texture silicone channel (ITER7)
- verdict Bayes (ITER8)
