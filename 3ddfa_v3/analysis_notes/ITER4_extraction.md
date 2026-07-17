# ITER4 — S1 Extraction helpers (completed)

Date: 2026-07-14
Scope: library-side extraction utilities (pose / crop / cache / metrics).
Does not require neural weights.

## Goals

1. Single pose-bucket dictionary (`pose_settings.json`)
2. Letterbox face crop (no aspect squeeze) + re-extract policy
3. Content-hash cache (not mtime)
4. selected_metrics fail-closed (no silent zeros)

## Files

| Path | Role |
|------|------|
| `util/pose_settings.json` | yaw ranges (newapp-compatible) |
| `util/pose_buckets.py` | classify / normalize / aliases |
| `util/letterbox.py` | letterbox resize + reextract policy |
| `util/extraction_cache.py` | blake2 content hash + npz cache |
| `util/selected_metrics.py` | fail-closed metric selection |
| `util/extraction.py` | orchestration record builder |
| `tests/test_iter4_extraction.py` | unit tests |

## Usage

```python
from util.pose_buckets import classify_pose_bucket
from util.extraction import build_extraction_record, build_face_crop_letterbox
from util.selected_metrics import select_metrics
from util.quality_gate import QualityGate

bucket = classify_pose_bucket(yaw_deg, pitch_deg, roll_deg)
q = QualityGate().evaluate(image_bgr, bbox=face_bbox)
rec = build_extraction_record(
    yaw_deg=yaw_deg, pitch_deg=pitch, roll_deg=roll,
    image=image_bgr, bbox=face_bbox,
    quality=q,
    metrics=computed,
    required_metric_keys=("bone_err", "orbit_L"),
)
assert rec.metrics_ok  # fail-closed
```

## Gate status

| Gate | Status |
|------|--------|
| Unified buckets match pose_settings | PASS |
| Letterbox preserves aspect | PASS (if cv2) |
| Re-extract if stretch/unknown | PASS |
| Cache by content hash | PASS |
| Metrics never silent-ok on missing | PASS |
| Full neural S1 e2e | SKIP (no weights) |

## Not in ITER4
- S2 calibration (ITER5)
- S3 compare engine (ITER6)
- Wiring into newapp FastAPI service
