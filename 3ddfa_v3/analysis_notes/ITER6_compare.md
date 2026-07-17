# ITER6 — Pair Compare / S3 (completed)

Date: 2026-07-14

## Goals

1. Visibility ∩ before alignment
2. Rigid Umeyama `allow_scale=False` (bone anchors when possible)
3. Bone-zone weighted metrics + summary
4. Optional SNR / person baseline hooks
5. Fail closed on insufficient shared visibility

## Files

| Path | Role |
|------|------|
| `util/compare.py` | PairCompareInput/Result, prepare, compare_pair |
| `tests/test_iter6_compare.py` | unit tests |

## Usage

```python
from util.compare import PairCompareInput, compare_pair

a = PairCompareInput(vertices=va, normals=na, vertices_camera=va_cam,
                     angles_deg=ang_a, pose_bucket="frontal", alpha_id=id_a)
b = PairCompareInput(vertices=vb, normals=nb, vertices_camera=vb_cam,
                     angles_deg=ang_b, pose_bucket="frontal", alpha_id=id_b)
res = compare_pair(a, b, predicted_noise=0.02)
assert res.status == "ok"
print(res.bone_raw_geometry_error, res.snr)
```

## Gate

| Gate | Status |
|------|--------|
| identical meshes ~0 error | PASS |
| shift increases error | PASS |
| insufficient visibility | PASS |
| bone-fit path | PASS (notes) |
| real multi-pose leave-one-out | SKIP (needs extracted meshes) |

## Not in ITER6
- Texture channel (ITER7)
- Verdict (ITER8)
- Report packaging (ITER9)
