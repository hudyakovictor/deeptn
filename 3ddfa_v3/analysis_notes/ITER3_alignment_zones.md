# ITER3 — Alignment + Zones (completed)

Date: 2026-07-14
Scope: library geometry utils inside `3ddfa_v3` (no pipeline compare rewrite).

## Goals

1. Pure Umeyama alignment (`allow_scale=False` forensic default)
2. BFM zone indices + zone metrics API without `core.*`
3. Expression exclusion helper
4. Bone-priority summary
5. Indices hash for topology provenance

## Changes

| Path | Role |
|------|------|
| `util/alignment.py` | `rigid_umeyama`, robust SVD, GPA, mesh align |
| `util/zones.py` | `compute_zone_metrics`, schema, expression mask |
| `util/zone_indices_data.py` | MACRO_BONE_INDICES + ZONE_CONFIG (from newapp) |
| `util/geom_utils.py` | weighted error / bounded score / face scale |
| `tests/test_iter3_alignment_zones.py` | unit tests |

## Defaults

- Forensic align: **no scale** (`allow_scale=False`)
- Soft zones excluded by default: `nose_wing_L/R`
- Expression exclude helper: lips (`upper_lip`, `lower_lip`)
- Core macro order prefers bone zones first when `exclusive_vertices=True`

## Usage

```python
from util.alignment import align_meshes_shared, rigid_umeyama
from util.zones import compute_zone_metrics, summarize_bone_priority_metrics, apply_expression_exclusion_mask
from util.visibility import compute_visibility

vis_a = compute_visibility(vertices_camera=va, normals_camera=na, ...)
vis_b = compute_visibility(vertices_camera=vb, normals_camera=nb, ...)
shared = vis_a.binary_mask & vis_b.binary_mask
shared = apply_expression_exclusion_mask(shared)

align = align_meshes_shared(va, vb, shared_mask=shared, allow_scale=False)
idx = np.where(shared)[0]
zones = compute_zone_metrics(
    aligned_points_a=align.source_aligned[idx],
    points_b=vb[idx],
    shared_indices=idx,
    shared_weights=vis_a.cosine_weights[idx],
    exclusive_vertices=False,  # or True for non-overlapping zone verts
)
summary = summarize_bone_priority_metrics(zones)
```

## Gate status

| Gate | Status |
|------|--------|
| Umeyama recovers known R,t | PASS |
| no-scale default | PASS |
| shared-mask align | PASS |
| zone indices < 35709 | PASS |
| identical meshes → ~0 error | PASS |
| shift detection | PASS |
| E2E on real recon meshes | SKIP (no weights/assets) |

## Not in ITER3
- pipeline S1 extraction (ITER4)
- calibration / compare engines
- verdict bands / policy thresholds beyond geometry helpers
