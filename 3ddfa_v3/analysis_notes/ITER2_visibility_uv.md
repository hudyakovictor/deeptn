# ITER2 — Visibility + UV + QualityGate (completed)

Date: 2026-07-14
Scope: library sensor layer inside `3ddfa_v3` (no pipeline S1 rewrite).

## Goals

1. Unified visibility API (analysis hard vs beauty soft)
2. Real `angle_threshold_deg` (fix 0.001 floor leak on analysis)
3. Package `uv_module` inside library tree with analysis/beauty split
4. `quality_gate` without `core.contracts`
5. Bridge from `ReconstructionResult` → UV recon_dict

## Changes

### New / updated
| Path | Role |
|------|------|
| `util/visibility.py` | vertex + triangle visibility; hard/soft modes; z-buffer; yaw fade |
| `util/quality_gate.py` | blur/noise/jpeg/motion; plain dict API |
| `uv_module/*` | full UV package copied from review_uv |
| `uv_module/visibility.py` | delegates to util; hard analysis threshold |
| `uv_module/hd_uv_generator.py` | dual weights; bake uses **analysis** only |
| `util/reconstruction_api.py` | `attach_visibility`, `recon_dict_for_uv` |
| `tests/test_iter2_visibility_uv.py` | unit tests |

### Critical fix
Pre-ITER2 triangle visibility used:
`max(cos^gamma, 0.001)` even for near-backfaces → occluded side leaked into UV analysis.

Now:
- **analysis**: `weight=0` if `cos < cos(threshold)`; floor default 0
- **beauty**: soft falloff; floor 0.001 allowed for hole-free render fill

### UV generate contract (unchanged return tuple)
```python
uv_analysis, uv_beauty, mask, conf, aux = HDUVTextureGenerator().generate(image, recon_dict)
# aux["tri_visibility_analysis"], aux["tri_visibility_beauty"]
```

## Gate status

| Gate | Status |
|------|--------|
| Analysis hard threshold | PASS (unit) |
| Beauty ≠ analysis | PASS (unit) |
| uv_module import | PASS |
| quality_gate no contracts | PASS (API) |
| quality_gate runtime | PASS if cv2 present; else skip |
| E2E bake with real mesh+photo | SKIP (no assets/weights in sandbox) |

## Usage

```python
from util.visibility import compute_visibility
from util.quality_gate import QualityGate
from util.reconstruction_api import attach_visibility, recon_dict_for_uv, run_reconstruction
from uv_module import HDUVTextureGenerator, HDUVConfig

q = QualityGate().evaluate(image_bgr, bbox=face_bbox)
rec = run_reconstruction(face_model, identity_only=True)
rec = attach_visibility(rec, angle_threshold_deg=75.0)
uv = HDUVTextureGenerator(HDUVConfig(device="cpu")).generate(image_rgb, recon_dict_for_uv(rec))
```

## Not in ITER2
- Umeyama / zones (ITER3)
- pipeline S1 extraction wiring (ITER4)
- full UV E2E with net_recon weights
