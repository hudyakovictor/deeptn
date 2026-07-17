# ITER1 — 3DDFA sensor foundation (completed)

Date: 2026-07-14
Scope: library sensor only (`3ddfa_v3`), no pipeline rewrite.

## Goals (from plan)

1. Export full alpha (`id/exp/alb/angle/sh/trans`)
2. `identity_only` + soft-neutral meshes
3. Remove in-place mutations (`to_camera`, UV process, landmark Y-flip)
4. `ReconstructionResult` contract + topology/basis hashes
5. Crop/camera metadata in artifact

## Changes

### `model/recon.py`
- `process_uv`: copy input, no mutation
- `to_camera`: clone before Z rewrite
- `forward(..., identity_only=False, neutral_expression=False, neutral_scale=0.1)`
- Always exports:
  - `alpha_raw`, `alpha_id`, `alpha_exp`, `alpha_exp_used`, `alpha_alb`, `alpha_angle`, `alpha_angle_deg`, `alpha_sh`, `alpha_trans`
  - `v3d_model`, `v3d_identity`, `v3d_transformed`
  - `normals_*`, `rotation_matrix`
  - `camera` contract (focal=1015, pp=112, size=224, distance=10)
  - `trans_params`, `schema_version`, `coordinate_spaces`, `expression_mode`
- `visible_idx` always computed/exported (numpy in result_dict; tensor kept for internal render)

### `util/io.py`
- Landmark Y-flip uses `.copy()` so `result_dict` landmarks stay crop-y-up

### New files
- `util/types.py` — `ReconstructionResult`, `CameraContract`, hash helpers
- `util/reconstruction_api.py` — convert/save/load/assert contract
- `tests/test_iter1_sensor.py` — unit tests without weights

## Gate status

| Gate | Status |
|------|--------|
| No in-place `to_camera` | PASS (unit) |
| No in-place `process_uv` | PASS (unit) |
| Landmark flip copy | PASS (unit) |
| Alpha export contract | PASS (unit via synthetic dict) |
| identity mesh field present | PASS (unit) |
| Artifact roundtrip | PASS (unit) |
| End-to-end with real weights | SKIPPED (assets not in sandbox zip) |

## How to use

```python
# After detector set face_model.input_img / trans_params:
result_dict = face_model.forward(identity_only=True)
# or
from util.reconstruction_api import run_reconstruction, save_reconstruction_artifact, assert_iter1_contract
rec = run_reconstruction(face_model, identity_only=True, image_path=path)
assert_iter1_contract(rec)
save_reconstruction_artifact(rec, "out/sample.npz")
```

## Not in ITER1 (next iterations)
- visibility soft/hard split (ITER2)
- uv_module package merge (ITER2)
- Umeyama / zones (ITER3)
- pipeline S1 integration (ITER4)

## Backups
- `model/recon.py.bak`
- `util/io.py.bak`
