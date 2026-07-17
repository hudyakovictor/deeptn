# New version: 50 key analyses and fixes

Date: 2026-07-14
Target: `/data/review_3ddfa`

## Result

- Required analyses: **50/50 passed after fixes**
- Additional checks: **10/10 passed**
- Total final audit: **60/60 passed**
- Regression: **ITER1–9 passed; ITER10 passed; metric catalog tests passed**
- Python compileall: **passed**

## Confirmed defects fixed

1. Removed duplicate `profile_trim_keep_ratio` definition and duplicate constants/imports in `geometry_metrics.py`.
2. Fixed `identity_scoring_config.json` path; per-bucket core/experimental/disabled filters no longer load an empty config.
3. Made geometry helpers fail safely for NaN/Inf; `bounded_score_from_error` now returns a conservative score for invalid input.
4. Rejected negative/non-finite metric weights instead of allowing mathematically invalid weighted errors.
5. Added explicit validation for zero/negative software z-buffer resolution and invalid tolerance.
6. Added explicit validation for zero/negative source and target letterbox dimensions.
7. Rejected negative/non-finite Bayesian priors and all-zero priors; output is limited to H0/H1/H2 and normalized.
8. Made catalog summary derive recovered count from the actual CSV rather than a stale hard-coded 1251 field.
9. Restored catalog family compatibility: `F_zone` relations and `F13` eye-mask diagnostics are now reachable.
10. Corrected reconstructed catalog routing: `pair_zone_*` → `pair_zone_residuals.py`/pair; dense residuals → pair/F9.
11. Registry/catalog now match exactly: **1251 CSV rows = 1251 unique specs**.
12. Added persistent audit runner `tests/audit_50_new_version.py`.

## Coverage areas

- Imports and compilation
- Duplicate definitions and unsafe exception syntax
- Nine pose buckets, aliases, boundaries, NaN/Inf yaw
- Catalog schema, names, sides, scopes, dynamic counts
- Registry uniqueness, implementation routing, per-view availability
- Identity-scoring bucket configuration
- Geometry numerical stability and degenerate input
- Visibility/frontface/backface/z-buffer/error handling
- Letterbox shape and dimension validation
- Cache/hash determinism
- Fail-closed selected metrics and texture
- Bayesian priors/posteriors
- Report fingerprint stability
- UV original-pixel gating and tiny-sample rejection
- Zone index/hash invariants
- Legacy runner degradation without crash

## Remaining limitation (not a failing test)

The old reference catalog reports **2199** metrics. The currently available local active CSV contains **1251** recovered unique metric definitions; all 1251 are now routable. The remaining **948** require replacing the reconstructed CSV with the original full `metric_catalog_active.csv` from `deeptn`; formulas were not invented.

## Commands

```bash
python3 tests/audit_50_new_version.py
python3 tests/run_all_iters.py
python3 tests/test_iter10_geometry_texture_metrics.py
python3 tests/test_metrics_catalog_restore.py
python3 -m compileall -q util uv_module tests
```
