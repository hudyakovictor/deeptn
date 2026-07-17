# 3DDFA-V3 Forensic Library Extensions (ITER1–9)

Implemented 2026-07-14 inside this tree (`util/`, `uv_module/`, `tests/`).

## Iterations

1. Sensor foundation — alpha export, identity mesh, no in-place bugs
2. Visibility + UV + quality_gate
3. Alignment (Umeyama) + zones
4. S1 extraction helpers (pose buckets, letterbox, cache, metrics fail-closed)
5. Calibration / noise floor
6. Pair compare
7. Texture / silicone channel (no 0.75 floor)
8. Verdict H0/H1/H2 (no calendar priors)
9. Report + acceptance suite

## Run tests

```bash
cd /data/review_3ddfa
python3 tests/run_all_iters.py
```

## Pipeline call order

```text
quality_gate → reconstruct(forward identity_only) → visibility → UV
  → extraction record (pose/letterbox/cache)
  → calibration noise predict
  → compare_pair
  → analyze_texture
  → render_verdict
  → build_report / save_report_json
```

## Not included (v1.1)

- MICA / Pixel3DMM offline validators
- Full newapp FastAPI wiring
- Training / finetune
