# ITER7 — Texture / silicone channel (completed)

Date: 2026-07-14

## Goals

1. Remove synthetic_prob floor ~0.75 saturation
2. Quality penalty toward neutral on bad images
3. Fail-closed on insufficient skin coverage
4. Fractal not sole/dominating evidence
5. Library-local (no reference JSON required for core path)

## Files

| Path | Role |
|------|------|
| `util/texture.py` | metrics, quality, synthetic score |
| `tests/test_iter7_texture.py` | unit tests |

## Usage

```python
from util.texture import analyze_texture

profile = analyze_texture(face_crop_bgr, skin_mask)
if profile.ok:
    print(profile.synthetic_prob, profile.raw_synthetic_prob, profile.reliability)
```

## Gate

| Gate | Status |
|------|--------|
| no hard 0.75 floor | PASS |
| smooth > textured synthetic | PASS (cv2) |
| empty mask fail-closed | PASS |
| quality penalty | PASS |
| real silicone holdout AUC | SKIP (no labeled corpus in sandbox) |

## Not in ITER7
- Verdict Bayes (ITER8)
- Full skin_authenticity reference v4 wiring (optional later)
- Report (ITER9)
