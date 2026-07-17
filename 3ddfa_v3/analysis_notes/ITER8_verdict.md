# ITER8 — Verdict (completed)

Date: 2026-07-14

## Goals

1. H0/H1/H2 posteriors from geometry SNR + texture (single evidence path)
2. **No calendar / era-named priors** forcing ORIGINAL
3. Publications / chronology only post-hoc (flags/citations)
4. Fail-closed on insufficient geometry
5. Texture+match ≠ automatic swap

## Files

| Path | Role |
|------|------|
| `util/verdict.py` | EvidenceBundle, render_verdict |
| `tests/test_iter8_verdict.py` | unit tests |

## Usage

```python
from util.verdict import EvidenceBundle, GeometryEvidenceMode, render_verdict

ev = EvidenceBundle(
    geometry_snr=0.5,
    texture_silicone=0.3,
    texture_reliability=0.8,
    shared_vertex_count=800,
    geometry_mode=GeometryEvidenceMode.CALIBRATED,
)
v = render_verdict(ev, citations=["optional_pub_id"])
print(v.status, v.probabilities, v.flags)
```

## Gate

| Gate | Status |
|------|--------|
| low SNR → same_person | PASS |
| high SNR not forced ORIGINAL by year | PASS |
| calendar only posthoc | PASS |
| insufficient data | PASS |
| sensitivity to priors documented via DEFAULT_PRIORS | PASS |

## Not in ITER8
- Report packaging (ITER9)
- Full fuzzy_bayes chronology graphs
- Multi-model double-counting (explicitly avoided)
