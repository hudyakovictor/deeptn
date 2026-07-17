# ITER9 — Report + hardening (completed)

Date: 2026-07-14

## Goals

1. Versioned forensic report JSON
2. Provenance manifest (hashes, modules, zone indices)
3. Acceptance gates (no calendar prior, posteriors sum, shared visibility)
4. End-to-end smoke across library stack
5. Master test runner ITER1–9

## Files

| Path | Role |
|------|------|
| `util/report.py` | build/save/load report + acceptance |
| `tests/test_iter9_report_acceptance.py` | unit + e2e smoke |
| `tests/run_all_iters.py` | full regression |

## Usage

```python
from util.report import build_report, save_report_json, build_provenance
from util.verdict import render_verdict, EvidenceBundle, GeometryEvidenceMode

report = build_report(
    photo_a="a.jpg", photo_b="b.jpg",
    compare=compare_dict,
    texture_a=tex_a, texture_b=tex_b,
    verdict=render_verdict(EvidenceBundle(...)).to_dict(),
    provenance=build_provenance(image_hashes={"a": "...", "b": "..."}),
)
save_report_json(report, "out/report.json")
assert report.acceptance["overall_pass"]
```

## Gate

| Gate | Status |
|------|--------|
| report roundtrip JSON | PASS |
| acceptance overall_pass on good report | PASS |
| fails without schema / no_calendar reason | PASS |
| e2e pose→compare→verdict→report | PASS |
| optional MICA/Pixel3DMM | DEFERRED (v1.1) |

## New version readiness

Library sensor+pipeline utils v1 (ITER1–9) is **code-complete** in `/data/review_3ddfa`.
Still needs real weights/assets + person_01..05 extract for production calibration numbers.
