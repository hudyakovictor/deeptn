"""ITER9 report packaging + acceptance suite."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.report import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    acceptance_checks,
    build_provenance,
    build_report,
    content_fingerprint,
    load_report_json,
    save_report_json,
)
from util.verdict import EvidenceBundle, GeometryEvidenceMode, render_verdict  # noqa: E402
from util.compare import PairCompareInput, compare_pair  # noqa: E402
from util.alignment import euler_to_rotation_matrix  # noqa: E402
from util.zones import MACRO_BONE_INDICES  # noqa: E402


def _synthetic_face(n=35709, seed=0):
    rng = np.random.default_rng(seed)
    verts = rng.normal(0, 0.02, size=(n, 3)).astype(np.float64)
    for zname, ids in MACRO_BONE_INDICES.items():
        for i, vid in enumerate(list(ids)[:40]):
            if 0 <= int(vid) < n:
                sx = -0.3 if zname.endswith("_L") else (0.3 if zname.endswith("_R") else 0.0)
                sy = 0.2 if "brow" in zname or "forehead" in zname else (-0.25 if "chin" in zname else 0.0)
                verts[int(vid)] = np.array([sx, sy, 0.5], dtype=np.float64)
    normals = verts - verts.mean(0)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    normals = normals.copy()
    normals[:, 2] = np.abs(normals[:, 2]) + 0.2
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return verts, normals.astype(np.float32)


def test_build_and_save_report(tmp_path=None):
    va, na = _synthetic_face(seed=1)
    a = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                         angles_deg=np.array([0.,0.,0.]), pose_bucket="frontal",
                         alpha_id=np.ones(80), person_id="p1", photo_id="a.jpg")
    b = PairCompareInput(vertices=va.copy(), normals=na.copy(), vertices_camera=va.copy(),
                         angles_deg=np.array([0.,0.,0.]), pose_bucket="frontal",
                         alpha_id=np.ones(80), person_id="p1", photo_id="b.jpg")
    cmp_res = compare_pair(a, b, min_shared=20)
    verd = render_verdict(EvidenceBundle(
        geometry_snr=0.4 if cmp_res.status == "ok" else None,
        geometry_error=cmp_res.bone_raw_geometry_error or cmp_res.raw_geometry_error,
        predicted_noise=0.02,
        texture_silicone=0.25,
        texture_reliability=0.8,
        shared_vertex_count=cmp_res.shared_count,
        geometry_mode=GeometryEvidenceMode.CALIBRATED if cmp_res.status == "ok" else GeometryEvidenceMode.UNAVAILABLE,
        id_cosine_distance=0.01,
    ))
    prov = build_provenance(image_hashes={"a": "aaa", "b": "bbb"}, topology_hash="t1")
    report = build_report(
        photo_a="a.jpg",
        photo_b="b.jpg",
        extraction_a={"image_hash": "aaa", "pose_bucket": "frontal", "metrics_ok": True},
        extraction_b={"image_hash": "bbb", "pose_bucket": "frontal", "metrics_ok": True},
        compare=cmp_res.to_dict(),
        texture_a={"ok": True, "synthetic_prob": 0.25, "raw_synthetic_prob": 0.22},
        texture_b={"ok": True, "synthetic_prob": 0.28, "raw_synthetic_prob": 0.24},
        verdict=verd.to_dict(),
        provenance=prov,
    )
    assert report.schema_version == REPORT_SCHEMA_VERSION
    assert report.acceptance.get("overall_pass") is True
    assert "no_calendar" in report.summary_text or "priors_not_calendar" in report.summary_text

    out = Path("/tmp/iter9_report.json")
    save_report_json(report, out)
    loaded = load_report_json(out)
    assert loaded["pair"]["photo_a"] == "a.jpg"
    assert loaded["verdict"]["probabilities"]
    fp1 = content_fingerprint(loaded)
    fp2 = content_fingerprint(load_report_json(out))
    assert fp1 == fp2


def test_acceptance_fails_without_schema():
    bad = {"pair": {"photo_a": "a", "photo_b": "b"}, "verdict": {}, "provenance": {}}
    acc = acceptance_checks(bad)
    assert acc["overall_pass"] is False
    assert acc["has_schema"]["pass"] is False


def test_acceptance_rejects_missing_no_calendar_reason_when_probs():
    # verdict with probs but without no_calendar_prior reasoning should fail gate
    rep = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pair": {"photo_a": "a", "photo_b": "b"},
        "verdict": {
            "status": "same_person",
            "probabilities": {"H0": 0.7, "H1": 0.1, "H2": 0.2},
            "reasoning": ["geometry_snr=0.2"],
        },
        "compare": {"status": "ok", "shared_count": 100},
        "provenance": {"schema_version": "x", "library_modules": ["util.verdict"], "notes": ["no_calendar_forced_verdict"]},
        "texture": {},
    }
    acc = acceptance_checks(rep)
    assert acc["no_calendar_prior_claim"]["pass"] is False


def test_end_to_end_library_stack_smoke():
    """Mini acceptance: pose → compare → verdict → report."""
    from util.pose_buckets import classify_pose_bucket
    from util.calibration import linear_snr
    from util.selected_metrics import select_metrics

    assert classify_pose_bucket(0.0) == "frontal"
    assert linear_snr(0.05, 0.01) > 1.0
    sm = select_metrics({"a": 1.0}, ["a"])
    assert sm.ok
    va, na = _synthetic_face(seed=9)
    mesh = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                            angles_deg=np.array([0.,0.,0.]), pose_bucket="frontal")
    res = compare_pair(mesh, mesh, min_shared=20)
    assert res.status == "ok"
    v = render_verdict(EvidenceBundle(
        geometry_error=res.raw_geometry_error,
        predicted_noise=0.02,
        geometry_snr=linear_snr(res.raw_geometry_error or 0.0, 0.02),
        texture_silicone=0.2,
        texture_reliability=0.7,
        shared_vertex_count=res.shared_count,
        geometry_mode=GeometryEvidenceMode.CALIBRATED,
    ))
    r = build_report(photo_a="x", photo_b="y", compare=res.to_dict(), verdict=v.to_dict())
    assert r.acceptance["overall_pass"] is True


def test_modules_parse():
    ast.parse((ROOT / "util" / "report.py").read_text())


if __name__ == "__main__":
    test_build_and_save_report()
    test_acceptance_fails_without_schema()
    test_acceptance_rejects_missing_no_calendar_reason_when_probs()
    test_end_to_end_library_stack_smoke()
    test_modules_parse()
    print("ALL ITER9 UNIT TESTS PASSED")
