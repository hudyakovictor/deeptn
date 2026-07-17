"""ITER11 tests: snapshot stage (extract once, compute metrics later)."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from util.snapshot import (  # noqa: E402
    SNAPSHOT_SCHEMA_VERSION,
    CanonTransform,
    PhotoSnapshot,
    canon_transform_from_pose,
    landmarks_table,
    load_snapshot,
    save_snapshot,
    to_metric_context,
)

rng = np.random.default_rng(7)
FAILED = 0


def run(name, fn):
    global FAILED
    try:
        fn()
        print(f"PASS {name}")
    except Exception as e:
        FAILED += 1
        print(f"FAIL {name} :: {type(e).__name__}: {e}")


def req(x, msg="condition failed"):
    if not x:
        raise AssertionError(msg)


N = 35709
VERTS = (rng.normal(0, 1, (N, 3)) * np.array([8.0, 10.0, 4.0])).astype(np.float32)
NORMALS = np.zeros((N, 3), dtype=np.float32)
NORMALS[:, 2] = 1.0
TRIS = np.asarray([[i, i + 1, i + 2] for i in range(0, 300, 3)], dtype=np.int64)
LMK = rng.normal(0, 5, (106, 3)).astype(np.float32)


def make_snapshot(**over):
    kw = dict(
        photo_id="p01_f001",
        image_path="person_01/f001.jpg",
        pose_bucket="frontal",
        yaw_deg=2.5,
        pitch_deg=-1.0,
        roll_deg=0.5,
        vertices_raw=VERTS,
        triangles=TRIS,
        canon=canon_transform_from_pose(-1.0, 2.5, 0.5, "frontal"),
        normals_raw=NORMALS,
        alpha_id=rng.normal(0, 1, 80).astype(np.float32),
        exp_params=rng.normal(0, 1, 64).astype(np.float32),
        landmarks_106_raw=LMK,
        visibility_weights=np.ones(N, dtype=np.float32),
        quality={"gate": "ok", "band": "high"},
        extras={"focal": 1015.0, "source_schema": "3ddfa_v3_iter1_v1"},
    )
    kw.update(over)
    return PhotoSnapshot(**kw)


def t01_roundtrip():
    snap = make_snapshot()
    with tempfile.TemporaryDirectory() as td:
        p = save_snapshot(snap, Path(td) / "p01_f001.npz")
        req(p.exists() and p.stat().st_size > 0)
        loaded = load_snapshot(p)
    req(loaded.schema_version == SNAPSHOT_SCHEMA_VERSION)
    req(loaded.photo_id == snap.photo_id and loaded.pose_bucket == "frontal")
    req(np.allclose(loaded.vertices_raw, snap.vertices_raw))
    req(np.array_equal(loaded.triangles, snap.triangles))
    req(np.allclose(loaded.landmarks_106_raw, LMK))
    req(np.allclose(loaded.alpha_id, snap.alpha_id))
    req(np.allclose(loaded.canon.rotation, snap.canon.rotation))
    req(loaded.quality == {"gate": "ok", "band": "high"})
    req(abs(float(loaded.extras["focal"]) - 1015.0) < 1e-9)


def t02_canon_derivation_matches_alignment():
    """canon = undo observed pose, apply bucket canonical pose."""
    from util.alignment import canonical_angles_deg_for_bucket, euler_to_rotation_matrix

    neutral = rng.normal(0, 1, (500, 3))
    pitch, yaw, roll, bucket = -3.0, 18.0, 2.0, "left_threequarter_light"
    R_pose = euler_to_rotation_matrix(np.deg2rad(np.array([pitch, yaw, roll])))
    raw = neutral @ R_pose.T
    canon = canon_transform_from_pose(pitch, yaw, roll, bucket)
    R_canon = euler_to_rotation_matrix(
        np.deg2rad(np.asarray(canonical_angles_deg_for_bucket(bucket), dtype=float))
    )
    expected = neutral @ R_canon.T
    req(np.allclose(canon.apply_points(raw), expected, atol=1e-8), "canon mapping wrong")


def t03_canon_never_baked_in():
    snap = make_snapshot(canon=canon_transform_from_pose(-1.0, 25.0, 0.5, "frontal"))
    vc1 = snap.vertices_canon
    req(not np.allclose(vc1, snap.vertices_raw, atol=1e-6), "canon should differ from raw")
    # re-deriving with a different canon definition needs no re-extraction
    snap.canon = canon_transform_from_pose(-1.0, 25.0, 0.5, "left_threequarter_light")
    vc2 = snap.vertices_canon
    req(not np.allclose(vc1, vc2, atol=1e-6), "changing canon must change derived verts")
    req(np.allclose(np.linalg.norm(vc1, axis=1), np.linalg.norm(vc2, axis=1), atol=1e-3),
        "rotation-only canon must preserve norms")


def t04_normals_stay_unit():
    snap = make_snapshot(canon=canon_transform_from_pose(-3.0, 18.0, 2.0, "frontal"))
    nc = snap.normals_canon
    req(np.allclose(np.linalg.norm(nc, axis=1), 1.0, atol=1e-6))


def t05_landmark_table_both_spaces():
    snap = make_snapshot()
    tab = landmarks_table(snap)
    req(set(tab) == {"raw", "canon"})
    req(tab["raw"].shape == (106, 3) and tab["canon"].shape == (106, 3))
    req(np.allclose(tab["canon"], snap.canon.apply_points(tab["raw"])))


def t06_metric_context_runs_legacy_runner():
    from util.legacy_metrics.runner import compute_single_photo_metrics

    snap = make_snapshot()
    with tempfile.TemporaryDirectory() as td:
        p = save_snapshot(snap, Path(td) / "s.npz")
        loaded = load_snapshot(p)
    ctx = to_metric_context(loaded)
    vals, errs = compute_single_photo_metrics(ctx)
    req(len(vals) >= 100, f"only {len(vals)} values, errors={errs[:3]}")
    bad = [v.spec.name for v in vals if v.value is None or not math.isfinite(float(v.value))]
    req(not bad, f"non-finite: {bad[:5]}")


def t07_snapshot_determinism_for_calibration():
    """Same snapshot -> identical metric values (calibration == main analysis)."""
    from util.legacy_metrics.runner import compute_single_photo_metrics

    snap = make_snapshot()
    with tempfile.TemporaryDirectory() as td:
        p = save_snapshot(snap, Path(td) / "s.npz")
        v1, _ = compute_single_photo_metrics(to_metric_context(load_snapshot(p)))
        v2, _ = compute_single_photo_metrics(to_metric_context(load_snapshot(p)))
    m1 = {(v.spec.name, v.spec.implementation): float(v.value) for v in v1}
    m2 = {(v.spec.name, v.spec.implementation): float(v.value) for v in v2}
    req(m1 == m2, "metric values differ across identical snapshot loads")


def t08_schema_mismatch_fails_closed():
    snap = make_snapshot()
    snap.schema_version = "snapshot_v0_old"
    with tempfile.TemporaryDirectory() as td:
        p = save_snapshot(snap, Path(td) / "s.npz")
        try:
            load_snapshot(p)
        except ValueError as e:
            req("schema" in str(e).lower())
        else:
            raise AssertionError("old schema accepted")


def t09_validation_rejects_bad_inputs():
    for bad_kwargs in (
        {"vertices_raw": np.full((10, 3), np.nan, dtype=np.float32),
         "triangles": np.zeros((0, 3), dtype=np.int64), "normals_raw": None,
         "visibility_weights": None},
        {"triangles": np.asarray([[0, 1, N + 5]], dtype=np.int64)},
        {"visibility_weights": np.ones(3, dtype=np.float32)},
    ):
        try:
            make_snapshot(**bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid input: {list(bad_kwargs)}")
    try:
        CanonTransform(rotation=np.eye(3) * 2.0, translation=np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("non-orthonormal rotation accepted")


def t10_storage_budget():
    snap = make_snapshot()
    with tempfile.TemporaryDirectory() as td:
        p = save_snapshot(snap, Path(td) / "s.npz")
        size_mb = p.stat().st_size / 1e6
    req(size_mb < 3.0, f"snapshot too large: {size_mb:.2f} MB")


TESTS = [
    ("01_roundtrip", t01_roundtrip),
    ("02_canon_derivation_matches_alignment", t02_canon_derivation_matches_alignment),
    ("03_canon_never_baked_in", t03_canon_never_baked_in),
    ("04_normals_stay_unit", t04_normals_stay_unit),
    ("05_landmark_table_both_spaces", t05_landmark_table_both_spaces),
    ("06_metric_context_runs_legacy_runner", t06_metric_context_runs_legacy_runner),
    ("07_snapshot_determinism_for_calibration", t07_snapshot_determinism_for_calibration),
    ("08_schema_mismatch_fails_closed", t08_schema_mismatch_fails_closed),
    ("09_validation_rejects_bad_inputs", t09_validation_rejects_bad_inputs),
    ("10_storage_budget", t10_storage_budget),
]

for name, fn in TESTS:
    run(name, fn)

print(f"SUMMARY iter11 total={len(TESTS)} fail={FAILED}")
raise SystemExit(1 if FAILED else 0)
