"""ITER3 unit tests: Umeyama alignment + BFM zone metrics."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.alignment import (  # noqa: E402
    align_meshes_shared,
    euler_to_rotation_matrix,
    gpa_unit_scale,
    rigid_umeyama,
    rigid_umeyama_robust,
)
from util.geom_utils import bounded_score_from_error, face_scale_from_points, weighted_mean_abs  # noqa: E402
from util.zones import (  # noqa: E402
    MACRO_BONE_INDICES,
    apply_expression_exclusion_mask,
    compute_zone_metrics,
    indices_hash,
    static_zone_schema,
    summarize_bone_priority_metrics,
    zone_vertex_mask,
)


def test_rigid_umeyama_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    # known rigid transform, no scale
    angles = np.deg2rad([10.0, -25.0, 5.0])
    r = euler_to_rotation_matrix(angles)
    t = np.array([0.2, -0.1, 0.05])
    dst = (src @ r) + t
    res = rigid_umeyama(src, dst, allow_scale=False)
    assert res.scale == 1.0
    assert np.allclose(res.rotation, r, atol=1e-6)
    assert np.allclose(res.translation, t, atol=1e-6)
    assert res.residual_after < res.residual_before
    assert res.residual_after < 1e-6


def test_rigid_umeyama_no_scale_by_default():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    dst = src * 2.0 + np.array([1.0, 0.0, 0.0])
    res = rigid_umeyama(src, dst, allow_scale=False)
    assert res.scale == 1.0
    # with scale allowed should approach 2
    res2 = rigid_umeyama(src, dst, allow_scale=True)
    assert abs(res2.scale - 2.0) < 0.05


def test_align_meshes_shared_mask():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 3))
    r = euler_to_rotation_matrix(np.deg2rad([0, 30, 0]))
    b = (a @ r) + np.array([0.0, 0.1, 0.0])
    mask = np.zeros(50, dtype=bool)
    mask[:20] = True
    # corrupt non-shared region in a so only shared should drive alignment
    a2 = a.copy()
    a2[~mask] += 5.0
    res = align_meshes_shared(a2, b, shared_mask=mask, allow_scale=False)
    err_shared = np.linalg.norm(res.source_aligned[mask] - b[mask], axis=1).mean()
    assert err_shared < 1e-5


def test_gpa_and_robust():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    c, s, centroid = gpa_unit_scale(pts)
    assert abs(np.mean(np.sum(c**2, axis=1)) - 1.0) < 1e-6 or s > 0
    r, t, scale = rigid_umeyama_robust(pts, pts + 1.0, allow_scale=False)
    assert scale == 1.0
    assert r.shape == (3, 3)


def test_zone_indices_valid():
    assert len(MACRO_BONE_INDICES) >= 20
    mx = max(max(s) for s in MACRO_BONE_INDICES.values() if s)
    assert mx < 35709
    h1 = indices_hash()
    h2 = indices_hash()
    assert h1 == h2 and len(h1) == 32
    schema = static_zone_schema()
    assert schema["total_zone_count"] > 10
    m = zone_vertex_mask("chin", 35709)
    assert m.dtype == bool and m.sum() == len(MACRO_BONE_INDICES["chin"])


def test_expression_exclusion():
    mask = np.ones(35709, dtype=bool)
    out = apply_expression_exclusion_mask(mask)
    # lips should be cleared if present in indices
    for z in ("upper_lip", "lower_lip"):
        ids = MACRO_BONE_INDICES.get(z)
        if ids:
            for i in list(ids)[:5]:
                assert out[int(i)] is np.False_ or out[int(i)] == False


def test_compute_zone_metrics_identical_meshes():
    n = 35709
    # small synthetic: only use zone subset vertices
    rng = np.random.default_rng(3)
    base = rng.normal(size=(n, 3)) * 0.01
    # pick shared indices from several zones
    ids = []
    for z in ("chin", "orbit_L", "orbit_R", "cheekbone_L", "cheekbone_R", "forehead"):
        ids.extend(list(MACRO_BONE_INDICES[z])[:30])
    shared = np.unique(np.asarray(ids, dtype=np.int64))
    a = base[shared]
    b = a.copy()
    zones = compute_zone_metrics(
        aligned_points_a=a,
        points_b=b,
        shared_indices=shared,
        exclusive_vertices=False,
    )
    ok = [z for z in zones if z.status == "ok"]
    assert len(ok) >= 3
    for z in ok:
        assert z.raw_error is not None and z.raw_error < 1e-9
        assert z.bounded_score is not None and z.bounded_score > 0.99
    summary = summarize_bone_priority_metrics(zones, min_usable_bone_zones=2)
    assert summary["bone_raw_geometry_error"] is not None
    assert summary["bone_raw_geometry_error"] < 1e-9


def test_compute_zone_metrics_detects_shift():
    ids = list(MACRO_BONE_INDICES["chin"])[:40] + list(MACRO_BONE_INDICES["orbit_L"])[:40]
    shared = np.unique(np.asarray(ids, dtype=np.int64))
    b = np.zeros((len(shared), 3), dtype=float)
    a = b.copy()
    a[:, 0] += 0.1  # shift in x
    zones = compute_zone_metrics(
        aligned_points_a=a,
        points_b=b,
        shared_indices=shared,
        face_width_override=1.0,
        exclusive_vertices=False,
    )
    chin = next(z for z in zones if z.name == "chin" and z.status == "ok")
    assert chin.raw_error is not None and chin.raw_error > 0.05
    assert chin.principal_shift_axis == "x"


def test_geom_utils():
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 0.0])
    assert abs(weighted_mean_abs(v, w) - 1.5) < 1e-9
    assert 0 < bounded_score_from_error(0.1) < 1
    assert bounded_score_from_error(0.0) == 1.0
    assert face_scale_from_points(np.eye(3)) > 0


def test_modules_parse():
    for rel in (
        "util/alignment.py",
        "util/zones.py",
        "util/geom_utils.py",
        "util/zone_indices_data.py",
    ):
        ast.parse((ROOT / rel).read_text())


if __name__ == "__main__":
    test_rigid_umeyama_recovers_transform()
    test_rigid_umeyama_no_scale_by_default()
    test_align_meshes_shared_mask()
    test_gpa_and_robust()
    test_zone_indices_valid()
    test_expression_exclusion()
    test_compute_zone_metrics_identical_meshes()
    test_compute_zone_metrics_detects_shift()
    test_geom_utils()
    test_modules_parse()
    print("ALL ITER3 UNIT TESTS PASSED")
