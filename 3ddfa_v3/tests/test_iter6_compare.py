"""ITER6 unit tests: pair compare visibility ∩, umeyama, bone score."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.alignment import euler_to_rotation_matrix  # noqa: E402
from util.compare import (  # noqa: E402
    PairCompareInput,
    compare_pair,
    geodesic_pose_distance,
    id_params_cosine_distance,
    pose_delta_deg,
    prepare_pair_alignment,
    score_aligned_pair,
    shared_vertex_indices,
)
from util.zones import MACRO_BONE_INDICES  # noqa: E402


def _sphere_normals(verts):
    n = verts - verts.mean(0)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    return n.astype(np.float32)


def _synthetic_face(n=35709, seed=0, yaw_deg=0.0):
    rng = np.random.default_rng(seed)
    # start from mild noise cloud then embed bone zones
    verts = rng.normal(0, 0.02, size=(n, 3)).astype(np.float64)
    # place a face-like structure along z facing camera
    for zname, ids in MACRO_BONE_INDICES.items():
        for i, vid in enumerate(list(ids)[:80]):
            if 0 <= int(vid) < n:
                # left zones negative x etc.
                sx = -0.3 if zname.endswith("_L") else (0.3 if zname.endswith("_R") else 0.0)
                sy = 0.2 if "brow" in zname or "forehead" in zname else (-0.25 if "chin" in zname or "jaw" in zname else 0.0)
                verts[int(vid)] = np.array([sx + 0.01 * (i % 5), sy + 0.01 * (i // 5), 0.5], dtype=np.float64)
    # rotate by yaw around Y
    R = euler_to_rotation_matrix(np.deg2rad([0.0, yaw_deg, 0.0]))
    verts = verts @ R.T
    normals = _sphere_normals(verts)
    # force front-ish normals +Z component positive for frontal
    normals = normals.copy()
    normals[:, 2] = np.abs(normals[:, 2]) + 0.2
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    return verts, normals


def test_shared_vertex_indices():
    a = np.array([True, True, False, True])
    b = np.array([True, False, True, True])
    idx = shared_vertex_indices(a, b)
    assert list(idx) == [0, 3]


def test_geodesic_and_pose_delta():
    R = euler_to_rotation_matrix(np.deg2rad([0, 30, 0]))
    d = geodesic_pose_distance(np.eye(3), R)
    assert 25 < d < 35
    assert pose_delta_deg([0, 0, 0], [0, 30, 0]) > 20


def test_id_cosine():
    a = np.ones(80)
    b = np.ones(80)
    assert abs(id_params_cosine_distance(a, b) - 0.0) < 1e-9
    c = np.zeros(80); c[0] = 1
    d = np.zeros(80); d[1] = 1
    assert id_params_cosine_distance(c, d) is not None


def test_score_aligned_identical():
    p = np.random.randn(40, 3)
    w = np.ones(40)
    pe, ps, re, rs = score_aligned_pair(p, p, w)
    assert pe < 1e-9 and re < 1e-9 and ps > 0.99


def test_compare_identical_meshes():
    va, na = _synthetic_face(seed=1, yaw_deg=0.0)
    mesh_a = PairCompareInput(
        vertices=va, normals=na, vertices_camera=va,
        angles_deg=np.array([0.0, 0.0, 0.0]), pose_bucket="frontal",
        alpha_id=np.ones(80), person_id="p1", photo_id="a",
    )
    mesh_b = PairCompareInput(
        vertices=va.copy(), normals=na.copy(), vertices_camera=va.copy(),
        angles_deg=np.array([0.0, 0.0, 0.0]), pose_bucket="frontal",
        alpha_id=np.ones(80), person_id="p1", photo_id="b",
    )
    res = compare_pair(mesh_a, mesh_b, min_shared=20)
    assert res.status == "ok", res
    assert res.shared_count >= 20
    assert res.raw_geometry_error is not None and res.raw_geometry_error < 1e-6
    assert res.bounded_similarity_score is not None and res.bounded_similarity_score > 0.99


def test_compare_detects_shift():
    # Rigid global translation is removed by Umeyama; use non-rigid local deformation.
    va, na = _synthetic_face(seed=2, yaw_deg=0.0)
    vb = va.copy()
    for vid in MACRO_BONE_INDICES["cheekbone_L"]:
        iv = int(vid)
        if 0 <= iv < vb.shape[0]:
            vb[iv] = vb[iv] * np.array([1.5, 1.0, 1.0])
    mesh_a = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                              angles_deg=np.array([0.,0.,0.]), pose_bucket="frontal")
    mesh_b = PairCompareInput(vertices=vb, normals=na, vertices_camera=vb,
                              angles_deg=np.array([0.,0.,0.]), pose_bucket="frontal")
    res = compare_pair(mesh_a, mesh_b, min_shared=20)
    assert res.status == "ok", res
    assert res.raw_geometry_error is not None and res.raw_geometry_error > 0.005
    cheek = [z for z in res.zones if z.get("name") == "cheekbone_L" and z.get("raw_error") is not None]
    if cheek:
        assert cheek[0]["raw_error"] > 0.03


def test_insufficient_visibility():
    n = 35709
    va = np.zeros((n, 3))
    # all normals point away (-Z) → invisible
    na = np.zeros((n, 3)); na[:, 2] = -1.0
    mesh_a = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                              angles_deg=np.array([0.,0.,0.]))
    mesh_b = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                              angles_deg=np.array([0.,0.,0.]))
    res = compare_pair(mesh_a, mesh_b, min_shared=50)
    assert res.status == "insufficient_shared_visibility"


def test_prepare_pair_alignment_ok():
    va, na = _synthetic_face(seed=3, yaw_deg=5.0)
    a = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                         angles_deg=np.array([0.,5.,0.]), pose_bucket="frontal")
    b = PairCompareInput(vertices=va, normals=na, vertices_camera=va,
                         angles_deg=np.array([0.,-5.,0.]), pose_bucket="frontal")
    prep = prepare_pair_alignment(a, b, min_shared=20)
    assert prep["status"] == "ok"
    assert prep["points_a_unit"].shape[0] == prep["shared_count"]


def test_modules_parse():
    ast.parse((ROOT / "util" / "compare.py").read_text())


if __name__ == "__main__":
    test_shared_vertex_indices()
    test_geodesic_and_pose_delta()
    test_id_cosine()
    test_score_aligned_identical()
    test_compare_identical_meshes()
    test_compare_detects_shift()
    test_insufficient_visibility()
    test_prepare_pair_alignment_ok()
    test_modules_parse()
    print("ALL ITER6 UNIT TESTS PASSED")
