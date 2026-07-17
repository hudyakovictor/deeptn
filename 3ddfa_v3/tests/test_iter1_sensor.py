"""Unit tests for ITER1 3DDFA sensor foundation (no weights / torch optional)."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.reconstruction_api import (  # noqa: E402
    assert_iter1_contract,
    load_reconstruction_artifact,
    result_dict_to_reconstruction,
    save_reconstruction_artifact,
)
from util.types import (  # noqa: E402
    SCHEMA_VERSION,
    CameraContract,
    ReconstructionResult,
    compute_topology_hash,
    hash_array,
)


def _load_process_uv_source():
    """Extract process_uv without importing torch-heavy recon module."""
    src = (ROOT / "model" / "recon.py").read_text()
    # Execute only process_uv definition + numpy import in isolation
    m = re.search(
        r"def process_uv\(uv_coords, uv_h = 224, uv_w = 224\):\n(?:    .*\n)+?(?=\ndef |\nclass )",
        src,
    )
    assert m, "process_uv not found in recon.py"
    ns = {"np": np}
    exec(m.group(0), ns)
    return ns["process_uv"]


def test_process_uv_does_not_mutate_input():
    process_uv = _load_process_uv_source()
    src = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    before = src.copy()
    out = process_uv(src, uv_h=224, uv_w=224)
    assert np.allclose(src, before), "process_uv mutated input"
    assert out.shape == (2, 3)
    assert np.isclose(out[0, 0], before[0, 0] * 223)


def test_to_camera_source_uses_clone():
    src = (ROOT / "model" / "recon.py").read_text()
    # Must clone; must not assign into face_shape[..., -1] directly on input name without clone
    assert "out = face_shape.clone()" in src
    assert "out[..., -1] = self.camera_distance - out[..., -1]" in src
    # Ensure old in-place pattern is gone
    assert "face_shape[..., -1] = self.camera_distance - face_shape[..., -1]" not in src


def test_forward_exports_alpha_and_identity():
    src = (ROOT / "model" / "recon.py").read_text()
    for token in (
        "identity_only",
        "neutral_expression",
        "'alpha_id'",
        "'alpha_exp'",
        "'alpha_alb'",
        "'alpha_angle'",
        "'alpha_sh'",
        "'alpha_trans'",
        "'v3d_identity'",
        "'schema_version'",
        "'camera'",
    ):
        assert token in src, f"missing {token} in recon.forward"


def test_io_landmark_flip_uses_copy():
    src = (ROOT / "util" / "io.py").read_text()
    assert "dtype=np.float64).copy()" in src
    # Old mutate-in-place pattern on result_dict slice should not remain as sole path
    assert "ITER1: always copy landmarks" in src


def test_landmark_flip_copy_semantics():
    ldm = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    orig = ldm.copy()
    work = np.asarray(ldm, dtype=np.float64).copy()
    work[:, 1] = 224 - 1 - work[:, 1]
    assert np.allclose(ldm, orig)
    assert not np.allclose(work, orig)


def test_reconstruction_result_from_dict_and_contract():
    n = 100
    rd = {
        "schema_version": SCHEMA_VERSION,
        "expression_mode": "identity_only",
        "v3d": np.zeros((1, n, 3), dtype=np.float32),
        "v2d": np.zeros((1, n, 2), dtype=np.float32),
        "v3d_model": np.ones((1, n, 3), dtype=np.float32),
        "v3d_identity": np.full((1, n, 3), 2.0, dtype=np.float32),
        "v3d_transformed": np.full((1, n, 3), 3.0, dtype=np.float32),
        "alpha_raw": np.zeros((1, 256), dtype=np.float32),
        "alpha_id": np.zeros((1, 80), dtype=np.float32),
        "alpha_exp": np.zeros((1, 64), dtype=np.float32),
        "alpha_exp_used": np.zeros((1, 64), dtype=np.float32),
        "alpha_alb": np.zeros((1, 80), dtype=np.float32),
        "alpha_angle": np.zeros((1, 3), dtype=np.float32),
        "alpha_angle_deg": np.zeros((1, 3), dtype=np.float32),
        "alpha_sh": np.zeros((1, 27), dtype=np.float32),
        "alpha_trans": np.zeros((1, 3), dtype=np.float32),
        "camera": {
            "model": "perspective_weak",
            "focal": 1015.0,
            "principal_point": (112.0, 112.0),
            "image_size": (224, 224),
            "camera_distance": 10.0,
        },
        "tri": np.array([[0, 1, 2]], dtype=np.int64),
        "visible_idx": np.ones(n, dtype=np.int64),
    }
    rec = result_dict_to_reconstruction(rd, image_path="/tmp/x.jpg")
    assert_iter1_contract(rec)
    assert rec.expression_mode == "identity_only"
    assert rec.vertices_identity is not None
    assert rec.vertices_identity.shape == (n, 3)
    assert rec.alpha_id.shape == (80,)
    assert rec.camera.focal == 1015.0


def test_artifact_roundtrip():
    out_dir = Path("/tmp/iter1_test_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = ReconstructionResult(
        schema_version=SCHEMA_VERSION,
        expression_mode="full",
        vertices_camera=np.random.randn(50, 3).astype(np.float32),
        vertices_model=np.random.randn(50, 3).astype(np.float32),
        vertices_identity=np.random.randn(50, 3).astype(np.float32),
        vertices_transformed=np.random.randn(50, 3).astype(np.float32),
        vertices_image=np.random.randn(50, 2).astype(np.float32),
        alpha_raw=np.random.randn(256).astype(np.float32),
        alpha_id=np.random.randn(80).astype(np.float32),
        alpha_exp=np.random.randn(64).astype(np.float32),
        alpha_exp_used=np.random.randn(64).astype(np.float32),
        alpha_alb=np.random.randn(80).astype(np.float32),
        alpha_angle=np.random.randn(3).astype(np.float32),
        alpha_angle_deg=np.random.randn(3).astype(np.float32),
        alpha_sh=np.random.randn(27).astype(np.float32),
        alpha_trans=np.random.randn(3).astype(np.float32),
        camera=CameraContract(),
        triangles=np.array([[0, 1, 2]], dtype=np.int64),
        visible_idx=np.ones(50, dtype=np.int64),
        topology_hash="abc",
        basis_hash="def",
        image_path="/data/x.jpg",
    )
    path = save_reconstruction_artifact(rec, out_dir / "sample.npz")
    loaded = load_reconstruction_artifact(path)
    assert_iter1_contract(loaded)
    assert np.allclose(loaded.alpha_id, rec.alpha_id)
    assert np.allclose(loaded.vertices_identity, rec.vertices_identity)
    assert loaded.topology_hash == "abc"


def test_topology_hash_stable():
    tri = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    h1 = compute_topology_hash(tri, n_vertices=4)
    h2 = compute_topology_hash(tri.copy(), n_vertices=4)
    assert h1 == h2
    tri2 = tri.copy()
    tri2[0, 0] = 1
    assert compute_topology_hash(tri2, n_vertices=4) != h1


def test_hash_array_changes_with_content():
    a = np.zeros((3, 3), dtype=np.float32)
    b = np.ones((3, 3), dtype=np.float32)
    assert hash_array(a) != hash_array(b)


def test_recon_py_parses():
    src = (ROOT / "model" / "recon.py").read_text()
    ast.parse(src)
    ast.parse((ROOT / "util" / "io.py").read_text())
    ast.parse((ROOT / "util" / "types.py").read_text())
    ast.parse((ROOT / "util" / "reconstruction_api.py").read_text())


if __name__ == "__main__":
    test_process_uv_does_not_mutate_input()
    test_to_camera_source_uses_clone()
    test_forward_exports_alpha_and_identity()
    test_io_landmark_flip_uses_copy()
    test_landmark_flip_copy_semantics()
    test_reconstruction_result_from_dict_and_contract()
    test_artifact_roundtrip()
    test_topology_hash_stable()
    test_hash_array_changes_with_content()
    test_recon_py_parses()
    print("ALL ITER1 UNIT TESTS PASSED")
