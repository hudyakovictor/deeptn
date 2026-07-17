"""ITER10: ported geometry bone metrics + expanded texture fields."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.geometry_metrics import (  # noqa: E402
    apply_expression_exclusion_to_metrics,
    calculate_coverage,
    extract_macro_bone_metrics,
)
from util.zones import MACRO_BONE_INDICES  # noqa: E402
from util.texture import TextureMetrics, extract_texture_metrics, analyze_texture  # noqa: E402


def _mesh():
    n = 35709
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.02, size=(n, 3))
    for name, ids in MACRO_BONE_INDICES.items():
        for i, vid in enumerate(list(ids)[:50]):
            sx = -0.3 if name.endswith("_L") else (0.3 if name.endswith("_R") else 0.0)
            sy = 0.35 if ("forehead" in name or "brow" in name) else (-0.35 if ("chin" in name or "jaw" in name) else 0.0)
            sz = 0.15 if "orbit" in name else 0.55
            v[int(vid)] = [sx + 0.01 * (i % 5), sy + 0.01 * (i // 5), sz]
    return v


def test_extract_macro_bone_metrics_core_keys():
    m, rel = extract_macro_bone_metrics(_mesh(), MACRO_BONE_INDICES, np.array([0.0, 0.0, 0.0]))
    assert rel > 0
    for k in (
        "cranial_face_index",
        "jaw_width_ratio",
        "interorbital_ratio",
        "canthal_tilt_L",
        "canthal_tilt_R",
        "orbit_depth_L_ratio",
        "orbit_depth_R_ratio",
        "nose_width_ratio",
        "nose_projection_ratio",
        "chin_projection_ratio",
    ):
        assert k in m, k
        assert m[k] is None or np.isfinite(float(m[k]))
    finite = [k for k, v in m.items() if isinstance(v, (int, float)) and np.isfinite(float(v))]
    assert len(finite) >= 40


def test_expression_exclusion_nulls_jaw_on_open():
    m, _ = extract_macro_bone_metrics(_mesh(), MACRO_BONE_INDICES, np.array([0.0, 0.0, 0.0]))
    exp = np.zeros(64)
    exp[0] = 0.5  # jaw open
    cleaned = apply_expression_exclusion_to_metrics(m, exp)
    assert cleaned.get("jaw_width_ratio") is None
    assert cleaned.get("gonial_angle_L") is None


def test_calculate_coverage_fallback():
    m, _ = extract_macro_bone_metrics(_mesh(), MACRO_BONE_INDICES, np.array([0.0, 0.0, 0.0]))
    c = calculate_coverage(m, "frontal")
    assert 0.0 < c <= 1.0


def test_texture_fields_parity():
    fields = set(TextureMetrics.__annotations__.keys())
    for k in (
        "lbp_uniformity",
        "glcm_energy",
        "glcm_correlation",
        "gabor_mean",
        "gabor_std",
        "pigmentation_index",
        "glcm_contrast_ratio",
        "quality_sharpness_score",
        "quality_noise_score",
        "quality_index",
    ):
        assert k in fields, k


def test_texture_extract_populates_new_fields():
    try:
        import cv2  # noqa: F401
    except Exception:
        print("SKIP texture extract: no cv2")
        return
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)
    mask = np.ones((160, 160), dtype=np.uint8) * 255
    metrics, cov = extract_texture_metrics(img, mask)
    assert cov > 0.9
    assert metrics.glcm_contrast is not None
    assert metrics.glcm_energy is not None
    assert metrics.glcm_correlation is not None
    assert metrics.gabor_mean is not None
    assert metrics.pigmentation_index is not None
    assert metrics.quality_index is not None
    prof = analyze_texture(img, mask)
    assert prof.ok
    assert prof.synthetic_prob is not None and 0.0 <= prof.synthetic_prob <= 1.0


def test_modules_parse():
    ast.parse((ROOT / "util" / "geometry_metrics.py").read_text())
    ast.parse((ROOT / "util" / "texture.py").read_text())


if __name__ == "__main__":
    test_extract_macro_bone_metrics_core_keys()
    test_expression_exclusion_nulls_jaw_on_open()
    test_calculate_coverage_fallback()
    test_texture_fields_parity()
    test_texture_extract_populates_new_fields()
    test_modules_parse()
    print("ALL ITER10 GEOMETRY/TEXTURE METRIC TESTS PASSED")
