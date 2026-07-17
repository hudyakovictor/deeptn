"""ITER7 texture channel tests: no floor, quality penalty, fail-closed."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.texture import (  # noqa: E402
    analyze_texture,
    extract_texture_metrics,
    quality_penalty_on_synthetic,
    score_synthetic_probability,
    TextureMetrics,
)


def _has_cv2():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def test_quality_penalty():
    assert quality_penalty_on_synthetic(0.8, "ambiguous") == 0.8  # keep high
    p = quality_penalty_on_synthetic(0.2, "ambiguous")
    assert 0.2 < p < 0.5  # pulled toward 0.5
    assert quality_penalty_on_synthetic(0.4, "normal") == 0.4


def test_no_hard_floor_on_score():
    # very "real" metrics → low synthetic, can be << 0.75
    m = TextureMetrics(
        lbp_uniformity=0.4,
        skin_micro_contrast=15.0,
        pore_proxy=10.0,
        specular_gloss=0.01,
        glcm_contrast=10.0,
        skin_shannon_entropy=4.5,
        fractal_dimension=2.4,
        matte_uniformity=0.3,
    )
    p, c = score_synthetic_probability(m)
    assert p is not None and p < 0.5
    # very "silicone" metrics
    m2 = TextureMetrics(
        lbp_uniformity=0.95,
        skin_micro_contrast=1.0,
        pore_proxy=0.5,
        specular_gloss=0.2,
        glcm_contrast=0.5,
        skin_shannon_entropy=1.5,
        fractal_dimension=1.8,
        matte_uniformity=0.95,
    )
    p2, _ = score_synthetic_probability(m2)
    assert p2 is not None and p2 > 0.6
    # range free: not forced to 0.75
    assert p2 != 0.75


def test_analyze_real_vs_smooth():
    if not _has_cv2():
        print("SKIP analyze: no cv2")
        return
    rng = np.random.default_rng(0)
    # textured real-like
    real = (rng.normal(120, 25, size=(256, 256))).clip(0, 255).astype(np.uint8)
    real = np.stack([real, real, real], axis=-1)
    # add high-frequency texture
    noise = rng.integers(0, 40, size=real.shape, dtype=np.int16)
    real = np.clip(real.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    mask = np.ones((256, 256), dtype=np.uint8) * 255

    # smooth silicone-like
    smooth = np.full((256, 256, 3), 180, dtype=np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    highlight = ((xx - 180) ** 2 + (yy - 80) ** 2) < 30 ** 2
    smooth[highlight] = (240, 240, 240)

    pr = analyze_texture(real, mask)
    ps = analyze_texture(smooth, mask)
    assert pr.ok and ps.ok
    assert pr.synthetic_prob is not None and ps.synthetic_prob is not None
    # smooth should score higher synthetic than noisy real
    assert ps.synthetic_prob > pr.synthetic_prob
    # neither forced to constant 0.75
    assert pr.synthetic_prob != 0.75 and ps.synthetic_prob != 0.75
    assert pr.raw_synthetic_prob is not None


def test_fail_closed_empty_mask():
    if not _has_cv2():
        print("SKIP fail-closed: no cv2")
        return
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    p = analyze_texture(img, mask)
    assert p.ok is False
    assert p.synthetic_prob is None
    assert p.reason == "insufficient_skin_coverage"


def test_extract_metrics_smoke():
    if not _has_cv2():
        print("SKIP extract: no cv2")
        return
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    m, cov = extract_texture_metrics(img, np.ones((128, 128), dtype=np.uint8) * 255)
    assert cov > 0.9
    assert m.laplacian_energy is not None


def test_modules_parse():
    ast.parse((ROOT / "util" / "texture.py").read_text())


if __name__ == "__main__":
    test_quality_penalty()
    test_no_hard_floor_on_score()
    test_analyze_real_vs_smooth()
    test_fail_closed_empty_mask()
    test_extract_metrics_smoke()
    test_modules_parse()
    print("ALL ITER7 UNIT TESTS PASSED")
