"""ITER7 texture / silicone channel for 3DDFA forensic pipeline.

Library-local, no reference JSON required for core metrics.
Fixes vs legacy:
  - no hard floor ~0.75 on synthetic_prob
  - quality penalty pulls ambiguous/low-quality toward neutral (not up)
  - fail-closed when mask/crop insufficient
  - fractal dimension not saturated as sole evidence
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "TextureMetrics",
    "TextureQuality",
    "TextureProfile",
    "extract_texture_metrics",
    "estimate_texture_quality",
    "score_synthetic_probability",
    "analyze_texture",
    "quality_penalty_on_synthetic",
]


def _cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("OpenCV (cv2) required for texture analysis") from exc


@dataclass
class TextureMetrics:
    lbp_uniformity: Optional[float] = None
    lbp_entropy: Optional[float] = None
    glcm_contrast: Optional[float] = None
    glcm_homogeneity: Optional[float] = None

    glcm_energy: Optional[float] = None
    glcm_correlation: Optional[float] = None
    gabor_mean: Optional[float] = None
    gabor_std: Optional[float] = None
    pigmentation_index: Optional[float] = None
    glcm_contrast_ratio: Optional[float] = None
    quality_sharpness_score: Optional[float] = None
    quality_noise_score: Optional[float] = None
    quality_index: Optional[float] = None
    laplacian_energy: Optional[float] = None
    specular_gloss: Optional[float] = None
    skin_micro_contrast: Optional[float] = None
    skin_shannon_entropy: Optional[float] = None
    fractal_dimension: Optional[float] = None
    autocorrelation_decay: Optional[float] = None
    pore_proxy: Optional[float] = None  # high-frequency residual energy
    matte_uniformity: Optional[float] = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TextureQuality:
    blur_level: float
    noise_level: float
    exposure_clipping: float
    skin_coverage: float
    overall: float
    mode: str  # normal | low_quality | ambiguous

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextureProfile:
    ok: bool
    synthetic_prob: Optional[float]
    raw_synthetic_prob: Optional[float]
    quality_adjusted_synthetic_prob: Optional[float]
    natural_score: Optional[float]
    reliability: float
    metrics: TextureMetrics = field(default_factory=TextureMetrics)
    quality: Optional[TextureQuality] = None
    contributions: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "synthetic_prob": self.synthetic_prob,
            "silicone_prob": self.synthetic_prob,
            "raw_synthetic_prob": self.raw_synthetic_prob,
            "quality_adjusted_synthetic_prob": self.quality_adjusted_synthetic_prob,
            "natural_score": self.natural_score,
            "reliability": self.reliability,
            "metrics": self.metrics.as_dict(),
            "quality": None if self.quality is None else self.quality.to_dict(),
            "contributions": self.contributions,
            "reason": self.reason,
            "status": "ok" if self.ok else "insufficient_texture",
        }


def _to_gray_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    cv2 = _cv2()
    img = np.asarray(image)
    if img.ndim == 2:
        gray = img.astype(np.float64)
    elif img.ndim == 3 and img.shape[2] == 4:
        # BGRA: alpha as mask if none provided
        bgr = img[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if mask is None:
            mask = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    if mask is None:
        m = np.ones(gray.shape[:2], dtype=bool)
        coverage = 1.0
    else:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = m > (10 if m.dtype != bool else 0)
        coverage = float(np.mean(m)) if m.size else 0.0
    return gray, m, coverage


def _lbp_uniformity_entropy(gray: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Simple 8-neighbor LBP uniformity ratio + entropy (no skimage required)."""
    g = gray.astype(np.uint8)
    h, w = g.shape
    if h < 3 or w < 3:
        return None, None
    c = g[1:-1, 1:-1].astype(np.int16)
    codes = np.zeros(c.shape, dtype=np.uint8)
    neighbors = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:],
        g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1], g[2:, 0:-2], g[1:-1, 0:-2],
    ]
    for i, nb in enumerate(neighbors):
        codes |= ((nb.astype(np.int16) >= c).astype(np.uint8) << i)
    m = mask[1:-1, 1:-1]
    if not np.any(m):
        return None, None
    vals = codes[m]
    # uniform LBP: at most 2 bitwise transitions
    def _is_uniform(v: int) -> bool:
        b = ((v << 1) | (v >> 7)) & 0xFF
        transitions = bin(v ^ b).count("1")
        return transitions <= 2
    # vectorized-ish via python for unique codes only
    uniq, counts = np.unique(vals, return_counts=True)
    total = float(counts.sum())
    uni = 0.0
    for u, c_ in zip(uniq.tolist(), counts.tolist()):
        if _is_uniform(int(u)):
            uni += float(c_)
    uniformity = uni / max(total, 1.0)
    p = counts.astype(np.float64) / max(total, 1.0)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))
    return float(uniformity), entropy


def _glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict[str, Optional[float]]:
    g = gray.astype(np.float64)
    m = mask
    empty = {"contrast": None, "homogeneity": None, "energy": None, "correlation": None}
    if np.count_nonzero(m) < 64:
        return empty
    lo, hi = float(np.percentile(g[m], 1)), float(np.percentile(g[m], 99))
    if hi <= lo + 1e-6:
        return {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0, "correlation": 1.0}
    q = np.clip(((g - lo) / (hi - lo) * (levels - 1)).astype(np.int32), 0, levels - 1)
    a = q[:, :-1][m[:, :-1] & m[:, 1:]]
    b = q[:, 1:][m[:, :-1] & m[:, 1:]]
    if a.size < 32:
        return empty
    glcm = np.zeros((levels, levels), dtype=np.float64)
    # vectorized bincount-style accumulation
    idx = a.ravel().astype(np.int64) * levels + b.ravel().astype(np.int64)
    bc = np.bincount(idx, minlength=levels * levels).astype(np.float64)
    glcm = bc.reshape(levels, levels)
    s = glcm.sum()
    if s <= 0:
        return empty
    glcm /= s
    ii, jj = np.indices(glcm.shape)
    contrast = float(np.sum(glcm * (ii - jj) ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + (ii - jj) ** 2)))
    energy = float(np.sum(glcm ** 2))
    mu_i = float(np.sum(ii * glcm))
    mu_j = float(np.sum(jj * glcm))
    sig_i = float(np.sqrt(np.sum(((ii - mu_i) ** 2) * glcm)) + 1e-12)
    sig_j = float(np.sqrt(np.sum(((jj - mu_j) ** 2) * glcm)) + 1e-12)
    correlation = float(np.sum(((ii - mu_i) * (jj - mu_j)) * glcm) / (sig_i * sig_j))
    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": float(np.clip(correlation, -1.0, 1.0)),
    }


def _glcm_contrast_homogeneity(gray: np.ndarray, mask: np.ndarray, levels: int = 16):
    """Backward-compatible wrapper."""
    f = _glcm_features(gray, mask, levels=levels)
    return f["contrast"], f["homogeneity"]


def _box_count_fd(img: np.ndarray, mask: np.ndarray) -> Optional[float]:
    vals = img[mask]
    if vals.size < 64:
        return None
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo + 1e-6:
        return 1.0
    norm = ((img - lo) / (hi - lo) * 255.0)
    sizes = [2, 4, 8, 16, 32]
    counts = []
    used = []
    for s in sizes:
        if s >= min(norm.shape):
            continue
        h, w = norm.shape
        nh, nw = h // s, w // s
        if nh < 2 or nw < 2:
            continue
        blocks = norm[: nh * s, : nw * s].reshape(nh, s, nw, s).max(axis=(1, 3))
        counts.append(float(np.count_nonzero(blocks > 0)))
        used.append(s)
    if len(counts) < 3:
        return None
    xs = np.log(np.asarray(used, dtype=np.float64))
    ys = np.log(np.asarray(counts, dtype=np.float64) + 1e-9)
    return float(-np.polyfit(xs, ys, 1)[0])



def _gabor_energy_stats(gray: np.ndarray, mask: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Lightweight multi-orientation Gabor energy via cos/sin kernels (no skimage)."""
    cv2 = _cv2()
    g = gray.astype(np.float32)
    m = mask
    if np.count_nonzero(m) < 64:
        return None, None
    energies = []
    for theta_deg in (0, 45, 90, 135):
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(g, cv2.CV_32F, kernel)
        energies.append(float(np.mean(np.abs(resp[m]))))
    arr = np.asarray(energies, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _pigmentation_index(image: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Color variance in a/b-like channels as pigmentation proxy."""
    cv2 = _cv2()
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    bgr = img[:, :, :3].astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    m = mask
    if np.count_nonzero(m) < 64:
        return None
    a = lab[:, :, 1][m]
    b = lab[:, :, 2][m]
    return float(np.std(a) + np.std(b))

def extract_texture_metrics(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[TextureMetrics, float]:
    """Extract forensic texture metrics from face crop (+ optional skin mask)."""
    cv2 = _cv2()
    gray, m, coverage = _to_gray_mask(image, mask)
    if coverage < 0.02 or np.count_nonzero(m) < 200:
        return TextureMetrics(), coverage

    metrics = TextureMetrics()
    uni, ent = _lbp_uniformity_entropy(gray, m)
    metrics.lbp_uniformity = uni
    metrics.lbp_entropy = ent

    glcm = _glcm_features(gray, m)
    metrics.glcm_contrast = glcm["contrast"]
    metrics.glcm_homogeneity = glcm["homogeneity"]
    metrics.glcm_energy = glcm["energy"]
    metrics.glcm_correlation = glcm["correlation"]

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    metrics.laplacian_energy = float(np.mean(np.abs(lap[m])))

    g_mean, g_std = _gabor_energy_stats(gray, m)
    metrics.gabor_mean = g_mean
    metrics.gabor_std = g_std
    metrics.pigmentation_index = _pigmentation_index(image, m)

    # specular proxy: bright hotspot density
    thr = float(np.percentile(gray[m], 95))
    metrics.specular_gloss = float(np.mean((gray >= thr) & m))

    # micro contrast: local std
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    local = np.abs(gray - blur)
    metrics.skin_micro_contrast = float(np.mean(local[m]))

    # shannon entropy of intensity hist
    hist, _ = np.histogram(gray[m], bins=32, range=(0, 255), density=True)
    hist = hist[hist > 0]
    metrics.skin_shannon_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

    fd = _box_count_fd(gray, m)
    metrics.fractal_dimension = fd

    roi = gray[m]
    if roi.size > 32:
        ac = float(np.corrcoef(roi[:-1], roi[1:])[0, 1])
        metrics.autocorrelation_decay = float(1.0 - max(ac, 0.0))
    else:
        metrics.autocorrelation_decay = None

    # pore proxy: high-pass residual energy
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 0.8)
    metrics.pore_proxy = float(np.std(hp[m]))

    # matte uniformity: inverse of micro contrast normalized
    metrics.matte_uniformity = float(np.clip(1.0 - (metrics.skin_micro_contrast or 0.0) / 25.0, 0.0, 1.0))

    # quality-linked scores (for parity with newapp TextureMetrics)
    sharp = float(np.clip((metrics.laplacian_energy or 0.0) / 30.0, 0.0, 1.0))
    noise = float(np.clip((metrics.pore_proxy or 0.0) / 15.0, 0.0, 1.0))
    metrics.quality_sharpness_score = sharp
    metrics.quality_noise_score = float(1.0 - noise)
    metrics.quality_index = float(np.clip(0.6 * sharp + 0.4 * (1.0 - noise), 0.0, 1.0))
    if metrics.glcm_contrast is not None and metrics.laplacian_energy is not None:
        metrics.glcm_contrast_ratio = float(metrics.glcm_contrast / (metrics.laplacian_energy + 1e-6))

    return metrics, coverage


def estimate_texture_quality(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> TextureQuality:
    cv2 = _cv2()
    gray, m, coverage = _to_gray_mask(image, mask)
    g = gray
    pixels = g[m] if np.any(m) else g.ravel()

    lap = cv2.Laplacian(g.astype(np.float32), cv2.CV_32F)
    blur_raw = float(np.var(lap[m])) if np.any(m) else float(np.var(lap))
    blur_level = float(np.clip(1.0 - blur_raw / 600.0, 0.0, 1.0))

    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    noise_raw = float(np.std(hp[m])) if np.any(m) else float(np.std(hp))
    noise_level = float(np.clip(noise_raw / 25.0, 0.0, 1.0))

    clip_lo = float(np.mean(pixels <= 5.0)) if pixels.size else 0.0
    clip_hi = float(np.mean(pixels >= 250.0)) if pixels.size else 0.0
    exposure_clipping = float(np.clip(clip_lo + clip_hi, 0.0, 1.0))

    overall = float(
        np.clip(
            0.40 * (1.0 - blur_level)
            + 0.25 * (1.0 - noise_level)
            + 0.15 * (1.0 - exposure_clipping)
            + 0.20 * min(coverage / 0.25, 1.0),
            0.0,
            1.0,
        )
    )
    if overall < 0.35 or blur_level > 0.90 or coverage < 0.05:
        mode = "ambiguous"
    elif overall < 0.55 or blur_level > 0.72 or noise_level > 0.22:
        mode = "low_quality"
    else:
        mode = "normal"

    return TextureQuality(
        blur_level=blur_level,
        noise_level=noise_level,
        exposure_clipping=exposure_clipping,
        skin_coverage=coverage,
        overall=overall,
        mode=mode,
    )


def quality_penalty_on_synthetic(raw_prob: float, mode: str) -> float:
    """Pull synthetic prob toward 0.5 when quality is poor; never invent high silicone."""
    p = float(np.clip(raw_prob, 0.0, 1.0))
    # if already strongly silicone, keep (don't hide mask evidence)
    if p >= 0.62:
        return p
    neutral = 0.5
    if mode == "ambiguous":
        return float(0.70 * p + 0.30 * neutral)
    if mode == "low_quality":
        return float(0.78 * p + 0.22 * neutral)
    return p


def score_synthetic_probability(metrics: TextureMetrics) -> Tuple[Optional[float], Dict[str, float]]:
    """Heuristic silicone/synthetic score in [0,1] without hard floor.

    High synthetic cues: high LBP uniformity, low micro-contrast/pores, high specular,
    low entropy, low GLCM contrast, very low fractal variability.
    """
    contrib: Dict[str, float] = {}
    scores = []

    if metrics.lbp_uniformity is not None:
        # real skin often mid uniformity; silicone very high
        s = float(np.clip((metrics.lbp_uniformity - 0.55) / 0.35, 0.0, 1.0))
        contrib["lbp_uniformity"] = s
        scores.append((s, 1.2))

    if metrics.skin_micro_contrast is not None:
        # low micro-contrast → synthetic
        s = float(np.clip(1.0 - metrics.skin_micro_contrast / 12.0, 0.0, 1.0))
        contrib["micro_contrast"] = s
        scores.append((s, 1.3))

    if metrics.pore_proxy is not None:
        s = float(np.clip(1.0 - metrics.pore_proxy / 8.0, 0.0, 1.0))
        contrib["pore_proxy"] = s
        scores.append((s, 1.1))

    if metrics.specular_gloss is not None:
        s = float(np.clip((metrics.specular_gloss - 0.02) / 0.15, 0.0, 1.0))
        contrib["specular"] = s
        scores.append((s, 0.9))

    if metrics.glcm_contrast is not None:
        # low contrast → plastic
        s = float(np.clip(1.0 - metrics.glcm_contrast / 8.0, 0.0, 1.0))
        contrib["glcm_contrast"] = s
        scores.append((s, 1.0))

    if metrics.skin_shannon_entropy is not None:
        s = float(np.clip(1.0 - (metrics.skin_shannon_entropy - 2.0) / 3.0, 0.0, 1.0))
        contrib["entropy"] = s
        scores.append((s, 0.8))

    if metrics.fractal_dimension is not None:
        # real skin FD often ~2.2-2.6; very low FD can be flat silicone
        # do NOT saturate: map gently
        s = float(np.clip((2.35 - metrics.fractal_dimension) / 0.6, 0.0, 1.0))
        contrib["fractal"] = s
        scores.append((s, 0.5))  # lower weight — avoid fractal domination

    if metrics.matte_uniformity is not None:
        s = float(np.clip((metrics.matte_uniformity - 0.5) / 0.4, 0.0, 1.0))
        contrib["matte_uniformity"] = s
        scores.append((s, 0.7))

    if not scores:
        return None, contrib

    wsum = sum(w for _, w in scores)
    raw = sum(s * w for s, w in scores) / max(wsum, 1e-8)
    # NO floor at 0.75 — full [0,1] range
    return float(np.clip(raw, 0.0, 1.0)), contrib


def analyze_texture(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    min_coverage: float = 0.05,
    min_pixels: int = 200,
) -> TextureProfile:
    """End-to-end texture analysis with quality adjustment (fail-closed)."""
    try:
        gray, m, coverage = _to_gray_mask(image, mask)
    except Exception as exc:
        return TextureProfile(ok=False, synthetic_prob=None, raw_synthetic_prob=None,
                              quality_adjusted_synthetic_prob=None, natural_score=None,
                              reliability=0.0, reason=f"load_error:{exc}")

    if coverage < min_coverage or int(np.count_nonzero(m)) < min_pixels:
        return TextureProfile(
            ok=False,
            synthetic_prob=None,
            raw_synthetic_prob=None,
            quality_adjusted_synthetic_prob=None,
            natural_score=None,
            reliability=0.0,
            reason="insufficient_skin_coverage",
        )

    metrics, coverage = extract_texture_metrics(image, mask)
    quality = estimate_texture_quality(image, mask)
    raw_prob, contrib = score_synthetic_probability(metrics)
    if raw_prob is None:
        return TextureProfile(
            ok=False,
            synthetic_prob=None,
            raw_synthetic_prob=None,
            quality_adjusted_synthetic_prob=None,
            natural_score=None,
            reliability=0.0,
            metrics=metrics,
            quality=quality,
            reason="insufficient_features",
        )

    adj = quality_penalty_on_synthetic(raw_prob, quality.mode)
    # reliability from quality + feature count
    n_feat = len(contrib)
    reliability = float(np.clip(0.5 * quality.overall + 0.5 * min(n_feat / 6.0, 1.0), 0.0, 1.0))
    if quality.mode == "ambiguous":
        reliability *= 0.6
    elif quality.mode == "low_quality":
        reliability *= 0.8

    return TextureProfile(
        ok=True,
        synthetic_prob=adj,  # primary reported = quality-adjusted, still no floor
        raw_synthetic_prob=raw_prob,
        quality_adjusted_synthetic_prob=adj,
        natural_score=float(1.0 - adj),
        reliability=reliability,
        metrics=metrics,
        quality=quality,
        contributions=contrib,
        reason="ok",
    )
