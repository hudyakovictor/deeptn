"""ITER3 pure geometry helpers for 3DDFA-V3 library."""
from __future__ import annotations

import numpy as np

BOUNDED_SCORE_GEOMETRY_GAIN = 5.0

__all__ = [
    "weighted_mean_abs",
    "bounded_score_from_error",
    "face_scale_from_points",
    "BOUNDED_SCORE_GEOMETRY_GAIN",
]


def weighted_mean_abs(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if weights.size != values.size:
        raise ValueError("values/weights length mismatch")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("values/weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        return float(np.mean(np.abs(values)))
    return float(np.sum(np.abs(values) * weights) / weight_sum)


def bounded_score_from_error(raw_error: float, gain: float = BOUNDED_SCORE_GEOMETRY_GAIN) -> float:
    """Readable similarity score in (0, 1]; does not flatten small geometry gaps."""
    error = float(raw_error)
    gain_f = float(gain)
    if not np.isfinite(error) or not np.isfinite(gain_f) or gain_f < 0.0:
        return 0.0
    return float(1.0 / (1.0 + gain_f * max(error, 0.0)))


def face_scale_from_points(points: np.ndarray) -> float:
    """3D percentile span scale (forensic-stable vs axis-wise width)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 1.0
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return 1.0
    q95 = np.percentile(pts, 95, axis=0)
    q05 = np.percentile(pts, 5, axis=0)
    scale = float(np.linalg.norm(q95 - q05))
    return max(scale, 1e-6)
