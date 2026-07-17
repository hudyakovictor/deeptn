from __future__ import annotations

import numpy as np

from .config import POSE_BINS


def classify_pose(yaw: float) -> tuple[str, float]:
    for name, lo, hi, canonical in POSE_BINS:
        if lo <= float(yaw) < hi:
            return name, canonical
    return "out_of_supported_range", float(np.clip(yaw, -70.0, 70.0))


def row_rotation_matrix(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    p, y, r = np.radians([pitch_deg, yaw_deg, roll_deg])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], np.float32)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], np.float32)
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], np.float32)
    return (rz @ ry @ rx).T.astype(np.float32)


def normalize_mesh(mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mesh = np.asarray(mesh, np.float32)
    center = mesh.mean(axis=0)
    centered = mesh - center
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("invalid full-mesh RMS scale")
    return (centered / scale).astype(np.float32), center.astype(np.float32), scale


def to_original_image(points_224: np.ndarray, trans_params: np.ndarray) -> np.ndarray:
    """Map 3DDFA image-plane coordinates to original top-left image coordinates."""
    q = np.asarray(points_224, np.float32).copy()
    q[:, 1] = 223.0 - q[:, 1]
    w0, h0, scale, cx, cy = map(float, np.asarray(trans_params).reshape(-1)[:5])
    w = max(int(w0 * scale), 1); h = max(int(h0 * scale), 1)
    left = int(w / 2 - 112 + (cx - w0 / 2) * scale)
    up = int(h / 2 - 112 + (h0 / 2 - cy) * scale)
    q[:, 0] = (q[:, 0] + left) / w * w0
    q[:, 1] = (q[:, 1] + up) / h * h0
    return q.astype(np.float32)


def reprojection_stats(projected: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    a = np.asarray(projected, np.float64); b = np.asarray(expected, np.float64)
    if a.shape != b.shape:
        raise ValueError(f"reprojection shape mismatch: {a.shape} vs {b.shape}")
    dist = np.linalg.norm(a - b, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(dist * dist))),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")


def unpack_mask(packed: np.ndarray, count: int) -> np.ndarray:
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[:count].astype(np.uint8)
