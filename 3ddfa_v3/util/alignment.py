"""ITER3 rigid alignment utilities for 3DDFA-V3 (pure numpy, no project deps).

Forensic default: rigid Umeyama with allow_scale=False.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "AlignmentResult",
    "CANONICAL_YAW_BY_VIEW_GROUP",
    "rigid_umeyama",
    "rigid_umeyama_robust",
    "euler_to_rotation_matrix",
    "gpa_unit_scale",
    "canonicalize_vertices_for_bucket",
    "canonical_angles_deg_for_bucket",
    "canonical_angles_deg_preserve_pose",
    "align_and_score_gpa",
    "align_meshes_shared",
]

CANONICAL_YAW_BY_VIEW_GROUP: dict[str, float] = {
    "frontal": 0.0,
    "left_threequarter_light": -22.5,
    "right_threequarter_light": 22.5,
    "left_threequarter_mid": -45.0,
    "right_threequarter_mid": 45.0,
    "left_threequarter_deep": -67.5,
    "right_threequarter_deep": 67.5,
    "left_profile": -90.0,
    "right_profile": 90.0,
}


@dataclass
class AlignmentResult:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    source_aligned: np.ndarray
    residual_before: float
    residual_after: float
    inlier_fraction: float = 1.0


def canonical_angles_deg_for_bucket(view_group: str) -> np.ndarray:
    target_yaw = float(CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    return np.array([0.0, target_yaw, 0.0], dtype=np.float32)


def canonical_angles_deg_preserve_pose(angles_deg: np.ndarray, view_group: str) -> np.ndarray:
    pitch, yaw, roll = np.asarray(angles_deg, dtype=np.float64).reshape(3)
    target_yaw = float(CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    return np.array([float(pitch), target_yaw, float(roll)], dtype=np.float32)


def rigid_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
    allow_scale: bool = False,
    robust_iterations: int = 4,
) -> AlignmentResult:
    """Weighted rigid Umeyama with iterative MAD outlier rejection."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError(f"source/target must be (N,3), got {source.shape} / {target.shape}")
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"source/target length mismatch: {source.shape[0]} vs {target.shape[0]}")
    if source.shape[0] < 4:
        raise ValueError("source/target must have at least 4 points")

    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != source.shape[0]:
            raise ValueError("weights length mismatch")

    active_weights = weights.copy()
    source_aligned = source.copy()
    rotation = np.eye(3)
    translation = np.zeros(3)
    scale = 1.0

    for iter_idx in range(int(robust_iterations)):
        weight_sum = float(np.sum(active_weights))
        if weight_sum <= 1e-8:
            active_weights = weights.copy()
            weight_sum = float(np.sum(active_weights))

        w = (active_weights / (weight_sum + 1e-8))[:, np.newaxis]
        source_mean = np.sum(source * w, axis=0)
        target_mean = np.sum(target * w, axis=0)
        centered_source = source - source_mean
        centered_target = target - target_mean
        m = centered_source.T @ (w * centered_target)

        if np.linalg.matrix_rank(m) < 3:
            raise ValueError(
                f"Degenerate alignment covariance (rank={np.linalg.matrix_rank(m)}) "
                f"for {source.shape[0]} shared points"
            )

        u, s, vh = np.linalg.svd(m)
        d = np.linalg.det(u @ vh)
        sign_matrix = np.diag([1.0, 1.0, float(np.sign(d) if d != 0 else 1.0)])
        rotation = u @ sign_matrix @ vh

        if allow_scale:
            # variance under the same normalized weights used for covariance m
            var_source = float(np.sum(w.reshape(-1) * np.sum(centered_source**2, axis=1)))
            # Umeyama scale: sum(singular values with reflection fix) / var
            sign_d = float(np.sign(d) if d != 0 else 1.0)
            scale = float((s[0] + s[1] + s[2] * sign_d) / var_source) if var_source > 1e-8 else 1.0
        else:
            scale = 1.0

        translation = target_mean - scale * (source_mean @ rotation)
        source_aligned = scale * (source @ rotation) + translation
        residuals = np.linalg.norm(source_aligned - target, axis=1)

        if iter_idx < robust_iterations - 1:
            med = np.median(residuals)
            mad = np.median(np.abs(residuals - med))
            std_est = 1.4826 * mad + 1e-6
            inliers = residuals <= (2.5 * std_est)
            if np.sum(inliers) < len(source) * 0.6:
                threshold = np.percentile(residuals, 60)
                inliers = residuals <= threshold
            active_weights = weights * inliers.astype(np.float64)

    residual_before = float(np.sum(np.linalg.norm(source - target, axis=1) * weights))
    residual_after = float(np.sum(np.linalg.norm(source_aligned - target, axis=1) * weights))
    inlier_fraction = float(np.mean(active_weights > 0)) if active_weights.size else 1.0

    return AlignmentResult(
        rotation=rotation.astype(np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        scale=float(scale),
        source_aligned=source_aligned.astype(np.float64),
        residual_before=residual_before,
        residual_after=residual_after,
        inlier_fraction=inlier_fraction,
    )


def euler_to_rotation_matrix(angles_rad: np.ndarray) -> np.ndarray:
    """3DDFA ZYX: angles = [pitch, yaw, roll] radians -> R = Rz @ Ry @ Rx."""
    pitch, yaw, roll = np.asarray(angles_rad, dtype=np.float64).reshape(3)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def gpa_unit_scale(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts, 1.0, np.zeros(3, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale < 1e-8:
        return centered, 1.0, centroid
    return centered / scale, scale, centroid


def canonicalize_vertices_for_bucket(
    vertices: np.ndarray,
    angles_deg: np.ndarray,
    view_group: str,
) -> np.ndarray:
    pitch, yaw, roll = np.asarray(angles_deg, dtype=np.float64).reshape(3)
    target_yaw = float(CANONICAL_YAW_BY_VIEW_GROUP.get(view_group, 0.0))
    r_current = euler_to_rotation_matrix(np.deg2rad([pitch, yaw, roll]))
    r_target = euler_to_rotation_matrix(np.deg2rad([pitch, target_yaw, roll]))
    r_align = r_current.T @ r_target
    verts = np.asarray(vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    return (verts - centroid) @ r_align + centroid


def rigid_umeyama_robust(
    src: np.ndarray,
    dst: np.ndarray,
    allow_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError("src/dst shape mismatch")
    n = src.shape[0]
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    src_var = float(np.mean(np.sum(src_c**2, axis=1)))
    h = (src_c.T @ dst_c) / max(n, 1)
    if np.linalg.matrix_rank(h) < 3:
        raise np.linalg.LinAlgError("covariance rank < 3")
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[2, :] *= -1
        r = vt.T @ u.T
    scale = float(np.sum(s) / (src_var + 1e-10)) if allow_scale else 1.0
    t = dst_mean - scale * (src_mean @ r.T)
    return r, t, scale


def align_and_score_gpa(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    verts_a = np.asarray(verts_a, dtype=np.float64)
    verts_b = np.asarray(verts_b, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    r, t, scale = rigid_umeyama_robust(verts_a[mask], verts_b[mask], allow_scale=False)
    verts_a_aligned = (scale * (verts_a @ r.T)) + t
    raw_errors = np.linalg.norm(verts_a_aligned - verts_b, axis=1)
    return verts_a_aligned, raw_errors


def align_meshes_shared(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    shared_mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    allow_scale: bool = False,
) -> AlignmentResult:
    a = np.asarray(verts_a, dtype=np.float64)
    b = np.asarray(verts_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"mesh shape mismatch {a.shape} vs {b.shape}")
    if shared_mask is None:
        src, tgt = a, b
        w = weights
    else:
        m = np.asarray(shared_mask, dtype=bool).reshape(-1)
        if m.shape[0] != a.shape[0]:
            raise ValueError("shared_mask length mismatch")
        src, tgt = a[m], b[m]
        w = None if weights is None else np.asarray(weights).reshape(-1)[m]
    partial = rigid_umeyama(src, tgt, weights=w, allow_scale=allow_scale)
    full_aligned = partial.scale * (a @ partial.rotation) + partial.translation
    res_before = float(np.mean(np.linalg.norm(a - b, axis=1)))
    res_after = float(np.mean(np.linalg.norm(full_aligned - b, axis=1)))
    return AlignmentResult(
        rotation=partial.rotation,
        translation=partial.translation,
        scale=partial.scale,
        source_aligned=full_aligned,
        residual_before=res_before,
        residual_after=res_after,
        inlier_fraction=partial.inlier_fraction,
    )
