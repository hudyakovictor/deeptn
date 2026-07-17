"""UV-module triangle visibility (ITER2).

Delegates to util.visibility when available; otherwise uses local hard/soft modes.

Critical fix vs pre-ITER2:
  - analysis mode enforces angle_threshold_deg as a real hard cut
  - no silent 0.001 floor that leaks occluded triangles into analysis UV
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

__all__ = ["compute_triangle_visibility", "compute_vertex_visibility"]

logger = logging.getLogger(__name__)

try:
    from util.visibility import (  # type: ignore
        compute_triangle_visibility as _lib_compute_triangle_visibility,
        map_triangle_weights_to_vertices as _lib_map_tri_to_vert,
    )

    _HAS_LIB = True
except Exception:  # pragma: no cover
    _HAS_LIB = False
    _lib_compute_triangle_visibility = None  # type: ignore
    _lib_map_tri_to_vert = None  # type: ignore


def compute_triangle_visibility(
    vertices_3d: np.ndarray,
    triangles: np.ndarray,
    view_dir: Optional[np.ndarray] = None,
    angle_threshold_deg: float = 75.0,
    gamma: float = 1.5,
    use_zbuffer: bool = False,
    vertices_2d: Optional[np.ndarray] = None,
    image_size: Optional[tuple[int, int]] = None,
    z_tolerance: float = 1e-3,
    occlusion_falloff: float = 0.1,
    mode: str = "analysis",
    min_weight_floor: Optional[float] = None,
) -> np.ndarray:
    """Return per-triangle visibility weights in [0, 1].

    mode:
      - "analysis": hard angle threshold, default floor 0
      - "beauty": soft falloff, default floor 0.001
    """
    if _HAS_LIB and _lib_compute_triangle_visibility is not None:
        return _lib_compute_triangle_visibility(
            vertices_3d=vertices_3d,
            triangles=triangles,
            view_dir=view_dir,
            angle_threshold_deg=angle_threshold_deg,
            gamma=gamma,
            use_zbuffer=use_zbuffer,
            vertices_2d=vertices_2d,
            image_size=image_size,
            z_tolerance=z_tolerance,
            occlusion_falloff=occlusion_falloff,
            mode=mode,
            min_weight_floor=min_weight_floor,
        )

    # Local fallback (same semantics as util.visibility)
    verts = np.asarray(vertices_3d, dtype=np.float32)
    if verts.ndim == 3 and verts.shape[0] == 1:
        verts = verts[0]
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")

    if view_dir is None:
        view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        view = np.asarray(view_dir, dtype=np.float32).reshape(3)
        n = float(np.linalg.norm(view))
        if n < 1e-8:
            raise ValueError("view_dir magnitude is too small")
        view = view / n

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norm_len.squeeze(-1) > 1e-8
    normals = np.divide(
        normals,
        np.maximum(norm_len, 1e-8),
        out=np.zeros_like(normals),
        where=norm_len > 1e-8,
    )
    cos_angle = np.clip((normals @ view).astype(np.float32), -1.0, 1.0)
    thr = float(np.cos(np.deg2rad(float(angle_threshold_deg))))
    weights = np.zeros(tris.shape[0], dtype=np.float32)
    mode_l = (mode or "analysis").lower()

    if mode_l == "beauty":
        floor = 0.001 if min_weight_floor is None else float(min_weight_floor)
        soft = np.power(np.maximum(cos_angle, 0.0), float(gamma)).astype(np.float32)
        weights[valid] = np.where(cos_angle[valid] > 0.0, np.maximum(soft[valid], floor), 0.0)
        weights[valid & (cos_angle < thr * 0.5)] = 0.0
    else:
        floor = 0.0 if min_weight_floor is None else float(min_weight_floor)
        front = valid & (cos_angle >= thr)
        scaled = np.clip((cos_angle - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
        if float(gamma) != 1.0:
            scaled = np.power(scaled, float(gamma))
        weights[front] = np.maximum(scaled[front], floor).astype(np.float32)

    if use_zbuffer and vertices_2d is not None and image_size is not None:
        _apply_centroid_zbuffer(
            weights=weights,
            mask=valid,
            verts=verts,
            tris=tris,
            vertices_2d=vertices_2d,
            image_size=image_size,
            z_tolerance=z_tolerance,
            occlusion_falloff=occlusion_falloff if mode_l == "beauty" else min(float(occlusion_falloff), 0.01),
        )

    logger.debug(
        "Visibility[%s]: valid=%d/%d thr=%.1fdeg",
        mode_l,
        int(valid.sum()),
        tris.shape[0],
        float(angle_threshold_deg),
    )
    return weights


def _apply_centroid_zbuffer(
    weights: np.ndarray,
    mask: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    vertices_2d: np.ndarray,
    image_size: tuple[int, int],
    z_tolerance: float,
    occlusion_falloff: float,
) -> None:
    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        return
    verts_2d = np.asarray(vertices_2d, dtype=np.float32)
    if verts_2d.ndim == 3 and verts_2d.shape[0] == 1:
        verts_2d = verts_2d[0]
    centroids_2d = verts_2d[tris].mean(axis=1)
    centroids_z = verts[tris].mean(axis=1)[:, 2]
    cx = np.clip(np.rint(centroids_2d[:, 0]).astype(int), 0, w - 1)
    cy = np.clip(np.rint(centroids_2d[:, 1]).astype(int), 0, h - 1)
    zbuffer = np.full((h, w), np.inf, dtype=np.float32)
    order = np.argsort(centroids_z)
    for idx in order:
        if not mask[idx]:
            continue
        x, y = int(cx[idx]), int(cy[idx])
        z = float(centroids_z[idx])
        if z < zbuffer[y, x]:
            zbuffer[y, x] = z
    occluded = mask & (centroids_z > zbuffer[cy, cx] + float(z_tolerance))
    weights[occluded] *= float(occlusion_falloff)


def compute_vertex_visibility(
    triangles: np.ndarray,
    tri_weights: np.ndarray,
    num_vertices: Optional[int] = None,
) -> np.ndarray:
    if _HAS_LIB and _lib_map_tri_to_vert is not None:
        return _lib_map_tri_to_vert(triangles, tri_weights, num_vertices=num_vertices)

    tris = np.asarray(triangles, dtype=np.int64)
    weights = np.asarray(tri_weights, dtype=np.float32).reshape(-1)
    if tris.shape[0] != weights.shape[0]:
        raise ValueError("triangles and tri_weights must have the same length")
    if num_vertices is None:
        num_vertices = int(tris.max()) + 1 if tris.size > 0 else 0
    vert_sum = np.zeros(num_vertices, dtype=np.float64)
    vert_count = np.zeros(num_vertices, dtype=np.float64)
    for j in range(3):
        np.add.at(vert_sum, tris[:, j], weights)
        np.add.at(vert_count, tris[:, j], 1.0)
    return (vert_sum / np.maximum(vert_count, 1e-6)).astype(np.float32)
