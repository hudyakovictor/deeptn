"""ITER2 unified visibility for 3DDFA-V3 forensic sensor.

Two modes:
  - analysis (hard): angle threshold is a real binary cut; min weight = 0
  - beauty (soft): gentle falloff; small floor allowed for hole-free rendering

Also provides vertex-level z-buffer + normal facing (pipeline-style).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "VisibilityResult",
    "compute_software_zbuffer_mask",
    "compute_vertex_visibility_from_normals",
    "compute_visibility",
    "compute_triangle_visibility",
    "map_triangle_weights_to_vertices",
    "get_visible_zones",
    "filter_metrics_by_pose",
]

DEFAULT_VIEW_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
Z_TOLERANCE_RATIO = 0.005


@dataclass
class VisibilityResult:
    binary_mask: np.ndarray  # (N,) bool
    cosine_weights: np.ndarray  # (N,) float32 analysis weights
    facing_cosines: np.ndarray  # (N,) float32
    visible_count: int
    beauty_weights: Optional[np.ndarray] = None  # (N,) soft weights
    triangle_weights_analysis: Optional[np.ndarray] = None
    triangle_weights_beauty: Optional[np.ndarray] = None
    angle_threshold_deg: float = 75.0
    mode_notes: str = "analysis=hard, beauty=soft"


def _as_Nx3(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {a.shape}")
    return a


def compute_software_zbuffer_mask(
    vertices_camera: np.ndarray,
    resolution: int = 512,
    z_tolerance_ratio: float = Z_TOLERANCE_RATIO,
) -> np.ndarray:
    """Per-vertex occlusion mask via sparse software z-buffer."""
    if int(resolution) <= 0:
        raise ValueError("resolution must be a positive integer")
    if not np.isfinite(z_tolerance_ratio) or float(z_tolerance_ratio) < 0.0:
        raise ValueError("z_tolerance_ratio must be finite and non-negative")
    resolution = int(resolution)
    verts = np.asarray(vertices_camera, dtype=np.float32)
    if verts.ndim == 3 and verts.shape[0] == 1:
        verts = verts[0]
    if verts.ndim != 2 or verts.shape[1] != 3:
        return np.zeros((verts.shape[0] if verts.ndim >= 1 else 0,), dtype=bool)

    finite_mask = np.isfinite(verts).all(axis=1)
    out = np.zeros((verts.shape[0],), dtype=bool)
    if not np.any(finite_mask):
        return out

    valid = verts[finite_mask]
    x, y, z = valid[:, 0], valid[:, 1], valid[:, 2]
    x_span = max(float(x.max() - x.min()), 1e-6)
    y_span = max(float(y.max() - y.min()), 1e-6)
    x_idx = np.clip(((x - x.min()) / x_span) * (resolution - 1), 0, resolution - 1).astype(np.int32)
    y_idx = np.clip(((y - y.min()) / y_span) * (resolution - 1), 0, resolution - 1).astype(np.int32)

    z_buffer = np.full((resolution, resolution), np.inf, dtype=np.float32)
    np.minimum.at(z_buffer, (y_idx, x_idx), z)

    z_min, z_max = float(z.min()), float(z.max())
    epsilon = max((z_max - z_min) * float(z_tolerance_ratio), 1e-6)
    visible_valid = z <= (z_buffer[y_idx, x_idx] + epsilon)
    out[finite_mask] = visible_valid
    return out


def compute_vertex_visibility_from_normals(
    vertices_camera: np.ndarray,
    normals_camera: np.ndarray,
    *,
    angle_threshold_deg: float = 75.0,
    use_zbuffer: bool = True,
    z_resolution: int = 512,
    z_tolerance_ratio: float = Z_TOLERANCE_RATIO,
    angles_deg: Optional[Sequence[float]] = None,
    renderer_visible: Optional[np.ndarray] = None,
    beauty_min_weight: float = 0.001,
    beauty_gamma: float = 1.5,
    analysis_gamma: float = 1.0,
) -> VisibilityResult:
    """Vertex-level visibility: hard analysis + soft beauty weights."""
    verts = _as_Nx3(vertices_camera, "vertices_camera")
    normals = _as_Nx3(normals_camera, "normals_camera")
    if verts.shape[0] != normals.shape[0]:
        raise ValueError("vertices_camera and normals_camera length mismatch")

    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_u = np.divide(normals, np.maximum(nlen, 1e-8), out=np.zeros_like(normals), where=nlen > 1e-8)

    view = DEFAULT_VIEW_DIR
    facing_cosines = np.clip((normals_u @ view).astype(np.float32), -1.0, 1.0)

    thr = float(np.cos(np.deg2rad(float(angle_threshold_deg))))
    binary = facing_cosines >= thr

    if use_zbuffer:
        binary = binary & compute_software_zbuffer_mask(
            verts, resolution=z_resolution, z_tolerance_ratio=z_tolerance_ratio
        )

    if renderer_visible is not None:
        rv = np.asarray(renderer_visible)
        if rv.dtype != bool:
            # accept 0/1 indices or mask
            if rv.ndim == 1 and rv.shape[0] == verts.shape[0] and set(np.unique(rv.tolist()[: min(20, len(rv))])) <= {0, 1}:
                rv = rv.astype(bool)
            elif rv.ndim == 1 and rv.size and rv.max() < verts.shape[0] and rv.dtype.kind in "iu":
                mask = np.zeros(verts.shape[0], dtype=bool)
                mask[rv.astype(np.int64)] = True
                rv = mask
            else:
                rv = rv.astype(bool)
        if rv.shape[0] == verts.shape[0]:
            binary = binary & rv

    # Analysis weights: hard cut, zero floor
    denom = max(1e-6, 1.0 - thr)
    analysis_w = np.clip((facing_cosines - thr) / denom, 0.0, 1.0).astype(np.float32)
    if analysis_gamma != 1.0:
        analysis_w = np.power(analysis_w, float(analysis_gamma)).astype(np.float32)
    analysis_w = analysis_w * binary.astype(np.float32)

    # Beauty weights: soft, small floor for non-backfacing
    beauty_w = np.power(np.maximum(facing_cosines, 0.0), float(beauty_gamma)).astype(np.float32)
    beauty_w = np.where(facing_cosines > 0.0, np.maximum(beauty_w, float(beauty_min_weight)), 0.0).astype(
        np.float32
    )
    if use_zbuffer:
        # still kill fully occluded for beauty but keep soft facing
        zmask = compute_software_zbuffer_mask(
            verts, resolution=z_resolution, z_tolerance_ratio=z_tolerance_ratio
        )
        beauty_w = beauty_w * np.where(zmask, 1.0, 0.05).astype(np.float32)

    # Profile yaw fade (forensic anti-hallucination)
    analysis_w, beauty_w = _apply_yaw_fade(analysis_w, beauty_w, verts, angles_deg)

    return VisibilityResult(
        binary_mask=binary.astype(bool),
        cosine_weights=analysis_w,
        facing_cosines=facing_cosines,
        visible_count=int(np.count_nonzero(binary)),
        beauty_weights=beauty_w,
        angle_threshold_deg=float(angle_threshold_deg),
    )


def _apply_yaw_fade(
    analysis_w: np.ndarray,
    beauty_w: np.ndarray,
    verts: np.ndarray,
    angles_deg: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if angles_deg is None or len(angles_deg) < 2:
        return analysis_w, beauty_w
    try:
        yaw = float(angles_deg[1])
    except Exception:
        return analysis_w, beauty_w
    yaw_abs = abs(yaw)
    x_coords = verts[:, 0]
    if yaw > 0:
        turning_away = x_coords < 0
    else:
        turning_away = x_coords > 0
    if 45.0 <= yaw_abs <= 60.0:
        fade = float((60.0 - yaw_abs) / 15.0)
        analysis_w = analysis_w.copy()
        beauty_w = beauty_w.copy()
        analysis_w[turning_away] *= fade
        beauty_w[turning_away] *= max(fade, 0.2)
    elif yaw_abs > 60.0:
        analysis_w = analysis_w.copy()
        beauty_w = beauty_w.copy()
        analysis_w[turning_away] *= 0.0
        beauty_w[turning_away] *= 0.05
    return analysis_w, beauty_w


def compute_visibility(
    *,
    vertices_camera: np.ndarray,
    normals_camera: np.ndarray,
    angle_threshold_deg: float = 75.0,
    triangles: Optional[np.ndarray] = None,
    vertices_2d: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    angles_deg: Optional[Sequence[float]] = None,
    renderer_visible: Optional[np.ndarray] = None,
    use_zbuffer: bool = True,
) -> VisibilityResult:
    """Main entry: vertex visibility + optional triangle weights."""
    result = compute_vertex_visibility_from_normals(
        vertices_camera,
        normals_camera,
        angle_threshold_deg=angle_threshold_deg,
        use_zbuffer=use_zbuffer,
        angles_deg=angles_deg,
        renderer_visible=renderer_visible,
    )
    if triangles is not None:
        tri = np.asarray(triangles, dtype=np.int64)
        result.triangle_weights_analysis = compute_triangle_visibility(
            vertices_3d=vertices_camera,
            triangles=tri,
            angle_threshold_deg=angle_threshold_deg,
            mode="analysis",
            use_zbuffer=use_zbuffer and vertices_2d is not None and image_size is not None,
            vertices_2d=vertices_2d,
            image_size=image_size,
        )
        result.triangle_weights_beauty = compute_triangle_visibility(
            vertices_3d=vertices_camera,
            triangles=tri,
            angle_threshold_deg=max(angle_threshold_deg, 85.0),
            mode="beauty",
            use_zbuffer=use_zbuffer and vertices_2d is not None and image_size is not None,
            vertices_2d=vertices_2d,
            image_size=image_size,
        )
    return result


def compute_triangle_visibility(
    vertices_3d: np.ndarray,
    triangles: np.ndarray,
    view_dir: Optional[np.ndarray] = None,
    angle_threshold_deg: float = 75.0,
    gamma: float = 1.5,
    use_zbuffer: bool = False,
    vertices_2d: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    z_tolerance: float = 1e-3,
    occlusion_falloff: float = 0.1,
    mode: str = "analysis",
    min_weight_floor: Optional[float] = None,
) -> np.ndarray:
    """Per-triangle visibility weights.

    mode="analysis": hard angle threshold, floor=0 (no leak of occluded tris)
    mode="beauty": soft, floor default 0.001 for hole-free UV
    """
    verts = _as_Nx3(vertices_3d, "vertices_3d")
    tris = np.asarray(triangles, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")

    if view_dir is None:
        view = DEFAULT_VIEW_DIR.copy()
    else:
        view = np.asarray(view_dir, dtype=np.float32).reshape(3)
        n = float(np.linalg.norm(view))
        if n < 1e-8:
            raise ValueError("view_dir magnitude is too small")
        view = view / n

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
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
    mode = (mode or "analysis").lower()
    if mode == "beauty":
        floor = 0.001 if min_weight_floor is None else float(min_weight_floor)
        # soft: power of positive cos, keep tiny floor for frontish tris only
        soft = np.power(np.maximum(cos_angle, 0.0), float(gamma)).astype(np.float32)
        weights[valid] = np.where(cos_angle[valid] > 0.0, np.maximum(soft[valid], floor), 0.0)
        # still hard-kill extreme backfaces beyond threshold for stability
        weights[valid & (cos_angle < thr * 0.5)] = 0.0
    else:
        # analysis: REAL hard threshold (ITER2 fix)
        floor = 0.0 if min_weight_floor is None else float(min_weight_floor)
        front = valid & (cos_angle >= thr)
        if float(gamma) != 1.0:
            scaled = np.power(np.clip((cos_angle - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0), float(gamma))
        else:
            scaled = np.clip((cos_angle - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
        weights[front] = np.maximum(scaled[front], floor).astype(np.float32)

    if use_zbuffer:
        if vertices_2d is None or image_size is None:
            pass
        else:
            _apply_centroid_zbuffer(
                weights=weights,
                mask=valid,
                verts=verts,
                tris=tris,
                vertices_2d=vertices_2d,
                image_size=image_size,
                z_tolerance=z_tolerance,
                occlusion_falloff=occlusion_falloff if mode == "beauty" else min(occlusion_falloff, 0.01),
            )

    return weights


def _apply_centroid_zbuffer(
    weights: np.ndarray,
    mask: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    vertices_2d: np.ndarray,
    image_size: Tuple[int, int],
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

    # vectorized min into zbuffer
    order = np.argsort(centroids_z)
    for idx in order:
        if not mask[idx]:
            continue
        x, y = cx[idx], cy[idx]
        z = centroids_z[idx]
        if z < zbuffer[y, x]:
            zbuffer[y, x] = z

    occluded = mask & (centroids_z > zbuffer[cy, cx] + float(z_tolerance))
    weights[occluded] *= float(occlusion_falloff)


def map_triangle_weights_to_vertices(
    triangles: np.ndarray,
    tri_weights: np.ndarray,
    num_vertices: Optional[int] = None,
) -> np.ndarray:
    tris = np.asarray(triangles, dtype=np.int64)
    weights = np.asarray(tri_weights, dtype=np.float32).reshape(-1)
    if tris.shape[0] != weights.shape[0]:
        raise ValueError("triangles and tri_weights length mismatch")
    if num_vertices is None:
        num_vertices = int(tris.max()) + 1 if tris.size else 0
    vert_sum = np.zeros(num_vertices, dtype=np.float64)
    vert_count = np.zeros(num_vertices, dtype=np.float64)
    for j in range(3):
        np.add.at(vert_sum, tris[:, j], weights)
        np.add.at(vert_count, tris[:, j], 1.0)
    return (vert_sum / np.maximum(vert_count, 1e-6)).astype(np.float32)


def get_visible_zones(yaw: float, pitch: float) -> List[str]:
    visible = ["nasal_bridge", "chin", "forehead"]
    if yaw < 45.0:
        visible.extend(["left_eye", "left_zygomatic", "left_cheek"])
    if yaw > -45.0:
        visible.extend(["right_eye", "right_zygomatic", "right_cheek"])
    if pitch > -30.0:
        visible.extend(["jawline", "lower_lip"])
    return visible


def filter_metrics_by_pose(metrics: Dict[str, float], yaw: float, pitch: float) -> Dict[str, float]:
    zones = set(get_visible_zones(yaw, pitch))
    return {k: v for k, v in metrics.items() if k in zones}
