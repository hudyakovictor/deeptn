"""ITER6 pair-compare engine for 3DDFA forensic geometry.

Visibility intersection → rigid Umeyama (no scale) → bone-zone metrics.
Pure numpy; uses util.alignment / visibility / zones / calibration / geom_utils.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from util.alignment import (
    AlignmentResult,
    align_meshes_shared,
    euler_to_rotation_matrix,
    rigid_umeyama,
)
from util.calibration import apply_person_baseline, linear_snr
from util.geom_utils import bounded_score_from_error, face_scale_from_points, weighted_mean_abs
from util.pose_buckets import normalize_bucket_name
from util.visibility import VisibilityResult, compute_visibility
from util.zones import (
    MACRO_BONE_INDICES,
    apply_expression_exclusion_mask,
    compute_zone_metrics,
    summarize_bone_priority_metrics,
    zone_vertex_mask,
)

__all__ = [
    "ALIGNMENT_MIN_SHARED",
    "VISIBILITY_ANGLE_DEG",
    "PairCompareInput",
    "PairCompareResult",
    "shared_vertex_indices",
    "geodesic_pose_distance",
    "pose_delta_deg",
    "id_params_cosine_distance",
    "prepare_pair_alignment",
    "score_aligned_pair",
    "compare_pair",
]

ALIGNMENT_MIN_SHARED = 50
VISIBILITY_ANGLE_DEG = 75.0
BONE_CORE_ZONES = (
    "orbit_L", "orbit_R", "cheekbone_L", "cheekbone_R",
    "nose_bridge_tip", "chin", "brow_ridge_L", "brow_ridge_R",
    "jaw_angle_L", "jaw_angle_R",
)


@dataclass
class PairCompareInput:
    """Minimal mesh package for one photo (library-local, not pipeline ReconstructionResult)."""
    vertices: np.ndarray  # (N,3) preferred: identity or model-space for forensic
    normals: Optional[np.ndarray] = None  # (N,3) camera-space for visibility
    vertices_camera: Optional[np.ndarray] = None  # if None, use vertices for z-buffer
    angles_deg: Optional[np.ndarray] = None  # pitch, yaw, roll
    pose_bucket: str = "frontal"
    alpha_id: Optional[np.ndarray] = None
    visible_idx: Optional[np.ndarray] = None
    person_id: Optional[str] = None
    photo_id: Optional[str] = None
    quality_score: float = 1.0


@dataclass
class PairCompareResult:
    status: str
    shared_count: int
    pose_delta_deg: float
    raw_geometry_error: Optional[float] = None
    robust_geometry_error: Optional[float] = None
    bone_raw_geometry_error: Optional[float] = None
    bone_bounded_similarity_score: Optional[float] = None
    bounded_similarity_score: Optional[float] = None
    id_cosine_distance: Optional[float] = None
    alignment: Optional[Dict[str, Any]] = None
    zones: List[Dict[str, Any]] = field(default_factory=list)
    bone_summary: Dict[str, Any] = field(default_factory=dict)
    snr: Optional[float] = None
    predicted_noise: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def shared_vertex_indices(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> np.ndarray:
    shared = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool)
    return np.where(shared)[0].astype(np.int64)


def geodesic_pose_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle (deg) between rotations: arccos((tr(R_a^T R_b)-1)/2)."""
    R = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def pose_delta_deg(angles_a: Optional[np.ndarray], angles_b: Optional[np.ndarray]) -> float:
    if angles_a is None or angles_b is None:
        return 0.0
    try:
        R_a = euler_to_rotation_matrix(np.deg2rad(np.asarray(angles_a, dtype=np.float64).reshape(3)))
        R_b = euler_to_rotation_matrix(np.deg2rad(np.asarray(angles_b, dtype=np.float64).reshape(3)))
        return geodesic_pose_distance(R_a, R_b)
    except Exception:
        da = np.asarray(angles_a, dtype=np.float64).reshape(3) - np.asarray(angles_b, dtype=np.float64).reshape(3)
        wrapped = np.arctan2(np.sin(np.radians(da)), np.cos(np.radians(da))) * 180.0 / np.pi
        return float(np.linalg.norm(wrapped))


def id_params_cosine_distance(id_a: Optional[np.ndarray], id_b: Optional[np.ndarray]) -> Optional[float]:
    if id_a is None or id_b is None:
        return None
    a = np.asarray(id_a, dtype=np.float64).ravel()
    b = np.asarray(id_b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 10:
        return None
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-8:
        return None
    return float(1.0 - float(np.dot(a, b)) / norm)


def _visibility_for(mesh: PairCompareInput, angle_threshold_deg: float) -> VisibilityResult:
    verts_cam = mesh.vertices_camera if mesh.vertices_camera is not None else mesh.vertices
    normals = mesh.normals
    if normals is None:
        # fallback: assume facing +Z
        normals = np.zeros_like(verts_cam)
        normals[:, 2] = 1.0
    return compute_visibility(
        vertices_camera=verts_cam,
        normals_camera=normals,
        angle_threshold_deg=angle_threshold_deg,
        angles_deg=None if mesh.angles_deg is None else np.asarray(mesh.angles_deg).reshape(-1),
        renderer_visible=mesh.visible_idx,
        use_zbuffer=True,
    )


def _bone_shared_mask(shared_idx: np.ndarray, n_vertices: int) -> np.ndarray:
    bone = np.zeros(n_vertices, dtype=bool)
    for z in BONE_CORE_ZONES:
        bone |= zone_vertex_mask(z, n_vertices)
    # positions in shared list that are bone verts
    return bone[shared_idx]


def score_aligned_pair(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: np.ndarray,
    *,
    trim_keep_ratio: float = 0.85,
) -> Tuple[float, float, float, float]:
    """Return primary_err, primary_score, robust_err, robust_score (normalized by face scale)."""
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    scale = face_scale_from_points(b)
    dists = np.linalg.norm(a - b, axis=1) / max(scale, 1e-6)
    primary = weighted_mean_abs(dists, w)
    # trimmed robust: keep lowest-error fraction by weight
    order = np.argsort(dists)
    keep_n = max(int(np.ceil(len(dists) * float(trim_keep_ratio))), 4)
    keep = order[:keep_n]
    robust = weighted_mean_abs(dists[keep], w[keep])
    return (
        float(primary),
        bounded_score_from_error(float(primary)),
        float(robust),
        bounded_score_from_error(float(robust)),
    )


def prepare_pair_alignment(
    mesh_a: PairCompareInput,
    mesh_b: PairCompareInput,
    *,
    angle_threshold_deg: float = VISIBILITY_ANGLE_DEG,
    min_shared: int = ALIGNMENT_MIN_SHARED,
    exclude_expression: bool = True,
) -> Dict[str, Any]:
    """Visibility ∩ → unit clouds on shared verts. status ok | insufficient_*."""
    va = np.asarray(mesh_a.vertices, dtype=np.float64)
    vb = np.asarray(mesh_b.vertices, dtype=np.float64)
    if va.shape != vb.shape or va.ndim != 2 or va.shape[1] != 3:
        return {"status": "shape_mismatch", "shared_idx": np.zeros(0, dtype=np.int64)}

    vis_a = _visibility_for(mesh_a, angle_threshold_deg)
    vis_b = _visibility_for(mesh_b, angle_threshold_deg)

    mask_a = vis_a.binary_mask.copy()
    mask_b = vis_b.binary_mask.copy()
    if exclude_expression:
        mask_a = apply_expression_exclusion_mask(mask_a, n_vertices=va.shape[0])
        mask_b = apply_expression_exclusion_mask(mask_b, n_vertices=vb.shape[0])

    shared_idx = shared_vertex_indices(mask_a, mask_b)
    pose_delta = pose_delta_deg(mesh_a.angles_deg, mesh_b.angles_deg)

    if shared_idx.size < int(min_shared):
        return {
            "status": "insufficient_shared_visibility",
            "shared_idx": shared_idx,
            "pose_delta": pose_delta,
            "vis_a": vis_a,
            "vis_b": vis_b,
            "shared_count": int(shared_idx.size),
        }

    pts_a = va[shared_idx]
    pts_b = vb[shared_idx]
    ca = pts_a.mean(axis=0)
    cb = pts_b.mean(axis=0)
    scale_a = face_scale_from_points(pts_a)
    scale_b = face_scale_from_points(pts_b)
    if scale_a < 1e-8 or scale_b < 1e-8:
        return {
            "status": "insufficient_scale",
            "shared_idx": shared_idx,
            "pose_delta": pose_delta,
            "vis_a": vis_a,
            "vis_b": vis_b,
            "shared_count": int(shared_idx.size),
        }

    points_a_unit = (pts_a - ca) / scale_a
    points_b_unit = (pts_b - cb) / scale_b
    weights = np.minimum(vis_a.cosine_weights[shared_idx], vis_b.cosine_weights[shared_idx])
    weights = np.maximum(weights, 0.0).astype(np.float64)
    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(weights)

    bone_mask = _bone_shared_mask(shared_idx, va.shape[0])

    return {
        "status": "ok",
        "shared_idx": shared_idx,
        "shared_count": int(shared_idx.size),
        "pose_delta": pose_delta,
        "vis_a": vis_a,
        "vis_b": vis_b,
        "points_a_unit": points_a_unit,
        "points_b_unit": points_b_unit,
        "weights": weights,
        "bone_mask": bone_mask,
        "centroid_a": ca,
        "centroid_b": cb,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "bucket_a": normalize_bucket_name(mesh_a.pose_bucket),
        "bucket_b": normalize_bucket_name(mesh_b.pose_bucket),
    }


def compare_pair(
    mesh_a: PairCompareInput,
    mesh_b: PairCompareInput,
    *,
    angle_threshold_deg: float = VISIBILITY_ANGLE_DEG,
    min_shared: int = ALIGNMENT_MIN_SHARED,
    predicted_noise: Optional[float] = None,
    person_baselines: Optional[Mapping[str, Mapping[str, float]]] = None,
    trim_keep_ratio: float = 0.85,
) -> PairCompareResult:
    """Full pair compare: align on shared visible, score full + bone zones."""
    notes: List[str] = []
    prep = prepare_pair_alignment(
        mesh_a, mesh_b,
        angle_threshold_deg=angle_threshold_deg,
        min_shared=min_shared,
    )
    pose_delta = float(prep.get("pose_delta", 0.0))
    shared_count = int(prep.get("shared_count", 0))

    if prep.get("status") != "ok":
        return PairCompareResult(
            status=str(prep.get("status")),
            shared_count=shared_count,
            pose_delta_deg=pose_delta,
            notes=[str(prep.get("status"))],
            id_cosine_distance=id_params_cosine_distance(mesh_a.alpha_id, mesh_b.alpha_id),
        )

    pts_a = prep["points_a_unit"]
    pts_b = prep["points_b_unit"]
    weights = prep["weights"]
    bone_mask = prep["bone_mask"]
    shared_idx = prep["shared_idx"]

    # Align: prefer bone anchors when enough, else all shared
    if int(np.count_nonzero(bone_mask)) >= 12:
        fit_a, fit_b, fit_w = pts_a[bone_mask], pts_b[bone_mask], weights[bone_mask]
        notes.append("umeyama_fit=bone_anchors")
    else:
        fit_a, fit_b, fit_w = pts_a, pts_b, weights
        notes.append("umeyama_fit=all_shared")

    try:
        alignment = rigid_umeyama(fit_a, fit_b, weights=fit_w, allow_scale=False)
    except Exception as exc:
        return PairCompareResult(
            status="alignment_failed",
            shared_count=shared_count,
            pose_delta_deg=pose_delta,
            notes=[f"alignment_failed:{exc}"],
        )

    aligned = alignment.scale * (pts_a @ alignment.rotation) + alignment.translation
    primary, primary_score, robust, robust_score = score_aligned_pair(
        aligned, pts_b, weights, trim_keep_ratio=trim_keep_ratio
    )

    # Bone-only score
    bone_raw = None
    bone_score = None
    if int(np.count_nonzero(bone_mask)) >= 8:
        br, bs, _, _ = score_aligned_pair(
            aligned[bone_mask], pts_b[bone_mask], weights[bone_mask], trim_keep_ratio=0.9
        )
        bone_raw, bone_score = br, bs

    # Zone metrics on unit-aligned clouds (face_width ~1)
    zones = compute_zone_metrics(
        aligned_points_a=aligned,
        points_b=pts_b,
        shared_indices=shared_idx,
        shared_weights=weights,
        face_width_override=1.0,
        exclusive_vertices=False,
        min_zone_vertices=3,
    )
    bone_summary = summarize_bone_priority_metrics(zones, min_usable_bone_zones=4)
    if bone_raw is None and bone_summary.get("bone_raw_geometry_error") is not None:
        bone_raw = float(bone_summary["bone_raw_geometry_error"])
        bone_score = bone_summary.get("bone_bounded_similarity_score")

    # Person baseline on hybrid error
    hybrid = bone_raw if bone_raw is not None else robust
    bucket = prep.get("bucket_a") or "frontal"
    if mesh_a.person_id and mesh_b.person_id and person_baselines:
        hybrid = apply_person_baseline(
            float(hybrid), mesh_a.person_id, mesh_b.person_id, str(bucket), person_baselines
        )
        notes.append("person_baseline_applied" if mesh_a.person_id == mesh_b.person_id else "cross_person")

    snr = None
    if predicted_noise is not None:
        snr = linear_snr(float(hybrid), float(predicted_noise))

    id_dist = id_params_cosine_distance(mesh_a.alpha_id, mesh_b.alpha_id)

    return PairCompareResult(
        status="ok",
        shared_count=shared_count,
        pose_delta_deg=pose_delta,
        raw_geometry_error=primary,
        robust_geometry_error=robust,
        bone_raw_geometry_error=bone_raw,
        bone_bounded_similarity_score=None if bone_score is None else float(bone_score),
        bounded_similarity_score=primary_score,
        id_cosine_distance=id_dist,
        alignment={
            "scale": alignment.scale,
            "residual_before": alignment.residual_before,
            "residual_after": alignment.residual_after,
            "inlier_fraction": alignment.inlier_fraction,
            "rotation": alignment.rotation,
            "translation": alignment.translation,
        },
        zones=[z.to_dict() for z in zones if z.status == "ok"],
        bone_summary=bone_summary,
        snr=snr,
        predicted_noise=predicted_noise,
        notes=notes,
        payload={
            "bucket_a": prep.get("bucket_a"),
            "bucket_b": prep.get("bucket_b"),
            "hybrid_error": hybrid,
            "quality_a": mesh_a.quality_score,
            "quality_b": mesh_b.quality_score,
        },
    )
