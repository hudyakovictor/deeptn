"""ITER10 geometry identity metrics — port of newapp pipeline/scoring.py.

Library-local: no core.constants / core.policy. Uses util.zones MACRO indices helpers.
Primary API: extract_macro_bone_metrics(vertices, bone_indices, angles).
"""
from __future__ import annotations

import math
import numpy as np

from util.alignment import rigid_umeyama, AlignmentResult
from util.geom_utils import bounded_score_from_error, weighted_mean_abs, face_scale_from_points
from util.zones import canthus_points_from_orbit, MACRO_BONE_INDICES

# Local constants (were core.constants / core.policy / zones thresholds)
TRIMMED_KEEP_RATIO = 0.90
MIN_KEEP_N = 8
POSE_YAW_BILATERAL_OFF_DEG = 18.0
POSE_YAW_OCCLUDE_DEG = 55.0
THRESHOLD_JAW_OPEN = 0.35
THRESHOLD_SMILE = 0.35


def profile_trim_keep_ratio(pose_bucket: str | None) -> float:
    b = (pose_bucket or "").lower()
    if "profile" in b:
        return 0.75
    if "deep" in b:
        return 0.80
    if "mid" in b:
        return 0.85
    return TRIMMED_KEEP_RATIO


def ramus_vertical_height_ratio(
    ramus_pts: np.ndarray,
    gonion: np.ndarray | None,
    face_height: float,
    face_vertical: np.ndarray | None = None,
) -> float | None:
    """Высота ветви челюсти: gonion → верх ramus (не весь Y-span зоны jaw)."""
    ramus_pts = np.asarray(ramus_pts) if ramus_pts is not None else np.zeros((0, 3))
    if ramus_pts.size == 0 or gonion is None or face_height <= 1e-6:
        return None
    if face_vertical is not None:
        face_vertical_u = face_vertical / (np.linalg.norm(face_vertical) + 1e-8)
        gonion_y = float(np.dot(gonion, face_vertical_u))
        y_vals = np.dot(ramus_pts, face_vertical_u)
    else:
        gonion_y = float(gonion[1])
        y_vals = ramus_pts[:, 1]
    span = float(np.max(np.abs(y_vals - gonion_y)))
    if span <= 1e-6:
        return None
    return span / face_height


def temporal_fossa_points_from_orbit(orbit_pts: np.ndarray, side: str) -> np.ndarray:
    """Approximate temporal fossa samples from orbit extremes."""
    pts = np.asarray(orbit_pts, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    if side.lower().startswith("l"):
        # more lateral = smaller x in many BFM conventions for subject-left
        order = np.argsort(pts[:, 0])
        return pts[order[: max(3, len(order)//4)]]
    order = np.argsort(pts[:, 0])
    return pts[order[-max(3, len(order)//4) :]]

def _get_face_scale_from_points(points: np.ndarray) -> float:
    """
    [ИСПРАВЛЕНИЕ №3] Трехмерное вычисление масштаба лица.
    Использует евклидову норму между перцентилями вместо одномерных осей.
    """
    if points.shape[0] == 0:
        return 1.0
    
    # Берем перцентили по каждой из 3-х осей
    q_95 = np.percentile(points, 95, axis=0)
    q_05 = np.percentile(points, 5, axis=0)
    
    # 3D Евклидово расстояние между 95% и 5% границами
    scale = float(np.linalg.norm(q_95 - q_05))
    
    return max(scale, 1e-6)

def compute_interorbital_ratio(canthus_L_inner: np.ndarray, canthus_R_inner: np.ndarray, zygomatic_breadth: float) -> float:
    """
    [K-03] Вычисляет отношение межорбитального расстояния к скуловой ширине.
    """
    if np.allclose(canthus_L_inner, 0) or np.allclose(canthus_R_inner, 0) or zygomatic_breadth <= 1e-6:
        return 0.0
    interorbital_dist = float(np.linalg.norm(canthus_L_inner - canthus_R_inner))
    return interorbital_dist / zygomatic_breadth

def _robust_trimmed_3d_error(
    values: np.ndarray,
    weights: np.ndarray,
    keep_ratio: float | None = None,
    min_keep_n: int = MIN_KEEP_N
) -> float:
    """
    Robust trimmed mean for distance errors: keep low-error stable points, drop upper-tail outliers only.
    """
    if values.size == 0:
        return 0.0
    
    n = values.size
    if keep_ratio is None:
        spread = float(np.std(values))
        # High spread → trim more aggressively (lower keep_ratio).
        keep_ratio = 0.82 if spread > 0.06 else TRIMMED_KEEP_RATIO
    else:
        # S-04: yaw-based keep_ratio is a ceiling; high spread may trim further.
        spread = float(np.std(values))
        if spread > 0.06:
            keep_ratio = min(float(keep_ratio), 0.82)
    if n <= min_keep_n:
        return weighted_mean_abs(values, weights)
    
    # Upper-tail only: keep lowest-error stable anatomical points.
    n_keep = max(int(n * keep_ratio), min_keep_n)
    n_keep = min(n_keep, n)
    sorted_indices = np.argsort(np.abs(values))
    keep_indices = sorted_indices[:n_keep]
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[keep_indices] = True
        
    if not np.any(keep_mask):
        return weighted_mean_abs(values, weights)
        
    return weighted_mean_abs(values[keep_mask], weights[keep_mask])

def fit_best_plane(points: np.ndarray, *, symmetrize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    # Симметризация только для почти фронтальных ракурсов; на профиле зеркало по X=0 искажает нормаль.
    if symmetrize:
        mirrored_points = points.copy()
        mirrored_points[:, 0] = -mirrored_points[:, 0]
        combined = np.vstack([points, mirrored_points])
    else:
        combined = points
    
    if points.shape[0] < 3:
        return points.mean(axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float64)
    centroid = combined.mean(axis=0)
    centered = combined - centroid
    if float(np.linalg.norm(centered)) < 1e-8:
        return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0 or float(s[-1]) < 1e-10:
        return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8 or not np.isfinite(norm):
        return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return centroid, normal / norm

def score_aligned_pair(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: np.ndarray,
    reliability_weight: float = 1.0,
    trim_keep_ratio: float | None = None,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    [GEOM-03] Forensic 3D Scoring.

    [ITER-1] ИСПРАВЛЕНИЕ: Вычисляем истинное евклидово расстояние между точками,
    а не их проекцию на одну из плоскостей. Это избегает двойного смещения
    при непараллельных плоскостях из-за погрешности детекции позы.
    """
    scale_b = _get_face_scale_from_points(points_b)
    _centroid_b, plane_normal = fit_best_plane(points_b)

    # 1. Difference vector
    diffs = points_a - points_b

    # [ITER-1] ИСПРАВЛЕНИЕ: Честное L2 расстояние между выровненными вершинами
    # Вычисляем норму по оси X,Y,Z для каждой точки
    true_distances = np.linalg.norm(diffs, axis=1)

    # Normalize to face scale
    distances_normalized = true_distances / scale_b

    # Apply reliability weight (from texture/pose)
    effective_weights = weights * reliability_weight

    # Weighted mean
    primary_error = weighted_mean_abs(distances_normalized, effective_weights)

    # Robust variant (trimmed)
    robust_error = _robust_trimmed_3d_error(
        distances_normalized,
        effective_weights,
        keep_ratio=trim_keep_ratio,
    )

    return (
        primary_error,
        bounded_score_from_error(primary_error),
        robust_error,
        bounded_score_from_error(robust_error),
        plane_normal,
    )

def align_and_score(
    points_a: np.ndarray,
    points_b: np.ndarray,
    weights: np.ndarray,
    alignment_weights: np.ndarray | None = None,
    reliability_weight: float = 1.0,
    pose_bucket: str | None = None,
    yaw_max_deg: float | None = None,
    fit_points_a: np.ndarray | None = None,
    fit_points_b: np.ndarray | None = None,
    fit_weights: np.ndarray | None = None,
    score_mask: np.ndarray | None = None,
    trim_keep_ratio_override: float | None = None,
) -> tuple[AlignmentResult, float, float, float, float, np.ndarray]:
    """
    Full alignment and scoring pipeline.
    fit_points_* — sparse bone anchors for Umeyama; score uses points_a/points_b (optionally score_mask).
    """
    fa = fit_points_a if fit_points_a is not None else points_a
    fb = fit_points_b if fit_points_b is not None else points_b
    if fit_points_a is not None:
        fw = np.ones(len(fa), dtype=np.float64)
    else:
        fw = fit_weights if fit_weights is not None else alignment_weights
    # [AXIOM-02] Scale must be locked (False) for forensic comparison
    alignment = rigid_umeyama(fa, fb, weights=fw, allow_scale=False)
    aligned_all = alignment.scale * (points_a @ alignment.rotation) + alignment.translation
    alignment = AlignmentResult(
        rotation=alignment.rotation,
        translation=alignment.translation,
        scale=alignment.scale,
        source_aligned=aligned_all,
        residual_before=alignment.residual_before,
        residual_after=alignment.residual_after,
    )

    score_a = aligned_all[score_mask] if score_mask is not None else aligned_all
    score_b = points_b[score_mask] if score_mask is not None else points_b
    score_w = weights[score_mask] if score_mask is not None else weights

    if trim_keep_ratio_override is not None:
        trim_keep_ratio = trim_keep_ratio_override
    elif yaw_max_deg is not None:
        trim_frac = 0.10 + 0.10 * (min(abs(yaw_max_deg), 90.0) / 45.0)
        trim_frac = min(trim_frac, 0.35)
        trim_keep_ratio = 1.0 - trim_frac
    else:
        trim_keep_ratio = profile_trim_keep_ratio(pose_bucket)

    (
        primary_error,
        bounded_primary,
        robust_error,
        bounded_robust,
        plane_normal,
    ) = score_aligned_pair(
        score_a,
        score_b,
        score_w,
        reliability_weight=reliability_weight,
        trim_keep_ratio=trim_keep_ratio,
    )

    return (
        alignment,
        primary_error,
        bounded_primary,
        robust_error,
        bounded_robust,
        plane_normal,
    )

def calc_3d_angle(v1, vertex, v2):
    """Вычисляет истинный 3D-угол между тремя точками"""
    a = np.array(v1) - np.array(vertex)
    b = np.array(v2) - np.array(vertex)
    
    # Защита от деления на ноль
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return None
        
    cos_a = np.dot(a, b) / (norm_a * norm_b)
    # Защита от выхода за пределы [-1, 1] из-за погрешностей float
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _zone_vertex_array(vertices: np.ndarray, bone_indices: dict, name: str) -> np.ndarray:
    raw = bone_indices.get(name, [])
    if not raw:
        return np.zeros((0, 3))
    idx = np.asarray(list(raw), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < vertices.shape[0])]
    if idx.size == 0:
        return np.zeros((0, 3))
    pts = vertices[idx]
    return pts if np.isfinite(pts).all() else np.zeros((0, 3))


def _zone_mean_depth(pts: np.ndarray, reference: np.ndarray, normal: np.ndarray) -> float | None:
    if pts.shape[0] == 0:
        return None
    depths = np.dot(pts - reference, normal)
    val = float(np.mean(depths))
    return None if not np.isfinite(val) else val


def _zone_depth_spread(pts: np.ndarray, reference: np.ndarray, normal: np.ndarray) -> float | None:
    """Std signed depth across zone vertices — 3D concavity of orbit fossa."""
    if pts.shape[0] < 4:
        return None
    depths = np.dot(pts - reference, normal)
    val = float(np.std(depths))
    return None if not np.isfinite(val) else val


def _zone_span_along(pts: np.ndarray, axis: np.ndarray) -> float:
    if pts.shape[0] == 0:
        return 0.0
    axis_u = axis / (np.linalg.norm(axis) + 1e-8)
    projs = pts @ axis_u
    return float(np.max(projs) - np.min(projs))


def _face_horizontal_axis(face_plane_normal: np.ndarray) -> np.ndarray:
    up_vector = np.array([0.0, -1.0, 0.0])
    face_horizontal = np.cross(face_plane_normal, up_vector)
    norm = np.linalg.norm(face_horizontal)
    # [BUGFIX-10] Увеличиваем порог отслеживания вырождения до 0.05 (~2.8 градуса) для стабильности при сильном pitch
    if norm <= 0.05:
        # Проецируем мировую ось X на плоскость лица в качестве стабильной альтернативы
        alt_horiz = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), face_plane_normal) * face_plane_normal
        alt_norm = np.linalg.norm(alt_horiz)
        if alt_norm <= 1e-4:
            return np.array([1.0, 0.0, 0.0])
        return alt_horiz / alt_norm
    return face_horizontal / norm


def _face_vertical_axis(
    face_plane_normal: np.ndarray,
    chin_bottom: np.ndarray,
    forehead_top: np.ndarray,
) -> np.ndarray:
    """Вертикаль лица в плоскости симметрии — устойчива к yaw (не world Y)."""
    face_h = _face_horizontal_axis(face_plane_normal)
    vertical = np.cross(face_h, face_plane_normal)
    norm = np.linalg.norm(vertical)
    if norm <= 1e-8:
        return np.array([0.0, -1.0, 0.0])
    vertical = vertical / norm
    if float(np.dot(vertical, forehead_top - chin_bottom)) < 0:
        vertical = -vertical
    return vertical


def _signed_lateral_offset(
    point: np.ndarray,
    line_p1: np.ndarray,
    line_p2: np.ndarray,
    face_plane_normal: np.ndarray,
) -> float:
    line_vec = line_p2 - line_p1
    point_vec = point - line_p1
    cross_prod = np.cross(line_vec, point_vec)
    sign = float(np.sign(np.dot(cross_prod, face_plane_normal))) if np.linalg.norm(cross_prod) > 1e-8 else 1.0
    dist = np.linalg.norm(cross_prod) / (np.linalg.norm(line_vec) + 1e-8)
    return sign * dist


def _null_side_metrics(metrics: dict[str, float | None], side: str) -> None:
    for key in (
        f"canthal_tilt_{side}",
        f"canthal_tilt_3d_{side}",
        f"orbit_depth_{side}_ratio",
        f"orbit_width_{side}_ratio",
        f"brow_ridge_projection_{side}_ratio",
        f"temporal_depth_{side}_ratio",
        f"zygomatic_arch_height_{side}_ratio",
        f"palpebral_fissure_length_{side}_ratio",
        f"mandibular_body_length_{side}_ratio",
        f"ramus_height_{side}_ratio",
        f"ligament_orbital_{side}_depth_ratio",
        f"orbit_fossa_spread_{side}",
        f"gonial_angle_{side}",
    ):
        metrics[key] = None


def _finalize_bilateral_geometry(metrics: dict[str, float | None]) -> None:
    """Пересчитать/обнулить L+R метрики после yaw-маски (только видимые стороны)."""
    od_l = metrics.get("orbit_depth_L_ratio")
    od_r = metrics.get("orbit_depth_R_ratio")
    if od_l is not None and od_r is not None:
        metrics["orbit_depth_asymmetry_ratio"] = abs(float(od_l) - float(od_r))
    else:
        metrics["orbit_depth_asymmetry_ratio"] = None

    if od_l is None or od_r is None:
        metrics["orbit_vertical_asymmetry_ratio"] = None
        metrics["orbit_vertical_signed_ratio"] = None
        metrics["orbital_height_signed"] = None
        metrics["orbital_asymmetry_index"] = None
        metrics["orbital_perimeter_symmetry"] = None
        metrics["interorbital_ratio"] = None

    pf_l = metrics.get("palpebral_fissure_length_L_ratio")
    pf_r = metrics.get("palpebral_fissure_length_R_ratio")
    if pf_l is not None and pf_r is not None:
        metrics["palpebral_fissure_asymmetry_ratio"] = abs(float(pf_l) - float(pf_r)) / max(
            float(pf_l), float(pf_r), 1e-6
        )
    else:
        metrics["palpebral_fissure_asymmetry_ratio"] = None

    tilt_vals = [
        float(metrics[k])
        for k in ("canthal_tilt_L", "canthal_tilt_R")
        if metrics.get(k) is not None and np.isfinite(float(metrics[k]))
    ]
    if len(tilt_vals) >= 2:
        metrics["canthal_tilt_asymmetry_deg"] = abs(tilt_vals[0] - tilt_vals[1])
        metrics["canthal_tilt_mean_deg"] = float(np.mean(tilt_vals))
    elif len(tilt_vals) == 1:
        metrics["canthal_tilt_asymmetry_deg"] = None
        metrics["canthal_tilt_mean_deg"] = tilt_vals[0]
    else:
        metrics["canthal_tilt_asymmetry_deg"] = None
        metrics["canthal_tilt_mean_deg"] = None


CHIN_TIP_VERTEX_ID = 33838
SUBNASALE_VERTEX_ID = 245


def _zone_idx_array(vertices: np.ndarray, bone_indices: dict, name: str) -> np.ndarray:
    raw = bone_indices.get(name, [])
    if not raw:
        return np.array([], dtype=np.int64)
    arr = np.asarray(list(raw), dtype=np.int64)
    return arr[(arr >= 0) & (arr < vertices.shape[0])]


def _gonion_from_jaw_pts(jaw_pts: np.ndarray, *, exclude_y_above: float | None = None) -> np.ndarray | None:
    if jaw_pts.size == 0:
        return None
    pts = jaw_pts
    if exclude_y_above is not None:
        mask = pts[:, 1] <= float(exclude_y_above)
        if np.any(mask):
            pts = pts[mask]
    return pts[np.argmax(pts[:, 1])]


def _chin_bottom_point(vertices: np.ndarray, bone_indices: dict) -> np.ndarray | None:
    """Mental protuberance — не смешивать с gonion (jaw_angle_* в chin zone)."""
    if CHIN_TIP_VERTEX_ID < vertices.shape[0]:
        tip = vertices[CHIN_TIP_VERTEX_ID]
        if np.isfinite(tip).all() and not np.allclose(tip, 0):
            return tip

    gonion_ids = set(_zone_idx_array(vertices, bone_indices, "jaw_angle_L").tolist())
    gonion_ids.update(_zone_idx_array(vertices, bone_indices, "jaw_angle_R").tolist())
    chin_idx = _zone_idx_array(vertices, bone_indices, "chin")
    chin_idx = chin_idx[~np.isin(chin_idx, list(gonion_ids))]
    if chin_idx.size == 0:
        chin_idx = _zone_idx_array(vertices, bone_indices, "chin")
    if chin_idx.size == 0:
        return None
    return vertices[chin_idx[np.argmax(vertices[chin_idx, 1])]]


def _safe_vertex(vertices: np.ndarray, idx: int) -> np.ndarray | None:
    if 0 <= idx < vertices.shape[0]:
        candidate = vertices[idx]
        if np.isfinite(candidate).all() and not np.allclose(candidate, 0):
            return candidate
    return None


def _resolve_subnasale_point(
    vertices: np.ndarray,
    get_zone_centroid,
    nasion_pt: np.ndarray,
) -> np.ndarray:
    # Prefer fixed topology anchor when compatible with current mesh.
    from_vertex = _safe_vertex(vertices, SUBNASALE_VERTEX_ID)
    if from_vertex is not None:
        return from_vertex
    from_wings = (get_zone_centroid("nose_wing_L") + get_zone_centroid("nose_wing_R")) / 2.0
    if np.isfinite(from_wings).all() and not np.allclose(from_wings, 0):
        return from_wings
    return nasion_pt


def _visible_gonion(
    gonion_L: np.ndarray | None,
    gonion_R: np.ndarray | None,
    *,
    pose_yaw_deg: float | None,
    bfm_yaw: float,
) -> np.ndarray | None:
    yaw = float(pose_yaw_deg) if pose_yaw_deg is not None else float(bfm_yaw)
    if yaw > POSE_YAW_OCCLUDE_DEG:
        return gonion_R
    if yaw < -POSE_YAW_OCCLUDE_DEG:
        return gonion_L
    if gonion_L is not None and gonion_R is not None:
        return (gonion_L + gonion_R) / 2.0
    return gonion_L if gonion_L is not None else gonion_R


def extract_macro_bone_metrics(
    vertices: np.ndarray,
    bone_indices: dict[str, list[int] | frozenset[int]],
    angles: np.ndarray,
    *,
    pose_yaw_deg: float | None = None,
    exp_params: np.ndarray | None = None,
) -> tuple[dict[str, float], float]:
    """
    [GEOM-01] Extraction of stable forensic features.
    """
    def get_zone_centroid(name: str) -> np.ndarray:
        idx_raw = bone_indices.get(name, [])
        if not idx_raw:
            return np.zeros(3)
        idx = np.asarray(list(idx_raw), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < vertices.shape[0])]
        if idx.size == 0:
            return np.zeros(3)
        centroid = np.mean(vertices[idx], axis=0)
        return centroid if np.isfinite(centroid).all() else np.zeros(3)

    def _valid_point(point: np.ndarray) -> bool:
        return bool(point.shape == (3,) and np.isfinite(point).all() and not np.allclose(point, 0))

    def _idx(name: str) -> np.ndarray:
        """Convert bone_indices entry (frozenset/list) to valid numpy index array."""
        raw = bone_indices.get(name, [])
        if not raw:
            return np.array([], dtype=np.int64)
        arr = np.asarray(list(raw), dtype=np.int64)
        return arr[(arr >= 0) & (arr < vertices.shape[0])]

    # The caller canonicalizes vertices to a bucket-specific pose first.  Deep/profile
    # buckets legitimately use yaw beyond 60 degrees, so only reject impossible input.
    angles = angles.get("angles", [0.0, 0.0, 0.0]) if isinstance(angles, dict) else angles
    effective_yaw = float(pose_yaw_deg) if pose_yaw_deg is not None else (float(angles[1]) if len(angles) > 1 else 0.0)
    if abs(effective_yaw) > 95.0:
        return {}, 0.0

    cheek_L = get_zone_centroid('cheekbone_L')
    cheek_R = get_zone_centroid('cheekbone_R')
    if not _valid_point(cheek_L) or not _valid_point(cheek_R):
        return {}, 0.0
        
    # Compensate zygomatic_breadth for neural reconstruction shrinkage under yaw (Error 8)
    yaw_corr = 1.0 + 0.075 * (min(abs(effective_yaw), 60.0) / 60.0) ** 2
    zygomatic_breadth_raw = float(np.linalg.norm(cheek_L - cheek_R))
    zygomatic_breadth = zygomatic_breadth_raw * yaw_corr

    # Adjust zygomatic_breadth for soft tissue / weight gain (Error 44)
    soft_tissue_L = get_zone_centroid('cheek_soft_L')
    soft_tissue_R = get_zone_centroid('cheek_soft_R')
    if _valid_point(soft_tissue_L) and _valid_point(soft_tissue_R):
        thickness = float(np.linalg.norm(soft_tissue_L - cheek_L) + np.linalg.norm(soft_tissue_R - cheek_R)) / 2.0
        adjusted = zygomatic_breadth - 0.15 * thickness
        # Soft-tissue subtract must not erase yaw shrinkage compensation (S-01).
        zygomatic_breadth = max(adjusted, zygomatic_breadth_raw * yaw_corr * 0.92)

    if not np.isfinite(zygomatic_breadth) or zygomatic_breadth <= 1e-6:
        return {}, 0.0
    mid_cheek_z = float((cheek_L[2] + cheek_R[2]) / 2.0)
    
    # 2. Face Height
    forehead_idx = _idx('forehead')
    chin_bottom = _chin_bottom_point(vertices, bone_indices)
    if forehead_idx.size == 0 or chin_bottom is None:
        return {}, 0.0
        
    # 3DDFA/BFM: Y растёт вниз — лоб = min(Y), подбородок = max(Y)
    forehead_top = vertices[forehead_idx[np.argmin(vertices[forehead_idx, 1])]]
    face_height_raw = float(abs(float(forehead_top[1]) - float(chin_bottom[1])))
    
    # Compensate for mouth opening / jaw open distortion of face_height (Error 43)
    # Check the ratio of forehead-to-subnasale (upper face height) to total face height.
    # In a closed face, this ratio is typically >= 0.57. If the mouth is open, it drops.
    subnasale_pt = _resolve_subnasale_point(vertices, get_zone_centroid, get_zone_centroid('nose_bridge_tip'))
    if _valid_point(subnasale_pt):
        upper_face_height = float(abs(float(forehead_top[1]) - float(subnasale_pt[1])))
        ratio = upper_face_height / (face_height_raw + 1e-8)
        if ratio < 0.57 and upper_face_height > 1e-6:
            # Jaw open: estimate closed-mouth height; stronger correction when mouth is wider.
            canonical_upper = 0.58
            if ratio < 0.50:
                canonical_upper = 0.52
            elif ratio < 0.55:
                canonical_upper = 0.55
            face_height = min(upper_face_height / canonical_upper, face_height_raw)
        else:
            face_height = face_height_raw
    else:
        face_height = face_height_raw

    if not np.isfinite(face_height) or face_height <= 1e-6:
        return {}, 0.0
    
    # 3. Indices
    jaw_raw = float(np.linalg.norm(get_zone_centroid('jaw_angle_L') - get_zone_centroid('jaw_angle_R'))) / zygomatic_breadth
    metrics: dict[str, float | None] = {
        "cranial_face_index": float(np.clip(zygomatic_breadth / face_height, 0.5, 2.0)),
        "jaw_width_ratio": float(np.clip(jaw_raw, 0.4, 1.3)),
    }
    reliability = 1.0
    
    # 4. Orbital Complex
    orbit_L_idx = _idx('orbit_L')
    orbit_R_idx = _idx('orbit_R')
    orbit_L_pts = vertices[orbit_L_idx] if orbit_L_idx.size > 0 else np.zeros((0, 3))
    orbit_R_pts = vertices[orbit_R_idx] if orbit_R_idx.size > 0 else np.zeros((0, 3))
    
    def calc_tilt_3d_coronal(p_inner, p_outer, face_normal, side='L'):
        """
        Вычисляет наклон глазной щели строго в корональной плоскости лица.
        Устойчиво к yaw-вращениям до 70 градусов.
        """
        import math
        # Вектор от внутреннего угла к внешнему
        eye_vector = p_outer - p_inner

        # Для правого глаза инвертируем направление вектора,
        # чтобы угол измерялся в одной системе координат с левым
        if side == 'R':
            eye_vector = -eye_vector
        
        # Проецируем вектор на плоскость лица (удаляем Z-компоненту относительно нормали лица)
        eye_vector_proj = eye_vector - np.dot(eye_vector, face_normal) * face_normal
        
        # Нормализуем
        eye_vector_proj = eye_vector_proj / (np.linalg.norm(eye_vector_proj) + 1e-8)
        
        # Определяем горизонтальную ось лица с защитой от вырождения под экстремальным pitch (Error 23)
        up_vector = np.array([0, -1, 0]) 
        face_horizontal = np.cross(face_normal, up_vector)
        h_norm = np.linalg.norm(face_horizontal)
        if h_norm < 0.15:
            canonical_horiz = np.array([1.0, 0.0, 0.0])
            face_horizontal = canonical_horiz - np.dot(canonical_horiz, face_normal) * face_normal
            face_horizontal = face_horizontal / (np.linalg.norm(face_horizontal) + 1e-8)
        else:
            face_horizontal = face_horizontal / h_norm
        
        # Считаем угол между спроецированным вектором глаза и горизонталью лица
        # [BUGFIX-1] Вычисляем знаковый синус угла наклона с помощью скалярного произведения с нормалью лица
        sin_theta = np.dot(np.cross(face_horizontal, eye_vector_proj), face_normal)
        cos_theta = np.dot(face_horizontal, eye_vector_proj)
        
        return math.degrees(math.atan2(sin_theta, cos_theta))

    def calc_3d_perimeter(points_array):
        """Вычисляет периметр 2D выпуклой оболочки точек, спроецированных на плоскость лица."""
        if points_array.size == 0:
            return 0.0
        
        # [BUGFIX-9] Нам нужны два ортонормированных вектора в плоскости лица для 2D-проекции
        up_vector = np.array([0.0, -1.0, 0.0])
        h = np.cross(face_plane_normal, up_vector)
        h_norm = np.linalg.norm(h)
        if h_norm < 1e-6:
            # Вырождение: используем альтернативную горизонталь
            h = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), face_plane_normal) * face_plane_normal
            h = h / (np.linalg.norm(h) + 1e-8)
        else:
            h = h / h_norm
        
        # Вертикальный вектор лица
        v = np.cross(face_plane_normal, h)
        v = v / (np.linalg.norm(v) + 1e-8)
        
        # Проецируем 3D-точки контура в 2D на плоскость лица
        points_2d = np.column_stack((np.dot(points_array, h), np.dot(points_array, v)))
        
        # Convex hull perimeter (pure numpy monotone chain; no scipy)
        try:
            return float(_convex_hull_perimeter_2d(points_2d))
        except Exception:
            if len(points_2d) < 2:
                return 0.0
            # diameter * 2 fallback (same spirit as pdist max)
            dmax = 0.0
            for i in range(len(points_2d)):
                dif = points_2d - points_2d[i]
                dmax = max(dmax, float(np.max(np.linalg.norm(dif, axis=1))))
            if dmax > 0:
                return float(2.0 * dmax)
            min_coords = np.min(points_2d, axis=0)
            max_coords = np.max(points_2d, axis=0)
            diff = max_coords - min_coords
            return float(2.0 * np.linalg.norm(diff))

    def calc_point_to_line_distance(point, line_p1, line_p2):
        """Кратчайшее 3D расстояние от точки до прямой, заданной двумя точками."""
        line_vec = line_p2 - line_p1
        point_vec = point - line_p1
        cross_prod = np.cross(line_vec, point_vec)
        return np.linalg.norm(cross_prod) / (np.linalg.norm(line_vec) + 1e-8)

    # Canthal tilt: orbit X-extrema proxy (MACRO_BONE_INDICES has no canthus_* keys)
    canthus_L_inner, canthus_L_outer = canthus_points_from_orbit(orbit_L_pts, "L")
    canthus_R_inner, canthus_R_outer = canthus_points_from_orbit(orbit_R_pts, "R")

    # 1.3 Face Plane Overhaul: Собираем только стабильные лицевые вершины для расчета строгой нормали
    face_mask_indices = np.concatenate([
        _idx('nose_bridge_tip'), _idx('orbit_L'), _idx('orbit_R'),
        _idx('cheekbone_L'), _idx('cheekbone_R'), _idx('forehead')
    ])
    symmetrize_plane = abs(effective_yaw) <= POSE_YAW_BILATERAL_OFF_DEG
    if face_mask_indices.size > 0:
        face_vertices_only = vertices[face_mask_indices]
        _, face_plane_normal = fit_best_plane(face_vertices_only, symmetrize=symmetrize_plane)
    else:
        _, face_plane_normal = fit_best_plane(vertices, symmetrize=symmetrize_plane)

    # Гарантируем валидность нормали и направляем "наружу" (в сторону камеры Z+ в 3DDFA)
    norm_face_normal = float(np.linalg.norm(face_plane_normal))
    if norm_face_normal < 1e-6 or not np.isfinite(norm_face_normal):
        face_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        face_plane_normal = face_plane_normal / norm_face_normal
    if face_plane_normal[2] < 0:
        face_plane_normal = -face_plane_normal

    metrics["canthal_tilt_3d_L"] = calc_tilt_3d_coronal(canthus_L_inner, canthus_L_outer, face_plane_normal, side='L')
    metrics["canthal_tilt_3d_R"] = calc_tilt_3d_coronal(canthus_R_inner, canthus_R_outer, face_plane_normal, side='R')
    metrics["canthal_tilt_L"] = metrics["canthal_tilt_3d_L"]
    metrics["canthal_tilt_R"] = metrics["canthal_tilt_3d_R"]
    metrics["interorbital_ratio"] = compute_interorbital_ratio(canthus_L_inner, canthus_R_inner, zygomatic_breadth)
    
    def depth_along_normal(point: np.ndarray, reference: np.ndarray, normal: np.ndarray) -> float:
        """
        Сохраняем знак: положительное значение — точка выступает ВПЕРЕД от плоскости (normal).
        Отрицательное — точка утоплена ВНУТРЬ (например, орбиты).
        """
        return float(np.dot(point - reference, normal))

    mid_cheek_pt = (cheek_L + cheek_R) / 2.0

    if orbit_L_pts.size > 0:
        orbit_L_centroid = np.mean(orbit_L_pts, axis=0)
        metrics["orbit_depth_L_ratio"] = depth_along_normal(
            orbit_L_centroid, mid_cheek_pt, face_plane_normal
        ) / zygomatic_breadth
    if orbit_R_pts.size > 0:
        orbit_R_centroid = np.mean(orbit_R_pts, axis=0)
        metrics["orbit_depth_R_ratio"] = depth_along_normal(
            orbit_R_centroid, mid_cheek_pt, face_plane_normal
        ) / zygomatic_breadth

    _od_l = metrics.get("orbit_depth_L_ratio")
    _od_r = metrics.get("orbit_depth_R_ratio")
    if _od_l is not None and _od_r is not None:
        metrics["orbit_depth_asymmetry_ratio"] = abs(float(_od_l) - float(_od_r))
    else:
        metrics["orbit_depth_asymmetry_ratio"] = None
    
    # 5. Jaw/Gonial Angle — use jaw_angle_L/R (point landmarks)
    # [BUGFIX-11] Исключаем вершины подбородка, чтобы гонион не путался с подбородком
    chin_idx_set = set(_idx('chin').tolist()) - set(_idx('jaw_angle_L').tolist()) - set(_idx('jaw_angle_R').tolist())
    jaw_L_idx = np.array([idx for idx in _idx('jaw_angle_L') if idx not in chin_idx_set], dtype=np.int64)
    jaw_R_idx = np.array([idx for idx in _idx('jaw_angle_R') if idx not in chin_idx_set], dtype=np.int64)
    jaw_L_pts = vertices[jaw_L_idx] if jaw_L_idx.size > 0 else np.zeros((0, 3))
    jaw_R_pts = vertices[jaw_R_idx] if jaw_R_idx.size > 0 else np.zeros((0, 3))
    
    ramus_L = get_zone_centroid('jaw_L')
    ramus_R = get_zone_centroid('jaw_R')
    
    chin_y_cap = float(chin_bottom[1]) if chin_bottom is not None and np.isfinite(chin_bottom[1]) else None
    gonion_L = _gonion_from_jaw_pts(jaw_L_pts, exclude_y_above=chin_y_cap)
    gonion_R = _gonion_from_jaw_pts(jaw_R_pts, exclude_y_above=chin_y_cap)
    
    if gonion_L is not None and gonion_R is not None:
        if _valid_point(ramus_L):
            ang_L = calc_3d_angle(ramus_L, gonion_L, chin_bottom)
            clipped_ang_L = float(np.clip(ang_L, 110.0, 145.0)) if ang_L is not None else None
            metrics["gonial_angle_L"] = clipped_ang_L
            if ang_L is not None and clipped_ang_L != ang_L:
                metrics.setdefault("provenance", {})["gonial_angle_L_clipped"] = {
                    "raw_value": ang_L,
                    "clipped_value": clipped_ang_L,
                    "corrections": ["angle_clipping_110_145"],
                }
        else:
            metrics["gonial_angle_L"] = None
            reliability = min(reliability, 0.82)
        if _valid_point(ramus_R):
            ang_R = calc_3d_angle(ramus_R, gonion_R, chin_bottom)
            clipped_ang_R = float(np.clip(ang_R, 110.0, 145.0)) if ang_R is not None else None
            metrics["gonial_angle_R"] = clipped_ang_R
            if ang_R is not None and clipped_ang_R != ang_R:
                metrics.setdefault("provenance", {})["gonial_angle_R_clipped"] = {
                    "raw_value": ang_R,
                    "clipped_value": clipped_ang_R,
                    "corrections": ["angle_clipping_110_145"],
                }
        else:
            metrics["gonial_angle_R"] = None
            reliability = min(reliability, 0.82)
    elif gonion_R is not None:
        ang_R = calc_3d_angle(ramus_R, gonion_R, chin_bottom) if _valid_point(ramus_R) else None
        clipped_ang_R = float(np.clip(ang_R, 110.0, 145.0)) if ang_R is not None else None
        metrics["gonial_angle_R"] = clipped_ang_R
        metrics["gonial_angle_L"] = None
        reliability = min(reliability, 0.75)
        if ang_R is not None and clipped_ang_R != ang_R:
            metrics.setdefault("provenance", {})["gonial_angle_R_clipped"] = {
                "raw_value": ang_R,
                "clipped_value": clipped_ang_R,
                "corrections": ["angle_clipping_110_145"],
            }
    elif gonion_L is not None:
        ang_L = calc_3d_angle(ramus_L, gonion_L, chin_bottom) if _valid_point(ramus_L) else None
        clipped_ang_L = float(np.clip(ang_L, 110.0, 145.0)) if ang_L is not None else None
        metrics["gonial_angle_L"] = clipped_ang_L
        metrics["gonial_angle_R"] = None
        reliability = min(reliability, 0.75)
        if ang_L is not None and clipped_ang_L != ang_L:
            metrics.setdefault("provenance", {})["gonial_angle_L_clipped"] = {
                "raw_value": ang_L,
                "clipped_value": clipped_ang_L,
                "corrections": ["angle_clipping_110_145"],
            }
    else:
        # No jaw angle landmarks available
        metrics["gonial_angle_L"] = None
        metrics["gonial_angle_R"] = None
        reliability = min(reliability, 0.70)

    # 5b. Mandibular ramus length — gonion → mental, видимая сторона
    _gonion_for_ramus = _visible_gonion(
        gonion_L, gonion_R, pose_yaw_deg=pose_yaw_deg, bfm_yaw=float(angles[1])
    )

    if _gonion_for_ramus is not None:
        metrics["mandibular_ramus_length"] = (
            float(np.linalg.norm(_gonion_for_ramus - chin_bottom)) / face_height
        )
    else:
        metrics["mandibular_ramus_length"] = None

    nose_bridge = get_zone_centroid('nose_bridge_tip')
    nose_wing_L = get_zone_centroid('nose_wing_L')
    nose_wing_R = get_zone_centroid('nose_wing_R')
    
    if not np.allclose(nose_wing_L, 0) and not np.allclose(nose_wing_R, 0):
        metrics["nose_width_ratio"] = float(np.linalg.norm(nose_wing_L - nose_wing_R)) / zygomatic_breadth
    else:
        metrics["nose_width_ratio"] = None
    metrics["nose_projection_ratio"] = float(
        np.clip(
            depth_along_normal(nose_bridge, mid_cheek_pt, face_plane_normal) / zygomatic_breadth,
            0.0,
            0.35,
        )
    )
    
    forehead_centroid = get_zone_centroid('forehead')
    metrics["nasal_frontal_index"] = depth_along_normal(forehead_centroid, nose_bridge, face_plane_normal) / face_height
    
    # 7. Chin — зона chin, fallback на jaw_angle только если chin пуст
    chin_idx_direct = _idx('chin')
    if chin_idx_direct.size > 0:
        chin_pts = vertices[chin_idx_direct]
    else:
        chin_pts = vertices[np.concatenate([_idx('jaw_angle_L'), _idx('jaw_angle_R')])]
    chin_centroid = np.mean(chin_pts, axis=0) if chin_pts.size > 0 else chin_bottom
    metrics["chin_projection_ratio"] = depth_along_normal(chin_centroid, mid_cheek_pt, face_plane_normal) / zygomatic_breadth
    
    # 8. Orbit Centroid Ratio (to prevent overwriting step 4 interorbital_ratio)
    orbit_L_c = get_zone_centroid('orbit_L')
    orbit_R_c = get_zone_centroid('orbit_R')
    metrics["orbit_centroid_ratio"] = float(np.linalg.norm(orbit_L_c - orbit_R_c)) / zygomatic_breadth

    # 9. Forehead slope index (forehead tilt relative to brow ridge) & Glabella-Nasion angle
    forehead_c = get_zone_centroid('forehead')
    brow_L = get_zone_centroid('brow_ridge_L')
    brow_R = get_zone_centroid('brow_ridge_R')
    brow_c = (brow_L + brow_R) / 2.0
    
    glabella_pt = brow_c
    nasion_pt = get_zone_centroid('nose_bridge_tip')
    forehead_vec = glabella_pt - nasion_pt
    forehead_vec = forehead_vec / (np.linalg.norm(forehead_vec) + 1e-8)
    
    import math
    metrics["glabella_nasion_projection_angle"] = math.degrees(math.acos(
        np.clip(np.dot(forehead_vec, face_plane_normal), -1.0, 1.0)
    ))
    metrics["forehead_slope_index"] = float(metrics["glabella_nasion_projection_angle"] / 90.0)

    # 10. Nasofacial angle ratio (nose protrusion vs face height)
    nose_bridge_c = get_zone_centroid('nose_bridge_tip')
    metrics["nasofacial_angle_ratio"] = depth_along_normal(nose_bridge_c, mid_cheek_pt, face_plane_normal) / face_height

    # 11. Orbital asymmetry index & 3D Perimeter symmetry (3D perimeter ratio based)
    perimeter_L = calc_3d_perimeter(orbit_L_pts)
    perimeter_R = calc_3d_perimeter(orbit_R_pts)
    metrics["orbital_perimeter_symmetry"] = min(perimeter_L, perimeter_R) / (max(perimeter_L, perimeter_R) + 1e-8)
    metrics["orbital_asymmetry_index"] = float(1.0 - metrics["orbital_perimeter_symmetry"])

    # 11b. Orbital vertical asymmetry (face-local axis, pose-stable)
    orbit_L_c = get_zone_centroid('orbit_L')
    orbit_R_c = get_zone_centroid('orbit_R')
    if not np.allclose(orbit_L_c, 0) and not np.allclose(orbit_R_c, 0):
        face_vertical = _face_vertical_axis(face_plane_normal, chin_bottom, forehead_top)
        h_L = float(np.dot(orbit_L_c - chin_bottom, face_vertical))
        h_R = float(np.dot(orbit_R_c - chin_bottom, face_vertical))
        _signed_yaw = float(pose_yaw_deg) if pose_yaw_deg is not None else float(angles[1])
        if abs(_signed_yaw) <= 20.0:
            metrics["orbit_vertical_asymmetry_ratio"] = abs(h_L - h_R) / face_height
            metrics["orbit_vertical_signed_ratio"] = (h_L - h_R) / face_height
        else:
            metrics["orbit_vertical_asymmetry_ratio"] = metrics.get("orbit_depth_asymmetry_ratio")
            metrics["orbit_vertical_signed_ratio"] = None
        metrics["orbital_height_signed"] = metrics.get("orbit_vertical_signed_ratio")
    else:
        metrics["orbit_vertical_asymmetry_ratio"] = metrics.get("orbit_depth_asymmetry_ratio")
        metrics["orbit_vertical_signed_ratio"] = None
        metrics["orbital_height_signed"] = None
        reliability = min(reliability, 0.85)

    # 12. Gnathion midline deviation
    nasion_pt = get_zone_centroid('nose_bridge_tip')
    subnasale_pt = _resolve_subnasale_point(vertices, get_zone_centroid, nasion_pt)
    gnathion_pt = get_zone_centroid('chin')
    if np.allclose(gnathion_pt, 0):
        gnathion_pt = chin_bottom
        
    metrics["gnathion_midline_deviation_ratio"] = calc_point_to_line_distance(
        gnathion_pt, nasion_pt, subnasale_pt
    ) / zygomatic_breadth
    metrics["chin_offset_asymmetry"] = metrics["gnathion_midline_deviation_ratio"]

    # 12b. Extended zone-based bone metrics (multi-vertex regions, 3D depth/angles)
    face_horizontal = _face_horizontal_axis(face_plane_normal)
    nose_bridge_pts = _zone_vertex_array(vertices, bone_indices, "nose_bridge_tip")
    subnasale_pt = _resolve_subnasale_point(vertices, get_zone_centroid, nasion_pt)

    if not np.allclose(nasion_pt, 0) and not np.allclose(subnasale_pt, 0):
        metrics["nasal_length_ratio"] = float(np.linalg.norm(nasion_pt - subnasale_pt)) / face_height

    glabella_pt = (get_zone_centroid("brow_ridge_L") + get_zone_centroid("brow_ridge_R")) / 2.0
    if nose_bridge_pts.shape[0] > 0 and not np.allclose(glabella_pt, 0):
        bridge_tip = nose_bridge_pts[np.argmax(np.dot(nose_bridge_pts - mid_cheek_pt, face_plane_normal))]
        metrics["nose_bridge_length_ratio"] = float(np.linalg.norm(bridge_tip - glabella_pt)) / face_height

    if not np.allclose(subnasale_pt, 0):
        metrics["subnasale_projection_ratio"] = depth_along_normal(
            subnasale_pt, mid_cheek_pt, face_plane_normal
        ) / zygomatic_breadth

    nasion_depth = _zone_mean_depth(nose_bridge_pts, mid_cheek_pt, face_plane_normal)
    if nasion_depth is not None:
        metrics["nasion_zone_depth_ratio"] = nasion_depth / zygomatic_breadth

    cheek_L_pts = _zone_vertex_array(vertices, bone_indices, "cheekbone_L")
    cheek_R_pts = _zone_vertex_array(vertices, bone_indices, "cheekbone_R")
    depth_L = _zone_mean_depth(cheek_L_pts, mid_cheek_pt, face_plane_normal)
    depth_R = _zone_mean_depth(cheek_R_pts, mid_cheek_pt, face_plane_normal)
    if depth_L is not None and depth_R is not None:
        metrics["bizygomatic_depth_asymmetry"] = abs(depth_L - depth_R) / max(abs(depth_L), abs(depth_R), 1e-6)

    if (
        jaw_L_pts.size > 0
        and jaw_R_pts.size > 0
        and gonion_L is not None
        and gonion_R is not None
    ):
        lat_L = _signed_lateral_offset(gonion_L, nasion_pt, subnasale_pt, face_plane_normal)
        lat_R = _signed_lateral_offset(gonion_R, nasion_pt, subnasale_pt, face_plane_normal)
        metrics["gonial_width_asymmetry"] = abs(abs(lat_L) - abs(lat_R)) / max(abs(lat_L), abs(lat_R), 1e-6)
        lateral_L = _zone_span_along(jaw_L_pts, face_horizontal)
        lateral_R = _zone_span_along(jaw_R_pts, face_horizontal)
        if lateral_L > 0 and lateral_R > 0:
            metrics["bigonial_width_ratio"] = float(np.linalg.norm(gonion_L - gonion_R)) / zygomatic_breadth

    if not np.allclose(canthus_L_inner, 0) and not np.allclose(canthus_R_inner, 0):
        metrics["intercanthal_width_ratio"] = compute_interorbital_ratio(
            canthus_L_inner, canthus_R_inner, zygomatic_breadth
        )

    for side, orbit_pts, inner, outer in (
        ("L", orbit_L_pts, canthus_L_inner, canthus_L_outer),
        ("R", orbit_R_pts, canthus_R_inner, canthus_R_outer),
    ):
        if orbit_pts.size > 0:
            span = _zone_span_along(orbit_pts, face_horizontal)
            if span > 0:
                metrics[f"orbit_width_{side}_ratio"] = span / zygomatic_breadth
            spread = _zone_depth_spread(orbit_pts, mid_cheek_pt, face_plane_normal)
            if spread is not None:
                metrics[f"orbit_fossa_spread_{side}"] = spread / zygomatic_breadth

        brow_pts = _zone_vertex_array(vertices, bone_indices, f"brow_ridge_{side}")
        brow_depth = _zone_mean_depth(brow_pts, mid_cheek_pt, face_plane_normal)
        if brow_depth is not None:
            metrics[f"brow_ridge_projection_{side}_ratio"] = brow_depth / zygomatic_breadth

        temporal_pts = temporal_fossa_points_from_orbit(orbit_pts, side)
        if temporal_pts.shape[0] == 0:
            temporal_pts = _zone_vertex_array(vertices, bone_indices, f"temporal_{side}")
        temp_depth = _zone_mean_depth(temporal_pts, mid_cheek_pt, face_plane_normal)
        if temp_depth is not None:
            metrics[f"temporal_depth_{side}_ratio"] = temp_depth / zygomatic_breadth

        cheek_pts = _zone_vertex_array(vertices, bone_indices, f"cheekbone_{side}")
        if cheek_pts.shape[0] > 0:
            y_span = float(np.max(cheek_pts[:, 1]) - np.min(cheek_pts[:, 1]))
            metrics[f"zygomatic_arch_height_{side}_ratio"] = y_span / face_height

        if not np.allclose(inner, 0) and not np.allclose(outer, 0):
            interorbital = float(np.linalg.norm(canthus_L_inner - canthus_R_inner))
            denom = interorbital if interorbital > 1e-6 else zygomatic_breadth
            metrics[f"palpebral_fissure_length_{side}_ratio"] = (
                float(np.linalg.norm(outer - inner)) / denom
            )

        lig_pts = _zone_vertex_array(vertices, bone_indices, f"ligament_orbital_{side}")
        lig_depth = _zone_mean_depth(lig_pts, mid_cheek_pt, face_plane_normal)
        if lig_depth is not None:
            metrics[f"ligament_orbital_{side}_depth_ratio"] = lig_depth / zygomatic_breadth

        jaw_side_pts = jaw_L_pts if side == "L" else jaw_R_pts
        gonion_side = gonion_L if side == "L" else gonion_R
        if jaw_side_pts.size > 0 and gonion_side is None:
            gonion_side = _gonion_from_jaw_pts(jaw_side_pts, exclude_y_above=chin_y_cap)
        if gonion_side is not None:
            metrics[f"mandibular_body_length_{side}_ratio"] = (
                float(np.linalg.norm(gonion_side - chin_bottom)) / face_height
            )

        ramus_pts = _zone_vertex_array(vertices, bone_indices, f"jaw_{side}")
        face_vertical_val = _face_vertical_axis(face_plane_normal, chin_bottom, forehead_top)
        ramus_h = ramus_vertical_height_ratio(ramus_pts, gonion_side, face_height, face_vertical=face_vertical_val)
        if ramus_h is not None:
            # Compensate for camera pitch compression (Error 33)
            pitch_deg = float(angles[0]) if len(angles) > 0 else 0.0
            pitch_corr = 1.0
            if pitch_deg < 0.0:
                pitch_corr += 0.18 * (abs(pitch_deg) / 45.0) ** 2
            elif pitch_deg > 0.0:
                pitch_corr += 0.10 * (pitch_deg / 45.0) ** 2
            corrected = float(ramus_h * pitch_corr)
            if corrected < 0.05:
                metrics[f"ramus_height_{side}_ratio"] = None
            else:
                metrics[f"ramus_height_{side}_ratio"] = float(min(corrected, 0.49))

    _br_l = metrics.get("brow_ridge_projection_L_ratio")
    _br_r = metrics.get("brow_ridge_projection_R_ratio")
    if _br_l is not None and _br_r is not None:
        span = max(abs(float(_br_l)), abs(float(_br_r)), 0.05)
        metrics["brow_asymmetry_deg"] = float(np.degrees(np.arctan2(abs(float(_br_l) - float(_br_r)), span)))

    _pf_l = metrics.get("palpebral_fissure_length_L_ratio")
    _pf_r = metrics.get("palpebral_fissure_length_R_ratio")
    if _pf_l is not None and _pf_r is not None:
        _pf_lf, _pf_rf = float(_pf_l), float(_pf_r)
        metrics["palpebral_fissure_asymmetry_ratio"] = abs(_pf_lf - _pf_rf) / max(
            _pf_lf, _pf_rf, 1e-6
        )
    else:
        metrics["palpebral_fissure_asymmetry_ratio"] = None

    # --- Eye Aspect Ratio (EAR) ---
    def compute_ear(eye_pts):
        if eye_pts.size == 0:
            return None
        face_vertical_axis = _face_vertical_axis(face_plane_normal, chin_bottom, forehead_top)
        face_horizontal_axis = _face_horizontal_axis(face_plane_normal)
        height = np.max(np.dot(eye_pts, face_vertical_axis)) - np.min(np.dot(eye_pts, face_vertical_axis))
        width = np.max(np.dot(eye_pts, face_horizontal_axis)) - np.min(np.dot(eye_pts, face_horizontal_axis))
        if width < 1e-6:
            return None
        return float(height / width)

    eye_L_idx = _idx('left_eye')
    eye_R_idx = _idx('right_eye')
    if eye_L_idx.size > 0:
        metrics["eye_aspect_ratio_L"] = compute_ear(vertices[eye_L_idx])
    if eye_R_idx.size > 0:
        metrics["eye_aspect_ratio_R"] = compute_ear(vertices[eye_R_idx])

    # --- Lip Thickness Asymmetry ---
    upper_lip_idx = _idx('upper_lip')
    lower_lip_idx = _idx('lower_lip')
    if upper_lip_idx.size > 0 and lower_lip_idx.size > 0:
        upper_lip_pts = vertices[upper_lip_idx]
        lower_lip_pts = vertices[lower_lip_idx]
        face_vertical_axis = _face_vertical_axis(face_plane_normal, chin_bottom, forehead_top)
        
        upper_thickness = np.max(np.dot(upper_lip_pts, face_vertical_axis)) - np.min(np.dot(upper_lip_pts, face_vertical_axis))
        lower_thickness = np.max(np.dot(lower_lip_pts, face_vertical_axis)) - np.min(np.dot(lower_lip_pts, face_vertical_axis))
        
        metrics["lip_thickness_ratio"] = float(upper_thickness) / max(float(lower_thickness), 1e-6)
        metrics["upper_lip_height_ratio"] = float(upper_thickness) / face_height
        metrics["lower_lip_height_ratio"] = float(lower_thickness) / face_height

    # 13. Reliability — no yaw/pitch penalty; curated dataset uses pose-dependent metrics only.
    yaw_abs = abs(angles[1])
    pitch_abs = abs(angles[0])

    # Pitch guard: при наклоне головы > 20° подбородок геометрически недостоверен,
    # но только для околофронтальных ракурсов (abs(yaw) <= 30.0).
    # Для выраженных профилей подбородок и профиль носа видны идеально.
    if pitch_abs > 20.0 and yaw_abs <= 30.0:
        metrics["chin_projection_ratio"] = None
        metrics["gnathion_midline_deviation_ratio"] = None
        metrics["chin_offset_asymmetry"] = None

    # Mask occluded side: pose_yaw (HPE) если есть, иначе BFM angles[1]; порог 18° для лёгкого 3/4
    mask_yaw = float(pose_yaw_deg) if pose_yaw_deg is not None else float(angles[1])

    if abs(mask_yaw) > POSE_YAW_OCCLUDE_DEG:
        if mask_yaw < 0:  # левая щека к камере — правая орбита скрыта
            _null_side_metrics(metrics, "R")
        else:
            _null_side_metrics(metrics, "L")

    if abs(mask_yaw) > POSE_YAW_BILATERAL_OFF_DEG:
        for key in (
            "bizygomatic_depth_asymmetry",
            "bigonial_width_ratio",
            "intercanthal_width_ratio",
            "gonial_width_asymmetry",
            "interorbital_ratio",
            "chin_offset_asymmetry",
            "gnathion_midline_deviation_ratio",
        ):
            metrics[key] = None

    _finalize_bilateral_geometry(metrics)

    midface_parts: list[float] = []
    orbit_depth_vals: list[float] = []
    for key in ("orbit_depth_L_ratio", "orbit_depth_R_ratio"):
        val = metrics.get(key)
        if val is not None and np.isfinite(float(val)):
            orbit_depth_vals.append(float(val))
    if orbit_depth_vals:
        midface_parts.append(float(np.mean(orbit_depth_vals)))
    nose_val = metrics.get("nose_projection_ratio")
    if nose_val is not None and np.isfinite(float(nose_val)):
        midface_parts.append(float(nose_val))
    if midface_parts:
        metrics["midface_depth_index"] = float(np.mean(midface_parts))

    if metrics.get("bizygomatic_depth_asymmetry") is not None:
        metrics["skull_depth_asymmetry_index"] = metrics["bizygomatic_depth_asymmetry"]
    if metrics.get("midface_depth_index") is not None and face_height > 1e-6:
        metrics["midface_compactness"] = float(metrics["midface_depth_index"]) / (
            float(face_height) / float(zygomatic_breadth) + 1e-6
        )
    orbit_w = metrics.get("orbit_width_L_ratio") or metrics.get("orbit_width_R_ratio")
    cfi = metrics.get("cranial_face_index")
    if orbit_w is not None and cfi is not None and float(cfi) > 1e-6:
        metrics["orbit_skull_ratio"] = float(orbit_w) / float(cfi)
    if exp_params is not None and np.asarray(exp_params).size >= 3:
        ep = np.asarray(exp_params, dtype=float)
        jaw_intensity = abs(float(ep[0]))
        smile_intensity = max(abs(float(ep[1])), abs(float(ep[2])))
        metrics["expression_severity"] = float(
            max(
                jaw_intensity / THRESHOLD_JAW_OPEN if THRESHOLD_JAW_OPEN > 0 else 0.0,
                smile_intensity / THRESHOLD_SMILE if THRESHOLD_SMILE > 0 else 0.0,
                0.0,
            )
        )
    else:
        metrics["expression_severity"] = 0.0
    
    return metrics, reliability


def apply_expression_exclusion_to_metrics(metrics: dict, expression_params: np.ndarray) -> dict:
    """
    Удаляет зоны, искаженные мимикой.
    Исключённые метрики получают None — так их видит весь pipeline ниже.
    """
    if expression_params is None or expression_params.size < 3:
        return metrics.copy()
    
    
    # [BUGFIX] Правильные индексы: 0=jaw_open, 1-2=smile (согласно zones.py)
    jaw_open = abs(expression_params[0])
    smile_intensity = max(abs(expression_params[1]), abs(expression_params[2]))

    cleaned_metrics = metrics.copy()
    distorted_keys: list[str] = []

    # Улыбка: nose/cheek soft-tissue metrics; jaw width стабильнее при smile.
    if smile_intensity > THRESHOLD_SMILE:
        distorted_keys.extend([
            'nose_width_ratio',
            'chin_projection_ratio',
            'nasofacial_angle_ratio',
            'mandibular_ramus_length',
        ])
    if jaw_open > THRESHOLD_JAW_OPEN:
        distorted_keys.extend([
            'jaw_width_ratio',
            'chin_projection_ratio',
            'nasofacial_angle_ratio',
            'mandibular_ramus_length',
            'gonial_angle_L',
            'gonial_angle_R',
            'bigonial_width_ratio',
        ])

    for key in set(distorted_keys):
        if key in cleaned_metrics:
            cleaned_metrics[key] = None

    return cleaned_metrics


def _metric_coverage_weight(metric_key: str) -> float:
    """Resolve ZONE_WEIGHTS for metric keys with _ratio/_index suffixes."""
    ZONE_WEIGHTS = {}  # optional; coverage falls back to 1.0

    key = str(metric_key)
    if key in ZONE_WEIGHTS:
        return float(ZONE_WEIGHTS[key])

    if "palpebral_fissure" in key:
        if "_L" in key and "canthal_tilt_L" in ZONE_WEIGHTS:
            return float(ZONE_WEIGHTS["canthal_tilt_L"])
        if "_R" in key and "canthal_tilt_R" in ZONE_WEIGHTS:
            return float(ZONE_WEIGHTS["canthal_tilt_R"])

    candidates: list[str] = [key]
    for suffix in ("_ratio", "_index", "_deg", "_log", "_norm"):
        if key.endswith(suffix):
            candidates.append(key[: -len(suffix)])
    for side in ("_L", "_R"):
        if side in key:
            candidates.append(key.split(side)[0] + side)
            if key.endswith("_ratio"):
                candidates.append(key[: -len("_ratio")])

    for cand in candidates:
        if cand in ZONE_WEIGHTS:
            return float(ZONE_WEIGHTS[cand])
        if (cand.endswith("_L") or cand.endswith("_R")) and cand[:-2] in ZONE_WEIGHTS:
            return float(ZONE_WEIGHTS[cand[:-2]])

    if key.startswith("texture_"):
        return float(ZONE_WEIGHTS.get("texture_silicone_prob", 0.25))
    return 0.15


def calculate_coverage(cleaned_metrics: dict, pose_bucket: str) -> float:
    """Weighted coverage over expected keys for a pose bucket.

    Without project BUCKET_METRIC_KEYS, use all non-null numeric metrics present.
    """
    # Optional override map: bucket -> list of metric keys
    BUCKET_METRIC_KEYS: dict[str, list[str]] = {}

    expected_keys = list(BUCKET_METRIC_KEYS.get(pose_bucket, []))
    if not expected_keys:
        # fallback: all finite numeric keys currently present
        expected_keys = [
            k for k, v in cleaned_metrics.items()
            if isinstance(v, (int, float)) and v is not None and np.isfinite(float(v))
        ]
    if not expected_keys:
        return 0.0

    total_weight = 0.0
    covered_weight = 0.0
    for key in expected_keys:
        w = _metric_coverage_weight(key)
        total_weight += w
        val = cleaned_metrics.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            covered_weight += w
    if total_weight <= 0:
        return 0.0
    return covered_weight / total_weight
