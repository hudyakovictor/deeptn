"""ITER3 BFM zone metrics for 3DDFA-V3 (library-local).

Uses MACRO_BONE_INDICES / ZONE_CONFIG from zone_indices_data.
No project photo exclusions or pipeline ReconstructionResult dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from util.geom_utils import bounded_score_from_error, face_scale_from_points, weighted_mean_abs
from util.zone_indices_data import MACRO_BONE_INDICES, ZONE_CONFIG

__all__ = [
    "ZoneMetric",
    "MACRO_BONE_INDICES",
    "ZONE_CONFIG",
    "MACRO_ZONE_ORDER",
    "BASE_ZONE_NAMES",
    "DERIVED_ZONE_NAMES",
    "zone_analysis_role",
    "zone_bone_priority_class",
    "zone_bone_weight",
    "static_zone_schema",
    "apply_expression_exclusion_mask",
    "compute_zone_metrics",
    "summarize_bone_priority_metrics",
    "zone_vertex_mask",
    "canthus_points_from_orbit",
    "indices_hash",
]

MACRO_ZONE_ORDER = (
    "orbit_L", "orbit_R", "cheekbone_L", "cheekbone_R",
    "nose_bridge_tip", "nose_wing_L", "nose_wing_R",
    "brow_ridge_L", "brow_ridge_R", "chin",
    "jaw_angle_L", "jaw_angle_R", "jaw_L", "jaw_R", "forehead",
    "ligament_zygomatic_L", "ligament_zygomatic_R",
    "ligament_orbital_L", "ligament_orbital_R",
    "temporal_L", "temporal_R", "cheek_soft_L", "cheek_soft_R",
)

BASE_ZONE_NAMES = (
    "right_eye", "left_eye", "right_eyebrow", "left_eyebrow",
    "nose", "upper_lip", "lower_lip", "skin",
)
DERIVED_ZONE_NAMES = (
    "forehead", "brow_ridge_L", "brow_ridge_R", "orbit_L", "orbit_R",
    "nose_bridge_tip", "nose_wing_L", "nose_wing_R",
    "cheekbone_L", "cheekbone_R", "chin", "jaw_L", "jaw_R",
)

DEFAULT_EXPRESSION_EXCLUDE = frozenset({"upper_lip", "lower_lip"})
DEFAULT_SOFT_EXCLUDE = frozenset({"nose_wing_L", "nose_wing_R"})


@dataclass
class ZoneMetric:
    name: str
    status: str
    analysis_role: str
    bone_priority_class: str
    bone_weight: float
    raw_error: Optional[float]
    bounded_score: Optional[float]
    shared_vertex_count: int
    mean_shift: Optional[np.ndarray] = None
    principal_shift_axis: Optional[str] = None
    dominant_shift_direction: Optional[str] = None
    view_name: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.mean_shift is not None:
            d["mean_shift"] = np.asarray(self.mean_shift).tolist()
        return d


def zone_analysis_role(zone_name: str) -> str:
    return ZONE_CONFIG.get(zone_name, ("unknown", "unknown", 0.0))[0]


def zone_bone_priority_class(zone_name: str) -> str:
    return ZONE_CONFIG.get(zone_name, ("unknown", "unknown", 0.0))[1]


def zone_bone_weight(zone_name: str) -> float:
    return float(ZONE_CONFIG.get(zone_name, ("unknown", "unknown", 0.0))[2])


def static_zone_schema() -> dict[str, object]:
    zones = {}
    for z in list(BASE_ZONE_NAMES) + list(DERIVED_ZONE_NAMES):
        zones[z] = {
            "analysis_role": zone_analysis_role(z),
            "bone_priority_class": zone_bone_priority_class(z),
            "bone_weight": zone_bone_weight(z),
            "vertex_count": len(MACRO_BONE_INDICES.get(z, ())),
        }
    return {
        "base_zone_count": len(BASE_ZONE_NAMES),
        "derived_zone_count": len(DERIVED_ZONE_NAMES),
        "total_zone_count": len(zones),
        "zones": zones,
    }


def indices_hash() -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    h.update(b"zone_indices_v1")
    for name in sorted(MACRO_BONE_INDICES.keys()):
        h.update(name.encode())
        ids = sorted(int(i) for i in MACRO_BONE_INDICES[name])
        h.update(np.asarray(ids, dtype=np.int64).tobytes())
    return h.hexdigest()


def zone_vertex_mask(zone_name: str, n_vertices: int = 35709) -> np.ndarray:
    mask = np.zeros(n_vertices, dtype=bool)
    ids = MACRO_BONE_INDICES.get(zone_name)
    if not ids:
        return mask
    idx = np.fromiter((i for i in ids if 0 <= int(i) < n_vertices), dtype=np.int64)
    if idx.size:
        mask[idx] = True
    return mask


def apply_expression_exclusion_mask(
    vertex_mask: np.ndarray,
    *,
    exclude_zones: Optional[Iterable[str]] = None,
    n_vertices: Optional[int] = None,
) -> np.ndarray:
    m = np.asarray(vertex_mask, dtype=bool).copy()
    n = int(n_vertices or m.shape[0])
    zones = DEFAULT_EXPRESSION_EXCLUDE if exclude_zones is None else frozenset(exclude_zones)
    for z in zones:
        ids = MACRO_BONE_INDICES.get(z)
        if not ids:
            continue
        for vid in ids:
            if 0 <= int(vid) < n:
                m[int(vid)] = False
    return m


def canthus_points_from_orbit(orbit_pts: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(orbit_pts, dtype=np.float64)
    if pts.size == 0:
        z = np.zeros(3)
        return z, z
    order = np.argsort(pts[:, 0])
    left_pt, right_pt = pts[order[0]], pts[order[-1]]
    if side.lower().startswith("l"):
        return right_pt, left_pt
    return left_pt, right_pt


def _principal_axis_label(vec: np.ndarray) -> str:
    axes = ("x", "y", "z")
    i = int(np.argmax(np.abs(vec)))
    return axes[i]


def _direction_label(vec: np.ndarray) -> str:
    axis = _principal_axis_label(vec)
    i = {"x": 0, "y": 1, "z": 2}[axis]
    sign = "+" if vec[i] >= 0 else "-"
    return f"{sign}{axis}"


def _build_zone_metric(
    name: str,
    zone_ids: Sequence[int],
    shared_lookup: Dict[int, int],
    aligned_points_a: np.ndarray,
    points_b: np.ndarray,
    shared_weights: Optional[np.ndarray],
    min_zone_vertices: int,
    view_name: Optional[str],
    face_width: float,
    used_vertex_ids: set,
    exclusive: bool = True,
) -> ZoneMetric:
    role = zone_analysis_role(name)
    prio = zone_bone_priority_class(name)
    bw = zone_bone_weight(name)

    ids = []
    for vid in zone_ids:
        iv = int(vid)
        if exclusive and iv in used_vertex_ids:
            continue
        if iv in shared_lookup:
            ids.append(iv)
    if exclusive:
        used_vertex_ids.update(ids)

    positions = [shared_lookup[i] for i in ids if i in shared_lookup]
    if len(positions) < min_zone_vertices:
        return ZoneMetric(
            name=name, status="insufficient", analysis_role=role,
            bone_priority_class=prio, bone_weight=bw,
            raw_error=None, bounded_score=None,
            shared_vertex_count=len(positions), view_name=view_name,
        )

    pos = np.asarray(positions, dtype=np.int64)
    a = aligned_points_a[pos]
    b = points_b[pos]
    diffs = a - b
    errs = np.linalg.norm(diffs, axis=1)
    if shared_weights is not None:
        w = np.asarray(shared_weights, dtype=np.float64).reshape(-1)[pos]
    else:
        w = np.ones(len(pos), dtype=np.float64)
    raw = weighted_mean_abs(errs / max(face_width, 1e-6), w)
    mean_shift = np.average(diffs, axis=0, weights=w)
    return ZoneMetric(
        name=name, status="ok", analysis_role=role,
        bone_priority_class=prio, bone_weight=bw,
        raw_error=float(raw),
        bounded_score=bounded_score_from_error(float(raw)),
        shared_vertex_count=len(positions),
        mean_shift=mean_shift.astype(np.float64),
        principal_shift_axis=_principal_axis_label(mean_shift),
        dominant_shift_direction=_direction_label(mean_shift),
        view_name=view_name,
    )


def compute_zone_metrics(
    *,
    aligned_points_a: np.ndarray,
    points_b: np.ndarray,
    shared_indices: np.ndarray,
    shared_weights: Optional[np.ndarray] = None,
    min_zone_vertices: int = 3,
    view_name: Optional[str] = None,
    face_width_override: Optional[float] = None,
    exclude_zones: Optional[Iterable[str]] = None,
    annotation_groups: Optional[Sequence[np.ndarray]] = None,
    exclusive_vertices: bool = True,
    n_vertices: int = 35709,
) -> List[ZoneMetric]:
    aligned_points_a = np.asarray(aligned_points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    shared_indices = np.asarray(shared_indices, dtype=np.int64).reshape(-1)
    if aligned_points_a.shape != points_b.shape:
        raise ValueError("aligned_points_a / points_b shape mismatch")
    if aligned_points_a.shape[0] != shared_indices.shape[0]:
        raise ValueError("shared_indices length mismatch")

    if face_width_override is not None and face_width_override > 1e-5:
        face_width = float(face_width_override)
    else:
        face_width = face_scale_from_points(points_b)

    shared_lookup = {int(vid): i for i, vid in enumerate(shared_indices.tolist())}
    excluded = set(exclude_zones or ()) | set(DEFAULT_SOFT_EXCLUDE)
    used = set()
    zones: List[ZoneMetric] = []

    if annotation_groups is not None:
        for zone_name, zone_vertices in zip(BASE_ZONE_NAMES, annotation_groups):
            if zone_name in excluded:
                continue
            zids = [int(v) for v in np.asarray(zone_vertices).reshape(-1).tolist()]
            zones.append(_build_zone_metric(
                name=zone_name, zone_ids=zids, shared_lookup=shared_lookup,
                aligned_points_a=aligned_points_a, points_b=points_b,
                shared_weights=shared_weights, min_zone_vertices=min_zone_vertices,
                view_name=view_name, face_width=face_width,
                used_vertex_ids=used, exclusive=exclusive_vertices,
            ))

    ordered = [n for n in MACRO_ZONE_ORDER if n in MACRO_BONE_INDICES] + [
        n for n in MACRO_BONE_INDICES
        if n not in MACRO_ZONE_ORDER and n not in {"right_eyebrow", "left_eyebrow", "full_mesh"}
    ]
    for macro_name in ordered:
        if macro_name in excluded:
            continue
        ids = list(MACRO_BONE_INDICES[macro_name])
        zones.append(_build_zone_metric(
            name=macro_name, zone_ids=ids, shared_lookup=shared_lookup,
            aligned_points_a=aligned_points_a, points_b=points_b,
            shared_weights=shared_weights, min_zone_vertices=min_zone_vertices,
            view_name=view_name, face_width=face_width,
            used_vertex_ids=used, exclusive=exclusive_vertices,
        ))
    return zones


def summarize_bone_priority_metrics(
    zones: List[ZoneMetric],
    min_usable_bone_zones: int = 4,
) -> dict:
    bone = [
        z for z in zones
        if z.bone_priority_class in ("bone_priority_core", "bone_priority_supporting")
        and z.status == "ok" and z.raw_error is not None
    ]
    core = [z for z in bone if z.bone_priority_class == "bone_priority_core"]
    supporting = [z for z in bone if z.bone_priority_class == "bone_priority_supporting"]
    if not bone:
        return {
            "bone_raw_geometry_error": None,
            "bone_bounded_similarity_score": None,
            "bone_zone_count_usable": 0,
            "bone_zone_coverage_quality": "insufficient",
        }
    weights = np.array([z.bone_weight for z in bone], dtype=np.float64)
    errs = np.array([float(z.raw_error) for z in bone], dtype=np.float64)
    wsum = float(np.sum(weights))
    raw = float(np.sum(errs * weights) / wsum) if wsum > 1e-8 else float(np.mean(errs))
    usable = len(bone)
    if usable >= min_usable_bone_zones and len(core) >= max(2, min_usable_bone_zones // 2):
        quality = "good"
    elif usable > 0:
        quality = "partial"
    else:
        quality = "insufficient"
    return {
        "bone_raw_geometry_error": raw,
        "bone_bounded_similarity_score": bounded_score_from_error(raw),
        "bone_zone_count_usable": usable,
        "bone_zone_count_core_usable": len(core),
        "bone_zone_count_supporting_usable": len(supporting),
        "bone_zone_coverage_quality": quality,
        "bone_zone_weight_total": wsum,
        "top_bone_priority_zones": [
            {"name": z.name, "raw_error": z.raw_error, "bone_weight": z.bone_weight}
            for z in sorted(bone, key=lambda z: (-z.bone_weight, z.raw_error or 0))[:8]
        ],
    }
