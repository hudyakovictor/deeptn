from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import json

import numpy as np

from app.stage1.geometry import unpack_mask
from .anchor_policy import stable_anchor_indices
from .core import robust_rigid_align

MESH_COUNT = 35709
MESH_SCHEMA = "deeputin-stage2-dense-mesh-v1.1"
ZONE_INDEX_PATH = Path(__file__).with_name("mesh_zone_indices.json")

PRIORITY_ZONES = (
    "forehead", "brow_ridge_L", "brow_ridge_R", "orbit_L", "orbit_R",
    "nose_bridge_tip", "nose_wing_L", "nose_wing_R", "cheekbone_L", "cheekbone_R",
    "cheek_soft_L", "cheek_soft_R", "jaw_L", "jaw_R", "jaw_angle_L", "jaw_angle_R",
    "chin", "temporal_L", "temporal_R", "ligament_orbital_L", "ligament_orbital_R",
    "ligament_zygomatic_L", "ligament_zygomatic_R",
)

@lru_cache(maxsize=1)
def load_anatomical_zones() -> dict[str, np.ndarray]:
    if not ZONE_INDEX_PATH.is_file():
        return {}
    try:
        raw = json.loads(ZONE_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, np.ndarray] = {}
    for name, vals in raw.items():
        arr = np.asarray([int(v) for v in vals if 0 <= int(v) < MESH_COUNT], np.int64)
        if arr.size:
            out[str(name)] = np.unique(arr)
    return out


def _normalize(vertices: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, np.float32)
    center = np.nanmean(v, axis=0)
    x = v - center
    scale = float(np.sqrt(np.nanmean(np.sum(x * x, axis=1))))
    if not np.isfinite(scale) or scale < 1e-8:
        return x.astype(np.float32)
    return (x / scale).astype(np.float32)


def _load_mesh(record: Any) -> dict[str, Any]:
    directory = getattr(record, "record_dir", None)
    if directory is None:
        return {"status": "missing_record_dir"}
    p = Path(directory) / "reconstruction.npz"
    if not p.is_file():
        return {"status": "missing_reconstruction"}
    try:
        with np.load(p, allow_pickle=False) as z:
            if "vertices_identity_only" in z:
                vertices = _normalize(z["vertices_identity_only"])
                space = "identity_only_normalized_runtime"
            elif "vertices_object_normalized" in z:
                vertices = z["vertices_object_normalized"].astype(np.float32)
                space = "object_normalized"
            else:
                return {"status": "missing_vertices"}
            if "full_mesh_visible_packbits" in z:
                visible = unpack_mask(z["full_mesh_visible_packbits"], MESH_COUNT).astype(bool)
            else:
                visible = np.ones(MESH_COUNT, bool)
            normals = z["normals_object"].astype(np.float32) if "normals_object" in z else np.zeros_like(vertices)
        return {"status": "ok", "vertices": vertices, "visible": visible, "normals": normals, "space": space}
    except Exception as exc:
        return {"status": "load_error", "error": str(exc)}


def _subsample(ids: np.ndarray, max_points: int = 6000) -> np.ndarray:
    ids = np.asarray(ids, np.int64)
    if ids.size <= max_points:
        return ids
    step = int(np.ceil(ids.size / max_points))
    return ids[::step][:max_points]


def _zone_labels(vertices: np.ndarray) -> np.ndarray:
    qx = np.quantile(vertices[:, 0], [1 / 3, 2 / 3])
    qy = np.quantile(vertices[:, 1], [1 / 3, 2 / 3])
    xb = np.digitize(vertices[:, 0], qx)
    yb = np.digitize(vertices[:, 1], qy)
    names = np.array([[f"mesh_x_{x}_{y}" for x in ("low", "center", "high")] for y in ("low", "center", "high")])
    return names[yb, xb]


def _shape_descriptor(pts: np.ndarray) -> dict[str, float]:
    if len(pts) < 4:
        return {
            "mesh_shape_eig_ratio_1": 0.0,
            "mesh_shape_eig_ratio_2": 0.0,
            "mesh_shape_planarity": 0.0,
            "mesh_shape_linearity": 0.0,
            "mesh_geodesic_span_proxy": 0.0,
        }
    x = np.asarray(pts, np.float64) - np.mean(pts, axis=0)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 0.0))[::-1]
    s = float(np.sum(vals) + 1e-12)
    span = np.ptp(np.asarray(pts), axis=0)
    return {
        "mesh_shape_eig_ratio_1": float(vals[0] / s),
        "mesh_shape_eig_ratio_2": float(vals[1] / s),
        "mesh_shape_planarity": float((vals[1] - vals[2]) / max(vals[0], 1e-12)),
        "mesh_shape_linearity": float((vals[0] - vals[1]) / max(vals[0], 1e-12)),
        "mesh_geodesic_span_proxy": float(np.linalg.norm(span)),
    }


def dense_mesh_pair(a: Any, b: Any, output_dir: Path, pair_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute cautious dense mesh residual for one pair.

    This is a direct measurement channel, but currently uncalibrated unless a later
    mesh calibration model is added. It must not be interpreted as identity verdict.
    """
    ma = _load_mesh(a)
    mb = _load_mesh(b)
    if ma.get("status") != "ok" or mb.get("status") != "ok":
        return {
            "mesh_status": "unavailable",
            "mesh_error_a": ma.get("status"),
            "mesh_error_b": mb.get("status"),
            "mesh_evidence_level": "not_available",
        }, []

    va = ma["vertices"]
    vb = mb["vertices"]
    common = ma["visible"] & mb["visible"] & np.isfinite(va).all(axis=1) & np.isfinite(vb).all(axis=1)
    common_ids = np.flatnonzero(common)
    if common_ids.size < 1200:
        return {
            "mesh_status": "insufficient_visibility",
            "mesh_common_vertex_count": int(common_ids.size),
            "mesh_evidence_level": "insufficient",
        }, []

    fit_ids, anchor_meta = stable_anchor_indices(va, common_ids, max_points=6000, min_count=1200)
    if fit_ids.size < 1200:
        fit_ids = _subsample(common_ids, 6000)
        anchor_meta = {"anchor_policy": "fallback_mesh_all_common", "anchor_count": int(fit_ids.size), "anchor_fraction": float(fit_ids.size / max(common_ids.size, 1))}
    aligned_fit, rot, trans, alignment_meta = robust_rigid_align(vb[fit_ids], va[fit_ids], min_points=1200)
    aligned_all = vb @ rot + trans
    vectors = np.full((MESH_COUNT, 3), np.nan, np.float32)
    vectors[common] = aligned_all[common] - va[common]
    mag = np.full(MESH_COUNT, np.nan, np.float32)
    mag[common] = np.linalg.norm(vectors[common], axis=1)

    normals_a = np.asarray(ma.get("normals"), np.float32)
    normal_norm = np.linalg.norm(normals_a, axis=1, keepdims=True)
    normals_unit = np.divide(normals_a, np.maximum(normal_norm, 1e-8), out=np.zeros_like(normals_a), where=np.isfinite(normal_norm))
    p2plane = np.full(MESH_COUNT, np.nan, np.float32)
    p2plane[common] = np.sum(vectors[common] * normals_unit[common], axis=1)
    abs_p2plane = np.abs(p2plane)

    vals = mag[common]
    p2vals = abs_p2plane[common]
    labels = _zone_labels(va)
    anatomical_zones = load_anatomical_zones()
    zone_source = "anatomical_mesh_zone_indices_v1" if anatomical_zones else "coordinate_grid_fallback_v1"

    mesh_dir = output_dir / "mesh_motion"
    mesh_dir.mkdir(exist_ok=True)
    safe = pair_id.replace("/", "_")
    np.savez_compressed(
        mesh_dir / f"{safe}.npz",
        schema=np.asarray(MESH_SCHEMA),
        common_visible=common,
        vectors=vectors.astype(np.float16),
        magnitude=mag.astype(np.float16),
        point_to_plane_signed=p2plane.astype(np.float16),
        point_to_plane_abs=abs_p2plane.astype(np.float16),
        rotation=rot.astype(np.float32),
        translation=trans.astype(np.float32),
        anchor_indices=fit_ids.astype(np.int32),
        anchor_policy=np.asarray(str(anchor_meta.get("anchor_policy", "unknown"))),
        alignment_policy=np.asarray(str(alignment_meta.get("alignment_policy", "unknown"))),
        vertex_zone_labels=labels,
        anatomical_zone_names=np.asarray(list(anatomical_zones.keys())),
        zone_source=np.asarray(zone_source),
        vertex_space=np.asarray(str(ma.get("space"))),
    )

    zones: list[dict[str, Any]] = []
    if anatomical_zones:
        zone_iter = [(z, anatomical_zones[z]) for z in PRIORITY_ZONES if z in anatomical_zones]
        min_zone_vertices = 40
        for zone, ids in zone_iter:
            valid_ids = ids[(ids >= 0) & (ids < MESH_COUNT)]
            zmask = np.zeros(MESH_COUNT, bool)
            zmask[valid_ids] = True
            zmask &= common
            if int(zmask.sum()) < min_zone_vertices:
                zones.append({"pair_id": pair_id, "zone": str(zone), "mesh_zone_source": zone_source, "mesh_zone_status": "insufficient_visibility", "mesh_vertex_count": int(zmask.sum())})
                continue
            zvals = mag[zmask]
            zp2 = abs_p2plane[zmask]
            zp2s = p2plane[zmask]
            zv = vectors[zmask]
            shape = _shape_descriptor(va[zmask])
            zones.append({
                "pair_id": pair_id,
                "zone": str(zone),
                "mesh_zone_source": zone_source,
                "mesh_zone_status": "measured",
                "mesh_vertex_count": int(zmask.sum()),
                "mesh_rmse": float(np.sqrt(np.mean(zvals * zvals))),
                "mesh_median": float(np.median(zvals)),
                "mesh_p95": float(np.percentile(zvals, 95)),
                "mesh_point_to_plane_rmse": float(np.sqrt(np.mean(zp2 * zp2))),
                "mesh_point_to_plane_median": float(np.median(zp2)),
                "mesh_point_to_plane_p95": float(np.percentile(zp2, 95)),
                "mesh_point_to_plane_signed_median": float(np.median(zp2s)),
                **shape,
                "mesh_signed_x": float(np.median(zv[:, 0])),
                "mesh_signed_y": float(np.median(zv[:, 1])),
                "mesh_signed_z": float(np.median(zv[:, 2])),
            })
    else:
        for zone in sorted(set(labels[common])):
            zmask = common & (labels == zone)
            if int(zmask.sum()) < 100:
                continue
            zvals = mag[zmask]
            zp2 = abs_p2plane[zmask]
            zp2s = p2plane[zmask]
            zv = vectors[zmask]
            shape = _shape_descriptor(va[zmask])
            zones.append({
                "pair_id": pair_id,
                "zone": str(zone),
                "mesh_zone_source": zone_source,
                "mesh_zone_status": "measured",
                "mesh_vertex_count": int(zmask.sum()),
                "mesh_rmse": float(np.sqrt(np.mean(zvals * zvals))),
                "mesh_median": float(np.median(zvals)),
                "mesh_p95": float(np.percentile(zvals, 95)),
                "mesh_point_to_plane_rmse": float(np.sqrt(np.mean(zp2 * zp2))),
                "mesh_point_to_plane_median": float(np.median(zp2)),
                "mesh_point_to_plane_p95": float(np.percentile(zp2, 95)),
                "mesh_point_to_plane_signed_median": float(np.median(zp2s)),
                **shape,
                "mesh_signed_x": float(np.median(zv[:, 0])),
                "mesh_signed_y": float(np.median(zv[:, 1])),
                "mesh_signed_z": float(np.median(zv[:, 2])),
            })

    row = {
        "mesh_status": "measured_uncalibrated",
        "mesh_evidence_level": "direct_uncalibrated_dense_residual",
        "mesh_zone_source": zone_source,
        "mesh_anatomical_zone_count": len(anatomical_zones),
        "mesh_alignment_policy": alignment_meta.get("alignment_policy"),
        "mesh_alignment_trimmed_count": alignment_meta.get("trimmed_point_count", 0),
        "mesh_alignment_residual_before_median": alignment_meta.get("residual_before_median"),
        "mesh_alignment_residual_after_median": alignment_meta.get("residual_after_median"),
        "mesh_file": f"mesh_motion/{safe}.npz",
        "mesh_common_vertex_count": int(common_ids.size),
        "mesh_fit_vertex_count": int(fit_ids.size),
        "mesh_anchor_policy": str(anchor_meta.get("anchor_policy", "unknown")),
        "mesh_anchor_fraction": float(anchor_meta.get("anchor_fraction", 0.0)),
        "mesh_rmse": float(np.sqrt(np.mean(vals * vals))),
        "mesh_median": float(np.median(vals)),
        "mesh_p95": float(np.percentile(vals, 95)),
        "mesh_point_to_plane_rmse": float(np.sqrt(np.mean(p2vals * p2vals))),
        "mesh_point_to_plane_median": float(np.median(p2vals)),
        "mesh_point_to_plane_p95": float(np.percentile(p2vals, 95)),
        "mesh_point_to_plane_signed_median": float(np.median(p2plane[common])),
        "mesh_visible_fraction": float(common_ids.size / MESH_COUNT),
        "mesh_space_a": str(ma.get("space")),
        "mesh_space_b": str(mb.get("space")),
    }
    return row, zones
