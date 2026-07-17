from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .anchor_policy import stable_anchor_mask


@dataclass
class Record:
    record_id: str
    dataset_id: str
    date: str | None
    sequence: int
    pose_bin: str
    angles: np.ndarray
    ldm106: np.ndarray
    ldm134: np.ndarray
    visible106: np.ndarray
    visible134: np.ndarray
    alpha_id: np.ndarray
    alpha_exp: np.ndarray
    identity_only106: np.ndarray | None = None
    identity_only134: np.ndarray | None = None
    quality_status: str = "unknown"
    quality_texture_score: float = 0.0
    forehead_wrinkle_supported: bool = False
    quality_zones: dict[str, Any] = field(default_factory=dict)
    record_dir: str | None = None
    source_group: str = "unknown"
    source_sha256: str | None = None


@dataclass
class Comparison:
    status: str
    metrics: dict[str, float]
    zones: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch row-vector alignment source -> target, without scale."""
    cs = source.mean(axis=0); ct = target.mean(axis=0)
    x = source - cs; y = target - ct
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    translation = ct - cs @ rotation
    return source @ rotation + translation, rotation.astype(np.float32), translation.astype(np.float32)


def robust_rigid_align(
    source: np.ndarray,
    target: np.ndarray,
    *,
    trim_fraction: float = 0.15,
    max_iterations: int = 5,
    min_points: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Iteratively trimmed Kabsch alignment with diagnostics.

    Large local changes must not be absorbed into the global transform.  Each
    iteration fits on the lowest-residual subset and the final transform is
    applied to all source points.  No scale is estimated.
    """
    src = np.asarray(source, np.float64)
    dst = np.asarray(target, np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source/target must have equal (N,3) shape")
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    ids = np.flatnonzero(finite)
    if ids.size < min_points:
        aligned, rotation, translation = _rigid_align(src[finite], dst[finite])
        return aligned, rotation, translation, {
            "alignment_policy": "kabsch_insufficient_for_trimming",
            "fit_point_count": int(ids.size),
            "trimmed_point_count": 0,
            "iterations": 1,
        }
    keep = ids.copy()
    iterations = 0
    rotation = np.eye(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    trim_fraction = float(np.clip(trim_fraction, 0.0, 0.4))
    for iterations in range(1, max_iterations + 1):
        _, rotation, translation = _rigid_align(src[keep], dst[keep])
        all_aligned = src @ rotation + translation
        residual = np.linalg.norm(all_aligned[ids] - dst[ids], axis=1)
        keep_n = max(min_points, int(np.ceil(ids.size * (1.0 - trim_fraction))))
        new_keep = ids[np.argsort(residual, kind="stable")[:keep_n]]
        if np.array_equal(np.sort(new_keep), np.sort(keep)):
            keep = new_keep
            break
        keep = new_keep
    _, rotation, translation = _rigid_align(src[keep], dst[keep])
    aligned_all = src @ rotation + translation
    before = np.linalg.norm(src[ids] - dst[ids], axis=1)
    after = np.linalg.norm(aligned_all[ids] - dst[ids], axis=1)
    return aligned_all.astype(np.float32), rotation, translation, {
        "alignment_policy": "iteratively_trimmed_kabsch_v1",
        "fit_point_count": int(keep.size),
        "input_point_count": int(ids.size),
        "trimmed_point_count": int(ids.size - keep.size),
        "iterations": int(iterations),
        "residual_before_median": float(np.median(before)),
        "residual_after_median": float(np.median(after)),
        "residual_after_p95": float(np.percentile(after, 95)),
    }


def _stats(distance: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(distance * distance))),
        "median": float(np.median(distance)),
        "p95": float(np.percentile(distance, 95)),
        "max": float(np.max(distance)),
    }


def compare_landmarks(
    a: Record,
    b: Record,
    zone_map106: np.ndarray,
    zone_map134: np.ndarray,
    min_points106: int = 24,
    min_points134: int = 30,
) -> Comparison:
    common106 = np.asarray(a.visible106, bool) & np.asarray(b.visible106, bool)
    common134 = np.asarray(a.visible134, bool) & np.asarray(b.visible134, bool)
    diagnostics = {
        "common_visible106": int(common106.sum()), "common_visible134": int(common134.sum()),
        "coverage106": float(common106.mean()), "coverage134": float(common134.mean()),
        "pose_distance": float(np.linalg.norm((a.angles - b.angles) / np.array([15.0, 20.0, 15.0]))),
    }
    if common106.sum() < min_points106 or common134.sum() < min_points134:
        return Comparison("insufficient_visibility", {}, [], diagnostics)

    anchor106, anchor_meta106 = stable_anchor_mask(a.ldm106, common106, min_count=min_points106)
    anchor134, anchor_meta134 = stable_anchor_mask(a.ldm134, common134, min_count=min_points134)
    _, r106, t106, align106 = robust_rigid_align(b.ldm106[anchor106], a.ldm106[anchor106])
    _, r134, t134, align134 = robust_rigid_align(b.ldm134[anchor134], a.ldm134[anchor134])
    aligned106_all = b.ldm106 @ r106 + t106
    aligned134_all = b.ldm134 @ r134 + t134
    residual106 = aligned106_all[common106] - a.ldm106[common106]
    residual134 = aligned134_all[common134] - a.ldm134[common134]
    distance106 = np.linalg.norm(residual106, axis=1)
    distance134 = np.linalg.norm(residual134, axis=1)
    s106, s134 = _stats(distance106), _stats(distance134)
    metrics = {
        **{f"ldm106_{k}": v for k, v in s106.items()},
        **{f"ldm134_{k}": v for k, v in s134.items()},
        "alpha_id_l2": float(np.linalg.norm(a.alpha_id - b.alpha_id)),
        "alpha_exp_l2": float(np.linalg.norm(a.alpha_exp - b.alpha_exp)),
    }
    if a.identity_only134 is not None and b.identity_only134 is not None:
        _, ir, it, _ = robust_rigid_align(b.identity_only134[anchor134], a.identity_only134[anchor134])
        identity_aligned_all = b.identity_only134 @ ir + it
        metrics["identity_only_ldm134_rmse"] = float(np.sqrt(np.mean(np.sum((identity_aligned_all[common134] - a.identity_only134[common134]) ** 2, axis=1))))

    zones: list[dict[str, Any]] = []
    visible_indices134 = np.flatnonzero(common134)
    for zone_id in np.unique(zone_map134):
        local = zone_map134[visible_indices134] == zone_id
        if int(local.sum()) < 4:
            zones.append({"zone": str(zone_id), "status": "insufficient_visibility", "point_count": int(local.sum())})
            continue
        rv = residual134[local]; dist = np.linalg.norm(rv, axis=1)
        zones.append({
            "zone": str(zone_id), "status": "measured", "point_count": int(local.sum()),
            "rmse": float(np.sqrt(np.mean(dist * dist))), "median": float(np.median(dist)),
            "p95": float(np.percentile(dist, 95)),
            "signed_x": float(np.median(rv[:, 0])), "signed_y": float(np.median(rv[:, 1])),
            "signed_z": float(np.median(rv[:, 2])),
        })
    diagnostics.update({"rotation106": r106, "translation106": t106, "rotation134": r134, "translation134": t134,
                        "anchor106_count": anchor_meta106.get("anchor_count", 0), "anchor106_policy": anchor_meta106.get("anchor_policy", "unknown"),
                        "anchor134_count": anchor_meta134.get("anchor_count", 0), "anchor134_policy": anchor_meta134.get("anchor_policy", "unknown"),
                        "alignment106_policy": align106.get("alignment_policy"), "alignment106_trimmed_count": align106.get("trimmed_point_count", 0),
                        "alignment134_policy": align134.get("alignment_policy"), "alignment134_trimmed_count": align134.get("trimmed_point_count", 0),
                        "alignment134_residual_before_median": align134.get("residual_before_median"),
                        "alignment134_residual_after_median": align134.get("residual_after_median")})
    return Comparison("measured", metrics, zones, diagnostics)


def build_coordinate_zone_map(records: list[Record], landmark_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Nine reproducible coordinate zones; avoids unverified anatomical labels."""
    if not records:
        raise ValueError("cannot build zones without records")
    stack = np.stack([r.ldm106 if landmark_count == 106 else r.ldm134 for r in records[: min(200, len(records))]])
    template = np.median(stack, axis=0)
    qx = np.quantile(template[:, 0], [1 / 3, 2 / 3]); qy = np.quantile(template[:, 1], [1 / 3, 2 / 3])
    xb = np.digitize(template[:, 0], qx); yb = np.digitize(template[:, 1], qy)
    names = np.array([[f"x_{x}_{y}" for x in ("low", "center", "high")] for y in ("low", "center", "high")])
    labels = names[yb, xb]
    meta = {"version": "coordinate-grid-v1", "landmark_count": landmark_count, "qx": qx.tolist(), "qy": qy.tolist(),
            "warning": "Coordinate zones are not anatomical labels."}
    return labels, meta


def robust_reference(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray([v for v in values if np.isfinite(v)], np.float64)
    if arr.size == 0:
        return {"count": 0, "median": 0.0, "mad": 0.0, "p95": 0.0, "p99": 0.0}
    median = float(np.median(arr)); mad = float(np.median(np.abs(arr - median)))
    return {"count": int(arr.size), "median": median, "mad": mad,
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99))}


def calibrated_score(value: float, reference: dict[str, float | int], matched: list[float]) -> dict[str, float | str]:
    matched_arr = np.asarray([v for v in matched if np.isfinite(v)], np.float64)
    threshold = float(reference.get("p95", 0.0))
    if matched_arr.size:
        threshold = max(threshold, float(np.percentile(matched_arr, 95)))
    median = float(reference.get("median", 0.0)); mad = float(reference.get("mad", 0.0))
    z = float((value - median) / max(1.4826 * mad, 1e-8))
    if int(reference.get("count", 0)) < 7:
        status = "insufficient_calibration"
    elif value <= threshold:
        status = "within_calibration_noise"
    elif z < 3.5:
        status = "elevated_but_uncertain"
    else:
        status = "elevated"
    return {"calibration_median": median, "calibration_p95": threshold, "robust_z": z, "status": status}
