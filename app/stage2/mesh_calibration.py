from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .anchor_policy import stable_anchor_indices
from .core import Record, robust_rigid_align, robust_reference
from .mesh_dense import MESH_COUNT, _load_mesh, _subsample

MESH_CALIBRATION_SCHEMA = "deeputin-stage2-mesh-calibration-v1.0"
MESH_METRICS = (
    "mesh_rmse",
    "mesh_median",
    "mesh_p95",
    "mesh_point_to_plane_rmse",
    "mesh_point_to_plane_median",
    "mesh_point_to_plane_p95",
)


def _pose_distance(a: Record, b: Record) -> float:
    return float(np.linalg.norm((a.angles - b.angles) / np.array([15.0, 20.0, 15.0])))


def _mesh_metrics(a: Record, b: Record) -> dict[str, Any]:
    ma = _load_mesh(a)
    mb = _load_mesh(b)
    if ma.get("status") != "ok" or mb.get("status") != "ok":
        return {"status": "unavailable", "reason_a": ma.get("status"), "reason_b": mb.get("status")}
    va = ma["vertices"]
    vb = mb["vertices"]
    common = ma["visible"] & mb["visible"] & np.isfinite(va).all(axis=1) & np.isfinite(vb).all(axis=1)
    common_ids = np.flatnonzero(common)
    if common_ids.size < 1200:
        return {"status": "insufficient_visibility", "mesh_common_vertex_count": int(common_ids.size)}
    fit_ids, _ = stable_anchor_indices(va, common_ids, max_points=6000, min_count=1200)
    if fit_ids.size < 1200:
        fit_ids = _subsample(common_ids, 6000)
    _, rot, trans, _ = robust_rigid_align(vb[fit_ids], va[fit_ids], min_points=1200)
    aligned = vb @ rot + trans
    vectors = aligned[common] - va[common]
    mag = np.linalg.norm(vectors, axis=1)

    normals_a = np.asarray(ma.get("normals"), np.float32)
    normal_norm = np.linalg.norm(normals_a, axis=1, keepdims=True)
    normals_unit = np.divide(normals_a, np.maximum(normal_norm, 1e-8), out=np.zeros_like(normals_a), where=np.isfinite(normal_norm))
    p2plane_signed = np.sum((aligned - va)[common] * normals_unit[common], axis=1)
    p2plane = np.abs(p2plane_signed)
    return {
        "status": "measured",
        "mesh_common_vertex_count": int(common_ids.size),
        "mesh_fit_vertex_count": int(fit_ids.size),
        "mesh_rmse": float(np.sqrt(np.mean(mag * mag))),
        "mesh_median": float(np.median(mag)),
        "mesh_p95": float(np.percentile(mag, 95)),
        "mesh_point_to_plane_rmse": float(np.sqrt(np.mean(p2plane * p2plane))),
        "mesh_point_to_plane_median": float(np.median(p2plane)),
        "mesh_point_to_plane_p95": float(np.percentile(p2plane, 95)),
        "mesh_point_to_plane_signed_median": float(np.median(p2plane_signed)),
    }


@dataclass
class MeshNoiseReference:
    schema: str
    status: str
    references: dict[str, dict[str, dict[str, float | int]]]
    pair_count: int
    unavailable_count: int
    pose_counts: dict[str, int]


class MeshNoiseModel:
    """Same-person dense mesh noise model.

    Calibration datasets often lack full Stage-1 reconstruction meshes. In that case
    this model reports `unavailable` and Stage 2 keeps dense mesh as direct but
    uncalibrated support. If calibration meshes are present, it produces p95/MAD
    references by pose_bin for mesh_rmse and point-to-plane metrics.
    """

    def __init__(self, records: list[Record], *, max_pairs_per_pose: int = 400):
        self.records = records
        self.max_pairs_per_pose = int(max_pairs_per_pose)
        self.reference = self._build()

    def _build(self) -> MeshNoiseReference:
        groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
        for r in self.records:
            groups[(r.dataset_id, r.pose_bin)].append(r)

        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        pair_count = 0
        unavailable = 0
        pose_counts: dict[str, int] = defaultdict(int)
        for (_, pose), rs in groups.items():
            rs = sorted(rs, key=lambda r: (float(r.angles[1]), float(r.angles[0]), r.sequence))
            local_pairs = 0
            for off in (1, 2):
                for a, b in zip(rs, rs[off:]):
                    if local_pairs >= self.max_pairs_per_pose:
                        break
                    if _pose_distance(a, b) > 2.5:
                        continue
                    m = _mesh_metrics(a, b)
                    if m.get("status") != "measured":
                        unavailable += 1
                        continue
                    pair_count += 1
                    local_pairs += 1
                    pose_counts[pose] += 1
                    for k in MESH_METRICS:
                        v = m.get(k)
                        if v is not None and np.isfinite(float(v)):
                            values[pose][k].append(float(v))
        refs = {pose: {metric: robust_reference(vals) for metric, vals in metrics.items()} for pose, metrics in values.items()}
        status = "available" if pair_count >= 7 else "unavailable"
        return MeshNoiseReference(
            schema=MESH_CALIBRATION_SCHEMA,
            status=status,
            references=refs,
            pair_count=pair_count,
            unavailable_count=unavailable,
            pose_counts=dict(pose_counts),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.reference.schema,
            "status": self.reference.status,
            "pair_count": self.reference.pair_count,
            "unavailable_count": self.reference.unavailable_count,
            "pose_counts": self.reference.pose_counts,
            "references": self.reference.references,
            "metrics": list(MESH_METRICS),
            "policy": "If unavailable, dense mesh remains direct_uncalibrated support only.",
        }

    def score(self, pose_bin: str, mesh_row: dict[str, Any]) -> dict[str, Any]:
        refs = self.reference.references.get(pose_bin, {})
        out: dict[str, Any] = {
            "mesh_calibration_status": self.reference.status if refs else "insufficient_calibration",
            "mesh_calibrated_metric_count": 0,
            "mesh_calibrated_elevated_count": 0,
            "mesh_max_robust_z": 0.0,
            "mesh_calibrated_summary": "",
        }
        if mesh_row.get("mesh_status") not in {"measured_uncalibrated", "measured_calibrated"}:
            out["mesh_calibration_status"] = "not_measured"
            return out
        summaries: list[str] = []
        max_z = 0.0
        elevated = 0
        count = 0
        for metric in MESH_METRICS:
            ref = refs.get(metric)
            val = mesh_row.get(metric)
            if not ref or val is None or int(ref.get("count", 0)) < 7:
                continue
            median = float(ref.get("median", 0.0))
            mad = float(ref.get("mad", 0.0))
            p95 = float(ref.get("p95", 0.0))
            z = float((float(val) - median) / max(1.4826 * mad, 1e-8))
            count += 1
            max_z = max(max_z, z)
            status = "within_mesh_noise" if float(val) <= p95 else ("mesh_elevated_but_uncertain" if z < 3.5 else "mesh_elevated")
            if status == "mesh_elevated":
                elevated += 1
            out[f"{metric}_calibration_median"] = median
            out[f"{metric}_calibration_p95"] = p95
            out[f"{metric}_robust_z"] = z
            out[f"{metric}_calibrated_status"] = status
            summaries.append(f"{metric}:{status}:z={z:.2f}")
        out["mesh_calibrated_metric_count"] = count
        out["mesh_calibrated_elevated_count"] = elevated
        out["mesh_max_robust_z"] = max_z
        out["mesh_calibrated_summary"] = "|".join(summaries)
        if count == 0:
            out["mesh_calibration_status"] = "insufficient_calibration"
        elif elevated:
            out["mesh_calibration_status"] = "mesh_elevated"
        else:
            out["mesh_calibration_status"] = "within_mesh_noise"
        return out
