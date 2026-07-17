from __future__ import annotations

from typing import Any
import numpy as np

SCHEMA = "deeputin-stage2-pose-leakage-diagnostic-v1.0"
METRICS = (
    "ldm106_rmse",
    "ldm134_rmse",
    "p95_point_z",
    "identity_only_motion_rmse",
    "mesh_rmse",
    "mesh_point_to_plane_rmse",
)


def _finite_pairs(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    for row in rows:
        try:
            pose = float(row.get("pose_distance"))
            value = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        if np.isfinite(pose) and np.isfinite(value):
            x.append(pose); y.append(value)
    return np.asarray(x, np.float64), np.asarray(y, np.float64)


def pose_leakage_diagnostic(rows: list[dict[str, Any]], *, min_count: int = 12) -> dict[str, Any]:
    """Check whether residuals still grow with pose difference after normalization.

    This is a diagnostic, not a correction. A strong positive rank correlation means
    the metric may retain pose leakage and should be interpreted conservatively.
    """
    results: dict[str, Any] = {}
    flagged: list[str] = []
    for metric in METRICS:
        x, y = _finite_pairs(rows, metric)
        if x.size < min_count or np.unique(x).size < 3 or np.unique(y).size < 3:
            results[metric] = {"status": "insufficient_data", "count": int(x.size)}
            continue
        # Spearman without scipy: Pearson correlation of stable ranks.
        xr = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
        yr = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
        rho = float(np.corrcoef(xr, yr)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        status = "pose_leakage_candidate" if rho >= 0.45 else ("weak_pose_dependence" if rho >= 0.25 else "no_strong_pose_dependence")
        if status == "pose_leakage_candidate":
            flagged.append(metric)
        results[metric] = {"status": status, "count": int(x.size), "spearman_rho": rho, "linear_slope": slope}
    return {
        "schema": SCHEMA,
        "status": "pose_leakage_candidates_present" if flagged else "no_strong_pose_leakage_detected",
        "flagged_metrics": flagged,
        "metrics": results,
        "policy": "Diagnostic only; does not subtract effects or alter primary measurements.",
    }
