from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import CalibrationModel
from .core import Record

SENSITIVITY_SCHEMA = "deeputin-stage2-calibration-sensitivity-v1.0"
CORE_METRICS = (
    "ldm134_rmse",
    "ldm134_p95",
    "identity_only_ldm134_rmse",
    "alpha_id_l2",
    "alpha_exp_l2",
)


def leave_one_dataset_sensitivity(records: list[Record], zone106: np.ndarray, zone134: np.ndarray) -> dict[str, Any]:
    """Compute leave-one-calibration-dataset-out p95 sensitivity.

    This is a compact robustness diagnostic. It does not change thresholds in v1;
    it tells Stage 2/Stage 3 whether the threshold is stable across calibration persons.
    """
    datasets = sorted({r.dataset_id for r in records})
    if len(datasets) < 3:
        return {
            "schema": SENSITIVITY_SCHEMA,
            "status": "insufficient_calibration_datasets",
            "dataset_count": len(datasets),
            "entries": [],
        }
    entries: list[dict[str, Any]] = []
    for holdout in datasets:
        subset = [r for r in records if r.dataset_id != holdout]
        if not subset:
            continue
        try:
            model = CalibrationModel(subset, zone106, zone134)
        except Exception as exc:
            entries.append({"holdout_dataset": holdout, "status": "failed", "error": str(exc)})
            continue
        for pose, metrics in model.references.items():
            for metric, ref in metrics.items():
                if metric not in CORE_METRICS and not str(metric).startswith("zone::"):
                    continue
                entries.append({
                    "holdout_dataset": holdout,
                    "pose_bin": pose,
                    "metric": metric,
                    "count": int(ref.get("count", 0)),
                    "median": float(ref.get("median", 0.0)),
                    "mad": float(ref.get("mad", 0.0)),
                    "p95": float(ref.get("p95", 0.0)),
                    "status": "ok" if int(ref.get("count", 0)) >= 7 else "low_count",
                })

    # Aggregate dispersion by pose/metric.
    grouped: dict[tuple[str, str], list[float]] = {}
    for e in entries:
        if e.get("status") not in {"ok", "low_count"}:
            continue
        grouped.setdefault((str(e.get("pose_bin")), str(e.get("metric"))), []).append(float(e.get("p95", 0.0)))
    summary: list[dict[str, Any]] = []
    for (pose, metric), vals in sorted(grouped.items()):
        arr = np.asarray(vals, np.float64)
        med = float(np.median(arr)) if arr.size else 0.0
        spread = float((np.max(arr) - np.min(arr)) / max(med, 1e-8)) if arr.size else 0.0
        summary.append({
            "pose_bin": pose,
            "metric": metric,
            "holdout_count": int(arr.size),
            "p95_median": med,
            "p95_min": float(np.min(arr)) if arr.size else 0.0,
            "p95_max": float(np.max(arr)) if arr.size else 0.0,
            "relative_spread": spread,
            "stability": "stable" if spread <= 0.35 and arr.size >= max(3, len(datasets) - 1) else "unstable_or_sparse",
        })
    return {
        "schema": SENSITIVITY_SCHEMA,
        "status": "complete",
        "dataset_count": len(datasets),
        "datasets": datasets,
        "entry_count": len(entries),
        "summary_count": len(summary),
        "summary": summary,
        "entries": entries,
        "policy": "diagnostic only; thresholds are not changed in this version",
    }
