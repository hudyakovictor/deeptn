"""Full forensic metrics catalog (legacy restore).

Old project truth (`metric_coverage_report.json`):
  - catalog_count / registry_count = **2199**
  - per-view single counts e.g. frontal=1814, light=1187, mid/deep/profile=1025
  - pair metrics per view + chronology

This module exposes:
  - target catalog size and family breakdown
  - recovered metric names (from stability + identity config; partial until original CSV returns)
  - distribution across 9 pose buckets
  - helpers to list metrics for a view
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_LEG = Path(__file__).resolve().parent / "legacy_metrics"
_DIST = _LEG / "view_metric_distribution.json"
_COV = _LEG / "metric_coverage_report.json"
_STAB = _LEG / "stability_pruned_names.json"
_CSV = _LEG / "metric_catalog_active.csv"

NINE_BUCKETS = (
    "frontal",
    "left_threequarter_light",
    "right_threequarter_light",
    "left_threequarter_mid",
    "right_threequarter_mid",
    "left_threequarter_deep",
    "right_threequarter_deep",
    "left_profile",
    "right_profile",
)

__all__ = [
    "NINE_BUCKETS",
    "catalog_summary",
    "target_catalog_count",
    "recovered_metric_names",
    "stability_allowed_names",
    "metrics_for_bucket",
    "view_distribution",
    "load_active_catalog_rows",
]


@lru_cache(maxsize=1)
def _dist() -> dict:
    if _DIST.exists():
        return json.loads(_DIST.read_text(encoding="utf-8"))
    if _COV.exists():
        cov = json.loads(_COV.read_text(encoding="utf-8"))
        return {
            "catalog_count_target": cov.get("catalog_count", 0),
            "single_by_bucket": cov.get("single_by_bucket", {}),
            "pair_by_bucket": cov.get("pair_by_bucket", {}),
            "catalog_by_family": cov.get("catalog_by_family", {}),
            "views": {},
        }
    return {}


def target_catalog_count() -> int:
    d = _dist()
    return int(d.get("catalog_count_target") or 2199)


def catalog_summary() -> Dict[str, Any]:
    d = _dist()
    cov = json.loads(_COV.read_text(encoding="utf-8")) if _COV.exists() else {}
    stab = json.loads(_STAB.read_text(encoding="utf-8")) if _STAB.exists() else {}
    recovered = len(recovered_metric_names())
    target = target_catalog_count()
    return {
        "target_catalog_count": target_catalog_count(),
        "registry_count_reported": cov.get("registry_count"),
        "recovered_names": recovered,
        "stability_allowed": len(stab.get("allowed") or []),
        "stability_excluded": len(stab.get("excluded") or []),
        "gap_to_full_catalog": max(0, target - recovered),
        "single_by_bucket": d.get("single_by_bucket") or cov.get("single_by_bucket"),
        "pair_by_bucket": d.get("pair_by_bucket") or cov.get("pair_by_bucket"),
        "catalog_by_family": d.get("catalog_by_family") or cov.get("catalog_by_family"),
        "chronology_count": cov.get("chronology_count"),
        "nine_buckets": list(NINE_BUCKETS),
        "active_csv": str(_CSV) if _CSV.exists() else None,
        "note": (
            "Full old catalog is 2199 metrics (not 234). "
            "234 was only GEOMETRY_CORE_METRICS subset. "
            f"The active CSV currently contains {recovered} unique metric names."
        ),
    }


def stability_allowed_names() -> List[str]:
    if not _STAB.exists():
        return []
    return list(json.loads(_STAB.read_text(encoding="utf-8")).get("allowed") or [])


def recovered_metric_names() -> List[str]:
    rows = load_active_catalog_rows()
    if rows:
        return sorted({r["metric_name"] for r in rows if r.get("metric_name")})
    return sorted(set(stability_allowed_names()))


def load_active_catalog_rows() -> List[dict]:
    if not _CSV.exists():
        return []
    import csv
    with _CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def view_distribution() -> Dict[str, Any]:
    d = _dist()
    return {
        "single_by_bucket": d.get("single_by_bucket", {}),
        "pair_by_bucket": d.get("pair_by_bucket", {}),
        "identity_selected_by_view": {
            b: (d.get("views") or {}).get(b, {})
            for b in NINE_BUCKETS
        },
    }


def metrics_for_bucket(
    bucket: str,
    *,
    tier: str = "selected",
) -> List[str]:
    """Return metric names for a pose bucket.

    tier:
      - selected: core+experimental minus disabled (identity scoring config)
      - core / experimental / disabled
      - coverage: not a name list — use view_distribution()["single_by_bucket"] counts
    """
    views = (_dist().get("views") or {})
    info = views.get(bucket) or {}
    if tier == "selected":
        return list(info.get("selected") or [])
    if tier in ("core", "experimental", "disabled"):
        return list(info.get(tier) or [])
    if tier == "all_recovered":
        return recovered_metric_names()
    return list(info.get("selected") or [])
