"""ITER4 fail-closed selected_metrics handling.

Never silently treat missing metrics as zeros / success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    "MetricStatus",
    "SelectedMetricsResult",
    "select_metrics",
    "assert_metrics_present",
    "merge_metric_dicts_fail_closed",
]


@dataclass
class MetricStatus:
    key: str
    present: bool
    value: Any = None
    reason: str = ""


@dataclass
class SelectedMetricsResult:
    ok: bool
    values: Dict[str, Any] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    nulls: List[str] = field(default_factory=list)
    statuses: List[MetricStatus] = field(default_factory=list)
    fail_open_attempted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "values": self.values,
            "missing": list(self.missing),
            "nulls": list(self.nulls),
            "fail_open_attempted": self.fail_open_attempted,
            "status": "ok" if self.ok else "insufficient_metrics",
        }


def _is_null(v: Any) -> bool:
    if v is None:
        return True
    try:
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray) and v.size == 0:
            return True
        if isinstance(v, (np.floating, float)) and not np.isfinite(float(v)):
            return True
    except Exception:
        pass
    return False


def select_metrics(
    source: Mapping[str, Any] | None,
    required_keys: Sequence[str],
    *,
    optional_keys: Sequence[str] = (),
    allow_fail_open: bool = False,
    fill_value: Any = 0.0,
) -> SelectedMetricsResult:
    """Select metrics fail-closed by default.

    If allow_fail_open=True, missing keys filled with fill_value but
    result.ok stays False and fail_open_attempted=True (never silent success).
    """
    src = dict(source or {})
    values: Dict[str, Any] = {}
    missing: List[str] = []
    nulls: List[str] = []
    statuses: List[MetricStatus] = []
    fail_open = False

    for key in required_keys:
        if key not in src:
            missing.append(key)
            statuses.append(MetricStatus(key=key, present=False, reason="missing"))
            if allow_fail_open:
                values[key] = fill_value
                fail_open = True
            continue
        val = src[key]
        if _is_null(val):
            nulls.append(key)
            statuses.append(MetricStatus(key=key, present=False, value=val, reason="null"))
            if allow_fail_open:
                values[key] = fill_value
                fail_open = True
            continue
        values[key] = val
        statuses.append(MetricStatus(key=key, present=True, value=val, reason="ok"))

    for key in optional_keys:
        if key not in src:
            statuses.append(MetricStatus(key=key, present=False, reason="optional_missing"))
            continue
        val = src[key]
        if _is_null(val):
            statuses.append(MetricStatus(key=key, present=False, value=val, reason="optional_null"))
            continue
        values[key] = val
        statuses.append(MetricStatus(key=key, present=True, value=val, reason="ok"))

    ok = len(missing) == 0 and len(nulls) == 0
    # fail-open never upgrades ok to True
    if fail_open:
        ok = False
    return SelectedMetricsResult(
        ok=ok,
        values=values,
        missing=missing,
        nulls=nulls,
        statuses=statuses,
        fail_open_attempted=fail_open,
    )


def assert_metrics_present(
    source: Mapping[str, Any] | None,
    required_keys: Sequence[str],
) -> Dict[str, Any]:
    res = select_metrics(source, required_keys, allow_fail_open=False)
    if not res.ok:
        raise ValueError(
            f"insufficient_metrics missing={res.missing} nulls={res.nulls}"
        )
    return res.values


def merge_metric_dicts_fail_closed(
    *dicts: Mapping[str, Any],
    required_keys: Optional[Sequence[str]] = None,
) -> SelectedMetricsResult:
    """Left-to-right merge; later keys override. Then fail-closed select if required."""
    merged: Dict[str, Any] = {}
    for d in dicts:
        if d:
            merged.update(dict(d))
    if required_keys is None:
        # all non-null keys present → ok if any values
        values = {k: v for k, v in merged.items() if not _is_null(v)}
        nulls = [k for k, v in merged.items() if _is_null(v)]
        return SelectedMetricsResult(
            ok=len(values) > 0 and len(nulls) == 0,
            values=values,
            missing=[],
            nulls=nulls,
            statuses=[MetricStatus(k, k in values, merged.get(k), "ok" if k in values else "null") for k in merged],
        )
    return select_metrics(merged, required_keys, allow_fail_open=False)
