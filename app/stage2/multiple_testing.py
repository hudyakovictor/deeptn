from __future__ import annotations

from math import erfc, sqrt
from typing import Any

MT_SCHEMA = "deeputin-stage2-multiple-testing-v1.0"


def _p_from_z(z: float) -> float:
    try:
        z = abs(float(z))
        return max(0.0, min(1.0, erfc(z / sqrt(2.0))))
    except Exception:
        return 1.0


def _bh_qvalues(items: list[tuple[int, float]]) -> dict[int, float]:
    if not items:
        return {}
    ordered = sorted(items, key=lambda x: x[1])
    m = len(ordered)
    q = [1.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx, p = ordered[rank - 1]
        val = min(prev, p * m / rank)
        q[rank - 1] = val
        prev = val
    return {ordered[i][0]: max(0.0, min(1.0, q[i])) for i in range(m)}


def apply_pair_fdr(rows: list[dict[str, Any]], *, z_key: str = "p95_point_z", q_threshold: float = 0.10) -> dict[str, Any]:
    tests: list[tuple[int, float]] = []
    for i, r in enumerate(rows):
        z = r.get(z_key)
        if z is None:
            continue
        p = _p_from_z(float(z))
        r["mt_p_approx"] = p
        tests.append((i, p))
    qmap = _bh_qvalues(tests)
    significant = 0
    for i, q in qmap.items():
        rows[i]["mt_q_value"] = q
        rows[i]["mt_significant_fdr10"] = bool(q <= q_threshold)
        significant += int(q <= q_threshold)
    return {
        "schema": MT_SCHEMA,
        "scope": "pair_metrics",
        "test_count": len(tests),
        "q_threshold": q_threshold,
        "significant_count": significant,
        "method": "Benjamini-Hochberg on approximate two-sided normal p from robust z-like score",
    }


def apply_zone_fdr(zones: list[dict[str, Any]], *, z_key: str = "robust_z", q_threshold: float = 0.10) -> dict[str, Any]:
    tests: list[tuple[int, float]] = []
    for i, zrow in enumerate(zones):
        if zrow.get("status") != "measured" and zrow.get("mesh_zone_status") != "measured":
            continue
        z = zrow.get(z_key)
        if z is None:
            continue
        p = _p_from_z(float(z))
        zrow["mt_p_approx"] = p
        tests.append((i, p))
    qmap = _bh_qvalues(tests)
    significant = 0
    for i, q in qmap.items():
        zones[i]["mt_q_value"] = q
        zones[i]["mt_significant_fdr10"] = bool(q <= q_threshold)
        significant += int(q <= q_threshold)
    return {
        "schema": MT_SCHEMA,
        "scope": "zone_metrics",
        "test_count": len(tests),
        "q_threshold": q_threshold,
        "significant_count": significant,
        "method": "Benjamini-Hochberg on approximate p-values",
    }
