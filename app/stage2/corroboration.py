from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

SCHEMA = "deeputin-stage2-cross-bin-corroboration-v1.0"
CANDIDATE_STATES = {
    "persistent_geometric_change",
    "coherent_jump_candidate",
    "alpha_id_jump_candidate",
    "rapid_change_candidate",
    "persistent_rapid_change_candidate",
    # Legacy compatibility while old result sets are migrated.
    "biologically_improbable_rate_candidate",
    "persistent_biologically_improbable_change",
}


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def apply_cross_bin_corroboration(rows: list[dict[str, Any]], *, window_days: int = 45) -> dict[str, Any]:
    """Annotate blind candidates with independent pose-bin support.

    Cross-bin rows never contribute to the primary residual. They only corroborate
    already-frozen same-bin adjacent results inside a bounded temporal window.
    """
    adjacent = [r for r in rows if r.get("pair_type") == "adjacent"]
    events: list[dict[str, Any]] = []
    for row in adjacent:
        d = _date(row.get("date_b"))
        if d is None:
            row["cross_bin_corroboration_status"] = "date_unavailable"
            row["cross_bin_support_count"] = 0
            continue
        supports = []
        for other in adjacent:
            if other is row or other.get("pose_bin") == row.get("pose_bin"):
                continue
            od = _date(other.get("date_b"))
            if od is None or abs((od - d).days) > int(window_days):
                continue
            if str(other.get("status")) not in CANDIDATE_STATES:
                continue
            if bool(other.get("quality_limited")):
                continue
            supports.append(other)
        pose_bins = sorted({str(x.get("pose_bin")) for x in supports})
        source_groups = sorted({str(x.get("source_group_b")) for x in supports if str(x.get("source_group_b") or "unknown") != "unknown"})
        row["cross_bin_support_count"] = len(supports)
        row["cross_bin_support_pose_count"] = len(pose_bins)
        row["cross_bin_support_pose_bins"] = "|".join(pose_bins)
        row["cross_bin_independent_source_count"] = len(source_groups)
        if str(row.get("status")) not in CANDIDATE_STATES:
            status = "not_a_candidate"
        elif len(pose_bins) >= 2:
            status = "corroborated_multiple_pose_bins"
        elif len(pose_bins) == 1:
            status = "corroborated_one_pose_bin"
        else:
            status = "not_corroborated"
        row["cross_bin_corroboration_status"] = status
        if status.startswith("corroborated"):
            events.append({
                "pair_id": row.get("pair_id"),
                "date": row.get("date_b"),
                "primary_pose_bin": row.get("pose_bin"),
                "support_pair_ids": [x.get("pair_id") for x in supports],
                "support_pose_bins": pose_bins,
                "independent_source_groups": source_groups,
                "status": status,
            })
    return {
        "schema": SCHEMA,
        "window_days": int(window_days),
        "event_count": len(events),
        "events": events,
        "policy": "Secondary corroboration only; never changes primary same-bin measurements or thresholds.",
    }


def aggregate_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate same target-date observations without pretending files are independent."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair_type") == "adjacent" and row.get("date_b"):
            groups[str(row["date_b"])].append(row)
    out: list[dict[str, Any]] = []
    for event_date, group in sorted(groups.items()):
        poses = sorted({str(r.get("pose_bin")) for r in group})
        known_sources = sorted({str(r.get("source_group_b")) for r in group if str(r.get("source_group_b") or "unknown") != "unknown"})
        candidates = [r for r in group if str(r.get("status")) in CANDIDATE_STATES]
        quality_limited = sum(bool(r.get("quality_limited")) for r in group)
        out.append({
            "event_date": event_date,
            "comparison_count": len(group),
            "pose_bin_count": len(poses),
            "pose_bins": "|".join(poses),
            "known_source_group_count": len(known_sources),
            "source_groups": "|".join(known_sources),
            "candidate_count": len(candidates),
            "quality_limited_count": quality_limited,
            "independence_status": "multiple_known_sources" if len(known_sources) >= 2 else ("single_known_source" if len(known_sources) == 1 else "source_unknown"),
            "candidate_pair_ids": "|".join(str(r.get("pair_id")) for r in candidates),
        })
    return out
