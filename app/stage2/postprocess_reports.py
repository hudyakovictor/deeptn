from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.stage1.utils import atomic_json, sha256_file, write_csv

POSTPROCESS_SCHEMA = "deeputin-stage2-postprocess-v1.0"
FORBIDDEN_PUBLIC_TERMS = (
    "двойник", "подмена", "силикон", "маска", "другой человек",
    "double", "impostor", "silicone", "mask", "different person",
)
CANDIDATE_STATES = {
    "persistent_geometric_change",
    "persistent_geometric_change_candidate",
    "alpha_id_change_candidate",
    "reversible_change_candidate",
    "rate_change_candidate",
    "persistent_rate_change_candidate",
    "same_day_conflict_candidate",
    "quality_limited",
}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _write_manual_review_queue(out: Path, rows: list[dict[str, Any]]) -> int:
    queue: list[dict[str, Any]] = []
    for r in rows:
        state = str(r.get("evidence_state") or "")
        if state not in CANDIDATE_STATES:
            continue
        priority = 0.0
        priority += _num(r.get("p95_point_z"))
        priority += 1.5 * _num(r.get("mesh_max_robust_z"))
        priority += 2.0 * int(bool(r.get("baseline_return")))
        priority += 1.0 * int(bool(r.get("quality_limited")))
        priority += 0.5 * _num(r.get("lead_priority"))
        queue.append({
            "pair_id": r.get("pair_id"),
            "review_priority": round(priority, 4),
            "evidence_state": state,
            "status": r.get("status"),
            "pose_bin": r.get("pose_bin"),
            "date_a": r.get("date_a"),
            "date_b": r.get("date_b"),
            "photo_a": r.get("photo_a"),
            "photo_b": r.get("photo_b"),
            "p95_point_z": r.get("p95_point_z"),
            "mesh_max_robust_z": r.get("mesh_max_robust_z"),
            "quality_limited": r.get("quality_limited"),
            "lead_overlap": r.get("lead_overlap"),
            "review_reason": "candidate_or_limited_evidence_requires_human_check",
        })
    queue.sort(key=lambda x: float(x.get("review_priority") or 0.0), reverse=True)
    write_csv(out / "manual_review_queue.csv", queue or [{"status": "no_review_candidates"}])
    return len(queue)


def _write_public_safety(out: Path, evidence_packets: list[dict[str, Any]]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for pkt in evidence_packets:
        text = str(pkt).lower()
        for term in FORBIDDEN_PUBLIC_TERMS:
            if term.lower() in text:
                hits.append({"pair_id": pkt.get("pair_id"), "term": term})
    report = {
        "schema": POSTPROCESS_SCHEMA,
        "status": "pass" if not hits else "fail",
        "forbidden_term_hit_count": len(hits),
        "hits": hits[:100],
        "policy": "Evidence packets must remain observation-based and avoid public identity/material verdict language.",
    }
    atomic_json(out / "public_safety_report.json", report)
    return report


def _write_degraded_modules(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for r in rows:
        if r.get("quality_limited"):
            counts["quality_limited"] += 1
        if str(r.get("mesh_status")) not in {"measured_uncalibrated", "measured_calibrated"}:
            counts["mesh_unavailable_or_insufficient"] += 1
        if str(r.get("mesh_calibration_status")) in {"insufficient_calibration", "unavailable"}:
            counts["mesh_calibration_insufficient"] += 1
        if str(r.get("texture_image_status")) != "measured":
            counts["texture_image_unavailable"] += 1
        if str(r.get("point_motion_status")) == "insufficient_calibration":
            counts["point_motion_calibration_insufficient"] += 1
    report = {
        "schema": POSTPROCESS_SCHEMA,
        "pair_count": len(rows),
        "counts": dict(counts),
        "degraded_fraction": {k: (v / max(len(rows), 1)) for k, v in counts.items()},
    }
    atomic_json(out / "degraded_modules.json", report)
    return report


def _write_mesh_shape_summary(out: Path, mesh_zones: list[dict[str, Any]]) -> int:
    by_zone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in mesh_zones:
        if r.get("mesh_zone_status") == "measured":
            by_zone[str(r.get("zone"))].append(r)
    rows: list[dict[str, Any]] = []
    for zone, vals in sorted(by_zone.items()):
        rows.append({
            "zone": zone,
            "count": len(vals),
            "median_planarity": float(np.median([_num(v.get("mesh_shape_planarity")) for v in vals])),
            "median_linearity": float(np.median([_num(v.get("mesh_shape_linearity")) for v in vals])),
            "median_geodesic_span_proxy": float(np.median([_num(v.get("mesh_geodesic_span_proxy")) for v in vals])),
            "median_point_to_plane_p95": float(np.median([_num(v.get("mesh_point_to_plane_p95")) for v in vals])),
            "policy": "shape/spectral-lite proxy summary; not calibrated verdict",
        })
    write_csv(out / "mesh_shape_summary.csv", rows or [{"status": "no_mesh_shape_rows"}])
    return len(rows)


def _write_texture_summary(out: Path, texture_zone_rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in texture_zone_rows if r.get("texture_zone_usable")]
    by_zone = Counter(str(r.get("zone")) for r in measured)
    report = {
        "schema": POSTPROCESS_SCHEMA,
        "texture_zone_row_count": len(texture_zone_rows),
        "usable_texture_zone_row_count": len(measured),
        "usable_by_zone": dict(by_zone),
        "policy": "image-space texture deltas are technical observations and require quality/exposure caveats",
    }
    atomic_json(out / "texture_summary.json", report)
    return report


def _write_status_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    status = Counter(str(r.get("status")) for r in rows)
    evidence = Counter(str(r.get("evidence_state")) for r in rows)
    all_keys = sorted(set(status) | set(evidence))
    write_csv(out / "status_summary.csv", [{"label": k, "status_count": status.get(k, 0), "evidence_state_count": evidence.get(k, 0)} for k in all_keys] or [{"status": "no_pairs"}])


def _write_gate_report(out: Path, rows: list[dict[str, Any]], changes: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = len(rows)
    if pair_count < 10:
        gate = "next_gate_10_photos_or_pairs"
    elif pair_count < 100:
        gate = "next_gate_100_photos_or_pairs"
    else:
        gate = "ready_for_full_run_if_error_rate_ok"
    report = {
        "schema": POSTPROCESS_SCHEMA,
        "pair_count": pair_count,
        "change_point_count": len(changes),
        "recommended_next_gate": gate,
        "quality_limited_fraction": sum(bool(r.get("quality_limited")) for r in rows) / max(pair_count, 1),
        "mesh_measured_fraction": sum(str(r.get("mesh_status")) in {"measured_uncalibrated", "measured_calibrated"} for r in rows) / max(pair_count, 1),
    }
    atomic_json(out / "gate_report.json", report)
    return report


def _write_artifact_index(out: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in {"artifact_index.json"}:
            entries.append({"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    report = {"schema": POSTPROCESS_SCHEMA, "artifact_count": len(entries), "artifacts": entries}
    atomic_json(out / "artifact_index.json", report)
    return report


def _write_stage3_input_summary(out: Path, rows: list[dict[str, Any]], changes: list[dict[str, Any]]) -> dict[str, Any]:
    top = sorted(rows, key=lambda r: (_num(r.get("p95_point_z")) + _num(r.get("mesh_max_robust_z"))), reverse=True)[:25]
    report = {
        "schema": POSTPROCESS_SCHEMA,
        "top_pair_ids": [r.get("pair_id") for r in top],
        "change_pair_ids": [c.get("pair_id") for c in changes],
        "report_policy": "Stage3 should render observations, limitations, and alternatives; no identity/material verdict.",
    }
    atomic_json(out / "stage3_input_summary.json", report)
    return report


def write_postprocess_reports(
    out: Path,
    *,
    rows: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    mesh_zones: list[dict[str, Any]],
    texture_zone_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    review_count = _write_manual_review_queue(out, rows)
    public_safety = _write_public_safety(out, evidence_packets)
    degraded = _write_degraded_modules(out, rows)
    mesh_shape_count = _write_mesh_shape_summary(out, mesh_zones)
    texture_summary = _write_texture_summary(out, texture_zone_rows)
    _write_status_summary(out, rows)
    gate = _write_gate_report(out, rows, changes)
    _write_stage3_input_summary(out, rows, changes)
    artifact_index = _write_artifact_index(out)
    evidence_chain = {
        "schema": POSTPROCESS_SCHEMA,
        "evidence_packet_count": len(evidence_packets),
        "manual_review_count": review_count,
        "public_safety_status": public_safety.get("status"),
        "artifact_count": artifact_index.get("artifact_count"),
    }
    atomic_json(out / "evidence_chain_manifest.json", evidence_chain)
    return {
        "manual_review_count": review_count,
        "public_safety_status": public_safety.get("status"),
        "mesh_shape_summary_count": mesh_shape_count,
        "texture_usable_zone_row_count": texture_summary.get("usable_texture_zone_row_count", 0),
        "gate": gate.get("recommended_next_gate"),
        "degraded_counts": degraded.get("counts", {}),
        "postprocess_outputs": [
            "manual_review_queue.csv", "public_safety_report.json", "degraded_modules.json",
            "mesh_shape_summary.csv", "texture_summary.json", "status_summary.csv",
            "gate_report.json", "stage3_input_summary.json", "artifact_index.json",
            "evidence_chain_manifest.json",
        ],
    }
