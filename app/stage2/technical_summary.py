from __future__ import annotations

from collections import Counter
from typing import Any

SUMMARY_SCHEMA = "deeputin-stage2-technical-summary-v1.0"


def build_technical_summary(rows: list[dict[str, Any]], changes: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(str(r.get("status")) for r in rows)
    evidence_counts = Counter(str(r.get("evidence_state")) for r in rows)
    quality_limited = sum(bool(r.get("quality_limited")) for r in rows)
    mesh_measured = sum(str(r.get("mesh_status")) in {"measured_uncalibrated", "measured_calibrated"} for r in rows)
    texture_ready = sum(str(r.get("texture_pair_status")) == "texture_ready" for r in rows)
    return {
        "schema": SUMMARY_SCHEMA,
        "pair_count": len(rows),
        "change_point_count": len(changes),
        "status_counts": dict(status_counts),
        "evidence_state_counts": dict(evidence_counts),
        "quality_limited_pair_count": quality_limited,
        "mesh_measured_pair_count": mesh_measured,
        "texture_ready_pair_count": texture_ready,
        "manifest_core": {
            "schema_version": manifest.get("schema_version"),
            "main_record_count": manifest.get("main_record_count"),
            "calibration_record_count": manifest.get("calibration_record_count"),
            "mesh_calibration_status": manifest.get("mesh_calibration_status"),
        },
        "public_safety": "observations only; no identity/medical/material verdict",
    }
