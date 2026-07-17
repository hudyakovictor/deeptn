from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.stage2.leads import load_leads
from app.stage1.utils import atomic_json, sha256_file, sha256_json, write_csv

SCHEMA = "deeputin-stage2b-private-corroboration-v1.0"
SIGNIFICANT_STATES = {
    "persistent_geometric_change",
    "persistent_geometric_change_candidate",
    "rate_change_candidate",
    "persistent_rate_change_candidate",
    "texture_line_change_candidate",
    "reversible_change_candidate",
    "same_day_conflict_candidate",
}
WEAK_STATES = {"elevated_uncertain", "quality_limited", "expression_dominated"}
NO_SUPPORT_STATES = {"within_noise", "insufficient_visibility", "insufficient_calibration", "unsupported_pose"}


def utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Stage2BConfig:
    stage2_root: Path
    output_dir: Path
    prior_root: Path | None = None
    overwrite: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "policy": "private corroboration only; never modifies blind Stage2 measurements",
            "prior_root": str(self.prior_root) if self.prior_root else None,
        }


class Stage2BEngine:
    def __init__(self, cfg: Stage2BConfig):
        self.cfg = cfg

    def run(self) -> dict[str, Any]:
        out = self.cfg.output_dir
        if out.exists() and any(out.iterdir()) and not self.cfg.overwrite:
            raise FileExistsError(f"output exists: {out}")
        out.mkdir(parents=True, exist_ok=True)

        stage2_manifest_path = self.cfg.stage2_root / "analysis_manifest.json"
        evidence_path = self.cfg.stage2_root / "evidence_packets.json"
        if not stage2_manifest_path.is_file():
            raise FileNotFoundError(stage2_manifest_path)
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)

        stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8")).get("packets", [])
        if self.cfg.prior_root:
            leads = load_leads(self.cfg.prior_root)
        else:
            lead_path = self.cfg.stage2_root / "lead_registry.json"
            leads = json.loads(lead_path.read_text(encoding="utf-8")) if lead_path.is_file() else {"status": "not_provided", "dates": {}}

        date_registry = leads.get("dates", {}) or {}
        rows: list[dict[str, Any]] = []
        for pkt in evidence:
            dates = [d for d in (pkt.get("date_a"), pkt.get("date_b")) if d]
            matched = [(d, date_registry[d]) for d in dates if d in date_registry]
            if not matched:
                continue
            state = str(pkt.get("evidence_state") or "")
            priority = sum(int(x.get("priority", 0)) for _, x in matched)
            regions = sorted({r for _, x in matched for r in x.get("regions", [])})
            events = sorted({r for _, x in matched for r in x.get("events", [])})
            if state in SIGNIFICANT_STATES and priority >= 4:
                status = "confirmed_independently"
            elif state in SIGNIFICANT_STATES or state in WEAK_STATES:
                status = "partially_supported"
            elif state in NO_SUPPORT_STATES:
                status = "contradicted" if priority >= 10 and state == "within_noise" else "not_supported"
            else:
                status = "insufficient_data"
            rows.append({
                "pair_id": pkt.get("pair_id"),
                "corroboration_status": status,
                "evidence_state": state,
                "status": pkt.get("status"),
                "pose_bin": pkt.get("pose_bin"),
                "date_a": pkt.get("date_a"),
                "date_b": pkt.get("date_b"),
                "photo_a": pkt.get("photo_a"),
                "photo_b": pkt.get("photo_b"),
                "prior_dates": "|".join(d for d, _ in matched),
                "prior_priority": priority,
                "prior_regions": "|".join(regions),
                "prior_events": "|".join(events),
                "policy": "private_only_does_not_modify_blind_stage2",
            })

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["corroboration_status"]] = counts.get(r["corroboration_status"], 0) + 1

        write_csv(out / "corroboration_results.csv", rows or [{"status": "no_prior_overlap"}])
        atomic_json(out / "private_summary.json", {
            "schema_version": SCHEMA,
            "created_at_utc": utc(),
            "stage2_manifest_sha256": sha256_file(stage2_manifest_path),
            "stage2_evidence_sha256": sha256_file(evidence_path),
            "stage2_schema": stage2_manifest.get("schema_version"),
            "prior_status": leads.get("status"),
            "prior_date_count": leads.get("date_count", len(date_registry)),
            "evidence_packet_count": len(evidence),
            "prior_overlap_pair_count": len(rows),
            "status_counts": counts,
            "policy": "This private module checks whether frozen blind Stage2 evidence overlaps prior leads. It never changes raw residuals, thresholds, or public conclusions.",
        })
        manifest = {
            "schema_version": SCHEMA,
            "status": "complete",
            "created_at_utc": utc(),
            "config_hash": sha256_json(self.cfg.payload()),
            "stage2_manifest_sha256": sha256_file(stage2_manifest_path),
            "prior_status": leads.get("status"),
            "corroboration_row_count": len(rows),
            "outputs": ["corroboration_results.csv", "private_summary.json"],
        }
        atomic_json(out / "stage2b_manifest.json", manifest)
        return manifest
