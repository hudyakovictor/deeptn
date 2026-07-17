from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "deeputin-private-hypothesis-retest-v1.0"

# Private-only source inventory. Payloads are copied losslessly into a ledger;
# legacy thresholds/posteriors are never used as current calibration.
SOURCE_SPECS: tuple[tuple[str, str], ...] = (
    ("hypothesis_explanations", "final_inference/hypothesis_explanations.json"),
    ("photo_forensic_verdicts", "final_inference/photo_forensic_verdicts.json"),
    ("chronology_events", "final_inference/chronology_events.json"),
    ("top_hypothesis_breaks", "final_inference/top_hypothesis_breaks.json"),
    ("top_identity_breaks", "final_inference/top_identity_breaks.json"),
    ("top_evidence_packets", "final_inference/top_evidence_packets.json"),
    ("canonical_anomaly_index", "final_inference/canonical_anomaly_index.json"),
    ("cross_bucket_consensus", "final_inference/cross_bucket_consensus.json"),
    ("h0_contradictions", "final_inference/h0_contradictions.json"),
    ("morphing_candidates", "final_inference/morphing_candidates.json"),
    ("era_breakpoints", "final_inference/era_breakpoints.json"),
    ("bucket_consistency", "final_inference/bucket_consistency_report.json"),
    ("texture_natural_baseline", "final_inference/texture_natural_baseline.json"),
    ("mesh_noise_reduction", "final_inference/mesh_noise_reduction_report.json"),
    ("calibration_health", "final_inference/calibration_health_report.json"),
    ("metric_coverage", "final_inference/metric_coverage_matrix.json"),
)

HYPOTHESIS_FAMILIES: tuple[dict[str, str], ...] = (
    {"id": "H0_same_technical_consistency", "label": "same-subject technical consistency", "channel": "geometry+texture+chronology"},
    {"id": "H1_synthetic_texture_candidate", "label": "synthetic/retouch texture candidate", "channel": "texture+quality"},
    {"id": "H2_different_geometry_candidate", "label": "different-geometry candidate", "channel": "identity-only geometry"},
    {"id": "H_uncertain", "label": "insufficient or conflicting evidence", "channel": "quality+calibration"},
    {"id": "biological_corridor", "label": "longitudinal technical corridor", "channel": "chronology"},
    {"id": "shape_return_A_B_A", "label": "A-B-A baseline return", "channel": "motion"},
    {"id": "change_point", "label": "calibrated change point", "channel": "chronology"},
    {"id": "rapid_change", "label": "rapid change candidate", "channel": "chronology"},
    {"id": "frozen_face", "label": "unusually low variation candidate", "channel": "geometry+expression"},
    {"id": "face_version_cluster", "label": "stepwise technical cluster", "channel": "geometry+texture"},
    {"id": "timeline_alternation", "label": "alternating technical clusters", "channel": "chronology"},
    {"id": "regional_age_desynchronization", "label": "regional temporal desynchronization", "channel": "texture+geometry"},
    {"id": "anatomy_consistency", "label": "cross-zone consistency", "channel": "geometry"},
    {"id": "zero_baseline_return", "label": "reference baseline return", "channel": "motion"},
    {"id": "singular_date", "label": "multi-channel singular date", "channel": "corroboration"},
    {"id": "cross_bucket_consensus", "label": "independent pose-bin support", "channel": "corroboration"},
    {"id": "struct_texture_divergence", "label": "structure-texture divergence", "channel": "geometry+texture"},
    {"id": "morphing_candidate", "label": "morphing/transition artifact candidate", "channel": "texture+geometry+quality"},
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _extract_records(obj: Any) -> list[Any]:
    """Extract record collections while preserving each original payload."""
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return [obj]
    preferred = ("entries", "verdicts", "events", "break_entries", "impossible_flags", "models", "metrics")
    out: list[Any] = []
    for key in preferred:
        value = obj.get(key)
        if isinstance(value, list):
            out.extend(value)
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, list):
                    out.extend({"source_group": subkey, "payload": item} for item in subvalue)
                else:
                    out.append({"source_group": subkey, "payload": subvalue})
    if out:
        return out
    # Small reports remain one lossless record rather than being discarded.
    return [obj]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _candidate_keys(payload: Any) -> tuple[set[str], set[str], set[str]]:
    photos: set[str] = set(); dates: set[str] = set(); metrics: set[str] = set()
    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                lk = k.lower()
                if lk in {"photo_id", "photo_id_a", "photo_id_b", "calibration_photo_id"} and isinstance(v, str): photos.add(v)
                if lk in {"date", "date_str", "date_a", "date_b"} and isinstance(v, str): dates.add(v[:10])
                if ("metric" in lk or lk in {"carrier", "feature"}) and isinstance(v, str): metrics.add(v)
                walk(v, k)
        elif isinstance(value, list):
            for item in value: walk(item, key)
    walk(payload)
    return photos, dates, metrics


def _retest_record(payload: Any, current_pairs: list[dict[str, str]], current_metric_names: set[str]) -> dict[str, Any]:
    photos, dates, metrics = _candidate_keys(payload)
    matched = [r for r in current_pairs if r.get("photo_a") in photos or r.get("photo_b") in photos or r.get("date_a", "")[:10] in dates or r.get("date_b", "")[:10] in dates]
    metric_hits = sorted(metrics & current_metric_names)
    if matched:
        states = [r.get("evidence_state") or r.get("status") or "unknown" for r in matched]
        strong = sum("persistent" in s or "candidate" in s or "conflict" in s for s in states)
        limited = sum("insufficient" in s or "quality_limited" in s or "uncertain" in s for s in states)
        result = "technical_anomaly_candidate" if strong else ("inconclusive" if limited else "within_current_noise_or_no_strong_change")
        status = "retested_with_current_alignment"
        reason = "matched_current_pair_or_date"
    else:
        states = []
        result = "not_tested_no_current_matching_data"
        status = "pending_missing_current_data"
        reason = "legacy_target_preserved; requires new Stage-1 extraction/current Stage-2 pair"
    return {
        "status": status, "result": result, "reason": reason,
        "matched_pair_count": len(matched), "current_states": sorted(set(states)),
        "legacy_photo_target_count": len(photos), "legacy_date_target_count": len(dates),
        "legacy_metric_name_count": len(metrics), "current_metric_name_hits": metric_hits,
    }


@dataclass(frozen=True)
class PrivateHypothesisConfig:
    analysis_root: Path
    legacy_archive_root: Path
    output_dir: Path
    minimum_import_coverage: float = 0.95


class PrivateHypothesisEngine:
    """Private retest layer; deliberately excluded from the public Stage-3 report."""
    def __init__(self, config: PrivateHypothesisConfig): self.config = config

    def run(self) -> dict[str, Any]:
        c = self.config; c.output_dir.mkdir(parents=True, exist_ok=True)
        current_pairs = _read_csv(c.analysis_root / "pair_metrics.csv")
        metric_catalog_path = c.analysis_root / "metric_catalog.json"
        metric_catalog = json.loads(metric_catalog_path.read_text(encoding="utf-8")) if metric_catalog_path.is_file() else {"metrics": []}
        current_metric_names = {str(x.get("name")) for x in metric_catalog.get("metrics", []) if x.get("name")}
        legacy_manifest_path = c.legacy_archive_root / "final_inference" / "final_inference_manifest.json"
        legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8")) if legacy_manifest_path.is_file() else {}

        ledger_path = c.output_dir / "legacy_hypothesis_ledger.jsonl"
        result_path = c.output_dir / "hypothesis_retest_results.jsonl"
        source_reports: list[dict[str, Any]] = []; total = imported = retested = pending = 0
        with ledger_path.open("w", encoding="utf-8") as ledger, result_path.open("w", encoding="utf-8") as results:
            for source_name, relative in SOURCE_SPECS:
                path = c.legacy_archive_root / relative
                if not path.is_file():
                    source_reports.append({"source": source_name, "path": relative, "status": "missing", "record_count": 0})
                    continue
                obj = json.loads(path.read_text(encoding="utf-8")); records = _extract_records(obj)
                source_imported = 0; source_retested = 0
                for index, payload in enumerate(records):
                    total += 1
                    envelope = {"source": source_name, "source_path": relative, "source_index": index, "payload": payload}
                    ledger.write(json.dumps(envelope, ensure_ascii=False, allow_nan=False) + "\n")
                    imported += 1; source_imported += 1
                    retest = _retest_record(payload, current_pairs, current_metric_names)
                    result = {"source": source_name, "source_index": index, **retest}
                    results.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    if retest["status"] == "retested_with_current_alignment": retested += 1; source_retested += 1
                    else: pending += 1
                source_reports.append({"source": source_name, "path": relative, "status": "imported", "record_count": len(records), "imported_count": source_imported, "retested_count": source_retested, "sha256": _sha256(path)})
        coverage = imported / max(total, 1)
        manifest = {
            "schema": SCHEMA, "created_at_utc": _utc(),
            "privacy": "private_hypothesis_layer_not_in_public_stage3",
            "status": "complete" if coverage >= c.minimum_import_coverage else "coverage_below_minimum",
            "minimum_import_coverage": c.minimum_import_coverage,
            "source_record_count": total, "imported_record_count": imported, "import_coverage_fraction": coverage,
            "retested_with_current_alignment_count": retested, "pending_missing_current_data_count": pending,
            "hypothesis_family_count": len(HYPOTHESIS_FAMILIES), "hypothesis_families": list(HYPOTHESIS_FAMILIES),
            "legacy_alignment": {"model_version": legacy_manifest.get("model_version"), "reference_use_mesh_alignment": legacy_manifest.get("reference_use_mesh_alignment"), "mesh_config_hash": legacy_manifest.get("mesh_config_hash")},
            "current_alignment_policy": "iteratively_trimmed_kabsch_v1_no_scale",
            "range_policy": "All legacy thresholds, posteriors and numeric ranges are historical only. Current ranges must be re-estimated from current calibration after the alignment change.",
            "source_reports": source_reports,
            "outputs": [ledger_path.name, result_path.name, "private_hypothesis_manifest.json", "hypothesis_coverage.json"],
            "interpretation_boundary": "Technical retest only; no automatic identity, mask, surgery, material, or medical conclusion.",
        }
        coverage_report = {
            "schema": "deeputin-private-hypothesis-coverage-v1.0", "source_count": len(SOURCE_SPECS),
            "available_source_count": sum(r["status"] == "imported" for r in source_reports),
            "source_record_count": total, "imported_record_count": imported, "coverage_fraction": coverage,
            "current_pair_count": len(current_pairs), "current_metric_registry_count": len(current_metric_names),
            "retested_count": retested, "pending_count": pending,
            "coverage_claim": "lossless_import_of_available_hypothesis_records",
        }
        (c.output_dir / "private_hypothesis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (c.output_dir / "hypothesis_coverage.json").write_text(json.dumps(coverage_report, indent=2, ensure_ascii=False), encoding="utf-8")
        (c.output_dir / "README_PRIVATE.md").write_text(
            "# Private hypothesis retest\n\nThis directory is intentionally excluded from public Stage 3. Legacy claims are targets for retesting, not calibration truth. Numeric ranges are re-estimated after the new robust alignment. Pending means missing current data, not confirmation or rejection.\n",
            encoding="utf-8",
        )
        return manifest
