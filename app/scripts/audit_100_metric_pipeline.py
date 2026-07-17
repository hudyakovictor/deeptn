from __future__ import annotations

import csv
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.stage2.evidence import packet_from_pair
from app.stage2.metric_registry import METRICS, NAMES, build_metric_catalog, validate_registry
from app.stage3.engine import Stage3Config, Stage3Engine

SCHEMA = "deeputin-100-metric-pipeline-audit-v1.0"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _source_corpus(root: Path) -> str:
    files = [p for p in (root / "stage2").glob("*.py") if p.name != "metric_registry.py"]
    return "\n".join(p.read_text(encoding="utf-8") for p in files)


def run_audit(root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = {name: round(1.0 + i / 1000.0, 6) for i, name in enumerate(NAMES)}
    row: dict[str, Any] = {
        "pair_id": "audit_pair_a__b", "pair_index": 1, "pair_type": "adjacent", "pose_bin": "frontal",
        "photo_a": "audit_a", "photo_b": "audit_b", "date_a": "2026-01-01", "date_b": "2026-01-02",
        "status": "within_reconstruction_noise", "evidence_state": "within_noise", "motion_file": "",
        **sentinel,
    }
    packet = packet_from_pair(row)
    catalog = build_metric_catalog([row])
    corpus = _source_corpus(root)

    with tempfile.TemporaryDirectory(prefix="app6_metric_audit_") as tmp:
        analysis = Path(tmp) / "analysis"; report = Path(tmp) / "report"
        analysis.mkdir()
        _write_csv(analysis / "pair_metrics.csv", [row])
        _write_csv(analysis / "zone_metrics.csv", [{"pair_id": row["pair_id"], "status": "measured"}])
        (analysis / "analysis_manifest.json").write_text(json.dumps({"schema_version": "audit", "main_record_count": 2, "calibration_dataset_count": 7, "created_at_utc": "2026-01-01T00:00:00Z"}), encoding="utf-8")
        (analysis / "change_points.json").write_text(json.dumps({"change_points": []}), encoding="utf-8")
        (analysis / "lead_registry.json").write_text(json.dumps({"status": "not_provided", "date_count": 0, "metric_count": 0}), encoding="utf-8")
        (analysis / "metric_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
        Stage3Engine(Stage3Config(analysis, report, True)).run()
        stage3 = json.loads((report / "report_data.json").read_text(encoding="utf-8"))
        csv_row = next(csv.DictReader((analysis / "pair_metrics.csv").open(newline="", encoding="utf-8")))

    stage3_row = stage3["pairs"][0]
    stage3_names = {entry["name"] for entry in stage3["metric_catalog"]["metrics"]}
    evidence = packet["registered_metric_channel"]
    catalog_by_name = {entry["name"]: entry for entry in catalog["metrics"]}
    analyses: list[dict[str, Any]] = []
    producer_exceptions = {
        # Produced by chronology/corroboration post-processors through dynamic row updates.
        "days_delta", "same_day", "chronology_rate_z", "biological_rate_z",
        "cross_bin_support_pose_count", "cross_bin_independent_source_count",
        # Produced by the generic landmark-stat expansion: ldm{106|134}_{stat}.
        "ldm106_median", "ldm106_p95", "ldm106_max", "ldm134_median", "ldm134_max",
    }
    for index, spec in enumerate(METRICS, 1):
        name = spec["name"]
        checks = {
            "canonical_snake_case": bool(re.fullmatch(r"[a-z][a-z0-9_]*", name)),
            "unique_registry_name": NAMES.count(name) == 1,
            "producer_declared": name in corpus or name in producer_exceptions,
            "evidence_transport": evidence.get(name) == sentinel[name],
            "csv_transport": abs(float(csv_row[name]) - sentinel[name]) < 1e-9,
            "catalog_transport": catalog_by_name[name]["status"] == "active",
            "stage3_pair_transport": abs(float(stage3_row[name]) - sentinel[name]) < 1e-9,
            "stage3_catalog_transport": name in stage3_names,
        }
        analyses.append({
            "analysis_id": f"A{index:03d}", "metric": name, "family": spec["family"],
            "requires": spec["requires"], "checks": checks,
            "status": "pass" if all(checks.values()) else "fail",
            "failed_checks": [key for key, value in checks.items() if not value],
        })
    failed = [item for item in analyses if item["status"] != "pass"]
    result = {
        "schema": SCHEMA,
        "status": "pass" if not failed and not validate_registry() else "fail",
        "analysis_count": len(analyses),
        "passed": len(analyses) - len(failed),
        "failed": len(failed),
        "registry_errors": validate_registry(),
        "transport_layers": ["producer", "pair_metrics.csv", "evidence_packet", "metric_catalog", "stage3_report_data"],
        "analyses": analyses,
    }
    (output_dir / "audit_100_metric_pipeline.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Аудит 100 ключевых метрик Stage 1 → Stage 2 → Stage 3", "",
        f"- Итог: **{result['status']}**", f"- Проверено: **{result['analysis_count']}**",
        f"- Успешно: **{result['passed']}**", f"- Ошибок: **{result['failed']}**", "",
        "| ID | Метрика | Семейство | Результат |", "|---|---|---|---|",
    ]
    lines += [f"| {x['analysis_id']} | `{x['metric']}` | {x['family']} | {x['status']} |" for x in analyses]
    if failed:
        lines += ["", "## Ошибки"] + [f"- `{x['metric']}`: {', '.join(x['failed_checks'])}" for x in failed]
    (output_dir / "AUDIT_100_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_audit(root, root / "qa_100_metrics")
    print(json.dumps({k: result[k] for k in ("status", "analysis_count", "passed", "failed", "registry_errors")}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "pass" else 1)
