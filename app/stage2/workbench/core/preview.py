from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.stage2.workbench.core.contracts import canonical_hash
from app.stage2.workbench.core.dag import analyze_paths


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    op: Literal["set", "unset", "pin", "unpin", "restore_auto", "restore_preset"]
    path: str
    value: Any = None


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    schema_version: Literal["stage2-preview-request-v1.0"] = Field(default="stage2-preview-request-v1.0", alias="schema", serialization_alias="schema")
    request_id: str
    idempotency_key: str
    session_id: str
    dataset_lock_id: str
    base_run_id: str | None = None
    base_config_id: str
    config_patch: tuple[ConfigPatch, ...] = ()
    step: str
    mode: Literal["impact", "preview", "full_check", "compare"]
    scope: dict[str, Any] = {}
    client_revision: int = Field(ge=0)


@dataclass(frozen=True)
class DatasetImpactIndex:
    photo_count: int
    pair_count: int
    event_count: int
    parameter_to_photos: dict[str, frozenset[str]]
    parameter_to_pairs: dict[str, frozenset[str]]
    parameter_to_events: dict[str, frozenset[str]]


class PreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    schema_version: Literal["stage2-preview-response-v1.0"] = Field(default="stage2-preview-response-v1.0", alias="schema", serialization_alias="schema")
    request_id: str
    server_revision: int
    status: Literal["complete", "partial", "failed", "cancelled"]
    change_class: str
    effective_config_hash: str
    config_diff: tuple[dict[str, Any], ...]
    impact: dict[str, Any]
    cache: dict[str, Any]
    warnings: tuple[dict[str, Any], ...] = ()
    blockers: tuple[dict[str, Any], ...] = ()
    log_correlation_ids: tuple[str, ...] = ()


def analyze_change(
    request: PreviewRequest,
    index: DatasetImpactIndex,
    approved_steps: Iterable[str] = (),
    server_revision: int = 1,
) -> PreviewResponse:
    paths = tuple(patch.path for patch in request.config_patch)
    dag_impact = analyze_paths(paths, approved_steps)
    affected_photos: set[str] = set()
    affected_pairs: set[str] = set()
    affected_events: set[str] = set()
    for path in paths:
        affected_photos.update(index.parameter_to_photos.get(path, ()))
        affected_pairs.update(index.parameter_to_pairs.get(path, ()))
        affected_events.update(index.parameter_to_events.get(path, ()))
    impact = {
        **asdict(dag_impact),
        "affected_photo_count": len(affected_photos),
        "affected_pair_count": len(affected_pairs),
        "affected_event_count": len(affected_events),
        "affected_photo_ids": sorted(affected_photos),
        "affected_pair_ids": sorted(affected_pairs),
        "affected_event_ids": sorted(affected_events),
        "coverage_before": {"photos": index.photo_count, "pairs": index.pair_count},
        "requires_full_consistency_check": dag_impact.change_class.value != "A",
    }
    config_diff = tuple(patch.model_dump(mode="json") for patch in request.config_patch)
    blockers: list[dict[str, Any]] = []
    if dag_impact.change_class.value == "D":
        blockers.append({
            "code": "STAGE1_CHANGE_REQUIRES_NEW_DATASET_LOCK",
            "severity": "blocker",
            "message_ru": "Изменение относится к Stage 1 и требует новой версии замороженного датасета.",
        })
    response_hash = canonical_hash({
        "request": request.model_dump(mode="json"),
        "impact": impact,
    })
    return PreviewResponse(
        request_id=request.request_id,
        server_revision=server_revision,
        status="complete",
        change_class=dag_impact.change_class.value,
        effective_config_hash=response_hash,
        config_diff=config_diff,
        impact=impact,
        cache={
            "raw_value_hits": 0,
            "raw_value_misses": len(affected_pairs) if dag_impact.change_class.value == "C" else 0,
            "status_reclassification_count": len(affected_pairs) if dag_impact.change_class.value == "B" else 0,
        },
        blockers=tuple(blockers),
        log_correlation_ids=(f"preview:{request.request_id}",),
    )
