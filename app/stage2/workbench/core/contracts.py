from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False, by_alias=True)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False, by_alias=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def canonical_hash(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class DatasetStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    FROZEN = "frozen"
    INVALIDATED = "invalidated"


class StepStatus(str, Enum):
    DRAFT = "draft"
    COMPUTED = "computed"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class ResultStatus(str, Enum):
    WITHIN_RANGE = "within_calibration_range"
    CANDIDATE = "candidate_with_alternatives"
    STRONG_CANDIDATE = "strong_technical_candidate"
    QUALITY_LIMITED = "quality_limited"
    CALIBRATION_LIMITED = "calibration_limited"
    DISABLED = "disabled_by_config"
    MISSING = "missing_input"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"
    DRAFT_CHANGED = "draft_changed"


class ReviewState(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEW_STARTED = "review_started"
    AUTO_CHECKED = "auto_checked"
    HUMAN_REVIEWED = "human_reviewed"
    APPROVED_LOCKED = "approved_locked"


class DatasetLock(FrozenModel):
    schema_version: Literal["stage2-dataset-lock-v1.0"] = Field(default="stage2-dataset-lock-v1.0", alias="schema", serialization_alias="schema")
    dataset_id: str
    status: DatasetStatus
    manifest_hash: str
    photo_count: int = Field(ge=0)
    photo_ids_hash: str
    source_hashes_hash: str
    extraction_model: dict[str, Any]
    array_contracts: dict[str, Any]
    required_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    duplicate_groups: tuple[tuple[str, ...], ...] = ()
    frozen_at: datetime | None = None
    frozen_by: str | None = None

    @model_validator(mode="after")
    def frozen_has_no_missing_required(self) -> "DatasetLock":
        if self.status == DatasetStatus.FROZEN and self.missing_artifacts:
            raise ValueError("frozen dataset cannot have missing required artifacts")
        return self


class StepConfig(FrozenModel):
    schema_version: Literal["stage2-step-config-v1.0"] = Field(default="stage2-step-config-v1.0", alias="schema", serialization_alias="schema")
    config_id: str
    step: str
    base_config_id: str | None = None
    preset: str
    parameters: dict[str, Any]
    parameter_sources: dict[str, Literal["auto", "preset", "manual", "inherited"]]
    pinned_parameters: frozenset[str] = frozenset()
    enabled_metric_families: tuple[str, ...] = ()
    disabled_metric_families: tuple[str, ...] = ()
    dependency_config_hashes: dict[str, str] = {}
    status: StepStatus = StepStatus.DRAFT

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"config_id", "status"})
        return canonical_hash(payload)


class TimelineItem(FrozenModel):
    schema_version: Literal["stage2-timeline-item-v1.0"] = Field(default="stage2-timeline-item-v1.0", alias="schema", serialization_alias="schema")
    item_id: str
    item_type: Literal[
        "photo_point", "pair_point", "line_segment", "calibration_band",
        "event_marker", "interval_region", "ghost_photo", "ghost_value"
    ]
    track_id: str
    metric_id: str | None = None
    photo_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime | None = None
    x_anchor: Literal["photo_time", "pair_midpoint", "event_start", "event_end", "interval_span"]
    value: float | int | str | bool | None = None
    status: ResultStatus
    review_state: ReviewState = ReviewState.UNREVIEWED
    quality_state: str
    render: dict[str, Any] = {}
    trace_id: str

    @model_validator(mode="after")
    def scope_matches_photos(self) -> "TimelineItem":
        if self.item_type == "photo_point" and len(self.photo_ids) != 1:
            raise ValueError("photo_point requires exactly one photo")
        if self.item_type == "pair_point" and len(self.photo_ids) < 2:
            raise ValueError("pair_point requires at least A and B")
        if self.item_type == "pair_point" and self.x_anchor != "pair_midpoint":
            raise ValueError("pair_point must use pair_midpoint")
        return self


class MetricTrace(FrozenModel):
    schema_version: Literal["stage2-metric-trace-v1.0"] = Field(default="stage2-metric-trace-v1.0", alias="schema", serialization_alias="schema")
    trace_id: str
    metric_id: str
    subject_ids: tuple[str, ...]
    raw_value: float | int | str | bool | None
    normalized_value: float | int | None
    calibration_snapshot_id: str | None
    thresholds: dict[str, Any]
    quality_inputs: dict[str, Any]
    dependency_values: dict[str, Any]
    alternative_explanations: tuple[str, ...] = ()
    status: ResultStatus
    status_reasons: tuple[str, ...]
    config_id: str
    previous_trace_id: str | None = None
    change_reason: Literal[
        "quality_filter", "threshold_relaxation", "threshold_tightening",
        "new_calibration", "new_support", "missing_dependency", "manual_override"
    ] | None = None

    @model_validator(mode="after")
    def changed_trace_has_reason(self) -> "MetricTrace":
        if self.previous_trace_id and not self.change_reason:
            raise ValueError("changed trace requires change_reason")
        return self


class ApprovalGate(FrozenModel):
    schema_version: Literal["stage2-approval-gate-v1.0"] = Field(default="stage2-approval-gate-v1.0", alias="schema", serialization_alias="schema")
    gate_id: str
    step: str
    run_id: str
    checks: tuple[dict[str, Any], ...]
    blockers: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    manual_review_complete: bool
    status: Literal["blocked", "ready_for_review", "approved", "revoked", "superseded"]
    approved_at: datetime | None = None
    approved_by: str | None = None

    @model_validator(mode="after")
    def approved_is_clean(self) -> "ApprovalGate":
        if self.status == "approved" and (self.blockers or not self.manual_review_complete):
            raise ValueError("approved gate requires no blockers and completed manual review")
        return self


class ReleaseManifest(FrozenModel):
    schema_version: Literal["stage2-release-manifest-v1.0"] = Field(default="stage2-release-manifest-v1.0", alias="schema", serialization_alias="schema")
    release_id: str
    run_id: str
    dataset_lock_id: str
    dataset_hash: str
    approved_gate_ids: tuple[str, ...]
    config_hashes: dict[str, str]
    calibration_snapshot_ids: tuple[str, ...]
    metric_registry_hash: str
    artifact_hashes: dict[str, str]
    excluded_metrics: tuple[str, ...] = ()
    limitations: tuple[dict[str, Any], ...] = ()
    public_artifacts: tuple[str, ...] = ()
    private_artifacts: tuple[str, ...] = ()
    status: Literal["approved"] = "approved"

    @property
    def reproducibility_hash(self) -> str:
        return canonical_hash(self)
