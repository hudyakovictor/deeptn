from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Iterable

from app.stage2.workbench.core.contracts import canonical_hash


@dataclass(frozen=True)
class PairEvidence:
    evidence_id: str
    pair_id: str
    timestamp: datetime
    pose_bin: str
    source_group: str
    channel: str
    metric_family: str
    face_zone: str
    normalized_value: float | None
    threshold: float
    quality_state: str = "sufficient"


@dataclass(frozen=True)
class ClassifiedEvidence:
    evidence: PairEvidence
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class FusionItem:
    pair_id: str
    geometry_status: str
    texture_status: str
    combined_status: str
    member_evidence_ids: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CrossPoseEvent:
    event_key: str
    date: str
    face_zone: str
    pose_bins: tuple[str, ...]
    source_groups: tuple[str, ...]
    member_evidence_ids: tuple[str, ...]
    status: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class Observation:
    photo_id: str
    timestamp: datetime
    pose_bin: str
    source_group: str
    metric_id: str
    value: float
    quality_state: str = "sufficient"


@dataclass(frozen=True)
class ChronologyEvent:
    event_id: str
    event_type: str
    photo_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    magnitude: float
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisSnapshot:
    geometry: tuple[ClassifiedEvidence, ...]
    texture: tuple[ClassifiedEvidence, ...]
    fusion: tuple[FusionItem, ...]
    cross_pose: tuple[CrossPoseEvent, ...]
    chronology: tuple[ChronologyEvent, ...]
    result_hash: str


def classify_evidence(items: Iterable[PairEvidence]) -> tuple[ClassifiedEvidence, ...]:
    result = []
    for item in items:
        if item.normalized_value is None:
            status, reasons = "missing_input", ("normalized_value_missing",)
        elif item.quality_state != "sufficient":
            status, reasons = "quality_limited", ("quality_not_sufficient",)
        elif abs(item.normalized_value) >= item.threshold:
            status, reasons = "candidate_with_alternatives", ("threshold_exceeded",)
        else:
            status, reasons = "within_calibration_range", ("within_threshold",)
        result.append(ClassifiedEvidence(item, status, reasons))
    return tuple(sorted(result, key=lambda x: x.evidence.evidence_id))


def fuse_channels(items: Iterable[ClassifiedEvidence]) -> tuple[FusionItem, ...]:
    grouped: dict[str, list[ClassifiedEvidence]] = {}
    for item in items:
        grouped.setdefault(item.evidence.pair_id, []).append(item)
    result = []
    candidate = {"candidate_with_alternatives", "strong_technical_candidate"}
    for pair_id, members in sorted(grouped.items()):
        geometry = [m for m in members if m.evidence.channel == "geometry"]
        texture = [m for m in members if m.evidence.channel == "texture"]
        g_statuses = {m.status for m in geometry}
        t_statuses = {m.status for m in texture}
        g_candidate = bool(g_statuses & candidate)
        t_candidate = bool(t_statuses & candidate)
        limitations = sorted({m.status for m in members if m.status in {"quality_limited", "calibration_limited", "missing_input"}})
        if not geometry and not texture:
            combined = "not_applicable"
        elif not geometry:
            combined = "texture_only_candidate" if t_candidate else "one_channel_limited"
            limitations.append("geometry_missing")
        elif not texture:
            combined = "geometry_only_candidate" if g_candidate else "one_channel_limited"
            limitations.append("texture_missing")
        elif g_candidate and t_candidate:
            combined = "channels_agree_candidate"
        elif g_candidate or t_candidate:
            other_statuses = t_statuses if g_candidate else g_statuses
            if other_statuses & {"within_calibration_range"}:
                combined = "channels_conflict"
            else:
                combined = "one_channel_limited"
        elif g_statuses == {"within_calibration_range"} and t_statuses == {"within_calibration_range"}:
            combined = "both_within_range"
        else:
            combined = "one_channel_limited"
        result.append(FusionItem(
            pair_id, ",".join(sorted(g_statuses)) or "missing",
            ",".join(sorted(t_statuses)) or "missing", combined,
            tuple(sorted(m.evidence.evidence_id for m in members)), tuple(sorted(set(limitations))),
        ))
    return tuple(result)


def build_cross_pose_events(
    items: Iterable[ClassifiedEvidence], *, minimum_pose_bins: int = 2,
    minimum_independent_sources: int = 2,
) -> tuple[CrossPoseEvent, ...]:
    candidate = {"candidate_with_alternatives", "strong_technical_candidate"}
    grouped: dict[tuple[str, str], list[ClassifiedEvidence]] = {}
    for item in items:
        if item.status in candidate:
            key = (item.evidence.timestamp.date().isoformat(), item.evidence.face_zone)
            grouped.setdefault(key, []).append(item)
    events = []
    for (date, zone), members in sorted(grouped.items()):
        poses = tuple(sorted({m.evidence.pose_bin for m in members}))
        sources = tuple(sorted({m.evidence.source_group for m in members}))
        limitations = []
        if len(poses) < minimum_pose_bins:
            limitations.append("insufficient_pose_bins")
        if len(sources) < minimum_independent_sources:
            limitations.append("insufficient_independent_sources")
        status = "cross_pose_support" if not limitations else "insufficient_independence"
        events.append(CrossPoseEvent(
            f"crosspose:{date}:{zone}", date, zone, poses, sources,
            tuple(sorted(m.evidence.evidence_id for m in members)), status, tuple(limitations),
        ))
    return tuple(events)


def detect_rapid_changes(
    observations: Iterable[Observation], *, threshold: float, max_gap_days: int,
) -> tuple[ChronologyEvent, ...]:
    grouped: dict[tuple[str, str], list[Observation]] = {}
    for item in observations:
        if item.quality_state == "sufficient":
            grouped.setdefault((item.metric_id, item.pose_bin), []).append(item)
    events = []
    for (metric_id, pose_bin), members in sorted(grouped.items()):
        members.sort(key=lambda x: (x.timestamp, x.photo_id))
        for a, b in zip(members, members[1:]):
            gap = (b.timestamp - a.timestamp).days
            magnitude = abs(b.value - a.value)
            if 0 <= gap <= max_gap_days and magnitude >= threshold:
                event_hash = canonical_hash(["rapid", metric_id, pose_bin, a.photo_id, b.photo_id])
                events.append(ChronologyEvent(
                    "rapid:" + event_hash.split(":", 1)[1][:20], "rapid_change",
                    (a.photo_id, b.photo_id), a.timestamp, b.timestamp, magnitude,
                    "rapid_change_candidate", ("magnitude_above_threshold", "gap_within_limit"),
                ))
    return tuple(events)


def detect_persistence(
    observations: Iterable[Observation], *, baseline_count: int = 2,
    threshold: float, minimum_followups: int = 2,
) -> tuple[ChronologyEvent, ...]:
    members = sorted((x for x in observations if x.quality_state == "sufficient"), key=lambda x: (x.timestamp, x.photo_id))
    if len(members) < baseline_count + 1 + minimum_followups:
        return ()
    baseline = median([x.value for x in members[:baseline_count]])
    events = []
    for idx in range(baseline_count, len(members) - minimum_followups):
        current = members[idx]
        followups = members[idx + 1:idx + 1 + minimum_followups]
        if abs(current.value - baseline) >= threshold and all(abs(x.value - baseline) >= threshold for x in followups):
            event_hash = canonical_hash(["persistent", current.photo_id, *[x.photo_id for x in followups]])
            events.append(ChronologyEvent(
                "persistent:" + event_hash.split(":", 1)[1][:20], "persistent_change",
                tuple([current.photo_id] + [x.photo_id for x in followups]),
                current.timestamp, followups[-1].timestamp, abs(current.value - baseline),
                "persistent_change_candidate", ("followups_remain_outside_baseline",),
            ))
            break
    return tuple(events)


def detect_aba(
    observations: Iterable[Observation], *, forward_threshold: float,
    return_tolerance: float, max_total_gap_days: int | None = None,
) -> tuple[ChronologyEvent, ...]:
    members = sorted((x for x in observations if x.quality_state == "sufficient"), key=lambda x: (x.timestamp, x.photo_id))
    events = []
    for i in range(len(members) - 2):
        for j in range(i + 1, len(members) - 1):
            for k in range(j + 1, len(members)):
                a, b, a2 = members[i], members[j], members[k]
                if a.metric_id != b.metric_id or a.metric_id != a2.metric_id or a.pose_bin != b.pose_bin or a.pose_bin != a2.pose_bin:
                    continue
                total_gap = (a2.timestamp - a.timestamp).days
                if max_total_gap_days is not None and total_gap > max_total_gap_days:
                    continue
                forward = b.value - a.value
                backward = a2.value - b.value
                if abs(forward) >= forward_threshold and forward * backward < 0 and abs(a2.value - a.value) <= return_tolerance:
                    event_hash = canonical_hash(["aba", a.photo_id, b.photo_id, a2.photo_id])
                    events.append(ChronologyEvent(
                        "aba:" + event_hash.split(":", 1)[1][:20], "aba_return",
                        (a.photo_id, b.photo_id, a2.photo_id), a.timestamp, a2.timestamp,
                        abs(forward), "aba_candidate", ("direction_reversed", "returned_within_tolerance"),
                    ))
    return tuple(events)


def run_analysis_snapshot(
    pair_evidence: Iterable[PairEvidence], observations: Iterable[Observation],
    *, rapid_threshold: float = 3.0, rapid_max_gap_days: int = 120,
    persistence_threshold: float = 3.0, aba_forward_threshold: float = 3.0,
    aba_return_tolerance: float = 1.0,
) -> AnalysisSnapshot:
    pair_evidence = tuple(pair_evidence)
    observations = tuple(observations)
    classified = classify_evidence(pair_evidence)
    geometry = tuple(x for x in classified if x.evidence.channel == "geometry")
    texture = tuple(x for x in classified if x.evidence.channel == "texture")
    fusion = fuse_channels(classified)
    cross_pose = build_cross_pose_events(classified)
    chronology = (
        detect_rapid_changes(observations, threshold=rapid_threshold, max_gap_days=rapid_max_gap_days)
        + detect_persistence(observations, threshold=persistence_threshold)
        + detect_aba(observations, forward_threshold=aba_forward_threshold, return_tolerance=aba_return_tolerance)
    )
    payload = {
        "geometry": [x.__dict__ for x in geometry], "texture": [x.__dict__ for x in texture],
        "fusion": [x.__dict__ for x in fusion], "cross_pose": [x.__dict__ for x in cross_pose],
        "chronology": [x.__dict__ for x in chronology],
    }
    return AnalysisSnapshot(geometry, texture, fusion, cross_pose, chronology, canonical_hash(payload))
