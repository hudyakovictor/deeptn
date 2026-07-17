from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from statistics import median
from typing import Iterable

from app.stage2.workbench.core.contracts import canonical_hash
from app.stage2.workbench.recommendation.engine import PhotoSummary, RecommendedProposal, build_recommended


@dataclass(frozen=True)
class ChannelSuitability:
    photo_id: str
    geometry: str
    texture: str
    chronology: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    metric_id: str
    pose_bin: str
    value: float
    source_dataset: str
    source_group: str
    quality_sufficient: bool = True


@dataclass(frozen=True)
class CalibrationRange:
    metric_id: str
    pose_bin: str
    count: int
    median: float
    mad: float
    p95: float
    source_dataset_count: int
    source_group_count: int
    status: str


@dataclass(frozen=True)
class VerticalSlice1Result:
    recommendation: RecommendedProposal
    suitability: tuple[ChannelSuitability, ...]
    calibration_ranges: tuple[CalibrationRange, ...]
    quality_blockers: tuple[str, ...]
    calibration_blockers: tuple[str, ...]
    quality_ready: bool
    calibration_ready: bool
    result_hash: str


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires values")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def classify_photos(photos: Iterable[PhotoSummary], parameters: dict) -> tuple[ChannelSuitability, ...]:
    p = parameters["quality"]["admission"]
    result = []
    for photo in photos:
        common_reasons = []
        if photo.quality < p["minimum_quality"]:
            common_reasons.append("quality_below_minimum")
        if photo.visibility < p["minimum_visibility"]:
            common_reasons.append("visibility_below_minimum")
        if photo.abs_yaw > p["maximum_abs_yaw"] or photo.abs_pitch > p["maximum_abs_pitch"] or photo.abs_roll > p["maximum_abs_roll"]:
            common_reasons.append("pose_outside_range")
        if photo.alignment_residual is not None and photo.alignment_residual > p["maximum_alignment_residual"]:
            common_reasons.append("alignment_residual_high")
        geometry = "included" if not common_reasons else "quality_limited"
        texture_reasons = list(common_reasons)
        if photo.texture_roi_suitability is None:
            texture_reasons.append("texture_roi_missing")
        elif photo.texture_roi_suitability < p["minimum_texture_roi_suitability"]:
            texture_reasons.append("texture_roi_below_minimum")
        texture = "included" if not texture_reasons else "quality_limited"
        chronology = "included"
        result.append(ChannelSuitability(photo.photo_id, geometry, texture, chronology, tuple(sorted(set(texture_reasons + common_reasons)))))
    return tuple(sorted(result, key=lambda item: item.photo_id))


def build_calibration_ranges(records: Iterable[CalibrationRecord], *, minimum_records: int = 8, minimum_sources: int = 2) -> tuple[CalibrationRange, ...]:
    grouped: dict[tuple[str, str], list[CalibrationRecord]] = {}
    for record in records:
        if record.quality_sufficient:
            grouped.setdefault((record.metric_id, record.pose_bin), []).append(record)
    ranges = []
    for (metric_id, pose_bin), members in sorted(grouped.items()):
        values = [m.value for m in members]
        center = median(values)
        mad = median([abs(value - center) for value in values])
        datasets = {m.source_dataset for m in members}
        groups = {m.source_group for m in members}
        status = "active" if len(values) >= minimum_records and len(datasets) >= minimum_sources else "calibration_limited"
        ranges.append(CalibrationRange(metric_id, pose_bin, len(values), center, mad, _quantile(values, 0.95), len(datasets), len(groups), status))
    return tuple(ranges)


def run_vertical_slice1(
    photos: Iterable[PhotoSummary], calibration_records: Iterable[CalibrationRecord],
    *, manual_review_complete: bool, minimum_coverage: float = 0.80,
    minimum_calibration_records: int = 8, minimum_calibration_sources: int = 2,
) -> VerticalSlice1Result:
    photos = tuple(photos)
    recommendation = build_recommended(photos, minimum_coverage=minimum_coverage)
    suitability = classify_photos(photos, recommendation.parameters)
    calibration_ranges = build_calibration_ranges(calibration_records, minimum_records=minimum_calibration_records, minimum_sources=minimum_calibration_sources)
    geometry_included = sum(1 for item in suitability if item.geometry == "included")
    texture_known = sum(1 for photo in photos if photo.texture_roi_suitability is not None)
    quality_blockers = []
    if not manual_review_complete:
        quality_blockers.append("MANUAL_REVIEW_INCOMPLETE")
    if geometry_included / len(photos) < minimum_coverage * 0.70:
        quality_blockers.append("GEOMETRY_COVERAGE_TOO_LOW")
    if texture_known == 0:
        quality_blockers.append("TEXTURE_SUITABILITY_UNAVAILABLE")
    calibration_blockers = []
    if not calibration_ranges:
        calibration_blockers.append("NO_CALIBRATION_RANGES")
    if any(item.status != "active" for item in calibration_ranges):
        calibration_blockers.append("CALIBRATION_COVERAGE_LIMITED")
    if len({record.source_dataset for record in calibration_records}) < minimum_calibration_sources:
        calibration_blockers.append("CALIBRATION_SOURCE_COUNT_LOW")
    payload = {
        "proposal_hash": recommendation.proposal_hash,
        "suitability": [item.__dict__ for item in suitability],
        "calibration_ranges": [item.__dict__ for item in calibration_ranges],
        "quality_blockers": quality_blockers,
        "calibration_blockers": calibration_blockers,
    }
    return VerticalSlice1Result(
        recommendation, suitability, calibration_ranges,
        tuple(sorted(set(quality_blockers))), tuple(sorted(set(calibration_blockers))),
        not quality_blockers, not calibration_blockers, canonical_hash(payload),
    )
