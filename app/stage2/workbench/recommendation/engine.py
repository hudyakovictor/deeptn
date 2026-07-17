from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from statistics import median
from typing import Iterable

from app.stage2.workbench.core.contracts import canonical_hash


@dataclass(frozen=True)
class PhotoSummary:
    photo_id: str
    pose_bin: str
    quality: float
    visibility: float
    abs_yaw: float
    abs_pitch: float
    abs_roll: float
    alignment_residual: float | None = None
    texture_roi_suitability: float | None = None
    source_group: str | None = None


@dataclass(frozen=True)
class ParameterProposal:
    path: str
    value: float | int
    confidence: str
    reason_codes: tuple[str, ...]
    coverage_retained: float


@dataclass(frozen=True)
class RecommendedProposal:
    proposal_id: str
    parameters: dict
    parameter_proposals: tuple[ParameterProposal, ...]
    presets: dict[str, dict]
    warnings: tuple[str, ...]
    proposal_hash: str


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def _coverage(values: Iterable[float], predicate) -> float:
    values = tuple(values)
    return sum(1 for value in values if predicate(value)) / len(values) if values else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_recommended(photos: Iterable[PhotoSummary], *, minimum_coverage: float = 0.80) -> RecommendedProposal:
    photos = tuple(photos)
    if not photos:
        raise ValueError("cannot recommend config without photos")
    if not 0.5 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0.5, 1.0]")
    qualities = [p.quality for p in photos]
    visibility = [p.visibility for p in photos]
    yaw = [p.abs_yaw for p in photos]
    pitch = [p.abs_pitch for p in photos]
    roll = [p.abs_roll for p in photos]
    align = [p.alignment_residual for p in photos if p.alignment_residual is not None]
    texture = [p.texture_roi_suitability for p in photos if p.texture_roi_suitability is not None]
    lower_q = 1.0 - minimum_coverage
    quality_min = _quantile(qualities, lower_q)
    visibility_min = _quantile(visibility, lower_q)
    yaw_max = _quantile(yaw, minimum_coverage)
    pitch_max = _quantile(pitch, minimum_coverage)
    roll_max = _quantile(roll, minimum_coverage)
    alignment_max = _quantile(align, 0.95) if align else 1.0
    texture_min = _quantile(texture, lower_q) if texture else 0.0
    params = {
        "quality": {"admission": {
            "minimum_quality": round(quality_min, 6),
            "minimum_visibility": round(visibility_min, 6),
            "maximum_abs_yaw": round(yaw_max, 6),
            "maximum_abs_pitch": round(pitch_max, 6),
            "maximum_abs_roll": round(roll_max, 6),
            "maximum_alignment_residual": round(alignment_max, 6),
            "minimum_texture_roi_suitability": round(texture_min, 6),
        }},
        "geometry": {"decision": {"robust_z": 3.5}},
        "texture": {"decision": {"robust_z": 3.5}},
        "cross_pose": {"minimum_independent_sources": 2},
    }
    proposals = (
        ParameterProposal("quality.admission.minimum_quality", params["quality"]["admission"]["minimum_quality"], "high" if len(photos) >= 20 else "medium", ("coverage_quantile",), _coverage(qualities, lambda x: x >= quality_min)),
        ParameterProposal("quality.admission.minimum_visibility", params["quality"]["admission"]["minimum_visibility"], "high" if len(photos) >= 20 else "medium", ("coverage_quantile",), _coverage(visibility, lambda x: x >= visibility_min)),
        ParameterProposal("quality.admission.maximum_abs_yaw", params["quality"]["admission"]["maximum_abs_yaw"], "medium", ("pose_coverage_boundary",), _coverage(yaw, lambda x: x <= yaw_max)),
        ParameterProposal("quality.admission.maximum_abs_pitch", params["quality"]["admission"]["maximum_abs_pitch"], "medium", ("pose_coverage_boundary",), _coverage(pitch, lambda x: x <= pitch_max)),
        ParameterProposal("quality.admission.maximum_abs_roll", params["quality"]["admission"]["maximum_abs_roll"], "medium", ("pose_coverage_boundary",), _coverage(roll, lambda x: x <= roll_max)),
        ParameterProposal("quality.admission.maximum_alignment_residual", params["quality"]["admission"]["maximum_alignment_residual"], "medium" if align else "unavailable", ("alignment_p95",) if align else ("missing_alignment_data",), _coverage(align, lambda x: x <= alignment_max) if align else 0.0),
        ParameterProposal("quality.admission.minimum_texture_roi_suitability", params["quality"]["admission"]["minimum_texture_roi_suitability"], "medium" if texture else "unavailable", ("coverage_quantile",) if texture else ("missing_texture_data",), _coverage(texture, lambda x: x >= texture_min) if texture else 0.0),
    )
    def variant(strictness: float) -> dict:
        base = params["quality"]["admission"]
        return {"quality": {"admission": {
            "minimum_quality": round(_clamp(base["minimum_quality"] + 0.15 * strictness, 0.0, 1.0), 6),
            "minimum_visibility": round(_clamp(base["minimum_visibility"] + 0.12 * strictness, 0.0, 1.0), 6),
            "maximum_abs_yaw": round(max(0.0, base["maximum_abs_yaw"] - 10.0 * strictness), 6),
            "maximum_abs_pitch": round(max(0.0, base["maximum_abs_pitch"] - 8.0 * strictness), 6),
            "maximum_abs_roll": round(max(0.0, base["maximum_abs_roll"] - 8.0 * strictness), 6),
            "maximum_alignment_residual": round(max(0.0, base["maximum_alignment_residual"] * (1.0 - 0.20 * strictness)), 6),
            "minimum_texture_roi_suitability": round(_clamp(base["minimum_texture_roi_suitability"] + 0.15 * strictness, 0.0, 1.0), 6),
        }}}
    presets = {
        "recommended": params,
        "balanced": params,
        "strict": variant(1.0),
        "sensitive": variant(-0.7),
        "best_photos": variant(1.5),
    }
    warnings = []
    if len(photos) < 20:
        warnings.append("LOW_SAMPLE_COUNT")
    if not align:
        warnings.append("ALIGNMENT_DATA_UNAVAILABLE")
    if not texture:
        warnings.append("TEXTURE_DATA_UNAVAILABLE")
    proposal_hash = canonical_hash({"parameters": params, "presets": presets, "warnings": warnings})
    return RecommendedProposal("proposal:" + proposal_hash.split(":", 1)[1][:24], params, proposals, presets, tuple(warnings), proposal_hash)
