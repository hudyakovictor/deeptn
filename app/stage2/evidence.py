from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .metric_registry import metric_channel

EVIDENCE_SCHEMA = "deeputin-stage2-evidence-v1.0"

STATUS_TO_EVIDENCE_STATE = {
    "within_reconstruction_noise": "within_noise",
    "within_calibration_noise": "within_noise",
    "scattered_or_uncertain": "elevated_uncertain",
    "elevated_but_uncertain": "elevated_uncertain",
    "coherent_jump_candidate": "persistent_geometric_change_candidate",
    "persistent_geometric_change": "persistent_geometric_change",
    "baseline_return_candidate": "reversible_change_candidate",
    "alpha_id_jump_candidate": "alpha_id_change_candidate",
    "expression_dominated": "expression_dominated",
    "same_day_structural_conflict": "same_day_conflict_candidate",
    "biologically_improbable_rate_candidate": "rate_change_candidate",
    "persistent_biologically_improbable_change": "persistent_rate_change_candidate",
    "rapid_change_candidate": "rate_change_candidate",
    "persistent_rapid_change_candidate": "persistent_rate_change_candidate",
    "insufficient_visibility": "insufficient_visibility",
    "insufficient_calibration": "insufficient_calibration",
    "unsupported_pose": "unsupported_pose",
}


def evidence_state(status: str, *, quality_limited: bool = False) -> str:
    if quality_limited and status not in {"within_reconstruction_noise", "within_calibration_noise"}:
        return "quality_limited"
    return STATUS_TO_EVIDENCE_STATE.get(status, "elevated_uncertain")


def alternative_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("quality_limited"):
        reasons.append("low_or_missing_quality")
    if row.get("expression_influence", 0.0) >= 0.45:
        reasons.append("expression_or_soft_tissue_influence")
    if row.get("common_visible134", 999) < 60:
        reasons.append("limited_landmark_visibility")
    if row.get("matched_calibration_sets", 999) < 3:
        reasons.append("limited_matched_calibration")
    if row.get("pose_distance", 0.0) > 2.5:
        reasons.append("large_pose_distance")
    if row.get("baseline_return"):
        reasons.append("baseline_return_or_reversible_motion")
    if row.get("alpha_exp_status") == "elevated":
        reasons.append("expression_coefficient_jump")
    if row.get("alpha_id_status") == "elevated":
        reasons.append("alpha_id_shape_channel_jump_candidate")
    if row.get("texture_image_status") == "measured":
        reasons.append("image_space_texture_channel_available")
    if row.get("mesh_calibration_status") == "insufficient_calibration":
        reasons.append("dense_mesh_insufficient_calibration")
    elif row.get("mesh_status") == "measured_uncalibrated":
        reasons.append("dense_mesh_uncalibrated_support_only")
    if row.get("biological_rate_status"): 
        reasons.append("short_interval_rate_flag_requires_review")
    return reasons


@dataclass
class EvidencePacket:
    schema_version: str
    pair_id: str
    evidence_state: str
    status: str
    pair_type: str
    pose_bin: str
    photo_a: str
    photo_b: str
    date_a: str | None
    date_b: str | None
    primary_zone_or_family: str
    calibration: dict[str, Any]
    quality: dict[str, Any]
    measurements: dict[str, Any]
    registered_metric_channel: dict[str, Any]
    alternative_explanations: list[str]
    source_files: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def packet_from_pair(row: dict[str, Any]) -> dict[str, Any]:
    quality = {
        "quality_limited": bool(row.get("quality_limited")),
        "photo_a_texture_score": row.get("quality_texture_score_a"),
        "photo_b_texture_score": row.get("quality_texture_score_b"),
        "photo_a_quality_status": row.get("quality_status_a"),
        "photo_b_quality_status": row.get("quality_status_b"),
        "quality_zone_common_count": row.get("quality_zone_common_count"),
        "quality_zone_usable_common_count": row.get("quality_zone_usable_common_count"),
        "quality_zone_usable_common": row.get("quality_zone_usable_common"),
        "quality_zone_pair_limited": row.get("quality_zone_pair_limited"),
    }
    calibration = {
        "primary_robust_z": row.get("primary_robust_z"),
        "primary_calibration_p95": row.get("primary_calibration_p95"),
        "matched_calibration_sets": row.get("matched_calibration_sets"),
    }
    measurements = {
        "p95_point_z": row.get("p95_point_z"),
        "significant_point_fraction": row.get("significant_point_fraction"),
        "coherent_motion_fraction": row.get("coherent_motion_fraction"),
        "descriptor_top_families": row.get("descriptor_top_families"),
        "identity_only_motion_rmse": row.get("identity_only_motion_rmse"),
        "expression_influence": row.get("expression_influence"),
        "baseline_return": row.get("baseline_return"),
        "baseline_return_opposite_fraction": row.get("baseline_return_opposite_fraction"),
        "baseline_return_median_cosine": row.get("baseline_return_median_cosine"),
        "alpha_id_status": row.get("alpha_id_status"),
        "alpha_id_robust_z": row.get("alpha_id_robust_z"),
        "alpha_exp_status": row.get("alpha_exp_status"),
        "alpha_exp_robust_z": row.get("alpha_exp_robust_z"),
        "mesh_status": row.get("mesh_status"),
        "mesh_evidence_level": row.get("mesh_evidence_level"),
        "mesh_rmse": row.get("mesh_rmse"),
        "mesh_p95": row.get("mesh_p95"),
        "mesh_point_to_plane_rmse": row.get("mesh_point_to_plane_rmse"),
        "mesh_point_to_plane_p95": row.get("mesh_point_to_plane_p95"),
        "mesh_point_to_plane_signed_median": row.get("mesh_point_to_plane_signed_median"),
        "mesh_visible_fraction": row.get("mesh_visible_fraction"),
        "mesh_anchor_policy": row.get("mesh_anchor_policy"),
        "mesh_anchor_fraction": row.get("mesh_anchor_fraction"),
        "ldm134_anchor_policy": row.get("ldm134_anchor_policy"),
        "ldm134_anchor_count": row.get("ldm134_anchor_count"),
        "mesh_calibration_status": row.get("mesh_calibration_status"),
        "mesh_max_robust_z": row.get("mesh_max_robust_z"),
        "mesh_calibrated_elevated_count": row.get("mesh_calibrated_elevated_count"),
        "texture_image_status": row.get("texture_image_status"),
        "texture_image_usable_zone_count": row.get("texture_image_usable_zone_count"),
        "texture_image_max_laplacian_delta": row.get("texture_image_max_laplacian_delta"),
        "texture_image_max_gradient_delta": row.get("texture_image_max_gradient_delta"),
        "texture_image_max_lbp_chi2": row.get("texture_image_max_lbp_chi2"),
        "texture_image_max_glcm_contrast_delta": row.get("texture_image_max_glcm_contrast_delta"),
        "texture_image_max_high_frequency_delta": row.get("texture_image_max_high_frequency_delta"),
        "texture_image_max_entropy_delta": row.get("texture_image_max_entropy_delta"),
        "texture_image_max_gabor_delta": row.get("texture_image_max_gabor_delta"),
        "texture_structure_registered_zone_count": row.get("texture_structure_registered_zone_count"),
        "texture_structure_max_ridge_delta": row.get("texture_structure_max_ridge_delta"),
        "texture_structure_min_registered_ssim": row.get("texture_structure_min_registered_ssim"),
        "texture_image_backend": row.get("texture_image_backend"),
    }
    pkt = EvidencePacket(
        schema_version=EVIDENCE_SCHEMA,
        pair_id=str(row.get("pair_id")),
        evidence_state=str(row.get("evidence_state") or evidence_state(str(row.get("status", "")), quality_limited=bool(row.get("quality_limited")))),
        status=str(row.get("status")),
        pair_type=str(row.get("pair_type")),
        pose_bin=str(row.get("pose_bin")),
        photo_a=str(row.get("photo_a")),
        photo_b=str(row.get("photo_b")),
        date_a=row.get("date_a"),
        date_b=row.get("date_b"),
        primary_zone_or_family=str(row.get("descriptor_top_families") or "ldm134_motion"),
        calibration=calibration,
        quality=quality,
        measurements=measurements,
        registered_metric_channel=metric_channel(row),
        alternative_explanations=alternative_reasons(row),
        source_files={"motion_file": row.get("motion_file")},
    )
    return pkt.to_dict()
