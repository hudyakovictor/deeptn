from __future__ import annotations

import math
import re
from typing import Any

SCHEMA = "deeputin-stage2-metric-registry-v1.0"

# Canonical Stage-2 contract. Exactly 100 high-value fields are tracked from
# production through pair CSV, evidence packet, metric catalog and Stage 3 data.
# Missing input is a status, never a fabricated zero measurement.
METRICS: tuple[dict[str, Any], ...] = tuple(
    {"name": name, "family": family, "requires": requires, "direction": direction}
    for family, requires, direction, names in (
        ("pair", "stage1_metadata", "metadata", (
            "pair_index", "days_delta", "same_day", "pose_distance", "common_visible106", "common_visible134",
            "coverage106", "coverage134", "matched_calibration_sets", "primary_robust_z", "primary_calibration_p95",
            "cross_bin_support_pose_count", "cross_bin_independent_source_count", "chronology_rate_z", "biological_rate_z",
        )),
        ("quality", "stage1_quality", "higher_is_better", (
            "quality_texture_score_a", "quality_texture_score_b", "quality_zone_common_count", "quality_zone_usable_common_count",
            "quality_zone_usable_common", "quality_zone_pair_limited", "quality_limited", "forehead_wrinkle_supported_a",
            "forehead_wrinkle_supported_b", "expression_influence",
        )),
        ("landmark", "stage1_landmarks", "higher_is_more_difference", (
            "ldm106_rmse", "ldm106_median", "ldm106_p95", "ldm106_max", "ldm134_rmse", "ldm134_median",
            "ldm134_p95", "ldm134_max", "identity_only_ldm134_rmse", "identity_only_motion_rmse",
            "ldm106_anchor_count", "ldm134_anchor_count", "alignment106_trimmed_count", "ldm134_alignment_trimmed_count",
            "alignment134_residual_before_median", "alignment134_residual_after_median", "significant_point_count",
            "significant_point_fraction", "coherent_motion_fraction", "median_point_z", "p95_point_z",
            "alpha_id_l2", "alpha_exp_l2", "alpha_id_robust_z", "alpha_exp_robust_z",
        )),
        ("descriptor", "stage1_landmarks", "higher_is_more_difference", (
            "descriptor_significant_fraction", "descriptor_landmark_fraction", "descriptor_p95_z",
            "baseline_return_opposite_fraction", "baseline_return_median_cosine", "baseline_return_magnitude_ratio",
            "baseline_return_common_vector_count", "cross_bin_support_count", "descriptor_top_families",
            "descriptor_top_counts",
        )),
        ("mesh", "stage1_dense_mesh", "higher_is_more_difference", (
            "mesh_common_vertex_count", "mesh_visible_fraction", "mesh_rmse", "mesh_median", "mesh_p95", "mesh_fit_vertex_count",
            "mesh_point_to_plane_rmse", "mesh_point_to_plane_median", "mesh_point_to_plane_p95",
            "mesh_point_to_plane_signed_median", "mesh_alignment_residual_before_median", "mesh_alignment_residual_after_median",
            "mesh_anatomical_zone_count", "mesh_anchor_fraction", "mesh_alignment_trimmed_count", "mesh_max_robust_z",
            "mesh_calibrated_elevated_count", "mesh_calibrated_metric_count", "mesh_shape_linearity", "mesh_shape_planarity",
        )),
        ("texture", "stage1_image_and_quality_masks", "higher_is_more_difference", (
            "texture_image_zone_count", "texture_image_usable_zone_count", "texture_image_max_laplacian_delta",
            "texture_image_max_gradient_delta", "texture_image_max_lbp_chi2", "texture_image_max_glcm_contrast_delta",
            "texture_image_max_high_frequency_delta", "texture_image_max_entropy_delta", "texture_image_max_gabor_delta",
            "texture_structure_registered_zone_count", "texture_structure_max_ridge_delta", "texture_structure_min_registered_ssim",
            "quality_zone_count", "usable_texture_zone_count", "min_usable_texture_score",
            "min_usable_texture_pixels", "texture_image_schema", "texture_pair_status",
            "texture_image_backend", "texture_image_status",
        )),
    )
    for name in names
)

if len(METRICS) != 100:
    raise RuntimeError(f"metric registry must contain exactly 100 entries, got {len(METRICS)}")

NAMES = tuple(m["name"] for m in METRICS)


def validate_registry() -> list[str]:
    errors: list[str] = []
    if len(set(NAMES)) != len(NAMES):
        errors.append("duplicate_metric_name")
    for name in NAMES:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            errors.append(f"invalid_metric_name:{name}")
        if name.endswith("_score") and "texture_score" not in name:
            errors.append(f"ambiguous_score_suffix:{name}")
    return errors


def _usable(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return True


def metric_channel(row: dict[str, Any]) -> dict[str, Any]:
    """Lossless registered metric projection for evidence/report transport."""
    return {name: row.get(name) for name in NAMES}


def build_metric_catalog(rows: list[dict[str, Any]], enabled: dict[str, bool] | None = None) -> dict[str, Any]:
    enabled = enabled or {}
    entries: list[dict[str, Any]] = []
    for spec in METRICS:
        name = spec["name"]
        values = [row.get(name) for row in rows if _usable(row.get(name))]
        if enabled.get(name, True) is False:
            status, reason = "disabled_by_config", "method_disabled_in_stage2_profile"
        elif not values:
            status, reason = "disabled_missing_data", f"required_input_unavailable:{spec['requires']}"
        else:
            status, reason = "active", "values_reached_output"
        entries.append({
            **spec,
            "status": status,
            "reason": reason,
            "pair_value_count": len(values),
            "pair_coverage_fraction": float(len(values) / max(len(rows), 1)),
        })
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "schema": SCHEMA,
        "registry_size": len(entries),
        "registry_errors": validate_registry(),
        "status_counts": counts,
        "metrics": entries,
        "policy": "Absent or disabled metrics retain canonical names and explicit status; missing values are never replaced by measurements.",
    }
