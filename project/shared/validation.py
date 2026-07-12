"""
DeepTN Validation System
========================
Soft validation for pipeline data integrity.

Instead of crashing on unexpected values, logs warnings and continues.
This catches data quality issues early without breaking the pipeline.

Usage:
    from deeputin.shared.validation import validate_texture_metrics, validate_info_json
    
    issues = validate_texture_metrics(metrics, photo_id="2020_01_01")
    for issue in issues:
        log.attention(issue)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .logging import setup_logger

log = setup_logger("validation")


# ─────────────────────────────────────────────
# Issue types
# ─────────────────────────────────────────────
class ValidationIssue:
    """A single validation issue."""

    def __init__(
        self,
        level: str,  # "attention", "warning", "error"
        field: str,
        message: str,
        value: Any = None,
        expected: Any = None,
        photo_id: str = "",
    ):
        self.level = level
        self.field = field
        self.message = message
        self.value = value
        self.expected = expected
        self.photo_id = photo_id

    def __str__(self):
        prefix = f"[{self.photo_id}] " if self.photo_id else ""
        val_str = f" (got {self.value}"
        if self.expected is not None:
            val_str += f", expected {self.expected}"
        val_str += ")"
        return f"{prefix}{self.field}: {self.message}{val_str}"

    def __repr__(self):
        return f"ValidationIssue({self.level}, {self.field!r})"


# ─────────────────────────────────────────────
# Texture metrics validation
# ─────────────────────────────────────────────

# Expected ranges for texture metrics (from 700 analyses)
TEXTURE_RANGES: Dict[str, Tuple[float, float, str]] = {
    # metric_name: (min, max, description)
    "tv_residual_sparsity": (0.5, 0.95, "TV residual sparsity"),
    "edge_tortuosity_mean": (0.8, 1.3, "Edge tortuosity"),
    "autocorr_decay_len": (5.0, 50.0, "Autocorrelation decay length"),
    "specular_elongation": (0.5, 30.0, "Specular elongation"),
    "glcm_diss_d3_aniso": (0.0, 0.5, "GLCM dissimilarity anisotropy"),
    "lacunarity": (1.0, 6.0, "Lacunarity"),
    "spectral_slope_beta": (1.0, 5.0, "Spectral slope beta"),
    "glrlm_sre": (0.1, 0.8, "GLRLM short run emphasis"),
    "glszm_small_area_emphasis": (0.2, 0.8, "GLSZM small area emphasis"),
    "fft_high_low_ratio": (0.0, 2.0, "FFT high/low ratio"),
    "wld_joint_entropy": (4.0, 7.5, "WLD joint entropy"),
    "lbp_r1_hist_entropy": (2.0, 4.5, "LBP histogram entropy"),
    "pore_density_r2_mpx": (5000.0, 60000.0, "Pore density per Mpx"),
    "hemoglobin_od_std": (0.0, 0.5, "Hemoglobin OD std"),
    "bimodality_ashman_D": (1.0, 6.0, "Bimodality Ashman D"),
    "pore_eccentricity_mean": (0.5, 1.0, "Pore eccentricity mean"),
    "ngtdm_coarseness": (0.0, 1.0, "NGTDM coarseness"),
    "seam_score": (0.0, 0.5, "Seam score"),
    "melanin_hemo_slope": (-2.0, 8.0, "Melanin-hemo slope"),
    "shannon_entropy_q32": (3.0, 6.0, "Shannon entropy Q32"),
    "specular_sharpness": (0.0, 10.0, "Specular sharpness"),
    "specular_dispersion": (0.0, 50.0, "Specular dispersion"),
    "sss_index": (-1.5, 1.5, "SSS index"),
    "gabor_f08_anisotropy": (0.0, 10.0, "Gabor anisotropy"),
    "dwt_haar_HH_LL_ratio": (0.0, 0.1, "DWT Haar HH/LL ratio"),
}

# Quality metric ranges
QUALITY_RANGES: Dict[str, Tuple[float, float]] = {
    "overall_quality": (0.0, 1.0),
    "sharpness_score": (0.0, 10000.0),
    "noise_level": (0.0, 15.0),
    "jpeg_blockiness": (0.5, 5.0),
}


def validate_texture_metrics(
    metrics: Dict[str, Any],
    photo_id: str = "",
    strict: bool = False,
) -> List[ValidationIssue]:
    """
    Validate texture metrics dict.
    
    Returns list of issues (empty if all OK).
    Does NOT raise exceptions — logs issues and continues.
    
    Args:
        metrics: dict of texture metric values
        photo_id: photo identifier for logging
        strict: if True, log as "warning" instead of "attention"
    """
    issues: List[ValidationIssue] = []
    level = "warning" if strict else "attention"

    # Check required metrics exist
    required = [
        "tv_residual_sparsity",
        "edge_tortuosity_mean",
        "autocorr_decay_len",
    ]
    for key in required:
        if key not in metrics:
            issues.append(ValidationIssue(
                "warning", key, "Required metric missing", photo_id=photo_id
            ))

    # Check each metric value
    for key, value in metrics.items():
        if key.startswith("_") or key.startswith("q_"):
            continue  # skip internal and quality metrics

        if value is None:
            issues.append(ValidationIssue(
                level, key, "Metric is None", value=None, photo_id=photo_id
            ))
            continue

        if not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                level, key, f"Non-numeric type: {type(value).__name__}",
                value=value, photo_id=photo_id
            ))
            continue

        if not math.isfinite(value):
            issues.append(ValidationIssue(
                level, key, "NaN or Inf", value=value, photo_id=photo_id
            ))
            continue

        # Check range
        if key in TEXTURE_RANGES:
            lo, hi, desc = TEXTURE_RANGES[key]
            if value < lo or value > hi:
                # Check if WAY out of range (error) or slightly out (attention)
                range_size = hi - lo
                if value < lo - range_size or value > hi + range_size:
                    issues.append(ValidationIssue(
                        "warning", key, f"Way outside expected range [{lo}, {hi}]",
                        value=f"{value:.4f}", expected=f"[{lo}, {hi}]",
                        photo_id=photo_id
                    ))
                else:
                    issues.append(ValidationIssue(
                        "attention", key, f"Outside expected range [{lo}, {hi}]",
                        value=f"{value:.4f}", expected=f"[{lo}, {hi}]",
                        photo_id=photo_id
                    ))

    return issues


# ─────────────────────────────────────────────
# Quality metrics validation
# ─────────────────────────────────────────────
def validate_quality_metrics(
    quality: Dict[str, Any],
    photo_id: str = "",
) -> List[ValidationIssue]:
    """Validate quality metrics."""
    issues: List[ValidationIssue] = []

    for key, (lo, hi) in QUALITY_RANGES.items():
        value = quality.get(key)
        if value is None:
            issues.append(ValidationIssue(
                "attention", key, "Quality metric missing", photo_id=photo_id
            ))
            continue

        if not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(ValidationIssue(
                "warning", key, f"Invalid quality value: {value}",
                photo_id=photo_id
            ))
            continue

        if value < lo or value > hi:
            issues.append(ValidationIssue(
                "attention", key, f"Outside expected [{lo}, {hi}]",
                value=f"{value:.3f}", photo_id=photo_id
            ))

    return issues


# ─────────────────────────────────────────────
# info.json validation
# ─────────────────────────────────────────────
REQUIRED_INFO_FIELDS = [
    "photo_id",
    "dataset",
    "pose",
    "quality",
]

REQUIRED_POSE_FIELDS = [
    "bucket",
    "yaw",
    "pitch",
    "roll",
]


def validate_info_json(
    info: Dict[str, Any],
    photo_id: str = "",
) -> List[ValidationIssue]:
    """Validate info.json structure."""
    issues: List[ValidationIssue] = []

    # Required top-level fields
    for field in REQUIRED_INFO_FIELDS:
        if field not in info:
            issues.append(ValidationIssue(
                "warning", field, "Required field missing from info.json",
                photo_id=photo_id
            ))

    # Pose fields
    pose = info.get("pose", {})
    if isinstance(pose, dict):
        for field in REQUIRED_POSE_FIELDS:
            if field not in pose:
                issues.append(ValidationIssue(
                    "attention", f"pose.{field}", "Pose field missing",
                    photo_id=photo_id
                ))

        # Validate bucket value
        valid_buckets = [
            "frontal", "left_threequarter_light", "right_threequarter_light",
            "left_threequarter_medium", "right_threequarter_medium",
            "left_threequarter_deep", "right_threequarter_deep",
            "left_profile", "right_profile", "unknown",
        ]
        bucket = pose.get("bucket", "")
        if bucket and bucket not in valid_buckets:
            issues.append(ValidationIssue(
                "attention", "pose.bucket", f"Unknown bucket value",
                value=bucket, expected=f"one of {valid_buckets}",
                photo_id=photo_id
            ))

        # Validate angles
        for angle in ["yaw", "pitch", "roll"]:
            val = pose.get(angle)
            if val is not None and isinstance(val, (int, float)):
                if abs(val) > 180:
                    issues.append(ValidationIssue(
                        "attention", f"pose.{angle}",
                        f"Angle > 180°",
                        value=f"{val:.1f}°",
                        photo_id=photo_id
                    ))

    # Quality
    quality = info.get("quality", {})
    if isinstance(quality, dict):
        issues.extend(validate_quality_metrics(quality, photo_id))

    return issues


# ─────────────────────────────────────────────
# Geometry metrics validation
# ─────────────────────────────────────────────
def validate_geometry_metrics(
    metrics: Dict[str, Any],
    photo_id: str = "",
) -> List[ValidationIssue]:
    """Validate geometry metrics dict."""
    issues: List[ValidationIssue] = []

    if not metrics:
        issues.append(ValidationIssue(
            "warning", "geometry", "Empty geometry metrics dict",
            photo_id=photo_id
        ))
        return issues

    nan_count = 0
    zero_count = 0
    total = 0

    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        total += 1

        if value is None:
            nan_count += 1
            continue

        if isinstance(value, float):
            if not math.isfinite(value):
                nan_count += 1
            elif value == 0.0:
                zero_count += 1
        elif not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                "attention", key, f"Non-numeric geometry value: {type(value).__name__}",
                photo_id=photo_id
            ))

    # Report summary issues
    if total > 0:
        nan_pct = nan_count / total * 100
        if nan_pct > 50:
            issues.append(ValidationIssue(
                "warning", "geometry",
                f"{nan_pct:.0f}% of geometry metrics are NaN/None ({nan_count}/{total})",
                photo_id=photo_id
            ))
        elif nan_pct > 20:
            issues.append(ValidationIssue(
                "attention", "geometry",
                f"{nan_pct:.0f}% of geometry metrics are NaN/None ({nan_count}/{total})",
                photo_id=photo_id
            ))

    return issues


# ─────────────────────────────────────────────
# JSON cleanliness — remove unnecessary fields
# ─────────────────────────────────────────────

# Fields to REMOVE from texture_metrics.json (unused downstream)
TEXTURE_NOISE_FIELDS = {
    "texture_feature_weights_json",  # Internal debug field
    "texture_noise_sigma",           # Duplicate of noise_level
    "specular_dispersion",           # Always 0.0
    "specular_sharpness",            # Always 0.0
}

# Fields to REMOVE from info.json (unused downstream)
INFO_NOISE_FIELDS = set()  # Currently all fields are used

# Fields to KEEP in texture_metrics.json (clean list)
TEXTURE_KEEP_FIELDS = {
    # Core Tier 1 (12)
    "tv_residual_sparsity",
    "lacunarity",
    "autocorr_decay_len",
    "wld_joint_entropy",
    "fft_high_low_ratio",
    "spectral_slope_beta",
    "glcm_diss_d3_aniso",
    "pore_density_r2_mpx",
    "hemoglobin_od_std",
    "bimodality_ashman_D",
    "glszm_small_area_emphasis",
    "edge_tortuosity_mean",
    # Tier 2 (8)
    "glrlm_sre",
    "ngtdm_coarseness",
    "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy",
    "shannon_entropy_q32",
    "gabor_f08_anisotropy",
    "pore_eccentricity_mean",
    "specular_elongation",
    # Physical (4)
    "seam_score",
    "sss_index",
    "melanin_hemo_slope",
    # Quality
    "overall_quality",
    "sharpness_score",
    "noise_level",
    "jpeg_blockiness",
    # Assessment
    "texture_assessability",
    "texture_unreliable",
    # Classification result
    "texture_skin_hint",
    "texture_skin_confidence",
    "texture_real_prob",
    "texture_silicone_prob",
}


def clean_texture_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove unnecessary fields from texture metrics for clean JSON output.
    Returns a new dict with only useful fields.
    """
    cleaned = {}
    removed = []

    for key, value in metrics.items():
        if key in TEXTURE_NOISE_FIELDS:
            removed.append(key)
            continue
        if key.startswith("_"):
            removed.append(key)
            continue
        # Skip _weight suffix fields (Tier2 weight indicators)
        if key.endswith("_weight"):
            removed.append(key)
            continue
        cleaned[key] = value

    return cleaned


def clean_info_json(info: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unnecessary fields from info.json."""
    cleaned = {}
    for key, value in info.items():
        if key in INFO_NOISE_FIELDS:
            continue
        if key.startswith("_"):
            continue
        cleaned[key] = value
    return cleaned


def clean_geometry_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove NaN/None values from geometry metrics for clean JSON output.
    Keeps only finite numeric values.
    """
    cleaned = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        cleaned[key] = value
    return cleaned


# ─────────────────────────────────────────────
# Composite validation
# ─────────────────────────────────────────────
def validate_all(
    info: Optional[Dict] = None,
    texture: Optional[Dict] = None,
    geometry: Optional[Dict] = None,
    photo_id: str = "",
) -> Dict[str, List[ValidationIssue]]:
    """
    Run all validations and return grouped issues.
    
    Returns:
        {"info": [...], "texture": [...], "geometry": [...]}
    """
    results: Dict[str, List[ValidationIssue]] = {
        "info": [],
        "texture": [],
        "geometry": [],
    }

    if info is not None:
        results["info"] = validate_info_json(info, photo_id)
    if texture is not None:
        results["texture"] = validate_texture_metrics(texture, photo_id)
    if geometry is not None:
        results["geometry"] = validate_geometry_metrics(geometry, photo_id)

    return results


def has_critical_issues(issues: Dict[str, List[ValidationIssue]]) -> bool:
    """Check if any validation group has error-level issues."""
    for group in issues.values():
        for issue in group:
            if issue.level == "error":
                return True
    return False


def count_issues(issues: Dict[str, List[ValidationIssue]]) -> Dict[str, int]:
    """Count issues by level across all groups."""
    counts = {"attention": 0, "warning": 0, "error": 0}
    for group in issues.values():
        for issue in group:
            if issue.level in counts:
                counts[issue.level] += 1
    return counts
