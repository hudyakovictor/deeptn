"""
TextureSkinClassifierV3 — Quality-compensated rule-based classifier.

Key changes from V2:
1. Quality-compensated normalization for quality-dependent metrics
2. Rebalanced weights based on Cohen's d and quality-independence
3. Fixed lacunarity direction (silicone has LOWER lacunarity, not higher)
4. Reduced fft_high_low_ratio weight from 2.20 to 0.40 (highly quality-dependent)
5. Increased tv_residual_sparsity weight to 2.00 (best Cohen's d=-1.245)
6. Added edge_tortuosity_mean and autocorr_decay_len as primary metrics
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MODEL_PATH = Path(__file__).parent / "skin_classifier_v3.pkl"

RULES_ONLY_DEFAULT = True

# =============================================================================
# QUALITY REGRESSION COEFFICIENTS
# Linear regression: metric = slope * quality + intercept
# Computed on REAL (true skin) calibration data
# =============================================================================

QUALITY_REGRESSION: Dict[str, Tuple[float, float]] = {
    # (slope, intercept) — how metric changes per unit of overall_quality
    "fft_high_low_ratio":        (+0.050, 0.130),
    "spectral_slope_beta":       (-0.450, 2.950),
    "tv_residual_sparsity":      (+0.060, 0.770),
    "lacunarity":                (+0.700, 2.180),
    "wld_joint_entropy":         (+0.200, 6.020),
    "glrlm_sre":                 (+0.150, 0.390),
    "glszm_small_area_emphasis": (+0.120, 0.400),
    "lbp_r1_hist_entropy":       (+0.400, 3.000),
    "pore_eccentricity_mean":    (+0.150, 0.770),
    "ngtdm_coarseness":          (-0.500, 0.470),
    "bimodality_ashman_D":       (+0.300, 3.000),
}

# Reference quality for compensation (median of calibration dataset)
REFERENCE_QUALITY = 0.50

# =============================================================================
# RULE DEFINITIONS V3
# Sorted by priority (Cohen's d * quality_independence)
# =============================================================================

# (metric_name, center, scale, weight, direction_hint)
# center = boundary between real and silicone
# scale = normalization factor
# weight = importance in logit
# direction: "below_is_silicone" or "above_is_silicone"

RULES_V3: List[Tuple[str, float, float, float, str]] = [
    # ── Tier A: High |d| + quality-independent (|r_q| < 0.3) ──
    ("tv_residual_sparsity",    0.790, 0.030, 2.00, "below_is_silicone"),
    ("edge_tortuosity_mean",    1.020, 0.040, 1.50, "below_is_silicone"),
    ("autocorr_decay_len",     25.0,   8.0,   1.20, "above_is_silicone"),

    # ── Tier B: Good |d|, moderate quality-dependence (compensated) ──
    ("glrlm_sre",               0.380, 0.100, 0.80, "below_is_silicone"),
    ("glszm_small_area_emphasis", 0.420, 0.060, 0.70, "below_is_silicone"),
    ("spectral_slope_beta",     3.20,  0.55,  0.60, "above_is_silicone"),
    ("lacunarity",              1.90,  0.35,  0.50, "below_is_silicone"),  # FIXED: was above!
    ("lbp_r1_hist_entropy",     3.15,  0.20,  0.50, "below_is_silicone"),

    # ── Tier C: Quality-robust supplementary ──
    ("specular_elongation",     2.50,  3.00,  0.40, "above_is_silicone"),
    ("glcm_diss_d3_aniso",      0.07,  0.055, 0.35, "above_is_silicone"),
    ("seam_score",              0.03,  0.07,  0.30, "above_is_silicone"),
    ("pore_eccentricity_mean",  0.84,  0.05,  0.25, "below_is_silicone"),

    # ── Tier D: Reduced weight (quality-dependent or low |d|) ──
    ("fft_high_low_ratio",      0.060, 0.050, 0.40, "below_is_silicone"),  # ↓ from 2.20
    ("wld_joint_entropy",       6.05,  0.35,  0.25, "below_is_silicone"),  # ↓ from 0.55
    ("pore_density_r2_mpx",   24000.0, 8000.0, 0.15, "below_is_silicone"), # ↓ from 0.55
    ("ngtdm_coarseness",        0.30,  0.15,  0.30, "above_is_silicone"),
    ("hemoglobin_od_std",       0.08,  0.04,  0.20, "below_is_silicone"),
    ("bimodality_ashman_D",     2.80,  0.50,  0.15, "below_is_silicone"),
]

# =============================================================================
# QUALITY GATE THRESHOLDS (unchanged from V2)
# =============================================================================

QUALITY_GATE_THRESHOLDS = {
    "overall_quality_min": 0.28,
    "sharpness_score_min": 25.0,
    "noise_level_max": 8.0,
    "jpeg_blockiness_max": 2.0,
}

EXTREME_QUALITY_GATE = {
    "overall_quality_min": 0.12,
    "sharpness_score_min": 0.1,
    "noise_level_max": 12.0,
    "jpeg_blockiness_max": 3.0,
}

# Feature lists for compatibility
TEXTURE_CORE_V2 = [name for name, _, _, _, _ in RULES_V3]
PHYSICAL_AUX = ["seam_score", "specular_sharpness", "sss_index", "melanin_hemo_slope"]
QUALITY_FEATURES = [
    "overall_quality", "sharpness_score", "noise_level", "jpeg_blockiness",
    "q_laplacian_var", "q_tenengrad", "q_noise_sigma", "q_jpeg_blockiness", "q_valid_patches",
]


class TextureSkinClassifierV3:
    """
    Rule-based texture classifier V3 with quality compensation.
    
    Key improvements over V2:
    - Quality-compensated normalization eliminates false positives on low-Q photos
    - Rebalanced weights based on actual separability (Cohen's d)
    - Fixed inverted lacunarity direction
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        rules_only: bool = RULES_ONLY_DEFAULT,
        quality_compensation: bool = True,
    ) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        self.rules_only = bool(rules_only)
        self.quality_compensation = quality_compensation
        self._pipeline = None
        self._feature_names: List[str] = []
        if (not self.rules_only) and path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._pipeline = data["pipeline"]
                self._feature_names = data.get("feature_names", [])
            except Exception:
                self._pipeline = None
                self._feature_names = []

    def classify(
        self,
        metrics: Dict[str, float],
        quality: Dict[str, float] | Any = None,
        reference: Dict | None = None,
        pose: Dict | None = None,
    ) -> Dict[str, Any]:
        q = {}
        if quality:
            q = quality if isinstance(quality, dict) else quality.__dict__ if hasattr(quality, "__dict__") else {}

        hard_gated, soft_gated, quality_reason = self._quality_state(q)

        if hard_gated:
            return self._empty_result(quality_reason, hard_gated=True)

        if (not self.rules_only) and self._pipeline is not None:
            return self._ml_classify(metrics, q, soft_gated, quality_reason)

        return self._heuristic_classify(metrics, q, soft_gated, quality_reason)

    # ──────────────────────────────────────────────────────────────────────
    # QUALITY STATE
    # ──────────────────────────────────────────────────────────────────────

    def _quality_state(self, q: Dict[str, float]) -> Tuple[bool, bool, str]:
        overall_q = float(q.get("overall_quality", 1.0) or 0.0)
        sharpness = float(q.get("sharpness_score", q.get("q_laplacian_var", 1000.0)) or 0.0)
        noise = float(q.get("noise_level", q.get("q_noise_sigma", 0.0)) or 0.0)
        blockiness = float(q.get("jpeg_blockiness", q.get("q_jpeg_blockiness", 1.0)) or 0.0)

        hard = (
            overall_q < EXTREME_QUALITY_GATE["overall_quality_min"]
            or sharpness < EXTREME_QUALITY_GATE["sharpness_score_min"]
            or noise > EXTREME_QUALITY_GATE["noise_level_max"]
            or blockiness > EXTREME_QUALITY_GATE["jpeg_blockiness_max"]
        )
        soft = (
            overall_q < QUALITY_GATE_THRESHOLDS["overall_quality_min"]
            or sharpness < QUALITY_GATE_THRESHOLDS["sharpness_score_min"]
            or noise > QUALITY_GATE_THRESHOLDS["noise_level_max"]
            or blockiness > QUALITY_GATE_THRESHOLDS["jpeg_blockiness_max"]
        )
        reason = f"q={overall_q:.2f}, sharp={sharpness:.0f}, noise={noise:.1f}, jpeg={blockiness:.2f}"
        return hard, soft, reason

    # ──────────────────────────────────────────────────────────────────────
    # QUALITY COMPENSATION
    # ──────────────────────────────────────────────────────────────────────

    def _compensate_quality(self, metric_name: str, raw_value: float, quality: float) -> float:
        """
        Remove quality trend from metric value.
        Returns quality-compensated value at reference quality level.
        """
        if not self.quality_compensation:
            return raw_value
        if metric_name not in QUALITY_REGRESSION:
            return raw_value

        slope, intercept = QUALITY_REGRESSION[metric_name]
        # Expected value at actual quality
        expected_at_quality = slope * quality + intercept
        # Expected value at reference quality
        expected_at_ref = slope * REFERENCE_QUALITY + intercept
        # Correction: shift value to reference quality
        correction = expected_at_ref - expected_at_quality
        return raw_value + correction

    # ──────────────────────────────────────────────────────────────────────
    # HEURISTIC CLASSIFY (main logic)
    # ──────────────────────────────────────────────────────────────────────

    def _heuristic_classify(
        self,
        metrics: Dict[str, float],
        q: Dict[str, float],
        soft_gated: bool,
        quality_reason: str,
    ) -> Dict[str, Any]:
        rules: List[Dict[str, Any]] = []
        logit = 0.0
        overall_quality = float(q.get("overall_quality", 0.5) or 0.5)

        def get_metric(name: str, default: float) -> float:
            try:
                v = float(metrics.get(name, default))
                return v if np.isfinite(v) else default
            except Exception:
                return default

        for metric_name, center, scale, weight, direction in RULES_V3:
            raw = get_metric(metric_name, center)

            # Apply quality compensation
            compensated = self._compensate_quality(metric_name, raw, overall_quality)

            # Compute normalized evidence [-1, +1]
            if direction == "below_is_silicone":
                # Below center → evidence for silicone (positive logit)
                normalized = (center - compensated) / scale
            else:
                # Above center → evidence for silicone (positive logit)
                normalized = (compensated - center) / scale

            # Clip to [-1, +1]
            value = float(np.clip(normalized, -1.0, 1.0))
            contribution = float(weight * value)
            logit += contribution

            rules.append({
                "metric": metric_name,
                "raw_value": raw,
                "compensated_value": compensated,
                "signed_evidence": contribution,
                "normalized": value,
                "weight": weight,
                "direction": direction,
                "why": f"{direction} center={center:.4f} scale={scale:.4f}",
            })

        # Soft gate discount
        if soft_gated:
            logit *= 0.85

        # Sigmoid → probability
        prob_silicone = float(1.0 / (1.0 + np.exp(-logit)))
        prob_real = float(1.0 - prob_silicone)
        confidence = float(max(prob_real, prob_silicone))

        # Adaptive threshold
        adaptive_thresh = self._adaptive_threshold(q, soft_gated)

        if confidence < adaptive_thresh:
            hint = "unknown"
        else:
            hint = "silicone" if prob_silicone >= 0.5 else "real"

        strongest = sorted(rules, key=lambda r: abs(float(r["signed_evidence"])), reverse=True)[:5]

        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": prob_real, "silicone": prob_silicone},
            "used_metrics": [str(r["metric"]) for r in rules],
            "model_loaded": False,
            "heuristic_fallback": True,
            "rule_based": True,
            "quality_gated": False,
            "quality_soft_gated": soft_gated,
            "quality_reason": quality_reason if soft_gated else "ok",
            "quality_threshold": adaptive_thresh,
            "heuristic_logit": float(logit),
            "heuristic_top_rules": strongest,
            "quality_compensation_enabled": self.quality_compensation,
            "version": "V3",
        }

    # ──────────────────────────────────────────────────────────────────────
    # ADAPTIVE THRESHOLD
    # ──────────────────────────────────────────────────────────────────────

    def _adaptive_threshold(self, q: Dict[str, float], soft_gated: bool) -> float:
        overall_q = float(q.get("overall_quality", 1.0) or 0.0)
        base = 0.58  # ↓ from 0.62 (V3: more aggressive)
        if soft_gated:
            base = max(base, 0.65)
        # Quality penalty: lower quality → higher threshold needed
        penalty = 0.12 * max(0.0, 0.50 - overall_q)
        return base + penalty

    # ──────────────────────────────────────────────────────────────────────
    # ML CLASSIFY (unchanged from V2)
    # ──────────────────────────────────────────────────────────────────────

    def _ml_classify(
        self,
        metrics: Dict[str, float],
        q: Dict[str, float],
        soft_gated: bool,
        quality_reason: str,
    ) -> Dict[str, Any]:
        feature_names = TEXTURE_CORE_V2 + PHYSICAL_AUX
        vector = []
        used = []
        overall_quality = float(q.get("overall_quality", 0.5) or 0.5)

        for name in feature_names:
            val = metrics.get(name)
            if val is None or not np.isfinite(float(val)):
                val = 0.0
            else:
                val = float(val)
                # Apply quality compensation to ML features too
                val = self._compensate_quality(name, val, overall_quality)
            vector.append(val)
            used.append(name)

        X = np.array([vector], dtype=np.float64)
        proba = self._pipeline.predict_proba(X)[0]
        prob_real = float(proba[0])
        prob_silicone = float(proba[1])

        hint = "silicone" if prob_silicone >= prob_real else "real"
        confidence = max(prob_real, prob_silicone)

        adaptive_thresh = self._adaptive_threshold(q, soft_gated)
        if confidence < adaptive_thresh:
            hint = "unknown"

        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": prob_real, "silicone": prob_silicone},
            "used_metrics": used,
            "model_loaded": True,
            "heuristic_fallback": False,
            "rule_based": False,
            "quality_gated": False,
            "quality_soft_gated": soft_gated,
            "quality_reason": quality_reason if soft_gated else "ok",
            "quality_threshold": adaptive_thresh,
            "heuristic_logit": 0.0,
            "heuristic_top_rules": [],
            "quality_compensation_enabled": self.quality_compensation,
            "version": "V3",
        }

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _empty_result(self, quality_reason: str, hard_gated: bool = False) -> Dict[str, Any]:
        return {
            "texture_skin_hint": "unknown",
            "texture_skin_confidence": 0.0,
            "posterior": {"real": 0.5, "silicone": 0.5},
            "used_metrics": [],
            "model_loaded": self._pipeline is not None,
            "heuristic_fallback": True,
            "rule_based": True,
            "quality_gated": hard_gated,
            "quality_soft_gated": False,
            "quality_reason": f"HARD_GATED: {quality_reason}" if hard_gated else quality_reason,
            "heuristic_logit": 0.0,
            "heuristic_top_rules": [],
            "quality_compensation_enabled": self.quality_compensation,
            "version": "V3",
        }


# Backward compatibility alias
TextureSkinClassifier = TextureSkinClassifierV3
