"""
TextureSkinClassifierV4 — Final optimized classifier based on 100+ statistical tests.

KEY FINDINGS IMPLEMENTED:
1. composite_3best (tv + edge_tort + autocorr) gives AUC=0.891, d=1.716 — BEST single feature
2. Leave-one-out ablation: spectral_slope_beta, glrlm_sre, glszm_sae, lbp_entropy are HARMFUL
3. Trimmed statistics show specular_elongation d jumps from 0.533 to 0.981
4. Chronological trend: silicone masks improving (specular↑, tortuosity↓ over years)
5. Quality-stratified: tv_sparsity, edge_tort, autocorr stable across quality bins
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

# =============================================================================
# FINAL METRICS CONFIG V4
# Based on ablation study + Cohen's d + quality-independence
# =============================================================================

# PRIMARY: Essential metrics (removing them hurts AUC)
PRIMARY_V4: List[Tuple[str, float, float, float, str, bool]] = [
    # (name, center, scale, weight, direction, quality_safe)
    ("tv_residual_sparsity",    0.796, 0.025, 2.50, "below_is_silicone", False),
    ("edge_tortuosity_mean",    1.035, 0.035, 2.20, "below_is_silicone", True),
    ("autocorr_decay_len",     24.0,   7.0,   1.80, "above_is_silicone", True),
    ("specular_elongation",     2.957, 2.50,  1.20, "above_is_silicone", True),
]

# SECONDARY: Helpful but less critical
SECONDARY_V4: List[Tuple[str, float, float, float, str, bool]] = [
    ("glcm_diss_d3_aniso",      0.068, 0.050, 0.70, "above_is_silicone", True),
    ("lacunarity",              2.050, 0.350, 0.60, "below_is_silicone", False),
    ("melanin_hemo_slope",      3.780, 1.50,  0.50, "below_is_silicone", True),
    ("pore_eccentricity_mean",  0.840, 0.040, 0.40, "below_is_silicone", False),
]

# HARMFUL (from ablation): DO NOT USE
# spectral_slope_beta  — Δ AUC = +0.005 when removed (HARMFUL)
# glrlm_sre           — Δ AUC = +0.008 when removed (HARMFUL)  
# glszm_small_area_emphasis — Δ AUC = +0.006 when removed (HARMFUL)
# lbp_r1_hist_entropy  — Δ AUC = +0.021 when removed (VERY HARMFUL)
# fft_high_low_ratio   — r_q(silicone) = +0.802 (DANGEROUS)

# SUPPLEMENTARY (low weight, context only)
SUPPLEMENTARY_V4: List[Tuple[str, float, float, float, str, bool]] = [
    ("seam_score",              0.021, 0.060, 0.30, "above_is_silicone", True),
    ("hemoglobin_od_std",       0.100, 0.040, 0.20, "below_is_silicone", True),
    ("ngtdm_coarseness",        0.208, 0.120, 0.25, "above_is_silicone", False),
]

# =============================================================================
# DERIVED COMPOSITE FEATURES (discovered in Analysis 8)
# =============================================================================
# composite_3best = ((1 - tv)/0.03 + (1.05 - et)/0.04 + (ac - 20)/8) / 3
# AUC = 0.891, Cohen's d = 1.716 — BEST single feature!

COMPOSITE_CENTER = 2.95  # Optimized: shifted toward silicone to reduce real FP
COMPOSITE_SCALE = 0.40   # Optimized: tighter scale for better separation
COMPOSITE_WEIGHT = 2.5   # Optimized via grid search

# ratio_tv_x_et = tv_residual_sparsity * edge_tortuosity_mean  
# AUC = 0.857, d = 1.488
RATIO_TV_ET_CENTER = 0.815
RATIO_TV_ET_SCALE = 0.040
RATIO_TV_ET_WEIGHT = 1.5

# ratio_ac_x_et_inv = autocorr_decay_len * (1.1 - edge_tortuosity_mean)
# AUC = 0.824, d = 1.299
RATIO_AC_ET_INV_CENTER = 1.80
RATIO_AC_ET_INV_SCALE = 1.0
RATIO_AC_ET_INV_WEIGHT = 1.0


# =============================================================================
# QUALITY REGRESSION (for compensation of non-safe metrics)
# =============================================================================
QUALITY_REGRESSION_V4 = {
    "tv_residual_sparsity":  (+0.055, 0.772),
    "lacunarity":            (+0.550, 2.150),
    "pore_eccentricity_mean": (+0.120, 0.780),
    "ngtdm_coarseness":      (-0.400, 0.430),
}

# Quality gate thresholds
EXTREME_QUALITY_GATE = {
    "overall_quality_min": 0.12,
    "sharpness_score_min": 0.1,
    "noise_level_max": 12.0,
    "jpeg_blockiness_max": 3.0,
}


class TextureSkinClassifierV4:
    """
    Final optimized texture classifier V4.
    
    Architecture:
    1. Compute composite_3best (tv + edge_tort + autocorr) — strongest single feature
    2. Compute ratio features — capture metric interactions
    3. Add individual metrics with ablation-optimized weights
    4. Quality compensation for non-safe metrics
    5. Sigmoid with adaptive threshold
    """

    def __init__(self, quality_compensation: bool = True):
        self.quality_compensation = quality_compensation

    def classify(
        self,
        metrics: Dict[str, float],
        quality: Dict[str, float] | Any = None,
    ) -> Dict[str, Any]:
        q = {}
        if quality:
            q = quality if isinstance(quality, dict) else (
                quality.__dict__ if hasattr(quality, "__dict__") else {}
            )

        overall_q = float(q.get("overall_quality", 0.5) or 0.5)

        # Quality hard gate
        if self._is_hard_gated(q):
            return self._empty_result("HARD_GATED: extreme low quality")

        logit = 0.0
        total_weight = 0.0
        rules = []

        def get(name, default):
            try:
                v = float(metrics.get(name, default))
                return v if np.isfinite(v) else default
            except:
                return default

        # ═══ STEP 1: Composite features ═══
        tv = get("tv_residual_sparsity", 0.80)
        et = get("edge_tortuosity_mean", 1.04)
        ac = get("autocorr_decay_len", 21.0)

        # composite_3best
        comp3 = ((1 - tv) / 0.03 + (1.05 - et) / 0.04 + (ac - 20) / 8) / 3
        comp3_evidence = (comp3 - COMPOSITE_CENTER) / COMPOSITE_SCALE
        comp3_evidence = np.clip(comp3_evidence, -2.0, 2.0)
        comp3_contrib = COMPOSITE_WEIGHT * comp3_evidence
        logit += comp3_contrib
        total_weight += COMPOSITE_WEIGHT
        rules.append({
            "metric": "composite_3best",
            "raw": comp3,
            "evidence": comp3_evidence,
            "contribution": comp3_contrib,
            "weight": COMPOSITE_WEIGHT,
        })

        # ratio_tv_x_et
        ratio_tv_et = tv * et
        r1_evidence = (RATIO_TV_ET_CENTER - ratio_tv_et) / RATIO_TV_ET_SCALE
        r1_evidence = np.clip(r1_evidence, -2.0, 2.0)
        r1_w = RATIO_TV_ET_WEIGHT * 0.50  # Apply INDIV_WEIGHT_MULT
        r1_contrib = r1_w * r1_evidence
        logit += r1_contrib
        total_weight += r1_w
        rules.append({
            "metric": "ratio_tv_x_et",
            "raw": ratio_tv_et,
            "evidence": r1_evidence,
            "contribution": r1_contrib,
            "weight": r1_w,
        })

        # ratio_ac_x_et_inv
        ratio_ac_et = ac * (1.1 - et)
        r2_evidence = (ratio_ac_et - RATIO_AC_ET_INV_CENTER) / RATIO_AC_ET_INV_SCALE
        r2_evidence = np.clip(r2_evidence, -2.0, 2.0)
        r2_w = RATIO_AC_ET_INV_WEIGHT * 0.50  # Apply INDIV_WEIGHT_MULT
        r2_contrib = r2_w * r2_evidence
        logit += r2_contrib
        total_weight += r2_w
        rules.append({
            "metric": "ratio_ac_x_et_inv",
            "raw": ratio_ac_et,
            "evidence": r2_evidence,
            "contribution": r2_contrib,
            "weight": r2_w,
        })

        # ═══ STEP 2: Individual metrics (ablation-optimized set) ═══
        all_individual = PRIMARY_V4 + SECONDARY_V4 + SUPPLEMENTARY_V4
        
        # Optimized multiplier for individual metrics (from grid search)
        INDIV_WEIGHT_MULT = 0.50
        
        for name, center, scale, weight, direction, q_safe in all_individual:
            raw = get(name, center)
            
            # Quality compensation for non-safe metrics
            if not q_safe and self.quality_compensation and name in QUALITY_REGRESSION_V4:
                slope, intercept = QUALITY_REGRESSION_V4[name]
                expected = slope * overall_q + intercept
                expected_ref = slope * 0.50 + intercept
                compensated = raw + (expected_ref - expected)
            else:
                compensated = raw
            
            if direction == "below_is_silicone":
                evidence = (center - compensated) / scale
            else:
                evidence = (compensated - center) / scale
            
            evidence = np.clip(evidence, -1.5, 1.5)
            w = weight * INDIV_WEIGHT_MULT
            contribution = w * evidence
            logit += contribution
            total_weight += w
            
            rules.append({
                "metric": name,
                "raw": raw,
                "compensated": compensated if not q_safe else raw,
                "evidence": evidence,
                "contribution": contribution,
                "weight": weight,
                "direction": direction,
            })

        # ═══ STEP 3: Normalize and sigmoid ═══
        if total_weight > 0:
            normalized_logit = logit / total_weight * 2.5
        else:
            normalized_logit = 0.0

        prob_silicone = float(1.0 / (1.0 + np.exp(-normalized_logit)))
        prob_real = float(1.0 - prob_silicone)
        confidence = float(max(prob_real, prob_silicone))

        # Adaptive threshold (optimized: base=0.50, weaker quality penalty)
        threshold = 0.50 + 0.05 * max(0, 0.5 - overall_q)

        if prob_silicone > threshold:
            hint = "silicone"
        elif prob_real > threshold:
            hint = "real"
        else:
            hint = "unknown"

        strongest = sorted(rules, key=lambda r: abs(float(r["contribution"])), reverse=True)[:8]

        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": prob_real, "silicone": prob_silicone},
            "heuristic_logit": float(normalized_logit),
            "quality_threshold": threshold,
            "composite_3best": comp3,
            "ratio_tv_x_et": ratio_tv_et,
            "ratio_ac_x_et_inv": ratio_ac_et,
            "top_rules": strongest,
            "version": "V4",
        }

    def _is_hard_gated(self, q: Dict) -> bool:
        overall_q = float(q.get("overall_quality", 1.0) or 0.0)
        sharpness = float(q.get("sharpness_score", 1000.0) or 0.0)
        noise = float(q.get("noise_level", 0.0) or 0.0)
        blockiness = float(q.get("jpeg_blockiness", 1.0) or 0.0)
        return (
            overall_q < EXTREME_QUALITY_GATE["overall_quality_min"]
            or sharpness < EXTREME_QUALITY_GATE["sharpness_score_min"]
            or noise > EXTREME_QUALITY_GATE["noise_level_max"]
            or blockiness > EXTREME_QUALITY_GATE["jpeg_blockiness_max"]
        )

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "texture_skin_hint": "unknown",
            "texture_skin_confidence": 0.0,
            "posterior": {"real": 0.5, "silicone": 0.5},
            "heuristic_logit": 0.0,
            "quality_threshold": 0.55,
            "composite_3best": 0.0,
            "ratio_tv_x_et": 0.0,
            "ratio_ac_x_et_inv": 0.0,
            "top_rules": [],
            "quality_gated": True,
            "quality_reason": reason,
            "version": "V4",
        }


# Backward compatibility
TextureSkinClassifier = TextureSkinClassifierV4
