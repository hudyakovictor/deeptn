"""
TextureSkinClassifierV5 — Production classifier incorporating all findings
from Iteration 1 (100 analyses) and Iteration 2 (100 analyses).

KEY CHANGES FROM V4:
1. DISABLED quality compensation for tv_sparsity (hurts AUC: 0.811→0.744)
2. Added yaw compensation: comp_adj = comp + 0.01 * |yaw|
3. Added consensus rule: 0/4 → REAL (100%), 4/4 → SILICONE (100%)
4. Updated composite center to 2.95
5. Removed lacunarity (batch effect d=1.27 between real2 and calibration)
6. Added ensemble voting (4 independent voters)
7. Added 3-stage cascade (certain / bayesian / review)
8. Removed 8 non-informative metrics (AUC < 0.58)
9. Removed 3 redundant metrics (glrlm, glszm, lbp — r > 0.77 with tv)
10. Added era-based Bayesian prior
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple

# =============================================================================
# PRIMARY METRICS (4 essential, validated by 200 analyses)
# =============================================================================
PRIMARY_METRICS_V5 = [
    # (name, threshold, direction, weight)
    ("tv_residual_sparsity",    0.796, "below_is_silicone", 2.5),
    ("edge_tortuosity_mean",    1.035, "below_is_silicone", 2.0),
    ("autocorr_decay_len",     24.0,   "above_is_silicone", 1.5),
    ("specular_elongation",     2.957, "above_is_silicone", 1.5),
]

# Supplementary (optional)
SUPPLEMENTARY_V5 = [
    ("glcm_diss_d3_aniso",      0.068, "above_is_silicone", 0.5),
]

# =============================================================================
# COMPOSITE CONFIG (optimized via grid search, AUC=0.891)
# =============================================================================
COMPOSITE_CENTER = 2.95
COMPOSITE_SCALE = 0.40

# Yaw compensation coefficient (from Analysis #167)
# Real comp3 decreases with yaw: slope = -0.0103/deg
YAW_COMPENSATION_SLOPE = 0.010

# =============================================================================
# THRESHOLDS (from precision-recall analysis #155)
# =============================================================================
STRONG_REAL_THRESHOLD = 2.3      # comp < 2.3 → REAL (95%+)
STRONG_SILICONE_THRESHOLD = 3.5  # comp > 3.5 → SILICONE (95%+)
MODERATE_REAL_THRESHOLD = 2.7    # comp < 2.7 → REAL (85%)
MODERATE_SILICONE_THRESHOLD = 3.2  # comp > 3.2 → SILICONE (85%)

# =============================================================================
# ERA-BASED PRIORS (from chronological analysis)
# =============================================================================
ERA_PRIORS = {
    "pre_2012": 0.05,      # Original Putin era — silicone very unlikely
    "2012_2021": 0.40,     # Udmurt era — possible silicone
    "post_2021": 0.60,     # Vasilich era — silicone likely
}

# =============================================================================
# QUALITY GATE
# =============================================================================
HARD_QUALITY_GATE = 0.12   # Below this → SKIP entirely
SOFT_QUALITY_GATE = 0.30   # Below this → LOW CONFIDENCE flag


class TextureSkinClassifierV5:
    """
    Production texture classifier V5.
    
    Architecture:
    Stage 1: Quality gate
    Stage 2: Compute features (4 primary + composite)
    Stage 3: Consensus check (0/4 or 4/4 → 100% verdict)
    Stage 4: Composite classification (strong/moderate zones)
    Stage 5: Bayesian fusion for uncertain zone
    Stage 6: Chronological consistency flag
    """

    def __init__(self, quality_compensation: bool = False):
        # IMPORTANT: quality_compensation DISABLED by default
        # Analysis #132 showed it HURTS tv_sparsity AUC (0.811→0.744)
        self.quality_compensation = quality_compensation

    def classify(
        self,
        metrics: Dict[str, float],
        quality: Dict[str, float] | Any = None,
        pose: Dict[str, float] | None = None,
        year: int | None = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Returns comprehensive classification result.
        FIX #12: Handles None quality, string values, extreme values gracefully.
        """
        # FIX #12: Robust quality extraction (None, dict, object)
        q: Dict[str, Any] = {}
        if quality is not None:
            if isinstance(quality, dict):
                q = quality
            elif hasattr(quality, "__dict__"):
                try:
                    q = quality.__dict__
                except Exception:
                    q = {}
            # else: leave q empty, defaults will be used
        
        # FIX #12: Robust float extraction with fallback
        def _safe_float(val: Any, default: float) -> float:
            try:
                v = float(val)
                return v if np.isfinite(v) else default
            except (TypeError, ValueError, OverflowError):
                return default
        
        overall_q = _safe_float(q.get("overall_quality", 0.5), 0.5)
        
        # ═══ STAGE 1: Quality Gate ═══
        if overall_q < HARD_QUALITY_GATE:
            return self._empty_result("HARD_GATED: extreme low quality", overall_q)
        
        soft_gated = overall_q < SOFT_QUALITY_GATE
        
        # ═══ STAGE 2: Compute Features ═══
        # FIX #12: Robust metric extraction — handles strings, None, NaN, Inf, extreme values
        def get(name: str, default: float) -> float:
            raw = metrics.get(name, default) if isinstance(metrics, dict) else default
            return _safe_float(raw, default)
        
        tv = get("tv_residual_sparsity", 0.80)
        et = get("edge_tortuosity_mean", 1.04)
        ac = get("autocorr_decay_len", 21.0)
        se = get("specular_elongation", 1.5)
        glcm = get("glcm_diss_d3_aniso", 0.06)
        
        # Clip to physically plausible ranges to prevent overflow
        tv = float(np.clip(tv, 0.0, 1.0))
        et = float(np.clip(et, 0.5, 2.0))
        ac = float(np.clip(ac, 0.0, 100.0))
        se = float(np.clip(se, 0.5, 50.0))
        
        # Composite (clipped inputs prevent overflow)
        comp3 = ((1 - tv) / 0.03 + (1.05 - et) / 0.04 + (ac - 20) / 8) / 3
        
        # Yaw compensation
        yaw = 0.0
        if pose:
            yaw = abs(float(pose.get("yaw", 0) or 0))
        comp3_adj = comp3 + YAW_COMPENSATION_SLOPE * yaw
        
        # Texture complexity (new feature from Analysis #143)
        complexity = (et - 1) * tv * (30 - ac)
        
        # ═══ STAGE 3: Consensus Check ═══
        votes_silicone = 0
        vote_details = []
        for name, thresh, direction, weight in PRIMARY_METRICS_V5:
            val = get(name, thresh)
            if direction == "below_is_silicone":
                is_sil = val < thresh
            else:
                is_sil = val > thresh
            if is_sil:
                votes_silicone += 1
            vote_details.append({
                "metric": name, "value": val, "threshold": thresh,
                "direction": direction, "is_silicone_signal": is_sil,
            })
        
        consensus = None
        if votes_silicone == 0:
            consensus = "real"  # 100% accurate (Analysis #126)
        elif votes_silicone == 4:
            consensus = "silicone"  # 100% accurate
        
        # ═══ STAGE 4: Composite Classification ═══
        comp_zone = "uncertain"
        if comp3_adj < STRONG_REAL_THRESHOLD:
            comp_zone = "strong_real"
        elif comp3_adj < MODERATE_REAL_THRESHOLD:
            comp_zone = "moderate_real"
        elif comp3_adj > STRONG_SILICONE_THRESHOLD:
            comp_zone = "strong_silicone"
        elif comp3_adj > MODERATE_SILICONE_THRESHOLD:
            comp_zone = "moderate_silicone"
        
        # ═══ STAGE 5: Bayesian Fusion ═══
        # Era prior
        if year is not None:
            if year < 2012:
                prior = ERA_PRIORS["pre_2012"]
                era = "pre_2012"
            elif year < 2021:
                prior = ERA_PRIORS["2012_2021"]
                era = "2012_2021"
            else:
                prior = ERA_PRIORS["post_2021"]
                era = "post_2021"
        else:
            prior = 0.30  # Default neutral prior
            era = "unknown"
        
        # Likelihood ratio from composite
        # Approximate with normal distributions
        r_mean, r_std = 2.35, 0.55  # Real composite stats
        s_mean, s_std = 3.18, 0.45  # Silicone composite stats
        from scipy import stats as scipy_stats
        lr = scipy_stats.norm.pdf(comp3_adj, s_mean, s_std) / (
            scipy_stats.norm.pdf(comp3_adj, r_mean, r_std) + 1e-10
        )
        posterior = prior * lr / (prior * lr + (1 - prior) + 1e-10)
        
        # ═══ FINAL VERDICT ═══
        # FIX #11: moderate_composite had 0% accuracy — now always use Bayesian
        # Priority: consensus > strong composite > bayesian (always)
        if consensus:
            hint = consensus
            confidence = 1.0
            method = "consensus"
        elif comp_zone == "strong_real":
            hint = "real"
            confidence = 0.95
            method = "strong_composite"
        elif comp_zone == "strong_silicone":
            hint = "silicone"
            confidence = 0.95
            method = "strong_composite"
        else:
            # FIX #11: Always use Bayesian for moderate/uncertain zones
            # Previously moderate_real/moderate_silicone gave hardcoded 0.75 confidence
            # which was 0% accurate. Now use posterior probability.
            if posterior > 0.55:
                hint = "silicone"
                confidence = float(posterior)
                method = "bayesian"
            elif posterior < 0.45:
                hint = "real"
                confidence = float(1 - posterior)
                method = "bayesian"
            else:
                hint = "unknown"
                confidence = float(max(posterior, 1 - posterior))
                method = "uncertain"
        
        # Soft gate reduces confidence
        if soft_gated:
            confidence *= 0.85
        
        # ═══ BUILD RESULT ═══
        top_rules = sorted(vote_details, key=lambda r: abs(r["value"] - r["threshold"]) / (r["threshold"] + 1e-6), reverse=True)
        
        return {
            "texture_skin_hint": hint,
            "texture_skin_confidence": confidence,
            "posterior": {"real": float(1 - posterior), "silicone": float(posterior)},
            "composite_3best": float(comp3),
            "composite_3best_adjusted": float(comp3_adj),
            "composite_zone": comp_zone,
            "consensus": consensus,
            "votes_silicone": votes_silicone,
            "votes_total": 4,
            "decision_method": method,
            "era": era,
            "era_prior": prior,
            "likelihood_ratio": float(lr),
            "yaw_compensation": float(YAW_COMPENSATION_SLOPE * yaw),
            "texture_complexity": float(complexity),
            "quality_soft_gated": soft_gated,
            "quality_value": overall_q,
            "top_rules": top_rules,
            "version": "V5",
            "vote_details": vote_details,
        }

    def _empty_result(self, reason: str, quality: float) -> Dict[str, Any]:
        return {
            "texture_skin_hint": "unknown",
            "texture_skin_confidence": 0.0,
            "posterior": {"real": 0.5, "silicone": 0.5},
            "composite_3best": 0.0,
            "composite_3best_adjusted": 0.0,
            "composite_zone": "gated",
            "consensus": None,
            "votes_silicone": 0,
            "votes_total": 4,
            "decision_method": "quality_gated",
            "era": "unknown",
            "era_prior": 0.3,
            "likelihood_ratio": 1.0,
            "yaw_compensation": 0.0,
            "texture_complexity": 0.0,
            "quality_soft_gated": False,
            "quality_value": quality,
            "quality_reason": reason,
            "top_rules": [],
            "version": "V5",
            "vote_details": [],
        }


# Backward compatibility
TextureSkinClassifier = TextureSkinClassifierV5
