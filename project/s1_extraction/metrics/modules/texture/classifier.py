from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MODEL_PATH = Path(__file__).parent / "skin_classifier_v2.pkl"

RULES_ONLY_DEFAULT = True

# Tier 1 + Tier 2 = 20 CORE_V2 metrics
TEXTURE_CORE_V2 = [
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
    "glrlm_sre",
    "ngtdm_coarseness",
    "dwt_haar_HH_LL_ratio",
    "lbp_r1_hist_entropy",
    "shannon_entropy_q32",
    "gabor_f08_anisotropy",
    "pore_eccentricity_mean",
    "specular_elongation",
]

PHYSICAL_AUX = [
    "seam_score",
    "specular_sharpness",
    "sss_index",
    "melanin_hemo_slope",
]

QUALITY_FEATURES = [
    "overall_quality", "sharpness_score", "noise_level", "jpeg_blockiness",
    "q_laplacian_var", "q_tenengrad", "q_noise_sigma", "q_jpeg_blockiness", "q_valid_patches",
]

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


class TextureSkinClassifierV2:
    """Rule-based texture classifier — works without trained ML model."""

    def __init__(self, model_path: str | Path | None = None, rules_only: bool = RULES_ONLY_DEFAULT) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        self.rules_only = bool(rules_only)
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

    def classify(self, metrics: Dict[str, float], quality: Dict[str, float] | Any = None, reference: Dict | None = None, pose: Dict | None = None) -> Dict[str, Any]:
        q = {}
        if quality:
            q = quality if isinstance(quality, dict) else quality.__dict__ if hasattr(quality, '__dict__') else {}

        hard_gated, soft_gated, quality_reason = self._quality_state(q)

        if hard_gated:
            return {
                "texture_skin_hint": "unknown",
                "texture_skin_confidence": 0.0,
                "posterior": {"real": 0.5, "silicone": 0.5},
                "used_metrics": [],
                "model_loaded": self._pipeline is not None,
                "heuristic_fallback": True,
                "rule_based": True,
                "quality_gated": True,
                "quality_soft_gated": False,
                "quality_reason": f"HARD_GATED: {quality_reason}",
                "heuristic_logit": 0.0,
                "heuristic_top_rules": [],
            }

        if (not self.rules_only) and self._pipeline is not None:
            return self._ml_classify(metrics, q, soft_gated, quality_reason)

        return self._heuristic_classify(metrics, q, soft_gated, quality_reason)

    def _quality_state(self, q: Dict[str, float]) -> Tuple[bool, bool, str]:
        overall_q = float(q.get("overall_quality", 1.0) or 0.0)
        sharpness = float(q.get("sharpness_score", q.get("q_laplacian_var", 1000.0)) or 0.0)
        noise = float(q.get("noise_level", q.get("q_noise_sigma", 0.0)) or 0.0)
        blockiness = float(q.get("jpeg_blockiness", q.get("q_jpeg_blockiness", 1.0)) or 0.0)

        hard = (
            overall_q < EXTREME_QUALITY_GATE["overall_quality_min"] or
            sharpness < EXTREME_QUALITY_GATE["sharpness_score_min"] or
            noise > EXTREME_QUALITY_GATE["noise_level_max"] or
            blockiness > EXTREME_QUALITY_GATE["jpeg_blockiness_max"]
        )
        soft = (
            overall_q < QUALITY_GATE_THRESHOLDS["overall_quality_min"] or
            sharpness < QUALITY_GATE_THRESHOLDS["sharpness_score_min"] or
            noise > QUALITY_GATE_THRESHOLDS["noise_level_max"] or
            blockiness > QUALITY_GATE_THRESHOLDS["jpeg_blockiness_max"]
        )
        reason = f"q={overall_q:.2f}, sharp={sharpness:.0f}, noise={noise:.1f}, jpeg={blockiness:.2f}"
        return hard, soft, reason

    @staticmethod
    def _clip_unit(x: float) -> float:
        if not np.isfinite(x):
            return 0.0
        return float(np.clip(x, -1.0, 1.0))

    def _add_rule(self, rules: List[Dict[str, float | str]], name: str, raw: float, weight: float, why: str) -> float:
        value = self._clip_unit(raw)
        contribution = float(weight * value)
        rules.append({
            "metric": name,
            "signed_evidence": contribution,
            "normalized": value,
            "why": why,
        })
        return contribution

    def _adaptive_threshold(self, q: Dict[str, float], soft_gated: bool) -> float:
        overall_q = float(q.get("overall_quality", 1.0) or 0.0)
        base = 0.62
        if soft_gated:
            base = max(base, 0.68)
        penalty = 0.15 * max(0.0, 0.50 - overall_q)
        return base + penalty

    def _heuristic_classify(self, metrics: Dict[str, float], q: Dict[str, float], soft_gated: bool, quality_reason: str) -> Dict[str, Any]:
        rules: List[Dict[str, float | str]] = []
        logit = 0.0

        def m(name: str, default: float) -> float:
            try:
                v = float(metrics.get(name, default))
                return v if np.isfinite(v) else default
            except Exception:
                return default

        logit += self._add_rule(
            rules, "fft_high_low_ratio",
            (0.095 - m("fft_high_low_ratio", 0.095)) / 0.055,
            2.20,
            "silicone: depressed chaotic micro high-frequency energy",
        )
        logit += self._add_rule(
            rules, "spectral_slope_beta",
            (m("spectral_slope_beta", 2.75) - 2.75) / 0.55,
            1.00,
            "silicone: steeper 1/f^beta spectrum / smoother material",
        )
        logit += self._add_rule(
            rules, "wld_joint_entropy",
            (6.15 - m("wld_joint_entropy", 6.15)) / 0.35,
            0.55,
            "silicone: lower local gradient-orientation entropy",
        )
        logit += self._add_rule(
            rules, "pore_density_r2_mpx",
            (28000.0 - m("pore_density_r2_mpx", 28000.0)) / 9000.0,
            0.55,
            "silicone: reduced small pore/blob density",
        )
        logit += self._add_rule(
            rules, "glrlm_sre",
            (0.42 - m("glrlm_sre", 0.42)) / 0.12,
            0.45,
            "silicone: fewer short runs / longer smooth runs",
        )
        logit += self._add_rule(
            rules, "glszm_small_area_emphasis",
            (0.44 - m("glszm_small_area_emphasis", 0.44)) / 0.08,
            0.45,
            "silicone: fewer small gray-level zones",
        )
        logit += self._add_rule(
            rules, "glcm_diss_d3_aniso",
            (m("glcm_diss_d3_aniso", 0.07) - 0.07) / 0.055,
            0.45,
            "silicone: stronger directional GLCM dissimilarity anisotropy",
        )
        logit += self._add_rule(
            rules, "lacunarity",
            (m("lacunarity", 2.0) - 2.0) / 0.28,
            0.35,
            "silicone: larger gaps / less fractal pore randomness",
        )
        logit += self._add_rule(
            rules, "seam_score",
            (m("seam_score", 0.03) - 0.03) / 0.07,
            0.35,
            "silicone: boundary/seam discontinuity",
        )
        logit += self._add_rule(
            rules, "tv_residual_sparsity",
            (0.79 - m("tv_residual_sparsity", 0.79)) / 0.035,
            0.35,
            "silicone: less sparse natural micro residual after TV denoise",
        )

        if soft_gated:
            logit *= 0.85

        prob_silicone = float(1.0 / (1.0 + np.exp(-logit)))
        prob_real = float(1.0 - prob_silicone)
        confidence = float(max(prob_real, prob_silicone))
        adaptive_thresh = max(self._adaptive_threshold(q, soft_gated), 0.62)

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
        }

    def _ml_classify(self, metrics: Dict[str, float], q: Dict[str, float], soft_gated: bool, quality_reason: str) -> Dict[str, Any]:
        feature_names = TEXTURE_CORE_V2 + PHYSICAL_AUX
        vector = []
        used = []
        for name in feature_names:
            val = metrics.get(name)
            if val is None or not np.isfinite(float(val)):
                val = 0.0
            vector.append(float(val))
            used.append(name)

        X = np.array([vector], dtype=np.float64)
        proba = self._pipeline.predict_proba(X)[0]
        prob_real = float(proba[0])
        prob_silicone = float(proba[1])

        hint = "silicone" if prob_silicone >= prob_real else "real"
        confidence = max(prob_real, prob_silicone)

        adaptive_thresh = max(self._adaptive_threshold(q, soft_gated), 0.62)
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
        }


TextureSkinClassifier = TextureSkinClassifierV2
