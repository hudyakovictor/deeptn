"""ITER8 forensic verdict engine (library-local).

Hypotheses:
  H0 — same person (genuine)
  H1 — identity swap / silicone-mask / impostor surface attack
  H2 — different person

Rules vs legacy:
  - NO era-named / calendar priors that force ORIGINAL by year
  - publications / chronology only as optional post-hoc flags (not prior mass)
  - single evidence path: geometry SNR + texture silicone (no double-count FR+FLAME)
  - fail-closed when geometry unavailable
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "Hypothesis",
    "ForensicStatus",
    "FuzzyLabel",
    "GeometryEvidenceMode",
    "EvidenceBundle",
    "VerdictResult",
    "DEFAULT_PRIORS",
    "SNR_UNCERTAIN",
    "SNR_SIGNAL",
    "normalize_priors",
    "geometry_likelihoods",
    "texture_likelihoods",
    "update_posteriors_log",
    "fuzzy_label_from_evidence",
    "render_verdict",
]

SNR_UNCERTAIN = 1.0
SNR_SIGNAL = 2.0

# Flat base priors — not year-dependent
DEFAULT_PRIORS = {"H0": 0.50, "H1": 0.05, "H2": 0.45}


class Hypothesis(str, Enum):
    H0 = "H0"  # same
    H1 = "H1"  # swap/mask
    H2 = "H2"  # different


class ForensicStatus(str, Enum):
    SAME_PERSON = "same_person"
    UNCERTAIN = "uncertain"
    DIFFERENT_PERSON = "different_person"
    IDENTITY_SWAP = "identity_swap"
    INSUFFICIENT_DATA = "insufficient_data"


class FuzzyLabel(str, Enum):
    STRONGLY_MATCHING = "strongly_matching"
    CONSISTENT = "consistent"
    INSUFFICIENT_DATA = "insufficient_data"
    WEAK_EVIDENCE = "weak_evidence"
    SUSPICIOUS_TEXTURE = "suspicious_texture"
    GEOMETRIC_MISMATCH = "geometric_mismatch"
    IDENTITY_ANOMALY = "identity_anomaly"


class GeometryEvidenceMode(str, Enum):
    CALIBRATED = "calibrated"
    LOCAL = "local"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


@dataclass
class EvidenceBundle:
    geometry_snr: Optional[float] = None
    geometry_error: Optional[float] = None
    predicted_noise: Optional[float] = None
    bone_error: Optional[float] = None
    texture_silicone: Optional[float] = None  # [0,1]
    texture_reliability: float = 0.0
    id_cosine_distance: Optional[float] = None
    shared_vertex_count: Optional[int] = None
    geometry_mode: GeometryEvidenceMode = GeometryEvidenceMode.UNAVAILABLE
    # optional post-hoc flags only (do NOT enter priors)
    posthoc_flags: List[str] = field(default_factory=list)
    # optional chronology delta years for flagging only
    delta_years: Optional[float] = None


@dataclass
class VerdictResult:
    status: ForensicStatus
    fuzzy_label: FuzzyLabel
    probabilities: Dict[str, float]
    confidence: float
    likelihoods: Dict[str, Dict[str, float]]
    priors: Dict[str, float]
    evidence: Dict[str, Any]
    flags: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # post-hoc only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "fuzzy_label": self.fuzzy_label.value,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "likelihoods": self.likelihoods,
            "priors": dict(self.priors),
            "evidence": self.evidence,
            "flags": list(self.flags),
            "reasoning": list(self.reasoning),
            "citations": list(self.citations),
        }


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def normalize_priors(priors: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    raw = DEFAULT_PRIORS if priors is None else priors
    p: Dict[str, float] = {}
    for k in ("H0", "H1", "H2"):
        value = float(raw.get(k, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"prior {k} must be finite and non-negative")
        p[k] = value
    s = p["H0"] + p["H1"] + p["H2"]
    if s <= 0.0:
        raise ValueError("at least one prior must be positive")
    return {k: float(p[k] / s) for k in ("H0", "H1", "H2")}


def geometry_likelihoods(
    snr: Optional[float],
    *,
    mode: GeometryEvidenceMode = GeometryEvidenceMode.CALIBRATED,
) -> Dict[str, float]:
    """Map geometry SNR → likelihoods for H0/H1/H2.

    H0 high at low SNR, H2 grows with SNR, H1 peaks in mid band (partial mismatch).
    """
    if snr is None or mode == GeometryEvidenceMode.UNAVAILABLE:
        return {"H0": 1.0, "H1": 1.0, "H2": 1.0}  # non-informative

    s = max(float(snr), 0.0)
    # soft shapes
    # H0: high when s << uncertain
    l0 = math.exp(-0.9 * max(s - 0.2, 0.0) ** 2) + 0.05
    # H2: logistic growth past signal threshold
    l2 = 0.05 + 0.95 / (1.0 + math.exp(-1.4 * (s - SNR_SIGNAL)))
    # H1: elevated in mid SNR (1..3) — geometry off but not maximal
    l1 = 0.08 + 0.75 * math.exp(-0.55 * (s - 1.6) ** 2)

    if mode == GeometryEvidenceMode.FALLBACK:
        # damp confidence of all but keep ranking
        l0, l1, l2 = 0.6 * l0 + 0.4, 0.6 * l1 + 0.4, 0.6 * l2 + 0.4

    return {"H0": float(l0), "H1": float(l1), "H2": float(l2)}


def texture_likelihoods(
    silicone_prob: Optional[float],
    *,
    reliability: float = 0.0,
    geometry_snr: Optional[float] = None,
) -> Dict[str, float]:
    """Texture evidence — single path, no second FR embedding.

    High silicone:
      - with geometry match (low SNR) → surface confound (makeup/lighting), weak H1
      - with geometry mismatch (high SNR) → H1/H2 prosthetic modifier
    """
    if silicone_prob is None or reliability <= 0.05:
        return {"H0": 1.0, "H1": 1.0, "H2": 1.0}

    tex = _clamp01(float(silicone_prob))
    rel = _clamp01(float(reliability))
    snr = 0.0 if geometry_snr is None else max(float(geometry_snr), 0.0)

    # base: natural texture supports H0 slightly, silicone supports H1
    l0 = 0.3 + 0.7 * (1.0 - tex)
    l1 = 0.15 + 0.85 * tex
    l2 = 0.4 + 0.3 * tex  # mild — different person need not be silicone

    if snr < SNR_UNCERTAIN and tex > 0.55:
        # geometry match + high texture → do NOT treat as swap; surface confound
        l1 = 0.15 + 0.25 * tex
        l0 = 0.45 + 0.4 * (1.0 - 0.5 * tex)
    elif snr >= SNR_SIGNAL and tex > 0.55:
        # geometry mismatch + silicone → boost H1 and H2
        l1 = 0.25 + 0.9 * tex
        l2 = 0.35 + 0.55 * tex

    # reliability attenuates toward 1
    def _att(L: float) -> float:
        return float((L ** rel) * (1.0 ** (1.0 - rel))) if rel < 1 else float(L)

    return {"H0": _att(l0), "H1": _att(l1), "H2": _att(l2)}


def id_likelihoods(id_cosine_distance: Optional[float]) -> Dict[str, float]:
    if id_cosine_distance is None:
        return {"H0": 1.0, "H1": 1.0, "H2": 1.0}
    d = max(float(id_cosine_distance), 0.0)
    # d~0 same, d~1 different
    l0 = math.exp(-3.0 * d) + 0.05
    l2 = 0.05 + 0.95 * (1.0 - math.exp(-2.5 * d))
    l1 = 0.15 + 0.5 * math.exp(-2.0 * (d - 0.35) ** 2)
    return {"H0": float(l0), "H1": float(l1), "H2": float(l2)}


def update_posteriors_log(
    priors: Mapping[str, float],
    likelihoods_list: Sequence[Mapping[str, float]],
) -> Dict[str, float]:
    log_p = {h: math.log(max(float(priors.get(h, 1e-12)), 1e-12)) for h in ("H0", "H1", "H2")}
    for lik in likelihoods_list:
        for h in ("H0", "H1", "H2"):
            log_p[h] += math.log(max(float(lik.get(h, 1.0)), 1e-12))
    m = max(log_p.values())
    exps = {h: math.exp(v - m) for h, v in log_p.items()}
    s = sum(exps.values())
    return {h: float(exps[h] / s) for h in ("H0", "H1", "H2")}


def fuzzy_label_from_evidence(
    post: Mapping[str, float],
    snr: Optional[float],
    texture: Optional[float],
    *,
    insufficient: bool = False,
) -> FuzzyLabel:
    if insufficient:
        return FuzzyLabel.INSUFFICIENT_DATA
    p0, p1, p2 = post["H0"], post["H1"], post["H2"]
    if p1 >= 0.45 and p1 >= p0 and p1 >= p2:
        return FuzzyLabel.IDENTITY_ANOMALY
    if p2 >= 0.55:
        return FuzzyLabel.GEOMETRIC_MISMATCH
    if texture is not None and texture >= 0.65 and (snr is None or snr < SNR_SIGNAL):
        return FuzzyLabel.SUSPICIOUS_TEXTURE
    if p0 >= 0.75 and (snr is None or snr < SNR_UNCERTAIN):
        return FuzzyLabel.STRONGLY_MATCHING
    if p0 >= 0.55:
        return FuzzyLabel.CONSISTENT
    if max(p0, p1, p2) < 0.5:
        return FuzzyLabel.WEAK_EVIDENCE
    return FuzzyLabel.WEAK_EVIDENCE


def _status_from_posteriors(post: Mapping[str, float], *,
                           insufficient: bool = False) -> ForensicStatus:
    if insufficient:
        return ForensicStatus.INSUFFICIENT_DATA
    p0, p1, p2 = post["H0"], post["H1"], post["H2"]
    # decision margins
    if p1 >= 0.40 and p1 >= p0 and p1 + 0.05 >= p2:
        return ForensicStatus.IDENTITY_SWAP
    if p0 >= 0.55 and p0 >= p2 and p0 >= p1:
        return ForensicStatus.SAME_PERSON
    if p2 >= 0.55 and p2 >= p0:
        return ForensicStatus.DIFFERENT_PERSON
    return ForensicStatus.UNCERTAIN


def render_verdict(
    evidence: EvidenceBundle,
    *,
    priors: Optional[Mapping[str, float]] = None,
    use_id_channel: bool = True,
    citations: Optional[Sequence[str]] = None,
) -> VerdictResult:
    """Produce forensic verdict from geometry/texture evidence.

    Chronology / publications may appear only in flags/citations (post-hoc),
    never as year-forced prior mass.
    """
    pr = normalize_priors(priors)
    flags = list(evidence.posthoc_flags)
    reasoning: List[str] = []

    insufficient = (
        evidence.geometry_mode == GeometryEvidenceMode.UNAVAILABLE
        or (
            evidence.geometry_snr is None
            and evidence.geometry_error is None
            and (evidence.shared_vertex_count is None or evidence.shared_vertex_count < 50)
        )
    )

    # compute SNR if only error+noise given
    snr = evidence.geometry_snr
    if snr is None and evidence.geometry_error is not None and evidence.predicted_noise is not None:
        from util.calibration import linear_snr
        snr = linear_snr(float(evidence.geometry_error), float(evidence.predicted_noise))

    if evidence.shared_vertex_count is not None and evidence.shared_vertex_count < 50:
        insufficient = True
        flags.append("insufficient_shared_visibility")

    lik_geom = geometry_likelihoods(None if insufficient else snr, mode=evidence.geometry_mode)
    lik_tex = texture_likelihoods(
        evidence.texture_silicone,
        reliability=evidence.texture_reliability,
        geometry_snr=None if insufficient else snr,
    )
    lik_list = [lik_geom, lik_tex]
    lik_id = {"H0": 1.0, "H1": 1.0, "H2": 1.0}
    if use_id_channel and evidence.id_cosine_distance is not None and not insufficient:
        lik_id = id_likelihoods(evidence.id_cosine_distance)
        lik_list.append(lik_id)
        reasoning.append("id_channel_used")
    else:
        reasoning.append("id_channel_skipped")

    # POST-HOC chronology flags only — do not multiply into priors
    if evidence.delta_years is not None and evidence.delta_years > 40:
        flags.append("large_time_gap_posthoc")
        reasoning.append("large_time_gap_not_used_as_prior")

    post = update_posteriors_log(pr, lik_list)
    status = _status_from_posteriors(post, insufficient=insufficient)
    label = fuzzy_label_from_evidence(
        post, None if insufficient else snr, evidence.texture_silicone, insufficient=insufficient
    )

    # confidence = margin of top vs second
    ordered = sorted(post.values(), reverse=True)
    confidence = float(ordered[0] - ordered[1]) if len(ordered) >= 2 else float(ordered[0])
    if insufficient:
        confidence = 0.0

    if snr is not None:
        reasoning.append(f"geometry_snr={snr:.3f}")
    if evidence.texture_silicone is not None:
        reasoning.append(f"texture_silicone={evidence.texture_silicone:.3f}")
    reasoning.append(f"posteriors H0={post['H0']:.3f} H1={post['H1']:.3f} H2={post['H2']:.3f}")
    reasoning.append("no_calendar_prior")

    return VerdictResult(
        status=status,
        fuzzy_label=label,
        probabilities=post,
        confidence=confidence,
        likelihoods={"geometry": lik_geom, "texture": lik_tex, "id": lik_id},
        priors=pr,
        evidence={
            "geometry_snr": snr,
            "geometry_error": evidence.geometry_error,
            "predicted_noise": evidence.predicted_noise,
            "bone_error": evidence.bone_error,
            "texture_silicone": evidence.texture_silicone,
            "texture_reliability": evidence.texture_reliability,
            "id_cosine_distance": evidence.id_cosine_distance,
            "shared_vertex_count": evidence.shared_vertex_count,
            "geometry_mode": evidence.geometry_mode.value,
            "delta_years": evidence.delta_years,
        },
        flags=flags,
        reasoning=reasoning,
        citations=list(citations or []),
    )
