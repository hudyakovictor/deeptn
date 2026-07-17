"""ITER5 calibration / noise-floor utilities for 3DDFA forensic pipeline.

Pure-numpy, no project contracts. Multi-person pose×quality noise model,
health buckets, effective sample size, person baselines.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from util.pose_buckets import ALL_BUCKETS, normalize_bucket_name

__all__ = [
    "CALIB_STABLE_CV",
    "CALIB_MARGINAL_CV",
    "CALIB_STABLE_SPREAD",
    "CALIB_MARGINAL_SPREAD",
    "CALIB_REF_QUALITY_THRESHOLD",
    "MIN_OBS_FOR_STABLE",
    "GEOMETRY_NOISE_CAP_RATIO",
    "mad",
    "effective_sample_size",
    "block_bootstrap_ci",
    "linear_snr",
    "metric_health_status",
    "build_person_baselines",
    "apply_person_baseline",
    "NoiseObservation",
    "ZoneNoiseProfile",
    "NoiseModel",
    "CalibrationSummary",
    "build_calibration_summary",
    "bucket_metric_health",
    "same_person_residual_stats",
    "pose_quality_cell_key",
]

CALIB_STABLE_CV = 0.25
CALIB_MARGINAL_CV = 0.50
CALIB_STABLE_SPREAD = 0.05
CALIB_MARGINAL_SPREAD = 0.12
CALIB_REF_QUALITY_THRESHOLD = 0.35
MIN_OBS_FOR_STABLE = 8
GEOMETRY_NOISE_CAP_RATIO = 0.35


def mad(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def effective_sample_size(n: int, ar1_rho: float = 0.0) -> float:
    """Effective N under AR(1) dependence: n_eff = n * (1-ρ)/(1+ρ)."""
    n = max(int(n), 0)
    if n == 0:
        return 0.0
    rho = float(np.clip(ar1_rho, -0.99, 0.99))
    return float(n * (1.0 - rho) / (1.0 + rho))


def block_bootstrap_ci(
    values: Sequence[float],
    *,
    block_size: int = 5,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Block bootstrap CI for the mean (pose-correlated frames)."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "n_eff": 0.0}
    rng = rng or np.random.default_rng(0)
    n = arr.size
    bs = max(int(block_size), 1)
    n_blocks = int(np.ceil(n / bs))
    means = []
    for _ in range(int(n_boot)):
        starts = rng.integers(0, max(n - bs + 1, 1), size=n_blocks)
        sample = np.concatenate([arr[s : s + bs] for s in starts])[:n]
        means.append(float(np.mean(sample)))
    means_a = np.asarray(means, dtype=np.float64)
    lo = float(np.quantile(means_a, alpha / 2))
    hi = float(np.quantile(means_a, 1.0 - alpha / 2))
    # crude AR(1) from lag-1 corr
    if n >= 3:
        x = arr[:-1] - arr[:-1].mean()
        y = arr[1:] - arr[1:].mean()
        denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y))) + 1e-12
        rho = float(np.sum(x * y) / denom)
    else:
        rho = 0.0
    return {
        "mean": float(np.mean(arr)),
        "lo": lo,
        "hi": hi,
        "n": int(n),
        "n_eff": effective_sample_size(n, rho),
        "ar1_rho": rho,
    }


def linear_snr(signal_error: float, noise_baseline: float) -> float:
    """Linear SNR = max(err - noise, 0) / max(noise, eps)."""
    safe_noise = max(abs(float(noise_baseline)), 0.005)
    safe_signal = max(float(signal_error) - safe_noise, 0.0)
    return safe_signal / safe_noise


def metric_health_status(
    robust_cv: float,
    spread: float,
    observation_count: int,
    *,
    min_obs: int = MIN_OBS_FOR_STABLE,
) -> str:
    if observation_count < max(3, min_obs // 2):
        return "insufficient"
    if observation_count < min_obs:
        # can still be marginal if tight
        if robust_cv <= CALIB_STABLE_CV and spread <= CALIB_STABLE_SPREAD:
            return "marginal"
        return "insufficient"
    if robust_cv <= CALIB_STABLE_CV and spread <= CALIB_STABLE_SPREAD:
        return "stable"
    if robust_cv <= CALIB_MARGINAL_CV or spread <= CALIB_MARGINAL_SPREAD:
        return "marginal"
    return "replace"


def pose_quality_cell_key(bucket: str, quality_band: str) -> str:
    return f"{normalize_bucket_name(bucket)}|{quality_band}"


def quality_band_from_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    s = float(score)
    if s >= 0.75:
        return "high"
    if s >= 0.45:
        return "mid"
    if s >= CALIB_REF_QUALITY_THRESHOLD:
        return "low"
    return "reject"


def build_person_baselines(
    samples: Sequence[Tuple[str, str, float]],
) -> Dict[str, Dict[str, float]]:
    """Median same-person residual per (person, bucket)."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for person, bucket, err in samples:
        if err is None:
            continue
        try:
            e = float(err)
        except Exception:
            continue
        if e != e:  # NaN
            continue
        buckets[str(person)][normalize_bucket_name(bucket)].append(e)
    out: Dict[str, Dict[str, float]] = {}
    for person, by_bucket in buckets.items():
        out[person] = {b: float(median(vals)) for b, vals in by_bucket.items() if vals}
    return out


def apply_person_baseline(
    band_error: float,
    person_a: Optional[str],
    person_b: Optional[str],
    bucket: str,
    baselines: Optional[Mapping[str, Mapping[str, float]]],
) -> float:
    if not baselines or not person_a or not person_b:
        return float(band_error)
    if person_a != person_b:
        return float(band_error)
    ba = (baselines.get(person_a) or {}).get(normalize_bucket_name(bucket))
    if ba is None:
        return float(band_error)
    return max(float(band_error) - float(ba), 0.0)


@dataclass
class NoiseObservation:
    photo_id_a: str
    photo_id_b: str
    person_id: str
    bucket: str
    pose_delta_mag: float
    zone_errors: Dict[str, float]
    quality_score: float = 1.0
    quality_band: str = "high"


@dataclass
class ZoneNoiseProfile:
    zone_name: str
    mean: float
    std: float
    count: int
    reliability: float
    n_eff: float = 0.0
    pose_weight: float = 0.02
    quality_weight: float = 0.01

    def predict_noise(self, pose_delta_mag: float, quality_degradation: float = 0.0) -> float:
        return float(
            self.mean
            + self.pose_weight * abs(float(pose_delta_mag))
            + self.quality_weight * abs(float(quality_degradation))
        )


class NoiseModel:
    """Same-person residual noise model by zone (+ optional pose×quality cells)."""

    def __init__(self):
        self.observations: List[NoiseObservation] = []
        self.zone_profiles: Dict[str, ZoneNoiseProfile] = {}
        self.cell_profiles: Dict[str, Dict[str, ZoneNoiseProfile]] = {}

    def add_observation(self, obs: NoiseObservation) -> None:
        self.observations.append(obs)
        self._rebuild()

    def extend(self, obs_list: Iterable[NoiseObservation]) -> None:
        self.observations.extend(list(obs_list))
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.observations:
            self.zone_profiles = {}
            self.cell_profiles = {}
            return
        all_zones: set[str] = set()
        for obs in self.observations:
            all_zones.update(obs.zone_errors.keys())

        self.zone_profiles = {}
        for zone in all_zones:
            errors = [obs.zone_errors[zone] for obs in self.observations if zone in obs.zone_errors]
            if not errors:
                continue
            self.zone_profiles[zone] = _profile_from_errors(zone, errors)

        # pose×quality cells
        cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for obs in self.observations:
            cell = pose_quality_cell_key(obs.bucket, obs.quality_band)
            for z, e in obs.zone_errors.items():
                cells[cell][z].append(float(e))
        self.cell_profiles = {}
        for cell, zmap in cells.items():
            self.cell_profiles[cell] = {
                z: _profile_from_errors(z, errs) for z, errs in zmap.items() if errs
            }

    def predict_noise(
        self,
        zone_name: str,
        pose_delta_mag: float = 0.0,
        *,
        bucket: Optional[str] = None,
        quality_band: Optional[str] = None,
        quality_degradation: float = 0.0,
    ) -> float:
        profile = None
        if bucket is not None and quality_band is not None:
            cell = pose_quality_cell_key(bucket, quality_band)
            profile = (self.cell_profiles.get(cell) or {}).get(zone_name)
        if profile is None:
            profile = self.zone_profiles.get(zone_name)
        if profile:
            return profile.predict_noise(pose_delta_mag, quality_degradation)
        return 0.015 * (1.0 + 0.05 * abs(float(pose_delta_mag)))

    def get_reliability(self, zone_name: str) -> float:
        p = self.zone_profiles.get(zone_name)
        return p.reliability if p else 0.5

    def to_dict(self) -> dict:
        return {
            "version": "iter5_v1",
            "n_observations": len(self.observations),
            "zone_profiles": {k: asdict(v) for k, v in self.zone_profiles.items()},
            "cell_profiles": {
                cell: {z: asdict(p) for z, p in zmap.items()}
                for cell, zmap in self.cell_profiles.items()
            },
        }


def _profile_from_errors(zone: str, errors: Sequence[float]) -> ZoneNoiseProfile:
    arr = np.asarray(list(errors), dtype=np.float64)
    med = float(np.median(arr))
    m = float(np.median(np.abs(arr - med))) if arr.size > 1 else max(med * 0.25, 0.001)
    std = float(np.std(arr)) if arr.size > 1 else m
    noise_score = m * 100.0 + std * 25.0
    reliability = 1.0 / (1.0 + np.exp(2.5 * (noise_score - 1.25)))
    n_eff = effective_sample_size(int(arr.size), ar1_rho=0.3)  # conservative default
    return ZoneNoiseProfile(
        zone_name=zone,
        mean=med,
        std=m,
        count=int(arr.size),
        reliability=float(np.clip(reliability, 0.05, 0.99)),
        n_eff=n_eff,
    )


@dataclass
class CalibrationSummary:
    observation_count: int
    bucket_coverage: Dict[str, int]
    stable_metrics: int
    marginal_metrics: int
    replace_metrics: int
    insufficient_metrics: int
    buckets: Dict[str, Any]
    metrics: List[dict]
    person_coverage: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "calibration_type": "pose_quality_noise_model_iter5",
            "observation_count": self.observation_count,
            "bucket_coverage": self.bucket_coverage,
            "stable_metrics": self.stable_metrics,
            "marginal_metrics": self.marginal_metrics,
            "replace_metrics": self.replace_metrics,
            "insufficient_metrics": self.insufficient_metrics,
            "buckets": self.buckets,
            "metrics": self.metrics,
            "person_coverage": self.person_coverage,
            "notes": self.notes,
        }


def build_calibration_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    quality_threshold: float = CALIB_REF_QUALITY_THRESHOLD,
    min_obs: int = MIN_OBS_FOR_STABLE,
) -> CalibrationSummary:
    """Build per-bucket metric health from extraction-like records.

    Expected record keys (flexible):
      status: "ready" (optional)
      bucket / pose_bucket
      person_id (optional)
      metrics: dict[str, float]
      quality: {overall_score, flags?}
      photo_id (optional)
      pose: {yaw, pitch} optional
    """
    extracted = []
    for r in records:
        if r.get("status") not in (None, "ready", "ok", "success"):
            continue
        if not r.get("metrics"):
            continue
        extracted.append(r)

    buckets: dict[str, Any] = {}
    flat_metrics: list[dict] = []
    person_coverage: dict[str, int] = defaultdict(int)
    notes: list[str] = []

    bucket_names = list(ALL_BUCKETS) + ["unclassified"]
    for bucket in bucket_names:
        bucket_records = [
            r
            for r in extracted
            if normalize_bucket_name(r.get("bucket") or r.get("pose_bucket")) == bucket
        ]
        for r in bucket_records:
            pid = str(r.get("person_id") or r.get("person") or "")
            if pid:
                person_coverage[pid] += 1

        # special-case notes for known weak cells
        if bucket.startswith("right") and len(bucket_records) > 0:
            persons = {str(r.get("person_id") or "") for r in bucket_records}
            if persons == {"person_03"} or (len(persons) == 1 and "03" in next(iter(persons))):
                notes.append(f"{bucket}: only person_03-like coverage (right-only risk)")

        metric_map: dict[str, Any] = {}
        best_reference = None
        best_reference_score = float("inf")
        values_by_metric: dict[str, list[float]] = defaultdict(list)

        for record in bucket_records:
            quality = record.get("quality") or {}
            flags = quality.get("flags") or {}
            if flags.get("QUALITY_REJECTED_TEXTURE"):
                continue
            qscore = float(quality.get("overall_score", 1.0))
            if qscore < quality_threshold:
                continue
            pose = record.get("pose") or {}
            pose_score = abs(float(pose.get("yaw", 0.0))) + abs(float(pose.get("pitch", 0.0)))
            # prefer higher quality, more frontal as reference
            score = pose_score + (1.0 - qscore)
            if score < best_reference_score:
                best_reference_score = score
                best_reference = record
            for key, value in (record.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and value == value:
                    values_by_metric[str(key)].append(float(value))

        for key, values in values_by_metric.items():
            if not values:
                continue
            med = float(median(values))
            spread = mad(values)
            robust_cv = spread / abs(med) if abs(med) > 1e-6 else spread
            n_eff = effective_sample_size(len(values), ar1_rho=0.5)
            # low-res person_02 style: if many values but high spread → replace/marginal
            status = metric_health_status(robust_cv, spread, len(values), min_obs=min_obs)
            # use n_eff for insufficient if heavily correlated
            if n_eff < max(3.0, min_obs / 2) and status == "stable":
                status = "marginal"
                notes.append(f"{bucket}/{key}: n_eff={n_eff:.1f} demoted stable→marginal")
            info = {
                "key": key,
                "median": med,
                "mad": spread,
                "robust_cv": robust_cv,
                "status": status,
                "observation_count": len(values),
                "n_eff": n_eff,
            }
            metric_map[key] = info
            flat_metrics.append({**info, "bucket": bucket})

        buckets[bucket] = {
            "bucket": bucket,
            "observation_count": len(bucket_records),
            "usable_observation_count": sum(
                1
                for r in bucket_records
                if float((r.get("quality") or {}).get("overall_score", 1.0)) >= quality_threshold
            ),
            "reference_photo_id": (best_reference or {}).get("photo_id"),
            "metrics": metric_map,
        }

    stable_count = sum(1 for item in flat_metrics if item["status"] == "stable")
    marginal_count = sum(1 for item in flat_metrics if item["status"] == "marginal")
    replace_count = sum(1 for item in flat_metrics if item["status"] == "replace")
    insuff_count = sum(1 for item in flat_metrics if item["status"] == "insufficient")

    return CalibrationSummary(
        observation_count=len(extracted),
        bucket_coverage={b: buckets[b]["observation_count"] for b in buckets},
        stable_metrics=stable_count,
        marginal_metrics=marginal_count,
        replace_metrics=replace_count,
        insufficient_metrics=insuff_count,
        buckets=buckets,
        metrics=flat_metrics,
        person_coverage=dict(person_coverage),
        notes=notes,
    )


def bucket_metric_health(summary: CalibrationSummary | Mapping[str, Any], bucket: str) -> List[dict]:
    if isinstance(summary, CalibrationSummary):
        data = summary.to_dict()
    else:
        data = dict(summary)
    b = normalize_bucket_name(bucket)
    metrics = (data.get("buckets") or {}).get(b, {}).get("metrics") or {}
    return [dict(v, bucket=b) for v in metrics.values()]


def same_person_residual_stats(
    residuals: Sequence[float],
    *,
    block_size: int = 5,
) -> Dict[str, float]:
    """Stats for same-person pose residual series (calibration gate)."""
    arr = [float(x) for x in residuals if x is not None and x == x]
    if not arr:
        return {"n": 0, "median": 0.0, "mad": 0.0, "mean": 0.0, "n_eff": 0.0}
    ci = block_bootstrap_ci(arr, block_size=block_size)
    return {
        "n": len(arr),
        "median": float(median(arr)),
        "mad": mad(arr),
        "mean": ci["mean"],
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
        "n_eff": ci["n_eff"],
        "ar1_rho": ci.get("ar1_rho", 0.0),
    }
