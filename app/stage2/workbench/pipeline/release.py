from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.stage2.workbench.core.contracts import ReleaseManifest, canonical_hash


class ReleaseBuildError(ValueError):
    pass


@dataclass(frozen=True)
class ApprovedSnapshot:
    step: str
    gate_id: str
    dataset_hash: str
    config_hash: str
    artifact_hashes: dict[str, str]
    limitations: tuple[dict, ...] = ()
    status: str = "approved_locked"

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PrivateHypothesisRecord:
    record_id: str
    hypothesis_family: str
    required_evidence_ids: tuple[str, ...]
    legacy_label: str | None = None
    legacy_score: float | None = None


@dataclass(frozen=True)
class PrivateRetestResult:
    record_id: str
    public_release_id: str
    current_evidence_ids: tuple[str, ...]
    status: str
    reason_codes: tuple[str, ...]


def build_release_from_snapshots(
    *, release_id: str, run_id: str, dataset_lock_id: str, dataset_hash: str,
    snapshots: Iterable[ApprovedSnapshot], required_steps: Iterable[str],
    calibration_snapshot_ids: Iterable[str], metric_registry_hash: str,
) -> ReleaseManifest:
    snapshots = tuple(snapshots)
    by_step = {snapshot.step: snapshot for snapshot in snapshots}
    required = tuple(required_steps)
    missing = set(required) - set(by_step)
    if missing:
        raise ReleaseBuildError(f"missing approved snapshots: {sorted(missing)}")
    for snapshot in snapshots:
        if snapshot.status != "approved_locked":
            raise ReleaseBuildError(f"snapshot is not approved_locked: {snapshot.step}")
        if snapshot.dataset_hash != dataset_hash:
            raise ReleaseBuildError(f"dataset hash mismatch: {snapshot.step}")
    config_hashes = {step: by_step[step].config_hash for step in required}
    artifact_hashes: dict[str, str] = {}
    limitations: list[dict] = []
    for step in required:
        snapshot = by_step[step]
        for name, value in snapshot.artifact_hashes.items():
            key = f"{step}.{name}"
            if key in artifact_hashes and artifact_hashes[key] != value:
                raise ReleaseBuildError(f"artifact collision: {key}")
            artifact_hashes[key] = value
        limitations.extend(snapshot.limitations)
    return ReleaseManifest(
        release_id=release_id,
        run_id=run_id,
        dataset_lock_id=dataset_lock_id,
        dataset_hash=dataset_hash,
        approved_gate_ids=tuple(by_step[step].gate_id for step in required),
        config_hashes=config_hashes,
        calibration_snapshot_ids=tuple(calibration_snapshot_ids),
        metric_registry_hash=metric_registry_hash,
        artifact_hashes=artifact_hashes,
        limitations=tuple(limitations),
        public_artifacts=tuple(sorted(artifact_hashes)),
        private_artifacts=(),
    )


def run_private_retest(
    records: Iterable[PrivateHypothesisRecord], *, public_release: ReleaseManifest,
    available_evidence_ids: Iterable[str], candidate_evidence_ids: Iterable[str] = (),
) -> tuple[PrivateRetestResult, ...]:
    available = set(available_evidence_ids)
    candidates = set(candidate_evidence_ids)
    results = []
    for record in sorted(records, key=lambda item: item.record_id):
        required = set(record.required_evidence_ids)
        current = tuple(sorted(required & available))
        missing = required - available
        if missing:
            status = "pending_missing_current_data"
            reasons = ("required_current_evidence_missing",)
        elif required & candidates:
            status = "technical_anomaly_candidate"
            reasons = ("current_approved_evidence_candidate",)
        else:
            status = "within_current_noise_or_no_strong_change"
            reasons = ("current_approved_evidence_not_candidate",)
        results.append(PrivateRetestResult(record.record_id, public_release.release_id, current, status, reasons))
    return tuple(results)


def private_retest_hash(results: Iterable[PrivateRetestResult], public_release: ReleaseManifest) -> str:
    return canonical_hash({
        "public_release_hash": public_release.reproducibility_hash,
        "results": [result.__dict__ for result in sorted(results, key=lambda item: item.record_id)],
    })
