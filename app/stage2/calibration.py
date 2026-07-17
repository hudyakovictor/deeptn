from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .core import Comparison, Record, compare_landmarks, robust_reference


class CalibrationModel:
    def __init__(self, records: list[Record], zone106: np.ndarray, zone134: np.ndarray):
        self.records = records; self.zone106 = zone106; self.zone134 = zone134
        self.by_dataset_bin: dict[tuple[str, str], list[Record]] = defaultdict(list)
        for record in records:
            self.by_dataset_bin[(record.dataset_id, record.pose_bin)].append(record)
        self.datasets = sorted({r.dataset_id for r in records})
        self.references: dict[str, dict[str, dict[str, float | int]]] = self._build_references()

    @staticmethod
    def _pose_distance(a: Record, b: Record) -> float:
        return float(np.linalg.norm((a.angles - b.angles) / np.array([15.0, 20.0, 15.0])))

    def _build_references(self) -> dict[str, dict[str, dict[str, float | int]]]:
        values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (dataset, pose_bin), group in self.by_dataset_bin.items():
            if len(group) < 2: continue
            group = sorted(group, key=lambda r: (float(r.angles[1]), float(r.angles[0]), r.sequence))
            for offset in (1, 2):
                for a, b in zip(group, group[offset:]):
                    if self._pose_distance(a, b) > 2.5: continue
                    comp = compare_landmarks(a, b, self.zone106, self.zone134)
                    if comp.status != "measured": continue
                    for key, value in comp.metrics.items(): values[pose_bin][key].append(value)
                    for zone in comp.zones:
                        if zone.get("status") == "measured":
                            values[pose_bin][f"zone::{zone['zone']}::rmse"].append(float(zone["rmse"]))
        return {pose: {metric: robust_reference(v) for metric, v in metrics.items()} for pose, metrics in values.items()}

    def _nearest(self, target: Record, dataset: str, exclude: str | None = None) -> Record | None:
        candidates = [r for r in self.by_dataset_bin.get((dataset, target.pose_bin), []) if r.record_id != exclude]
        if not candidates: return None
        def score(record: Record) -> float:
            pose = self._pose_distance(target, record)
            vis = abs(float(target.visible134.mean()) - float(record.visible134.mean()))
            return pose + 1.5 * vis
        return min(candidates, key=score)

    def matched_null(self, a: Record, b: Record) -> dict[str, list[float]]:
        values: dict[str, list[float]] = defaultdict(list)
        for dataset in self.datasets:
            ca = self._nearest(a, dataset)
            cb = self._nearest(b, dataset, exclude=ca.record_id if ca else None)
            if ca is None or cb is None: continue
            comp = compare_landmarks(ca, cb, self.zone106, self.zone134)
            if comp.status != "measured": continue
            for key, value in comp.metrics.items(): values[key].append(value)
            for zone in comp.zones:
                if zone.get("status") == "measured":
                    values[f"zone::{zone['zone']}::rmse"].append(float(zone["rmse"]))
        return dict(values)

    def reference(self, pose_bin: str, metric: str) -> dict[str, float | int]:
        return self.references.get(pose_bin, {}).get(metric, {"count": 0, "median": 0.0, "mad": 0.0, "p95": 0.0, "p99": 0.0})
