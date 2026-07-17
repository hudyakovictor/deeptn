from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_SCHEMA = "deeputin-stage2-baseline-return-v1.0"
CANDIDATE_STATUSES = {
    "coherent_jump_candidate",
    "persistent_geometric_change",
    "biologically_improbable_rate_candidate",
    "persistent_biologically_improbable_change",
    "rapid_change_candidate",
    "persistent_rapid_change_candidate",
}


def _load_vectors(output_dir: Path, row: dict[str, Any]) -> np.ndarray | None:
    rel = row.get("motion_file")
    if not rel:
        return None
    p = output_dir / str(rel)
    if not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            return z["ldm134_vectors"].astype(np.float32)
    except Exception:
        return None


def _reversal_stats(v1: np.ndarray, v2: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(v1).all(axis=1) & np.isfinite(v2).all(axis=1)
    if int(finite.sum()) < 12:
        return {"common_vector_count": int(finite.sum()), "median_cosine": 0.0, "opposite_fraction": 0.0, "magnitude_ratio": 0.0}
    a = v1[finite].astype(np.float64)
    b = v2[finite].astype(np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-8) & (nb > 1e-8)
    if int(ok.sum()) < 12:
        return {"common_vector_count": int(ok.sum()), "median_cosine": 0.0, "opposite_fraction": 0.0, "magnitude_ratio": 0.0}
    cos = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return {
        "common_vector_count": int(ok.sum()),
        "median_cosine": float(np.median(cos)),
        "opposite_fraction": float(np.mean(cos < -0.25)),
        "magnitude_ratio": float(np.median(nb[ok]) / max(float(np.median(na[ok])), 1e-8)),
    }


def apply_baseline_return(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Detect local A→B spike followed by B→C return in same pose-bin chronology.

    This is intentionally conservative and does not assert biology/identity. It marks a
    candidate as reversible when the next adjacent edge has broadly opposite motion.
    """
    by_pose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("pair_type") == "adjacent":
            by_pose[str(r.get("pose_bin"))].append(r)

    events: list[dict[str, Any]] = []
    for pose, group in by_pose.items():
        group.sort(key=lambda x: (str(x.get("date_a") or ""), str(x.get("date_b") or ""), int(x.get("pair_index") or 0)))
        for i, row in enumerate(group[:-1]):
            nxt = group[i + 1]
            # Need contiguous A->B then B->C.
            if row.get("photo_b") != nxt.get("photo_a"):
                continue
            if row.get("status") not in CANDIDATE_STATUSES:
                continue
            v1 = _load_vectors(output_dir, row)
            v2 = _load_vectors(output_dir, nxt)
            if v1 is None or v2 is None:
                continue
            stats = _reversal_stats(v1, v2)
            is_return = (
                int(stats["common_vector_count"]) >= 30
                and float(stats["opposite_fraction"]) >= 0.45
                and float(stats["median_cosine"]) <= -0.20
                and 0.35 <= float(stats["magnitude_ratio"]) <= 2.75
            )
            row["baseline_return_tested"] = True
            row["baseline_return_next_pair_id"] = nxt.get("pair_id")
            row["baseline_return_opposite_fraction"] = stats["opposite_fraction"]
            row["baseline_return_median_cosine"] = stats["median_cosine"]
            row["baseline_return_magnitude_ratio"] = stats["magnitude_ratio"]
            row["baseline_return_common_vector_count"] = stats["common_vector_count"]
            if is_return:
                row["baseline_return"] = True
                row["baseline_return_interpretation"] = "reversible_motion_candidate"
                if row.get("status") == "coherent_jump_candidate":
                    row["status"] = "baseline_return_candidate"
                events.append({
                    "pair_id": row.get("pair_id"),
                    "next_pair_id": nxt.get("pair_id"),
                    "pose_bin": pose,
                    "photo_a": row.get("photo_a"),
                    "photo_b": row.get("photo_b"),
                    "photo_c": nxt.get("photo_b"),
                    "date_a": row.get("date_a"),
                    "date_b": row.get("date_b"),
                    "date_c": nxt.get("date_b"),
                    **stats,
                    "interpretation": "A→B motion is substantially reversed by B→C; treat as reversible/quality/expression candidate unless independently persistent.",
                })
            else:
                row["baseline_return"] = False
    return {"schema": BASELINE_SCHEMA, "event_count": len(events), "events": events}
