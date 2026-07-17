from __future__ import annotations

import numpy as np

ANCHOR_SCHEMA = "deeputin-stage2-stable-anchor-policy-v1.0"


def stable_anchor_mask(points: np.ndarray, common_visible: np.ndarray, *, min_count: int = 24) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Choose conservative central-face anchors for pair alignment.

    This is a deterministic fallback policy until calibration-ranked anatomical anchors
    are introduced. It avoids using the full face for alignment, reducing the chance that
    soft peripheral/mouth/jaw changes are absorbed by the transform.
    """
    p = np.asarray(points, np.float32)
    common = np.asarray(common_visible, bool)
    anchor = np.zeros(len(p), bool)
    if p.ndim != 2 or p.shape[1] != 3 or int(common.sum()) < min_count:
        return common.copy(), {
            "anchor_policy": "fallback_all_common_insufficient_input",
            "anchor_count": int(common.sum()),
            "anchor_fraction": float(common.mean()) if common.size else 0.0,
        }
    qx1, qx2 = np.quantile(p[common, 0], [0.18, 0.82])
    qy1, qy2 = np.quantile(p[common, 1], [0.18, 0.78])
    qz1, qz2 = np.quantile(p[common, 2], [0.08, 0.92])
    anchor = common & (p[:, 0] >= qx1) & (p[:, 0] <= qx2) & (p[:, 1] >= qy1) & (p[:, 1] <= qy2) & (p[:, 2] >= qz1) & (p[:, 2] <= qz2)
    policy = "central_quantile_anchor_v1"
    if int(anchor.sum()) < min_count:
        # Relax vertical/depth gates before falling back to all common points.
        anchor = common & (p[:, 0] >= qx1) & (p[:, 0] <= qx2)
        policy = "central_x_anchor_relaxed_v1"
    if int(anchor.sum()) < min_count:
        anchor = common.copy()
        policy = "fallback_all_common_too_few_anchors"
    return anchor, {
        "anchor_policy": policy,
        "anchor_count": int(anchor.sum()),
        "anchor_fraction": float(anchor.sum() / max(int(common.sum()), 1)),
    }


def stable_anchor_indices(points: np.ndarray, common_indices: np.ndarray, *, max_points: int = 6000, min_count: int = 1200) -> tuple[np.ndarray, dict[str, float | int | str]]:
    common = np.asarray(common_indices, np.int64)
    mask = np.zeros(len(points), bool)
    mask[common[(common >= 0) & (common < len(points))]] = True
    anchor_mask, meta = stable_anchor_mask(points, mask, min_count=min_count)
    ids = np.flatnonzero(anchor_mask)
    if ids.size > max_points:
        step = int(np.ceil(ids.size / max_points))
        ids = ids[::step][:max_points]
        meta = dict(meta)
        meta["anchor_subsample_step"] = step
        meta["anchor_count_after_subsample"] = int(ids.size)
    return ids.astype(np.int64), meta
