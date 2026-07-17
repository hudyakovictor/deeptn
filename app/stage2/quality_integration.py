from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

QUALITY_INTEGRATION_SCHEMA = "deeputin-stage2-quality-integration-v1.0"


def load_quality_zone_summary(photo_dir: Path) -> dict[str, Any]:
    """Read Stage-1 quality_zones.npz into a compact Stage-2 summary.

    This does not recompute quality. It only exposes zone coverage/usable flags so
    Stage 2 can downgrade evidence produced from weak texture/visibility zones.
    """
    p = photo_dir / "quality_zones.npz"
    if not p.is_file():
        return {
            "status": "missing",
            "zone_count": 0,
            "usable_zone_count": 0,
            "usable_zone_names": [],
            "per_zone": {},
        }
    try:
        with np.load(p, allow_pickle=False) as z:
            names = [str(x) for x in z["zone_names"]]
            statuses = [str(x) for x in z["zone_status"]]
            texture_pixels = z["zone_texture_pixels"].astype(np.int64)
            visible_fraction = z["zone_visible_fraction"].astype(np.float32)
            texture_score = z["zone_texture_score"].astype(np.float32)
            sides = [str(x) for x in z["zone_sides"]] if "zone_sides" in z else ["unknown"] * len(names)
            types = [str(x) for x in z["zone_types"]] if "zone_types" in z else ["unknown"] * len(names)
    except Exception as exc:
        return {
            "status": "invalid",
            "error": str(exc),
            "zone_count": 0,
            "usable_zone_count": 0,
            "usable_zone_names": [],
            "per_zone": {},
        }

    per_zone: dict[str, dict[str, Any]] = {}
    usable: list[str] = []
    for i, name in enumerate(names):
        is_usable = bool(
            statuses[i].startswith("usable")
            and int(texture_pixels[i]) >= 2500
            and float(texture_score[i]) >= 0.35
            and float(visible_fraction[i]) >= 0.45
        )
        if is_usable:
            usable.append(name)
        per_zone[name] = {
            "status": statuses[i],
            "side": sides[i],
            "type": types[i],
            "texture_pixels": int(texture_pixels[i]),
            "visible_fraction": float(visible_fraction[i]),
            "texture_score_0_1": float(texture_score[i]),
            "usable": is_usable,
        }
    finite_scores = [v["texture_score_0_1"] for v in per_zone.values() if v["texture_pixels"] > 0]
    return {
        "schema": QUALITY_INTEGRATION_SCHEMA,
        "status": "loaded",
        "zone_count": len(names),
        "usable_zone_count": len(usable),
        "usable_zone_names": usable,
        "median_texture_score": float(np.median(finite_scores)) if finite_scores else 0.0,
        "min_visible_fraction": float(min((v["visible_fraction"] for v in per_zone.values()), default=0.0)),
        "per_zone": per_zone,
    }


def pair_quality_zone_overlap(a: Any, b: Any, pair_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qa = getattr(a, "quality_zones", {}) or {}
    qb = getattr(b, "quality_zones", {}) or {}
    za = qa.get("per_zone", {}) or {}
    zb = qb.get("per_zone", {}) or {}
    common = sorted(set(za) & set(zb))
    rows: list[dict[str, Any]] = []
    usable_common: list[str] = []
    for name in common:
        va = za[name]; vb = zb[name]
        usable = bool(va.get("usable") and vb.get("usable"))
        if usable:
            usable_common.append(name)
        rows.append({
            "pair_id": pair_id,
            "zone": name,
            "usable_both": usable,
            "status_a": va.get("status"),
            "status_b": vb.get("status"),
            "texture_pixels_a": va.get("texture_pixels", 0),
            "texture_pixels_b": vb.get("texture_pixels", 0),
            "visible_fraction_a": va.get("visible_fraction", 0.0),
            "visible_fraction_b": vb.get("visible_fraction", 0.0),
            "texture_score_a": va.get("texture_score_0_1", 0.0),
            "texture_score_b": vb.get("texture_score_0_1", 0.0),
        })
    summary = {
        "quality_zone_status_a": qa.get("status", "missing"),
        "quality_zone_status_b": qb.get("status", "missing"),
        "quality_zone_common_count": len(common),
        "quality_zone_usable_common_count": len(usable_common),
        "quality_zone_usable_common": "|".join(usable_common),
        "quality_zone_pair_limited": len(common) > 0 and len(usable_common) == 0,
    }
    return summary, rows
