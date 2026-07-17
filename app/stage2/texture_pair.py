from __future__ import annotations

from collections import defaultdict
from typing import Any

TEXTURE_SCHEMA = "deeputin-stage2-texture-pair-v1.0"


def summarize_texture_pairs(zone_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize Stage-1 quality-zone texture comparability per pair.

    This is not yet a full texture-difference module. It converts quality_zones pair
    coverage into a pair-level readiness table for future image-space texture/wrinkle
    analysis and public evidence caveats.
    """
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in zone_rows:
        pid = str(r.get("pair_id", ""))
        if pid:
            by_pair[pid].append(r)
    out: list[dict[str, Any]] = []
    for pid, rows in sorted(by_pair.items()):
        usable = [r for r in rows if r.get("usable_both")]
        if usable:
            min_score = min(float(r.get("texture_score_a", 0.0) or 0.0) for r in usable + [])
            min_score = min(min_score, min(float(r.get("texture_score_b", 0.0) or 0.0) for r in usable))
            min_pixels = min(min(int(r.get("texture_pixels_a", 0) or 0), int(r.get("texture_pixels_b", 0) or 0)) for r in usable)
            zones = "|".join(str(r.get("zone")) for r in usable)
            status = "texture_ready"
        else:
            min_score = 0.0
            min_pixels = 0
            zones = ""
            status = "texture_not_ready"
        out.append({
            "schema": TEXTURE_SCHEMA,
            "pair_id": pid,
            "texture_pair_status": status,
            "quality_zone_count": len(rows),
            "usable_texture_zone_count": len(usable),
            "usable_texture_zones": zones,
            "min_usable_texture_score": min_score,
            "min_usable_texture_pixels": min_pixels,
            "policy": "readiness only; no texture identity verdict",
        })
    return out
