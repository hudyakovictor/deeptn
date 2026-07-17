"""ITER4 unified pose-bucket taxonomy for 3DDFA-V3 / S1 extraction.

Single source of truth: util/pose_settings.json (same ranges as newapp pipeline).
No project env vars required for classification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "ALL_BUCKETS",
    "POSE_BUCKET_YAW_ORDER",
    "CANONICAL_YAW_BY_VIEW_GROUP",
    "load_pose_yaw_ranges",
    "classify_pose_bucket_from_yaw_ranges",
    "classify_pose_bucket",
    "visible_face_side_from_yaw",
    "is_known_bucket",
    "normalize_bucket_name",
]

ALL_BUCKETS = [
    "frontal",
    "left_threequarter_light",
    "right_threequarter_light",
    "left_threequarter_mid",
    "right_threequarter_mid",
    "left_threequarter_deep",
    "right_threequarter_deep",
    "left_profile",
    "right_profile",
]

POSE_BUCKET_YAW_ORDER = (
    "frontal",
    "left_threequarter_light",
    "right_threequarter_light",
    "left_threequarter_mid",
    "right_threequarter_mid",
    "left_threequarter_deep",
    "right_threequarter_deep",
    "left_profile",
    "right_profile",
)

# Keep in sync with util.alignment.CANONICAL_YAW_BY_VIEW_GROUP
CANONICAL_YAW_BY_VIEW_GROUP: dict[str, float] = {
    "frontal": 0.0,
    "left_threequarter_light": -22.5,
    "right_threequarter_light": 22.5,
    "left_threequarter_mid": -45.0,
    "right_threequarter_mid": 45.0,
    "left_threequarter_deep": -67.5,
    "right_threequarter_deep": 67.5,
    "left_profile": -90.0,
    "right_profile": 90.0,
}

_ALIASES = {
    "front": "frontal",
    "frontal_view": "frontal",
    "left_3_4_light": "left_threequarter_light",
    "right_3_4_light": "right_threequarter_light",
    "left_3_4": "left_threequarter_mid",
    "right_3_4": "right_threequarter_mid",
    "left_3_4_mid": "left_threequarter_mid",
    "right_3_4_mid": "right_threequarter_mid",
    "left_3_4_deep": "left_threequarter_deep",
    "right_3_4_deep": "right_threequarter_deep",
    "profile_left": "left_profile",
    "profile_right": "right_profile",
    "l_profile": "left_profile",
    "r_profile": "right_profile",
}

_POSE_YAW_RANGES: dict[str, dict[str, float]] | None = None
_SETTINGS_PATH = Path(__file__).resolve().parent / "pose_settings.json"


def load_pose_yaw_ranges(*, reload: bool = False, path: Optional[Path] = None) -> dict[str, dict[str, float]]:
    global _POSE_YAW_RANGES
    if _POSE_YAW_RANGES is None or reload or path is not None:
        p = Path(path) if path is not None else _SETTINGS_PATH
        data = json.loads(p.read_text(encoding="utf-8"))
        _POSE_YAW_RANGES = {str(k): {"min": float(v["min"]), "max": float(v["max"])} for k, v in data.items()}
    return _POSE_YAW_RANGES


def classify_pose_bucket_from_yaw_ranges(yaw_deg: float) -> str:
    yaw = float(yaw_deg)
    ranges = load_pose_yaw_ranges()
    for bucket in POSE_BUCKET_YAW_ORDER:
        info = ranges.get(bucket)
        if not info:
            continue
        if float(info["min"]) <= yaw <= float(info["max"]):
            return bucket
    return "unclassified"


def classify_pose_bucket(
    yaw_deg: float,
    pitch_deg: float | None = None,
    roll_deg: float | None = None,
    *,
    needs_manual_review: bool = False,
) -> str:
    """Bucket from canonical yaw (+ optional HPE false-profile guard)."""
    bucket = classify_pose_bucket_from_yaw_ranges(yaw_deg)
    if (
        bucket.endswith("_profile")
        and needs_manual_review
        and pitch_deg is not None
        and roll_deg is not None
        and abs(float(pitch_deg)) < 16.0
        and abs(float(roll_deg)) < 15.0
    ):
        prefix = "left" if bucket.startswith("left") else "right"
        bucket = f"{prefix}_threequarter_deep"
    return bucket


def visible_face_side_from_yaw(yaw_deg: float) -> str:
    bucket = classify_pose_bucket_from_yaw_ranges(yaw_deg)
    if bucket.startswith("left"):
        return "left"
    if bucket.startswith("right"):
        return "right"
    return "left" if float(yaw_deg) < 0 else "right"


def is_known_bucket(name: str) -> bool:
    return normalize_bucket_name(name) in ALL_BUCKETS


def normalize_bucket_name(name: str | None) -> str:
    if not name:
        return "unclassified"
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in ALL_BUCKETS:
        return key
    return "unclassified" if key not in ALL_BUCKETS else key
