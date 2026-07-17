"""ITER4 S1 extraction helpers for 3DDFA-V3 library.

Combines: pose bucket, quality gate, letterbox crop policy, content-hash cache.
Does not run the neural net (assets may be absent); orchestrates pure steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from util.extraction_cache import (
    content_hash_image_array,
    make_cache_key,
    ExtractionCache,
)
from util.letterbox import (
    FACE_CROP_HEIGHT,
    FACE_CROP_WIDTH,
    crop_bbox_with_margin,
    resize_letterbox,
    should_reextract_face_crop,
)
from util.pose_buckets import classify_pose_bucket, normalize_bucket_name
from util.selected_metrics import select_metrics

__all__ = [
    "ExtractionArtifacts",
    "build_face_crop_letterbox",
    "build_extraction_record",
    "cache_key_for_image",
]


@dataclass
class ExtractionArtifacts:
    pose_bucket: str
    yaw_deg: float
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    image_hash: Optional[str] = None
    cache_key: Optional[str] = None
    face_crop: Optional[np.ndarray] = None
    letterbox_meta: Optional[dict] = None
    quality: Optional[dict] = None
    metrics: Optional[dict] = None
    metrics_ok: bool = False
    reextract_face_crop: bool = False
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pose_bucket": self.pose_bucket,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "image_hash": self.image_hash,
            "cache_key": self.cache_key,
            "letterbox_meta": self.letterbox_meta,
            "quality": self.quality,
            "metrics": self.metrics,
            "metrics_ok": self.metrics_ok,
            "reextract_face_crop": self.reextract_face_crop,
            "payload": self.payload,
        }


def cache_key_for_image(
    image: np.ndarray,
    *,
    backbone: str = "resnet50",
    identity_only: bool = False,
    expression_mode: str = "full",
) -> tuple[str, str]:
    ih = content_hash_image_array(image)
    key = make_cache_key(
        image_hash=ih,
        backbone=backbone,
        identity_only=identity_only,
        expression_mode=expression_mode,
    )
    return ih, key


def build_face_crop_letterbox(
    image_bgr: np.ndarray,
    bbox: dict,
    *,
    target_w: int = FACE_CROP_WIDTH,
    target_h: int = FACE_CROP_HEIGHT,
    margin: float = 0.25,
) -> tuple[np.ndarray, dict, tuple]:
    crop, xyxy = crop_bbox_with_margin(image_bgr, bbox, margin=margin)
    out, meta = resize_letterbox(crop, target_w, target_h)
    return out, meta.to_dict(), xyxy


def build_extraction_record(
    *,
    yaw_deg: float,
    pitch_deg: float | None = None,
    roll_deg: float | None = None,
    image: np.ndarray | None = None,
    bbox: dict | None = None,
    existing_face_crop: np.ndarray | None = None,
    existing_crop_meta: dict | None = None,
    quality: dict | None = None,
    metrics: dict | None = None,
    required_metric_keys: tuple[str, ...] = (),
    backbone: str = "resnet50",
    identity_only: bool = False,
) -> ExtractionArtifacts:
    bucket = classify_pose_bucket(yaw_deg, pitch_deg, roll_deg)
    bucket = normalize_bucket_name(bucket)

    image_hash = None
    cache_key = None
    if image is not None:
        image_hash, cache_key = cache_key_for_image(
            image, backbone=backbone, identity_only=identity_only
        )

    reextract = should_reextract_face_crop(existing_face_crop, meta=existing_crop_meta)
    face_crop = existing_face_crop
    lb_meta = existing_crop_meta
    if reextract and image is not None and bbox is not None:
        face_crop, lb_meta, xyxy = build_face_crop_letterbox(image, bbox)
        reextract = True
        xy = {"crop_xyxy": xyxy}
    else:
        xy = {}

    metrics_ok = True
    metrics_out = dict(metrics or {})
    if required_metric_keys:
        sel = select_metrics(metrics, required_metric_keys, allow_fail_open=False)
        metrics_ok = sel.ok
        metrics_out = {
            **sel.to_dict(),
            "values": sel.values,
        }

    return ExtractionArtifacts(
        pose_bucket=bucket,
        yaw_deg=float(yaw_deg),
        pitch_deg=None if pitch_deg is None else float(pitch_deg),
        roll_deg=None if roll_deg is None else float(roll_deg),
        image_hash=image_hash,
        cache_key=cache_key,
        face_crop=face_crop,
        letterbox_meta=lb_meta,
        quality=quality,
        metrics=metrics_out,
        metrics_ok=metrics_ok,
        reextract_face_crop=reextract,
        payload=xy,
    )
