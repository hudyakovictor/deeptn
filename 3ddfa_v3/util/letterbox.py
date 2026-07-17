"""ITER4 letterbox resize / face-crop helpers (no aspect squeeze).

Texture re-extract policy: if a crop was produced by stretch-resize (legacy),
recompute with letterbox before metrics. Library never stretch-fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "FACE_CROP_WIDTH",
    "FACE_CROP_HEIGHT",
    "LetterboxMeta",
    "resize_letterbox",
    "resize_letterbox_gray",
    "letterbox_meta",
    "crop_bbox_with_margin",
    "should_reextract_face_crop",
]

FACE_CROP_WIDTH = 424
FACE_CROP_HEIGHT = 500


@dataclass
class LetterboxMeta:
    target_w: int
    target_h: int
    content_w: int
    content_h: int
    offset_x: int
    offset_y: int
    scale: float
    source_w: int
    source_h: int

    def to_dict(self) -> dict:
        return {
            "target_w": self.target_w,
            "target_h": self.target_h,
            "content_w": self.content_w,
            "content_h": self.content_h,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "scale": self.scale,
            "source_w": self.source_w,
            "source_h": self.source_h,
            "method": "letterbox",
        }


def _cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as exc:  # pragma: no cover
        raise ImportError("OpenCV (cv2) required for letterbox resize") from exc


def letterbox_meta(source_w: int, source_h: int, tw: int, th: int) -> LetterboxMeta:
    if source_w <= 0 or source_h <= 0:
        raise ValueError("source dimensions must be positive")
    if tw <= 0 or th <= 0:
        raise ValueError("target dimensions must be positive")
    scale = min(tw / source_w, th / source_h)
    nw = max(1, int(round(source_w * scale)))
    nh = max(1, int(round(source_h * scale)))
    return LetterboxMeta(
        target_w=tw,
        target_h=th,
        content_w=nw,
        content_h=nh,
        offset_x=(tw - nw) // 2,
        offset_y=(th - nh) // 2,
        scale=float(scale),
        source_w=int(source_w),
        source_h=int(source_h),
    )


def resize_letterbox(
    image: np.ndarray,
    tw: int = FACE_CROP_WIDTH,
    th: int = FACE_CROP_HEIGHT,
    *,
    pad_value: int = 0,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Fit image into tw×th without distorting aspect ratio."""
    cv2 = _cv2()
    img = np.asarray(image)
    if img.ndim == 2:
        gray, meta = resize_letterbox_gray(img, tw, th, pad_value=pad_value)
        return gray, meta
    h, w = img.shape[:2]
    meta = letterbox_meta(w, h, tw, th)
    if meta.content_w <= 0 or meta.content_h <= 0:
        return np.zeros((th, tw, img.shape[2]), dtype=img.dtype), meta
    resized = cv2.resize(img, (meta.content_w, meta.content_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, img.shape[2]), pad_value, dtype=img.dtype)
    y0, x0 = meta.offset_y, meta.offset_x
    canvas[y0 : y0 + meta.content_h, x0 : x0 + meta.content_w] = resized
    return canvas, meta


def resize_letterbox_gray(
    gray: np.ndarray,
    tw: int = FACE_CROP_WIDTH,
    th: int = FACE_CROP_HEIGHT,
    *,
    pad_value: int = 0,
) -> tuple[np.ndarray, LetterboxMeta]:
    cv2 = _cv2()
    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError("gray must be HxW")
    h, w = g.shape[:2]
    meta = letterbox_meta(w, h, tw, th)
    if meta.content_w <= 0 or meta.content_h <= 0:
        return np.zeros((th, tw), dtype=g.dtype), meta
    resized = cv2.resize(g, (meta.content_w, meta.content_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw), pad_value, dtype=g.dtype)
    y0, x0 = meta.offset_y, meta.offset_x
    canvas[y0 : y0 + meta.content_h, x0 : x0 + meta.content_w] = resized
    return canvas, meta


def crop_bbox_with_margin(
    image: np.ndarray,
    bbox: dict,
    *,
    margin: float = 0.25,
    target_aspect: Optional[float] = None,
) -> tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop image by bbox with margin; clamp to frame. Returns (crop, x1,y1,x2,y2)."""
    h, w = image.shape[:2]
    x = int(bbox.get("x", bbox.get("x_min", 0)) or 0)
    y = int(bbox.get("y", bbox.get("y_min", 0)) or 0)
    bw = int(bbox.get("w", bbox.get("width", 0)) or 0)
    bh = int(bbox.get("h", bbox.get("height", 0)) or 0)
    if bw <= 0 and bbox.get("x_max") is not None:
        bw = int(bbox["x_max"]) - x
    if bh <= 0 and bbox.get("y_max") is not None:
        bh = int(bbox["y_max"]) - y
    bw = max(bw, 1)
    bh = max(bh, 1)
    if target_aspect is None:
        target_aspect = FACE_CROP_WIDTH / float(FACE_CROP_HEIGHT)
    crop_w = int(max(bw * (1.0 + margin), bh * (1.0 + margin) * target_aspect, 1))
    crop_h = int(max(bh * (1.0 + margin), crop_w / target_aspect, 1))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 = max(0, x1 - (x2 - w))
        x2 = w
    if y2 > h:
        y1 = max(0, y1 - (y2 - h))
        y2 = h
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def should_reextract_face_crop(
    crop: Optional[np.ndarray],
    *,
    expected_w: int = FACE_CROP_WIDTH,
    expected_h: int = FACE_CROP_HEIGHT,
    meta: Optional[dict] = None,
) -> bool:
    """Policy: re-extract if missing, wrong size, or not letterboxed."""
    if crop is None:
        return True
    arr = np.asarray(crop)
    if arr.size == 0:
        return True
    if arr.shape[0] != expected_h or arr.shape[1] != expected_w:
        return True
    if meta is None:
        # unknown provenance → re-extract to be safe for texture metrics
        return True
    method = str(meta.get("method", "")).lower()
    if method not in ("letterbox", "letterboxed"):
        return True
    return False
