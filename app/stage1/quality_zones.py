from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import atomic_json

QUALITY_SCHEMA_VERSION = "deeputin-quality-v1.0"
QUALITY_ZONES_SCHEMA_VERSION = "deeputin-quality-zones-v1.0"
FOREHEAD_ZONE_VERSION = "forehead-fallback-v1"
FACE_W = 424
FACE_H = 500


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pack_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), np.uint8)
    return np.stack([np.packbits(np.asarray(m, np.uint8).reshape(-1), bitorder="little") for m in masks])


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _erode(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)


def _to_face_space(mask_original: np.ndarray, bbox_original: list[int], letterbox: dict[str, Any]) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox_original]
    H, W = mask_original.shape[:2]
    x1 = max(0, min(x, W - 1)); y1 = max(0, min(y, H - 1))
    x2 = max(0, min(x + w, W)); y2 = max(0, min(y + h, H))
    crop = mask_original[y1:y2, x1:x2].astype(np.uint8) * 255
    canvas = np.zeros((FACE_H, FACE_W), np.uint8)
    if crop.size == 0:
        return canvas.astype(bool)
    nw = int(letterbox.get("content_width", max(1, round(crop.shape[1] * float(letterbox.get("scale", 1.0))))))
    nh = int(letterbox.get("content_height", max(1, round(crop.shape[0] * float(letterbox.get("scale", 1.0))))))
    ox = int(letterbox.get("offset_x", 0)); oy = int(letterbox.get("offset_y", 0))
    resized = cv2.resize(crop, (max(1, nw), max(1, nh)), interpolation=cv2.INTER_NEAREST)
    y_end = min(FACE_H, oy + resized.shape[0]); x_end = min(FACE_W, ox + resized.shape[1])
    if y_end > oy and x_end > ox:
        canvas[oy:y_end, ox:x_end] = resized[: y_end - oy, : x_end - ox]
    return canvas > 0


def _texture_stats(bgr: np.ndarray, roi: np.ndarray) -> dict[str, float | int | bool | str]:
    roi = np.asarray(roi, bool)
    count = int(np.count_nonzero(roi))
    if count < 16:
        return {
            "texture_pixels": count,
            "laplacian_var": 0.0,
            "tenengrad_mean": 0.0,
            "highlight_fraction": 0.0,
            "shadow_fraction": 0.0,
            "texture_score_0_1": 0.0,
            "texture_usable": False,
            "quality_class": "insufficient_texture_area",
        }
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[roi]
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    lap_var = float(np.var(lap[roi]))
    ten = float(np.mean((gx * gx + gy * gy)[roi]))
    highlight = float(np.mean(vals >= 245))
    shadow = float(np.mean(vals <= 20))
    # Conservative technical score, not a verdict. It only helps gates later.
    sharp = float(np.clip(np.log1p(lap_var) / np.log1p(500.0), 0.0, 1.0))
    exposure_penalty = min(1.0, highlight * 4.0 + shadow * 4.0)
    score = float(np.clip(sharp * (1.0 - 0.6 * exposure_penalty), 0.0, 1.0))
    usable = bool(count >= 2500 and score >= 0.35 and highlight <= 0.20 and shadow <= 0.25)
    return {
        "texture_pixels": count,
        "laplacian_var": lap_var,
        "tenengrad_mean": ten,
        "highlight_fraction": highlight,
        "shadow_fraction": shadow,
        "texture_score_0_1": score,
        "texture_usable": usable,
        "quality_class": "usable" if usable else "weak_texture_quality",
    }


def _forehead_fallback_zones(
    skin_mask: np.ndarray,
    bbox_original: list[int],
    pose_bin: str,
) -> tuple[list[str], list[str], list[str], list[np.ndarray], np.ndarray, list[str], list[float]]:
    """Approximate forehead zones until mesh-zone projection is wired.

    This is intentionally marked as fallback in metadata. Production replacement should
    use anatomical mesh vertex/triangle zones projected through 3DDFA camera geometry.
    """
    H, W = skin_mask.shape[:2]
    x, y, w, h = [int(v) for v in bbox_original]
    x1 = max(0, min(x, W - 1)); y1 = max(0, min(y, H - 1))
    x2 = max(0, min(x + w, W)); y2 = max(0, min(y + h, H))
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    # Conservative upper-face band. Intersecting with skin mask removes most hair/background.
    fy1 = y1 + int(round(0.04 * bh))
    fy2 = y1 + int(round(0.32 * bh))
    fy1 = max(0, min(fy1, H)); fy2 = max(0, min(fy2, H))

    ranges = {
        "forehead_L": (0.12, 0.45),
        "forehead_center": (0.34, 0.66),
        "forehead_R": (0.55, 0.88),
    }
    names = ["forehead_center", "forehead_L", "forehead_R"]
    types = ["skin_texture", "skin_texture", "skin_texture"]
    sides = ["midline", "left", "right"]
    masks: list[np.ndarray] = []
    visible: list[float] = []
    status: list[str] = []

    for name in names:
        rx1, rx2 = ranges[name]
        zx1 = x1 + int(round(rx1 * bw)); zx2 = x1 + int(round(rx2 * bw))
        m = np.zeros((H, W), bool)
        if fy2 > fy1 and zx2 > zx1:
            m[fy1:fy2, zx1:zx2] = True
        m &= skin_mask.astype(bool)
        m = _erode(m, radius=3)
        masks.append(m)

        # Pose policy fallback. Real version should use renderer/triangle visibility.
        if pose_bin == "frontal":
            vf = 1.0
        elif pose_bin == "left_light":
            vf = 0.90 if name in {"forehead_center", "forehead_L"} else 0.55
        elif pose_bin == "right_light":
            vf = 0.90 if name in {"forehead_center", "forehead_R"} else 0.55
        else:
            vf = 0.0
        visible.append(float(vf))
        if pose_bin not in {"frontal", "left_light", "right_light"}:
            status.append("unsupported_pose")
        elif np.count_nonzero(m) < 2500:
            status.append("insufficient_texture_area")
        elif vf < 0.45:
            status.append("insufficient_visibility")
        else:
            status.append("usable_fallback_roi")
    return names, types, sides, masks, np.asarray(visible, np.float32), status, [fy1, fy2]


def build_quality_files(
    *,
    bgr: np.ndarray,
    hard_mask_original: np.ndarray | None,
    crop_meta: dict[str, Any],
    pose: dict[str, Any],
    photo_id: str,
    out: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Write quality.json and quality_zones.npz for Stage 1.

    Current implementation creates forehead zones for frontal/left_light/right_light using
    a conservative skin-mask fallback. The file contract is designed so a later mesh-zone
    projection can replace the masks without changing downstream wrinkle code.
    """
    if hard_mask_original is None or hard_mask_original.size == 0:
        summary = {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "photo_id": photo_id,
            "created_at_utc": _utc(),
            "status": "no_face_mask",
            "warnings": ["quality_zones_not_created_no_face_mask"],
        }
        atomic_json(out / "quality.json", summary)
        return {"quality": "quality.json"}, summary

    skin_mask = np.asarray(hard_mask_original > 0, bool)
    bbox = [int(v) for v in crop_meta.get("bbox_original", [0, 0, bgr.shape[1], bgr.shape[0]])]
    letterbox = crop_meta.get("letterbox", {}) or {}
    pose_bin = str(pose.get("pose_bin", "unknown"))

    names, types, sides, masks_original, visible_fraction, zone_status, forehead_band = _forehead_fallback_zones(
        skin_mask, bbox, pose_bin
    )
    masks_face = [_to_face_space(m, bbox, letterbox) for m in masks_original]
    texture_stats = [_texture_stats(bgr, m) for m in masks_original]

    zone_projected_area_px = np.asarray([int(np.count_nonzero(m)) for m in masks_original], np.int32)
    zone_visible_area_px = np.asarray([int(round(a * vf)) for a, vf in zip(zone_projected_area_px, visible_fraction)], np.int32)
    zone_skin_pixels = zone_projected_area_px.copy()
    zone_texture_pixels = np.asarray([int(s["texture_pixels"]) for s in texture_stats], np.int32)
    zone_texture_score = np.asarray([float(s["texture_score_0_1"]) for s in texture_stats], np.float32)
    zone_laplacian_var = np.asarray([float(s["laplacian_var"]) for s in texture_stats], np.float32)
    zone_tenengrad_mean = np.asarray([float(s["tenengrad_mean"]) for s in texture_stats], np.float32)
    zone_highlight_fraction = np.asarray([float(s["highlight_fraction"]) for s in texture_stats], np.float32)
    zone_shadow_fraction = np.asarray([float(s["shadow_fraction"]) for s in texture_stats], np.float32)
    zone_bbox_original = np.asarray([_mask_bbox(m) for m in masks_original], np.int32)
    zone_bbox_face = np.asarray([_mask_bbox(m) for m in masks_face], np.int32)

    np.savez_compressed(
        out / "quality_zones.npz",
        schema_version=np.asarray(QUALITY_ZONES_SCHEMA_VERSION),
        photo_id=np.asarray(photo_id),
        zone_version=np.asarray(FOREHEAD_ZONE_VERSION),
        roi_source=np.asarray("landmark_skin_fallback_pending_mesh_projection"),
        side_convention=np.asarray("image_space_fallback_pending_anatomical_mesh_zones"),
        pose_bin=np.asarray(pose_bin),
        yaw_pitch_roll=np.asarray([float(pose.get("yaw", 0.0)), float(pose.get("pitch", 0.0)), float(pose.get("roll", 0.0))], np.float32),
        original_shape=np.asarray(bgr.shape[:2], np.int32),
        face_shape=np.asarray([FACE_H, FACE_W], np.int32),
        crop_bbox_original=np.asarray(bbox, np.int32),
        letterbox_scale=np.asarray([float(letterbox.get("scale", 1.0))], np.float32),
        letterbox_offset=np.asarray([int(letterbox.get("offset_x", 0)), int(letterbox.get("offset_y", 0))], np.int32),
        letterbox_content_size=np.asarray([int(letterbox.get("content_width", FACE_W)), int(letterbox.get("content_height", FACE_H))], np.int32),
        forehead_band_y=np.asarray(forehead_band, np.int32),
        zone_names=np.asarray(names),
        zone_types=np.asarray(types),
        zone_sides=np.asarray(sides),
        zone_status=np.asarray(zone_status),
        zone_texture_roi_original_packbits=_pack_masks(masks_original),
        zone_texture_roi_face_packbits=_pack_masks(masks_face),
        zone_bbox_original=zone_bbox_original,
        zone_bbox_face=zone_bbox_face,
        zone_projected_area_px=zone_projected_area_px,
        zone_visible_area_px=zone_visible_area_px,
        zone_skin_pixels=zone_skin_pixels,
        zone_texture_pixels=zone_texture_pixels,
        zone_visible_fraction=visible_fraction,
        zone_skin_fraction=np.asarray([1.0 if a > 0 else 0.0 for a in zone_projected_area_px], np.float32),
        zone_texture_fraction=np.asarray([float(t / max(a, 1)) for t, a in zip(zone_texture_pixels, zone_projected_area_px)], np.float32),
        zone_laplacian_var=zone_laplacian_var,
        zone_tenengrad_mean=zone_tenengrad_mean,
        zone_highlight_fraction=zone_highlight_fraction,
        zone_shadow_fraction=zone_shadow_fraction,
        zone_texture_score=zone_texture_score,
        semantics=np.asarray(
            "quality_zones.npz: forehead ROI masks and texture quality arrays for Stage 1; "
            "current ROI source is fallback skin-mask forehead band for frontal/left_light/right_light; "
            "future mesh-projected anatomical zones should keep this file contract"
        ),
    )

    per_zone: dict[str, Any] = {}
    for i, name in enumerate(names):
        usable = bool(zone_status[i].startswith("usable") and texture_stats[i]["texture_usable"])
        qclass = "good" if usable else (zone_status[i] if not zone_status[i].startswith("usable") else texture_stats[i]["quality_class"])
        per_zone[name] = {
            "roi_source": "landmark_skin_fallback_pending_mesh_projection",
            "texture_pixels": int(zone_texture_pixels[i]),
            "visible_fraction": float(visible_fraction[i]),
            "skin_fraction": float(1.0 if zone_projected_area_px[i] > 0 else 0.0),
            "laplacian_var": float(zone_laplacian_var[i]),
            "tenengrad_mean": float(zone_tenengrad_mean[i]),
            "highlight_fraction": float(zone_highlight_fraction[i]),
            "shadow_fraction": float(zone_shadow_fraction[i]),
            "texture_score_0_1": float(zone_texture_score[i]),
            "texture_usable": usable,
            "quality_class": str(qclass),
        }

    supported_pose = pose_bin in {"frontal", "left_light", "right_light"}
    texture_scores = [float(v["texture_score_0_1"]) for v in per_zone.values() if v["texture_pixels"] >= 2500]
    summary = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "photo_id": photo_id,
        "created_at_utc": _utc(),
        "status": "complete",
        "supported_forehead_wrinkle_pose_v1": supported_pose,
        "pose": pose,
        "mask_quality": {
            "skin_pixels_original": int(np.count_nonzero(skin_mask)),
            "skin_coverage_original": float(np.mean(skin_mask)),
            "status": "ok" if np.any(skin_mask) else "empty_mask",
        },
        "global_texture_quality": {
            "texture_score_0_1": float(np.median(texture_scores)) if texture_scores else 0.0,
            "status": "usable" if texture_scores and np.median(texture_scores) >= 0.35 else "weak_or_insufficient",
        },
        "per_zone_quality": per_zone,
        "files": {"quality_zones": "quality_zones.npz"},
        "warnings": [] if supported_pose else ["forehead_wrinkle_v1_unsupported_pose"],
    }
    atomic_json(out / "quality.json", summary)
    return {"quality": "quality.json", "quality_zones": "quality_zones.npz"}, summary
