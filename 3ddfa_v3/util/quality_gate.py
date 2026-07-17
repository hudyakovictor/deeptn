"""ITER2 technical quality gate for 3DDFA-V3 (no project contracts).

Pure image metrics: blur / noise / jpeg blockiness / motion blur / over-smooth.
Returns plain dicts only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

BLUR_THRESHOLD_DEFAULT = 65.0
NOISE_THRESHOLD_DEFAULT = 2.5
MIN_FACE_TEXTURE_PX = 120

__all__ = [
    "QualityGate",
    "evaluate_image_array",
    "BLUR_THRESHOLD_DEFAULT",
    "NOISE_THRESHOLD_DEFAULT",
    "MIN_FACE_TEXTURE_PX",
]


def _reject(reason: str, issues: Optional[list[str]] = None) -> Dict[str, Any]:
    return {
        "success": False,
        "is_rejected": True,
        "overall_score": 0.0,
        "overall_quality": 0.0,
        "blur_value": 0.0,
        "sharpness_variance": 0.0,
        "noise_level": 0.0,
        "quality_scope": "rejected",
        "is_motion_blurred": False,
        "is_jpeg_blocky": False,
        "is_over_smoothed": False,
        "jpeg_blockiness": 1.0,
        "admissibility_reason": reason,
        "blocking_issues": list(issues or [reason]),
    }


def _jpeg_block_boundary_slices(grid_offset: int) -> tuple[slice, slice, slice, slice]:
    ox = int(grid_offset) % 8
    boundary_a = (7 - ox) % 8
    boundary_b = boundary_a + 1
    inside_a = (3 - ox) % 8
    inside_b = inside_a + 1
    return (
        slice(boundary_a, None, 8),
        slice(boundary_b, None, 8),
        slice(inside_a, None, 8),
        slice(inside_b, None, 8),
    )


def _jpeg_blockiness_score(gray: np.ndarray, grid_offset_x: int) -> float:
    h_g, w_g = gray.shape[:2]
    if h_g <= 16 or w_g <= 16:
        return 1.0
    b_a, b_b, i_a, i_b = _jpeg_block_boundary_slices(grid_offset_x)
    boundary_a = gray[:, b_a]
    boundary_b = gray[:, b_b]
    n_blocks = min(boundary_a.shape[1], boundary_b.shape[1])
    if n_blocks <= 0:
        return 1.0
    diff_grid_x = float(np.mean(np.abs(boundary_a[:, :n_blocks] - boundary_b[:, :n_blocks])))
    inside_a = gray[:, i_a]
    inside_b = gray[:, i_b]
    n_inside = min(inside_a.shape[1], inside_b.shape[1])
    if n_inside <= 0:
        return 1.0
    diff_inside_x = float(np.mean(np.abs(inside_a[:, :n_inside] - inside_b[:, :n_inside])))
    return diff_grid_x / (diff_inside_x + 1e-5)


def _laplacian_sharpness_denominator(min_face_dim: int) -> float:
    dim = max(int(min_face_dim), 64)
    return 400.0 * float(np.clip(dim / 224.0, 0.35, 2.5))


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for QualityGate")


def evaluate_image_array(
    img_bgr: np.ndarray,
    *,
    bbox: Optional[dict] = None,
    blur_threshold: float = BLUR_THRESHOLD_DEFAULT,
    noise_threshold: float = NOISE_THRESHOLD_DEFAULT,
    reject_motion_blur: bool = True,
) -> Dict[str, Any]:
    """Evaluate quality on a BGR uint8 image array."""
    _require_cv2()
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return _reject("INSUFFICIENT_DATA_EMPTY")

    img = img_bgr
    h, w = img.shape[:2]

    if bbox is not None:
        face_h = bbox.get("h") or bbox.get("height") or h
        face_w = bbox.get("w") or bbox.get("width") or w
        try:
            face_h = int(face_h)
            face_w = int(face_w)
        except Exception:
            face_h, face_w = h, w
        if face_h < MIN_FACE_TEXTURE_PX or face_w < MIN_FACE_TEXTURE_PX:
            return _reject(f"FACE_TOO_SMALL_{int(min(face_h, face_w))}px")

    quality_scope = "full_image"
    jpeg_grid_offset_x = 0
    if bbox is not None:
        x = int(bbox.get("x", bbox.get("x_min", 0)) or 0)
        y = int(bbox.get("y", bbox.get("y_min", 0)) or 0)
        bw = int(bbox.get("w", bbox.get("width", 0)) or 0)
        bh = int(bbox.get("h", bbox.get("height", 0)) or 0)
        if bw <= 0 and bbox.get("x_max") is not None:
            bw = int(bbox.get("x_max") or 0) - x
        if bh <= 0 and bbox.get("y_max") is not None:
            bh = int(bbox.get("y_max") or 0) - y
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x0 + max(0, bw)), min(h, y0 + max(0, bh))
        if x1 > x0 and y1 > y0:
            img = img[y0:y1, x0:x1]
            quality_scope = "face_crop"
            jpeg_grid_offset_x = x0 % 8

    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_min_dim = min(gray.shape[:2])

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    var_x = max(float(np.var(sobel_x)), 1e-5)
    var_y = max(float(np.var(sobel_y)), 1e-5)
    motion_ratio = max(var_x / var_y, var_y / var_x)
    is_motion_blurred = bool(motion_ratio > 3.0 and min(var_x, var_y) < 100.0)

    blockiness = _jpeg_blockiness_score(gray, jpeg_grid_offset_x)
    is_jpeg_blocky = bool(blockiness > 1.35)

    sharpness_score = float(np.clip(blur_score / _laplacian_sharpness_denominator(face_min_dim), 0.0, 1.0))
    if is_motion_blurred:
        sharpness_score *= 0.5
    if is_jpeg_blocky:
        sharpness_score *= 0.7

    median = cv2.medianBlur(gray, 3)
    noise_score = float(np.mean(np.abs(gray.astype(np.float32) - median.astype(np.float32))))
    noise_quality = float(np.clip(1.0 - (noise_score / 25.0), 0.0, 1.0))

    is_over_smoothed = bool(
        sharpness_score > 0.88
        and noise_quality > 0.82
        and blockiness < 1.08
        and not is_motion_blurred
    )
    overall_penalty = 0.75 if is_over_smoothed else 1.0
    if is_over_smoothed:
        sharpness_score *= 0.72

    overall_score = float(((sharpness_score * 0.7) + (noise_quality * 0.3)) * overall_penalty)

    # Threshold flags (informational; silicone may be over-smoothed and still admissible)
    is_rejected = bool(reject_motion_blur and is_motion_blurred)
    if blur_score < float(blur_threshold) * 0.15 and face_min_dim >= MIN_FACE_TEXTURE_PX:
        # extremely soft — flag but do not auto-reject (archives)
        pass

    return {
        "success": True,
        "is_rejected": is_rejected,
        "blur_value": blur_score,
        "sharpness_variance": blur_score,
        "noise_level": noise_score,
        "overall_score": overall_score,
        "overall_quality": overall_score,
        "quality_scope": quality_scope,
        "is_motion_blurred": is_motion_blurred,
        "is_jpeg_blocky": is_jpeg_blocky,
        "is_over_smoothed": is_over_smoothed,
        "jpeg_blockiness": float(blockiness),
        "sharpness_score": sharpness_score,
        "noise_quality": noise_quality,
        "blur_threshold": float(blur_threshold),
        "noise_threshold": float(noise_threshold),
        "admissibility_reason": "MOTION_BLUR" if is_rejected else "OK",
        "blocking_issues": ["MOTION_BLUR"] if is_rejected else [],
    }


class QualityGate:
    """Technical quality gate (library-local, no core.contracts)."""

    def __init__(
        self,
        blur_threshold: float = BLUR_THRESHOLD_DEFAULT,
        noise_threshold: float = NOISE_THRESHOLD_DEFAULT,
        reject_motion_blur: bool = True,
    ):
        self.blur_threshold = float(blur_threshold)
        self.noise_threshold = float(noise_threshold)
        self.reject_motion_blur = bool(reject_motion_blur)

    def evaluate(
        self,
        image: Union[str, Path, np.ndarray],
        bbox: Optional[dict] = None,
    ) -> Dict[str, Any]:
        _require_cv2()
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                return _reject("INSUFFICIENT_DATA_UNREADABLE")
        else:
            img = image
        return evaluate_image_array(
            img,
            bbox=bbox,
            blur_threshold=self.blur_threshold,
            noise_threshold=self.noise_threshold,
            reject_motion_blur=self.reject_motion_blur,
        )

    def evaluate_face_quality(
        self,
        img_full: np.ndarray,
        face_bbox: dict,
        skin_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        _require_cv2()
        x = int(face_bbox.get("x", 0))
        y = int(face_bbox.get("y", 0))
        w = int(face_bbox.get("w", face_bbox.get("width", 0)) or 0)
        h = int(face_bbox.get("h", face_bbox.get("height", 0)) or 0)
        if w < MIN_FACE_TEXTURE_PX or h < MIN_FACE_TEXTURE_PX:
            return _reject("FACE_TOO_SMALL", ["FACE_TOO_SMALL"])

        face_crop = img_full[y : y + h, x : x + w]
        if face_crop.size == 0:
            return _reject("EMPTY_CROP")

        if face_crop.ndim == 2:
            gray_crop = face_crop
        else:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        if skin_mask is not None:
            mask_crop = skin_mask[y : y + h, x : x + w]
            laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
            valid = laplacian[mask_crop > 0]
            if valid.size < 100:
                return _reject("INSUFFICIENT_SKIN", ["INSUFFICIENT_SKIN"])
            sharpness = float(np.var(valid))
            median_blurred = cv2.medianBlur(gray_crop, 3)
            noise_diff = np.abs(gray_crop.astype(np.int16) - median_blurred.astype(np.int16))
            noise_level = float(np.mean(noise_diff[mask_crop > 0]))
        else:
            sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            median_blurred = cv2.medianBlur(gray_crop, 3)
            noise_level = float(
                np.mean(np.abs(gray_crop.astype(np.int16) - median_blurred.astype(np.int16)))
            )

        return {
            "success": True,
            "is_rejected": False,
            "sharpness": sharpness,
            "noise_level": noise_level,
            "overall_score": float(
                np.clip(sharpness / _laplacian_sharpness_denominator(min(w, h)), 0, 1.0)
            ),
        }
