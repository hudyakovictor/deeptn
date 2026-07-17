from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import SEMANTIC_POLICY

CHANNEL_NAMES = (
    "right_eye", "left_eye", "right_eyebrow", "left_eyebrow",
    "nose", "upper_lip", "lower_lip", "skin",
)


@dataclass
class MaskBundle:
    channels_224: np.ndarray
    soft_224: np.ndarray
    hard_224: np.ndarray
    soft_original: np.ndarray | None
    hard_original: np.ndarray | None
    status: str
    error: str | None
    metadata: dict[str, Any]


def build_mask_bundle(channels: np.ndarray, trans_params: np.ndarray, image_shape: tuple[int, ...]) -> MaskBundle:
    a = np.asarray(channels, np.float32)
    if a.shape != (224, 224, 8):
        raise ValueError(f"semantic channels must be (224,224,8), got {a.shape}")
    face_skin = np.maximum(a[:, :, 7], a[:, :, 4])
    excluded = np.maximum.reduce([a[:, :, i] for i in (0, 1, 2, 3, 5, 6)])
    soft = np.clip(face_skin * (1.0 - np.clip(excluded, 0.0, 1.0)), 0.0, 1.0)
    hard = soft >= 0.50
    areas = {name: float(np.mean(a[:, :, i] >= 0.5)) for i, name in enumerate(CHANNEL_NAMES)}
    meta = {
        "policy": SEMANTIC_POLICY,
        "channel_names": list(CHANNEL_NAMES),
        "channel_area_fraction_224": areas,
        "soft_area_fraction_224": float(np.mean(soft > 0.05)),
        "hard_area_fraction_224": float(np.mean(hard)),
        "fallback_used": False,
    }
    try:
        from util.io import back_resize_crop_img
        h, w = image_shape[:2]
        soft_u8 = np.clip(soft * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(soft_u8[:, :, None], 3, axis=2)
        blank = np.zeros((h, w, 3), np.uint8)
        projected = back_resize_crop_img(rgb, trans_params, blank, resample_method=Image.BILINEAR)
        full_soft = np.asarray(projected[:, :, 0], np.uint8)
        full_hard = full_soft >= 128
        meta["soft_area_fraction_original"] = float(np.mean(full_soft > 10))
        meta["hard_area_fraction_original"] = float(np.mean(full_hard))
        return MaskBundle(a.astype(np.float16), soft, hard, full_soft, full_hard, "valid", None, meta)
    except Exception as exc:
        # Never stretch aligned 224x224 data over the full source image.
        meta["fallback_used"] = False
        return MaskBundle(a.astype(np.float16), soft, hard, None, None, "projection_failed", str(exc), meta)


def save_masks(bundle: MaskBundle, out_dir) -> dict[str, str]:
    """Save only semantic_channels.npz (no skin_mask_*.png)."""
    out_dir = out_dir
    np.savez_compressed(
        out_dir / "semantic_channels.npz",
        channels_224=bundle.channels_224,
        channel_names=np.asarray(CHANNEL_NAMES),
        skin_soft_224=bundle.soft_224.astype(np.float16),
        skin_hard_224=bundle.hard_224.astype(np.uint8),
    )
    return {"semantic_channels": "semantic_channels.npz"}
