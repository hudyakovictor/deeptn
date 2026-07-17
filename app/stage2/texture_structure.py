from __future__ import annotations

from typing import Any

import cv2
import numpy as np

SCHEMA = "deeputin-stage2-texture-structure-v1.0"
PATCH_SIZE = 192


def _patch(image: np.ndarray, mask: np.ndarray, size: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.asarray(mask, bool)
    ys, xs = np.where(mask)
    if ys.size < 256:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if y1 - y0 < 12 or x1 - x0 < 12:
        return None
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    local_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA if max(gray.shape) > size else cv2.INTER_CUBIC)
    local_mask = cv2.resize(local_mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    # Border erosion limits interpolation and crop-edge artifacts.
    local_mask = cv2.erode(local_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if int(local_mask.sum()) < 1000:
        return None
    return gray.astype(np.float32) / 255.0, local_mask


def register_patches(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    common = np.asarray(mask_a, bool) & np.asarray(mask_b, bool)
    if int(common.sum()) < 1000:
        return {"status": "insufficient_common_mask", "common_pixels": int(common.sum())}
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    aa = np.where(common, a, 0.0).astype(np.float32) * window
    bb = np.where(common, b, 0.0).astype(np.float32) * window
    shift, response = cv2.phaseCorrelate(aa, bb)
    sx, sy = float(shift[0]), float(shift[1])
    # Do not allow registration to explain a large anatomical displacement.
    max_shift = 0.04 * min(a.shape)
    if abs(sx) > max_shift or abs(sy) > max_shift or not np.isfinite([sx, sy, response]).all():
        return {"status": "registration_unstable", "shift_x": sx, "shift_y": sy, "response": float(response), "max_shift": float(max_shift)}
    matrix = np.array([[1.0, 0.0, sx], [0.0, 1.0, sy]], np.float32)
    aligned_b = cv2.warpAffine(b, matrix, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    aligned_mask_b = cv2.warpAffine(mask_b.astype(np.uint8), matrix, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
    overlap = mask_a & aligned_mask_b
    if int(overlap.sum()) < 1000:
        return {"status": "insufficient_registered_overlap", "shift_x": sx, "shift_y": sy, "response": float(response), "common_pixels": int(overlap.sum())}
    residual_before = np.abs(a[common] - b[common])
    residual_after = np.abs(a[overlap] - aligned_b[overlap])
    return {
        "status": "registered",
        "shift_x": sx,
        "shift_y": sy,
        "response": float(response),
        "overlap": overlap,
        "aligned_b": aligned_b,
        "residual_before_median": float(np.median(residual_before)),
        "residual_after_median": float(np.median(residual_after)),
        "registration_improvement": float(np.median(residual_before) - np.median(residual_after)),
    }


def _ssim(gray_a: np.ndarray, gray_b: np.ndarray, mask: np.ndarray) -> float:
    """Masked SSIM using the standard local Gaussian formulation."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_a * mu_b
    value = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / np.maximum((mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2), 1e-8)
    return float(np.mean(value[mask])) if int(mask.sum()) else 0.0


def _ridge_probability(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Multi-scale, dual-polarity Hessian ridge response without hard medical labels."""
    responses: list[np.ndarray] = []
    orientation_energy = np.zeros(4, np.float64)
    for sigma in (0.8, 1.4, 2.2):
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        trace = dxx + dyy
        disc = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4 * dxy ** 2, 0.0))
        l1 = 0.5 * (trace - disc)
        l2 = 0.5 * (trace + disc)
        # Both dark and light elongated structures are retained.
        ridge = np.maximum(np.abs(l1), np.abs(l2)) * np.exp(-np.minimum(np.abs(l1), np.abs(l2)) / np.maximum(np.maximum(np.abs(l1), np.abs(l2)), 1e-6))
        ridge[~mask] = 0.0
        scale = float(np.percentile(ridge[mask], 99)) if int(mask.sum()) else 1.0
        responses.append(np.clip(ridge / max(scale, 1e-8), 0.0, 1.0))
        angle = 0.5 * np.arctan2(2 * dxy, dxx - dyy)
        for i, center in enumerate((0.0, np.pi / 4, np.pi / 2, -np.pi / 4)):
            weight = np.maximum(np.cos(2 * (angle - center)), 0.0)
            orientation_energy[i] += float(np.mean((ridge * weight)[mask])) if int(mask.sum()) else 0.0
    probability = np.max(np.stack(responses), axis=0)
    orientation_energy /= max(float(orientation_energy.sum()), 1e-8)
    return probability, {f"ridge_orientation_{i}": float(v) for i, v in enumerate(orientation_energy)}


def _skeleton(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(256):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
        if cv2.countNonZero(image) == 0:
            break
    return skeleton > 0


def _skeleton_metrics(probability: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    vals = probability[mask]
    if vals.size < 1000:
        return {"skeleton_pixels": 0, "skeleton_component_count": 0, "skeleton_endpoint_count": 0, "skeleton_branchpoint_count": 0, "ridge_density": 0.0}
    threshold = float(np.percentile(vals, 88))
    binary = (probability >= threshold) & mask
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    skel = _skeleton(binary)
    neighbors = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16S, np.ones((3, 3), np.int16)) - skel.astype(np.int16)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), 8)
    lengths = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], np.int32)
    kept = lengths[lengths >= 5]
    return {
        "skeleton_pixels": int(skel.sum()),
        "skeleton_component_count": int(kept.size),
        "skeleton_endpoint_count": int(np.sum(skel & (neighbors == 1))),
        "skeleton_branchpoint_count": int(np.sum(skel & (neighbors >= 3))),
        "skeleton_longest_component": int(kept.max()) if kept.size else 0,
        "ridge_density": float(binary.sum() / max(int(mask.sum()), 1)),
    }


def compare_zone_structure(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray) -> dict[str, Any]:
    pa = _patch(image_a, mask_a)
    pb = _patch(image_b, mask_b)
    if pa is None or pb is None:
        return {"structure_status": "insufficient_roi"}
    gray_a, local_a = pa
    gray_b, local_b = pb
    reg = register_patches(gray_a, gray_b, local_a, local_b)
    if reg.get("status") != "registered":
        return {"structure_status": str(reg.get("status")), **{f"registration_{k}": v for k, v in reg.items() if k != "status"}}
    overlap = reg.pop("overlap")
    aligned_b = reg.pop("aligned_b")
    ridge_a, ori_a = _ridge_probability(gray_a, overlap)
    ridge_b, ori_b = _ridge_probability(aligned_b, overlap)
    skel_a = _skeleton_metrics(ridge_a, overlap)
    skel_b = _skeleton_metrics(ridge_b, overlap)
    ridge_delta = float(np.mean(np.abs(ridge_a[overlap] - ridge_b[overlap])))
    # Blur-matched robustness: compare again after the same slight blur on both.
    blur_a = cv2.GaussianBlur(gray_a, (0, 0), 0.8)
    blur_b = cv2.GaussianBlur(aligned_b, (0, 0), 0.8)
    ridge_blur_a, _ = _ridge_probability(blur_a, overlap)
    ridge_blur_b, _ = _ridge_probability(blur_b, overlap)
    ridge_blur_delta = float(np.mean(np.abs(ridge_blur_a[overlap] - ridge_blur_b[overlap])))
    orientation_delta = float(np.linalg.norm(np.asarray(list(ori_a.values())) - np.asarray(list(ori_b.values()))))
    return {
        "structure_status": "measured",
        "structure_schema": SCHEMA,
        **{f"registration_{k}": v for k, v in reg.items() if k != "status"},
        "registered_ssim": _ssim(gray_a, aligned_b, overlap),
        "ridge_map_delta": ridge_delta,
        "ridge_blur_matched_delta": ridge_blur_delta,
        "ridge_blur_stability": float(abs(ridge_delta - ridge_blur_delta)),
        "ridge_orientation_delta": orientation_delta,
        "skeleton_length_delta_abs": abs(int(skel_a["skeleton_pixels"]) - int(skel_b["skeleton_pixels"])),
        "skeleton_component_delta_abs": abs(int(skel_a["skeleton_component_count"]) - int(skel_b["skeleton_component_count"])),
        "skeleton_endpoint_delta_abs": abs(int(skel_a["skeleton_endpoint_count"]) - int(skel_b["skeleton_endpoint_count"])),
        "skeleton_branchpoint_delta_abs": abs(int(skel_a["skeleton_branchpoint_count"]) - int(skel_b["skeleton_branchpoint_count"])),
        "ridge_density_a": skel_a["ridge_density"],
        "ridge_density_b": skel_b["ridge_density"],
        "policy": "Technical line-structure comparison; not age, diagnosis, material, or identity inference.",
    }
