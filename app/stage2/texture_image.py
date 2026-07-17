from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .texture_structure import compare_zone_structure

TEXTURE_IMAGE_SCHEMA = "deeputin-stage2-image-texture-v1.2"
LBP_BINS = 10
GLCM_LEVELS = 16

try:  # Optional: use scikit-image when the production environment provides it.
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern  # type: ignore
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - exercised in the sandbox fallback path.
    graycomatrix = graycoprops = local_binary_pattern = None
    _HAS_SKIMAGE = False


def _unpack_mask(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    n = int(shape[0] * shape[1])
    arr = np.unpackbits(np.asarray(bits, np.uint8), bitorder="little")[:n]
    return arr.reshape(shape).astype(bool)


def _image_path(record: Any) -> Path | None:
    d = getattr(record, "record_dir", None)
    if d is None:
        return None
    d = Path(d)
    try:
        info = json.loads((d / "info.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    files = info.get("files") or {}
    rel = files.get("original") or files.get("face_crop")
    if not rel:
        return None
    p = d / str(rel)
    return p if p.is_file() else None


def _load_texture(record: Any) -> dict[str, Any]:
    d = getattr(record, "record_dir", None)
    if d is None:
        return {"status": "missing_record_dir"}
    d = Path(d)
    qz_path = d / "quality_zones.npz"
    img_path = _image_path(record)
    if img_path is None or not qz_path.is_file():
        return {"status": "missing_image_or_quality_zones"}
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "cannot_decode_image"}
    try:
        with np.load(qz_path, allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["original_shape"])
            if img.shape[:2] != shape:
                return {"status": "shape_mismatch", "image_shape": img.shape[:2], "zone_shape": shape}
            names = [str(x) for x in z["zone_names"]]
            statuses = [str(x) for x in z["zone_status"]]
            pixels = z["zone_texture_pixels"].astype(np.int64)
            score = z["zone_texture_score"].astype(np.float32)
            bits = z["zone_texture_roi_original_packbits"]
            masks = {name: _unpack_mask(bits[i], shape) for i, name in enumerate(names)}
        return {"status": "ok", "image": img, "zone_names": names, "zone_status": statuses, "zone_pixels": pixels, "zone_score": score, "masks": masks}
    except Exception as exc:
        return {"status": "load_error", "error": str(exc)}


def _lbp_histogram(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Uniform LBP histogram; numpy fallback keeps the module dependency-light."""
    if int(mask.sum()) < 64:
        return np.zeros(LBP_BINS, np.float32)
    if _HAS_SKIMAGE and local_binary_pattern is not None:
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp[mask], bins=np.arange(LBP_BINS + 1), range=(0, LBP_BINS), density=False)
    else:
        g = gray.astype(np.int16)
        center = g[1:-1, 1:-1]
        code = np.zeros_like(center, np.uint8)
        neighbors = (
            g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:],
            g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2],
        )
        for bit, nb in enumerate(neighbors):
            code |= ((nb >= center).astype(np.uint8) << bit)
        transitions = np.zeros_like(code, np.uint8)
        prev = (code >> 7) & 1
        for bit in range(8):
            cur = (code >> bit) & 1
            transitions += (cur != prev).astype(np.uint8)
            prev = cur
        pop = np.unpackbits(code.reshape(-1, 1), axis=1).sum(axis=1).reshape(code.shape)
        uniform = transitions <= 2
        lbp = np.full(gray.shape, 9, np.uint8)
        lbp[1:-1, 1:-1] = np.where(uniform, pop, 9).astype(np.uint8)
        hist, _ = np.histogram(lbp[mask], bins=np.arange(LBP_BINS + 1), range=(0, LBP_BINS), density=False)
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.0)


def _glcm_stats(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if int(mask.sum()) < 64:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}
    ys, xs = np.where(mask)
    y0, y1 = max(int(ys.min()) - 2, 0), min(int(ys.max()) + 3, gray.shape[0])
    x0, x1 = max(int(xs.min()) - 2, 0), min(int(xs.max()) + 3, gray.shape[1])
    roi = gray[y0:y1, x0:x1]
    roi_mask = mask[y0:y1, x0:x1]
    if roi.size == 0 or int(roi_mask.sum()) < 64:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}
    quant = np.clip((roi.astype(np.float32) / 256.0 * GLCM_LEVELS).astype(np.uint8), 0, GLCM_LEVELS - 1)
    if _HAS_SKIMAGE and graycomatrix is not None and graycoprops is not None:
        # Masked pixels are filled with the zone median to avoid measuring the bbox edge.
        filled = quant.copy()
        filled[~roi_mask] = int(np.median(quant[roi_mask]))
        glcm = graycomatrix(filled, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=GLCM_LEVELS, symmetric=True, normed=True)
        return {
            "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
            "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
            "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
            "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        }
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        src = quant[max(0, -dy): quant.shape[0] - max(0, dy), max(0, -dx): quant.shape[1] - max(0, dx)]
        dst = quant[max(0, dy): quant.shape[0] - max(0, -dy), max(0, dx): quant.shape[1] - max(0, -dx)]
        m1 = roi_mask[max(0, -dy): roi_mask.shape[0] - max(0, dy), max(0, -dx): roi_mask.shape[1] - max(0, dx)]
        m2 = roi_mask[max(0, dy): roi_mask.shape[0] - max(0, -dy), max(0, dx): roi_mask.shape[1] - max(0, -dx)]
        valid = m1 & m2
        if int(valid.sum()):
            pairs.append((src[valid], dst[valid]))
    if not pairs:
        return {"glcm_contrast": 0.0, "glcm_homogeneity": 0.0, "glcm_energy": 0.0, "glcm_correlation": 0.0}
    mat = np.zeros((GLCM_LEVELS, GLCM_LEVELS), np.float64)
    for src, dst in pairs:
        np.add.at(mat, (src.reshape(-1), dst.reshape(-1)), 1)
        np.add.at(mat, (dst.reshape(-1), src.reshape(-1)), 1)
    mat /= max(float(mat.sum()), 1.0)
    i, j = np.indices(mat.shape)
    contrast = float(np.sum(mat * (i - j) ** 2))
    homogeneity = float(np.sum(mat / (1.0 + np.abs(i - j))))
    energy = float(np.sqrt(np.sum(mat * mat)))
    mi = float(np.sum(i * mat)); mj = float(np.sum(j * mat))
    si = float(np.sqrt(np.sum(((i - mi) ** 2) * mat))); sj = float(np.sqrt(np.sum(((j - mj) ** 2) * mat)))
    corr = float(np.sum((i - mi) * (j - mj) * mat) / max(si * sj, 1e-8))
    return {"glcm_contrast": contrast, "glcm_homogeneity": homogeneity, "glcm_energy": energy, "glcm_correlation": corr}


def _frequency_ratio(gray: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 64:
        return 0.0
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if min(patch.shape) < 8:
        return 0.0
    patch = patch - float(np.mean(patch))
    spec = np.abs(np.fft.rfft2(patch)) ** 2
    yy, xx = np.indices(spec.shape)
    radius = np.sqrt((yy / max(spec.shape[0] - 1, 1)) ** 2 + (xx / max(spec.shape[1] - 1, 1)) ** 2)
    high = float(spec[radius > 0.35].sum())
    total = float(spec.sum())
    return high / max(total, 1e-8)


def _erode_roi(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove warp/hair/eyebrow-prone ROI borders using a scale-aware erosion."""
    m = np.asarray(mask, np.uint8)
    if int(m.sum()) < 64:
        return m.astype(bool), 0
    ys, xs = np.where(m > 0)
    radius = max(1, min(7, int(round(0.05 * min(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(m, kernel, iterations=1).astype(bool)
    return (eroded if int(eroded.sum()) >= 64 else m.astype(bool)), radius


def _entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    vals = gray[mask]
    if vals.size < 64:
        return 0.0
    hist = np.bincount(vals.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(float(hist.sum()), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _gabor_profile(gray: np.ndarray, mask: np.ndarray) -> list[float]:
    """Compact orientation profile; values are normalized mean absolute responses."""
    if int(mask.sum()) < 64:
        return [0.0] * 8
    src = gray.astype(np.float32) / 255.0
    values: list[float] = []
    for sigma, lambd in ((2.0, 4.0), (3.0, 7.0)):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            kernel = cv2.getGaborKernel((15, 15), sigma, theta, lambd, 0.55, 0, ktype=cv2.CV_32F)
            response = cv2.filter2D(src, cv2.CV_32F, kernel)
            values.append(float(np.mean(np.abs(response[mask]))))
    scale = max(max(values, default=0.0), 1e-8)
    return [float(v / scale) for v in values]


def _patch_profile(gray: np.ndarray, mask: np.ndarray, rows: int = 3, cols: int = 4) -> dict[str, float | int]:
    """Patch-grid aggregation prevents one highlight from dominating a whole ROI."""
    ys, xs = np.where(mask)
    if ys.size < 64:
        return {"usable_patch_count": 0, "patch_entropy_median": 0.0, "patch_entropy_mad": 0.0, "patch_hf_median": 0.0}
    y_edges = np.linspace(int(ys.min()), int(ys.max()) + 1, rows + 1).astype(int)
    x_edges = np.linspace(int(xs.min()), int(xs.max()) + 1, cols + 1).astype(int)
    entropies: list[float] = []
    high_freq: list[float] = []
    for yi in range(rows):
        for xi in range(cols):
            patch_mask = np.zeros_like(mask, bool)
            patch_mask[y_edges[yi]:y_edges[yi + 1], x_edges[xi]:x_edges[xi + 1]] = True
            patch_mask &= mask
            if int(patch_mask.sum()) < 64:
                continue
            entropies.append(_entropy(gray, patch_mask))
            high_freq.append(_frequency_ratio(gray, patch_mask))
    if not entropies:
        return {"usable_patch_count": 0, "patch_entropy_median": 0.0, "patch_entropy_mad": 0.0, "patch_hf_median": 0.0}
    med = float(np.median(entropies))
    return {"usable_patch_count": len(entropies), "patch_entropy_median": med, "patch_entropy_mad": float(np.median(np.abs(np.asarray(entropies) - med))), "patch_hf_median": float(np.median(high_freq))}


def _stats(img: np.ndarray, mask: np.ndarray) -> dict[str, float | int | list[float]]:
    if int(mask.sum()) < 64:
        return {"texture_pixels": int(mask.sum()), "gray_mean": 0.0, "gray_std": 0.0, "laplacian_var": 0.0, "gradient_energy": 0.0, "high_frequency_ratio": 0.0, "lbp_histogram": [0.0] * LBP_BINS, **_glcm_stats(np.zeros(mask.shape, np.uint8), mask)}
    mask, erosion_radius = _erode_roi(mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = gray[mask].astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    glcm = _glcm_stats(gray, mask)
    gabor = _gabor_profile(gray, mask)
    patches = _patch_profile(gray, mask)
    return {
        "texture_pixels": int(mask.sum()),
        "gray_mean": float(np.mean(vals)),
        "gray_std": float(np.std(vals)),
        "laplacian_var": float(np.var(lap[mask])),
        "gradient_energy": float(np.mean((gx * gx + gy * gy)[mask])),
        "high_frequency_ratio": float(_frequency_ratio(gray, mask)),
        "lbp_histogram": _lbp_histogram(gray, mask).tolist(),
        "local_entropy": _entropy(gray, mask),
        "gabor_profile": gabor,
        "roi_erosion_radius": erosion_radius,
        **patches,
        **glcm,
    }


def texture_pair_deltas(a: Any, b: Any, pair_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ta = _load_texture(a)
    tb = _load_texture(b)
    if ta.get("status") != "ok" or tb.get("status") != "ok":
        return {
            "texture_image_status": "unavailable",
            "texture_image_error_a": ta.get("status"),
            "texture_image_error_b": tb.get("status"),
        }, []
    rows: list[dict[str, Any]] = []
    max_lap_delta = 0.0
    max_grad_delta = 0.0
    max_lbp_chi2 = 0.0
    max_glcm_contrast_delta = 0.0
    max_high_freq_delta = 0.0
    max_entropy_delta = 0.0
    max_gabor_delta = 0.0
    max_ridge_delta = 0.0
    min_registered_ssim = 1.0
    registered_structure_count = 0
    usable_count = 0
    common = sorted(set(ta["zone_names"]) & set(tb["zone_names"]))
    for name in common:
        ma = ta["masks"][name]
        mb = tb["masks"][name]
        sa = _stats(ta["image"], ma)
        sb = _stats(tb["image"], mb)
        usable = bool(sa["texture_pixels"] >= 2500 and sb["texture_pixels"] >= 2500)
        if usable:
            usable_count += 1
        lap_delta = abs(float(sa["laplacian_var"]) - float(sb["laplacian_var"]))
        grad_delta = abs(float(sa["gradient_energy"]) - float(sb["gradient_energy"]))
        lbp_a = np.asarray(sa.get("lbp_histogram", [0.0] * LBP_BINS), np.float32)
        lbp_b = np.asarray(sb.get("lbp_histogram", [0.0] * LBP_BINS), np.float32)
        lbp_chi2 = float(0.5 * np.sum(((lbp_a - lbp_b) ** 2) / np.maximum(lbp_a + lbp_b, 1e-8)))
        glcm_contrast_delta = abs(float(sa["glcm_contrast"]) - float(sb["glcm_contrast"]))
        high_freq_delta = abs(float(sa["high_frequency_ratio"]) - float(sb["high_frequency_ratio"]))
        entropy_delta = abs(float(sa.get("local_entropy", 0.0)) - float(sb.get("local_entropy", 0.0)))
        ga = np.asarray(sa.get("gabor_profile", [0.0] * 8), np.float32)
        gb = np.asarray(sb.get("gabor_profile", [0.0] * 8), np.float32)
        gabor_delta = float(np.linalg.norm(ga - gb) / np.sqrt(max(len(ga), 1)))
        structure = compare_zone_structure(ta["image"], ma, tb["image"], mb) if usable else {"structure_status": "quality_gate_failed"}
        if structure.get("structure_status") == "measured":
            registered_structure_count += 1
            max_ridge_delta = max(max_ridge_delta, float(structure.get("ridge_map_delta", 0.0)))
            min_registered_ssim = min(min_registered_ssim, float(structure.get("registered_ssim", 1.0)))
        max_lap_delta = max(max_lap_delta, lap_delta)
        max_grad_delta = max(max_grad_delta, grad_delta)
        max_lbp_chi2 = max(max_lbp_chi2, lbp_chi2)
        max_glcm_contrast_delta = max(max_glcm_contrast_delta, glcm_contrast_delta)
        max_high_freq_delta = max(max_high_freq_delta, high_freq_delta)
        max_entropy_delta = max(max_entropy_delta, entropy_delta)
        max_gabor_delta = max(max_gabor_delta, gabor_delta)
        rows.append({
            "pair_id": pair_id,
            "zone": name,
            "texture_zone_usable": usable,
            "texture_pixels_a": sa["texture_pixels"],
            "texture_pixels_b": sb["texture_pixels"],
            "gray_mean_delta_abs": abs(float(sa["gray_mean"]) - float(sb["gray_mean"])),
            "gray_std_delta_abs": abs(float(sa["gray_std"]) - float(sb["gray_std"])),
            "laplacian_var_delta_abs": lap_delta,
            "gradient_energy_delta_abs": grad_delta,
            "lbp_chi2_delta": lbp_chi2,
            "glcm_contrast_delta_abs": glcm_contrast_delta,
            "glcm_homogeneity_delta_abs": abs(float(sa["glcm_homogeneity"]) - float(sb["glcm_homogeneity"])),
            "glcm_energy_delta_abs": abs(float(sa["glcm_energy"]) - float(sb["glcm_energy"])),
            "glcm_correlation_delta_abs": abs(float(sa["glcm_correlation"]) - float(sb["glcm_correlation"])),
            "high_frequency_ratio_delta_abs": high_freq_delta,
            "local_entropy_delta_abs": entropy_delta,
            "gabor_profile_rmse": gabor_delta,
            "usable_patch_count_a": sa.get("usable_patch_count", 0),
            "usable_patch_count_b": sb.get("usable_patch_count", 0),
            "patch_entropy_median_delta_abs": abs(float(sa.get("patch_entropy_median", 0.0)) - float(sb.get("patch_entropy_median", 0.0))),
            "patch_hf_median_delta_abs": abs(float(sa.get("patch_hf_median", 0.0)) - float(sb.get("patch_hf_median", 0.0))),
            "roi_erosion_radius_a": sa.get("roi_erosion_radius", 0),
            "roi_erosion_radius_b": sb.get("roi_erosion_radius", 0),
            **structure,
            "texture_backend": "scikit-image" if _HAS_SKIMAGE else "numpy_cv2_fallback",
            "policy": "image-space LBP/GLCM/frequency texture delta; quality/exposure/compression alternatives required",
        })
    status = "measured" if rows else "no_common_zones"
    return {
        "texture_image_status": status,
        "texture_image_zone_count": len(rows),
        "texture_image_usable_zone_count": usable_count,
        "texture_image_max_laplacian_delta": max_lap_delta,
        "texture_image_max_gradient_delta": max_grad_delta,
        "texture_image_max_lbp_chi2": max_lbp_chi2,
        "texture_image_max_glcm_contrast_delta": max_glcm_contrast_delta,
        "texture_image_max_high_frequency_delta": max_high_freq_delta,
        "texture_image_max_entropy_delta": max_entropy_delta,
        "texture_image_max_gabor_delta": max_gabor_delta,
        "texture_structure_registered_zone_count": registered_structure_count,
        "texture_structure_max_ridge_delta": max_ridge_delta,
        "texture_structure_min_registered_ssim": min_registered_ssim if registered_structure_count else 0.0,
        "texture_image_backend": "scikit-image" if _HAS_SKIMAGE else "numpy_cv2_fallback",
        "texture_image_schema": TEXTURE_IMAGE_SCHEMA,
    }, rows
