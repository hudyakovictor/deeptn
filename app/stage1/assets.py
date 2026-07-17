from __future__ import annotations
from .masks import CHANNEL_NAMES
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import to_original_image

CROP_WIDTH = 424
CROP_HEIGHT = 500
CROP_MARGIN = 0.25


def _bbox(points: np.ndarray, shape: tuple[int, ...], margin: float = CROP_MARGIN) -> list[int]:
    h, w = shape[:2]
    p = np.asarray(points, np.float32)
    x1, y1 = np.floor(p.min(axis=0)).astype(int)
    x2, y2 = np.ceil(p.max(axis=0)).astype(int)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    dx, dy = int(round(bw * margin)), int(round(bh * margin))
    x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
    x2, y2 = min(w, x2 + dx), min(h, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("invalid landmark crop")
    return [x1, y1, x2 - x1, y2 - y1]


def _letterbox(image: np.ndarray, width: int = CROP_WIDTH, height: int = CROP_HEIGHT) -> tuple[np.ndarray, dict[str, float]]:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((height, width, image.shape[2]), image.dtype)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, {"scale": float(scale), "offset_x": int(ox), "offset_y": int(oy), "content_width": int(nw), "content_height": int(nh)}


def save_image_assets(source: Path, bgr: np.ndarray, ldm106_original: np.ndarray, out: Path, save_original: bool = True) -> tuple[dict[str, str], dict[str, Any]]:
    files: dict[str, str] = {}
    if save_original:
        original_name = "original" + source.suffix.lower()
        shutil.copy2(source, out / original_name)
        files["original"] = original_name
    bbox = _bbox(ldm106_original[:, :2], bgr.shape)
    x, y, w, h = bbox
    crop = bgr[y:y + h, x:x + w]
    face, transform = _letterbox(crop)
    if not cv2.imwrite(str(out / "face_crop.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("failed to write face_crop.jpg")
    side = min(face.shape[:2]); yy = (face.shape[0] - side) // 2; xx = (face.shape[1] - side) // 2
    thumb = cv2.resize(face[yy:yy + side, xx:xx + side], (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "thumb.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
    files.update({"face_crop": "face_crop.jpg", "thumbnail": "thumb.jpg"})
    return files, {"bbox_original": bbox, "letterbox": transform, "crop_source": "ldm106_projection"}


def technical_quality(bgr: np.ndarray, face_bbox: list[int], mask: np.ndarray | None, combined_visible: np.ndarray) -> dict[str, float | int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_bbox
    face_gray = gray[y:y + h, x:x + w]
    lap = cv2.Laplacian(face_gray, cv2.CV_64F)
    gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    med = cv2.medianBlur(face_gray, 3)
    diff = np.abs(face_gray.astype(np.float32) - med.astype(np.float32))
    vx, vy = max(float(np.var(gx)), 1e-8), max(float(np.var(gy)), 1e-8)
    out: dict[str, float | int] = {
        "face_bbox_width": int(w), "face_bbox_height": int(h),
        "face_bbox_area_ratio": float(w * h / max(bgr.shape[0] * bgr.shape[1], 1)),
        "laplacian_variance": float(np.var(lap)),
        "tenengrad_mean": float(np.mean(gx * gx + gy * gy)),
        "noise_residual_mean": float(np.mean(diff)),
        "gradient_anisotropy": float(max(vx / vy, vy / vx)),
        "combined_visible_fraction": float(np.mean(np.asarray(combined_visible, bool))),
    }
    out["skin_mask_coverage"] = float(np.mean(mask > 0)) if mask is not None else 0.0
    return out


def _uv_domain_mask(uv: np.ndarray, triangles: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize the complete UV topology; independent from photo visibility."""
    h, w = shape
    coords = np.asarray(uv, np.float32)[:, :2].copy()
    if coords.size == 0:
        return np.zeros(shape, bool)
    if float(np.nanmax(np.abs(coords))) <= 1.5:
        coords[:, 0] *= max(w - 1, 1)
        coords[:, 1] = (1.0 - coords[:, 1]) * max(h - 1, 1)
    coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
    pixels = np.rint(coords).astype(np.int32)
    domain = np.zeros(shape, np.uint8)
    for tri in np.asarray(triangles, np.int64):
        cv2.fillConvexPoly(domain, pixels[tri], 1)
    return domain.astype(bool)


def _complete_uv_texture(
    texture: np.ndarray,
    observed: np.ndarray,
    is_original: np.ndarray,
    uv: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a hole-free visual texture without changing scientific masks.

    Observed pixels are preserved exactly. Hidden UV areas are filled first from
    the symmetric UV side and then by deterministic inpainting. The returned
    completion mask records which pixels are synthetic, so Stage 2 never treats
    them as observed evidence.
    """
    raw = np.asarray(texture, np.uint8)
    observed = np.asarray(observed, bool)
    is_original = np.asarray(is_original, bool)
    shape = raw.shape[:2]
    domain = _uv_domain_mask(uv, triangles, shape)
    source = observed & is_original & np.any(raw > 0, axis=2)
    completed = raw.copy()
    method = np.zeros(shape, np.uint8)  # 0 observed/outside, 2 inpainted, 3 median fallback

    filled = source
    remaining = domain & ~filled
    if np.any(remaining) and np.any(filled):
        completed = cv2.inpaint(completed, remaining.astype(np.uint8) * 255, 5.0, cv2.INPAINT_TELEA)
        method[remaining] = 2

    still_black = domain & (completed.mean(axis=2) < 10)
    if np.any(still_black):
        samples = completed[filled]
        fallback = np.median(samples, axis=0).astype(np.uint8) if samples.size else np.array([128, 128, 128], np.uint8)
        completed[still_black] = fallback
        method[still_black] = 3
    completed[~domain] = 0
    return completed, method, domain


def _confidence_preview(confidence_01: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Soft grayscale confidence preview; never threshold to black/white."""
    preview = np.round(np.clip(confidence_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview[~np.asarray(domain, bool)] = 0
    return preview


def _analysis_mesh_arrays(bundle: Any) -> dict[str, np.ndarray]:
    """Compact visibility-cut mesh for NPZ analysis; OBJ remains full topology."""
    triangles = np.asarray(bundle.triangles, np.int64)
    visible_vertices = np.asarray(bundle.combined_visible, bool)
    triangle_mask = np.all(visible_vertices[triangles], axis=1)
    full_triangle_indices = np.flatnonzero(triangle_mask).astype(np.int64)
    selected = triangles[triangle_mask]
    used = np.unique(selected.reshape(-1)).astype(np.int64) if selected.size else np.empty(0, np.int64)
    used_mask = np.zeros(len(visible_vertices), bool)
    used_mask[used] = True
    remap = np.full(len(visible_vertices), -1, np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_triangles = remap[selected] if selected.size else np.empty((0, 3), np.int64)
    return {
        "analysis_mesh_vertex_indices_full": used,
        "analysis_mesh_triangle_indices_full": full_triangle_indices,
        "analysis_mesh_vertices_object": np.asarray(bundle.vertices_object, np.float32)[used],
        "analysis_mesh_vertices_normalized": np.asarray(bundle.vertices_object_normalized, np.float32)[used],
        "analysis_mesh_normals_object": np.asarray(bundle.normals_object, np.float32)[used],
        "analysis_mesh_uv_coords": np.asarray(bundle.uv_coords, np.float32)[used],
        "analysis_mesh_triangles": compact_triangles,
        "analysis_mesh_vertex_mask_packbits": np.packbits(used_mask.astype(np.uint8), bitorder="little"),
        "analysis_mesh_triangle_mask_packbits": np.packbits(triangle_mask.astype(np.uint8), bitorder="little"),
    }


def save_uv_and_mesh(bgr: np.ndarray, bundle: Any, out: Path, uv_size: int) -> tuple[dict[str, str], dict[str, np.ndarray], dict[str, Any]]:
    from uv_module import HDUVConfig, HDUVTextureGenerator

    vertices_2d = to_original_image(bundle.vertices_image_224, bundle.trans_params)
    recon = {
        "vertices": bundle.vertices_camera,
        "vertices_3d": bundle.vertices_camera,
        "vertices_2d": vertices_2d,
        "triangles": bundle.triangles,
        "uv_coords": bundle.uv_coords,
        "normals_3d": bundle.normals_posed,
        "alpha_sh": bundle.alpha_sh,
    }
    cfg = HDUVConfig(uv_size=int(uv_size), super_sample=2, enable_delighting=False, force_all_triangles_visible=False, device="cpu")

    analysis, beauty, observed, confidence, aux = HDUVTextureGenerator(cfg).generate(bgr, recon)
    observed_bool = np.asarray(observed, bool)
    is_original_bool = np.asarray(aux.get("uv_is_original", observed), bool)
    confidence_f32 = np.asarray(confidence, np.float32)
    confidence_f32 = np.nan_to_num(confidence_f32, nan=0.0, posinf=1.0, neginf=0.0)
    if confidence_f32.size and float(np.max(confidence_f32)) > 1.5:
        confidence_01 = np.clip(confidence_f32 / 255.0, 0.0, 1.0)
    else:
        confidence_01 = np.clip(confidence_f32, 0.0, 1.0)
    confidence_u8 = np.round(confidence_01 * 255.0).astype(np.uint8)
    valid_threshold = 0.50
    valid_mask = observed_bool & is_original_bool & (confidence_01 >= valid_threshold)

    completed_texture, completion_method, uv_domain = _complete_uv_texture(
        np.asarray(beauty, np.uint8), observed_bool, is_original_bool,
        bundle.uv_coords, bundle.triangles,
    )
    if not cv2.imwrite(str(out / "uv_texture.png"), completed_texture):
        raise OSError("failed to write completed UV texture")
    if not cv2.imwrite(str(out / "uv_observed_texture.png"), np.asarray(beauty, np.uint8)):
        raise OSError("failed to write observed UV texture")
    if not cv2.imwrite(str(out / "uv_confidence.png"), _confidence_preview(confidence_01, uv_domain)):
        raise OSError("failed to write continuous UV confidence preview")

    tri_visibility = np.asarray(aux.get("tri_visibility", []), np.float16)
    np.savez_compressed(
        out / "uv.npz",
        texture_bgr=completed_texture,
        texture_observed_bgr=np.asarray(beauty, np.uint8),
        confidence=confidence_01.astype(np.float16),
        confidence_u8=confidence_u8,
        observed_mask=observed_bool,
        is_original_mask=is_original_bool,
        valid_mask=valid_mask,
        uv_domain_mask=uv_domain,
        completion_method=completion_method,
        synthetic_texture_mask=(completion_method > 0),
        tri_visibility=tri_visibility,
        uv_shape=np.asarray(observed_bool.shape, np.int32),
        valid_threshold=np.asarray([valid_threshold], np.float32),
        uv_coords=np.asarray(bundle.uv_coords, np.float32),
        semantics=np.asarray(
            "uv.npz: texture_bgr is hole-free visual/morphing texture; texture_observed_bgr is raw projection; "
            "synthetic_texture_mask marks completion pixels and must never be used as observed evidence; "
            "confidence is continuous 0..1; valid_mask = observed AND original AND confidence >= threshold"
        ),
    )

    uv_arrays = {
        "uv_observed_mask_packbits": np.packbits(observed_bool.astype(np.uint8).reshape(-1), bitorder="little"),
        "uv_is_original_packbits": np.packbits(is_original_bool.astype(np.uint8).reshape(-1), bitorder="little"),
        "uv_valid_mask_packbits": np.packbits(valid_mask.astype(np.uint8).reshape(-1), bitorder="little"),
        "uv_confidence": confidence_01.astype(np.float16),
        "tri_visibility": tri_visibility,
        "uv_shape": np.asarray(observed_bool.shape, np.int32),
        **_analysis_mesh_arrays(bundle),
    }
    uv_meta = {
        "observed_coverage": float(np.mean(observed_bool)),
        "original_coverage": float(np.mean(is_original_bool)),
        "valid_coverage": float(np.mean(valid_mask)),
        "valid_threshold": float(valid_threshold),
        "mean_confidence_observed": float(np.mean(confidence_01[observed_bool])) if np.any(observed_bool) else 0.0,
        "confidence_semantics": "uv_confidence.png is a soft grayscale preview; exact confidence remains 0..1 in uv.npz",
        "texture_semantics": "uv_texture.png is hole-free for morphing/display; uv_observed_texture.png and scientific masks preserve the raw observed projection",
        "synthetic_texture_fraction": float(np.mean((completion_method > 0) & uv_domain)) if np.any(uv_domain) else 0.0,
    }
    _write_obj(
        out / "mesh.obj", out / "mesh.mtl",
        bundle.vertices_object_normalized, bundle.normals_object,
        bundle.uv_coords, bundle.triangles,
        texture_reference="uv_texture.png",
    )
    files = {
        "uv_texture": "uv_texture.png",
        "uv_observed_texture": "uv_observed_texture.png",
        "uv_confidence": "uv_confidence.png",
        "uv_data": "uv.npz",
        "morphing_mesh": "mesh.obj",
        "morphing_mesh_material": "mesh.mtl",
    }
    return files, uv_arrays, uv_meta


def _write_obj(obj_path: Path, mtl_path: Path, vertices: np.ndarray, normals: np.ndarray, uv: np.ndarray, triangles: np.ndarray, texture_reference: str = "uv_texture.png") -> None:
    if len(vertices) != len(normals) or len(vertices) != len(uv):
        raise ValueError("OBJ vertex/normal/UV counts differ")
    mtl_path.write_text(f"newmtl face_material\nKa 0.2 0.2 0.2\nKd 0.8 0.8 0.8\nKs 0 0 0\nillum 2\nmap_Kd {texture_reference}\n", encoding="utf-8")
    with obj_path.open("w", encoding="utf-8") as f:
        f.write("mtllib mesh.mtl\nusemtl face_material\n")
        for x, y, z in vertices: f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for u, v in uv[:, :2]: f.write(f"vt {u:.8f} {v:.8f}\n")
        for x, y, z in normals: f.write(f"vn {x:.8f} {y:.8f} {z:.8f}\n")
        for tri in triangles:
            a, b, c = (int(x) + 1 for x in tri)
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")


def save_face_mask(bgr: np.ndarray, hard_mask: np.ndarray | None, bbox: list[int], out: Path) -> dict[str, str] | None:
    """
    Create and save:
      - face_mask.png: visual RGBA 424x500 face crop with skin mask in alpha;
      - face_mask.npz: lossless numeric mask bundle for future texture/quality analysis.
    
    Args:
        bgr: Full image BGR
        hard_mask: Full image size binary mask (bool or 0/255) or None if projection failed
        bbox: [x, y, w, h] face crop bbox in original image
        out: Output directory
    
    Returns:
        File mapping or None if mask unavailable
    """
    if hard_mask is None or hard_mask.size == 0:
        return None
    
    # Convert to uint8 if boolean
    if hard_mask.dtype == bool:
        hard_mask = hard_mask.astype(np.uint8) * 255
    elif hard_mask.dtype != np.uint8:
        hard_mask = np.clip(hard_mask, 0, 255).astype(np.uint8)
    
    x, y, w, h = bbox
    H, W = hard_mask.shape[:2]
    
    # Clamp bbox to image bounds
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(0, min(x + w, W))
    y2 = max(0, min(y + h, H))
    w = x2 - x1
    h = y2 - y1
    
    if w <= 0 or h <= 0:
        return None
    
    # Extract face crop and mask
    crop = bgr[y1:y2, x1:x2]
    mask_crop = hard_mask[y1:y2, x1:x2]
    
    # Letterbox to 424x500 (same as face_crop)
    face, transform = _letterbox(crop)
    mh, mw = mask_crop.shape[:2]
    scale = transform["scale"]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = transform["offset_x"], transform["offset_y"]
    
    # Resize mask with same letterbox transform
    mask_resized = cv2.resize(mask_crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_canvas = np.zeros((500, 424), np.uint8)
    if oy + nh <= 500 and ox + nw <= 424:
        mask_canvas[oy:oy + nh, ox:ox + nw] = mask_resized
    
    # Create RGBA visual preview.
    rgba = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_canvas
    if not cv2.imwrite(str(out / "face_mask.png"), rgba):
        raise OSError("failed to write face_mask.png")

    mask_original_bool = hard_mask > 0
    mask_crop_bool = mask_crop > 0
    mask_face_bool = mask_canvas > 0
    np.savez_compressed(
        out / "face_mask.npz",
        mask_original=mask_original_bool,
        mask_crop=mask_crop_bool,
        mask_face=mask_face_bool,
        mask_alpha_u8=mask_canvas,
        bbox_original=np.asarray([x1, y1, w, h], np.int32),
        original_shape=np.asarray(hard_mask.shape[:2], np.int32),
        crop_shape=np.asarray(mask_crop.shape[:2], np.int32),
        face_shape=np.asarray(mask_canvas.shape[:2], np.int32),
        letterbox_scale=np.asarray([float(scale)], np.float32),
        letterbox_offset=np.asarray([int(ox), int(oy)], np.int32),
        letterbox_content_size=np.asarray([int(nw), int(nh)], np.int32),
        skin_pixels_original=np.asarray([int(np.count_nonzero(mask_original_bool))], np.int64),
        skin_pixels_crop=np.asarray([int(np.count_nonzero(mask_crop_bool))], np.int64),
        skin_pixels_face=np.asarray([int(np.count_nonzero(mask_face_bool))], np.int64),
        skin_coverage_original=np.asarray([float(np.mean(mask_original_bool))], np.float32),
        skin_coverage_crop=np.asarray([float(np.mean(mask_crop_bool))], np.float32),
        skin_coverage_face=np.asarray([float(np.mean(mask_face_bool))], np.float32),
        semantics=np.asarray(
            "face_mask.npz: numeric skin/face mask bundle; mask_original is in original image space; "
            "mask_crop is the original-resolution bbox crop; mask_face is the 424x500 letterboxed preview alpha"
        ),
    )
    return {"face_mask": "face_mask.png", "face_mask_data": "face_mask.npz"}


def save_semantic_channels(bundle: Any, out: Path) -> str:
    """
    Save semantic_channels.npz from mask bundle.
    """
    np.savez_compressed(
        out / "semantic_channels.npz",
        channels_224=bundle.channels_224,
        channel_names=np.asarray(CHANNEL_NAMES),
        skin_soft_224=bundle.soft_224.astype(np.float16),
        skin_hard_224=bundle.hard_224.astype(np.uint8),
    )
    return "semantic_channels.npz"
