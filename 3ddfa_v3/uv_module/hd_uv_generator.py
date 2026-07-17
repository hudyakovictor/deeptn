from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .visibility import compute_triangle_visibility
from .uv_baker import UVBaker
from .detail import UVDetailEnhancer
from .inpaint_blend import UVBeautyPostprocessor

try:  # optional; keep generator usable when SH/de-lighting is unavailable
    from .delight import compute_shading_uv, albedo_from_texture
except Exception:  # pragma: no cover
    compute_shading_uv = None  # type: ignore
    albedo_from_texture = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class HDUVConfig:
    uv_size: int = 768
    super_sample: int = 2
    enable_delighting: bool = False
    enable_symmetry_fill: bool = True
    enable_detail_boost: bool = False
    detail_strength: float = 1.0
    unsharp_amount: float = 0.0
    detail_base_sigma_s_ratio: float = 0.02
    detail_base_sigma_r: float = 0.08
    use_barycentric_bake: bool = True
    force_all_triangles_visible: bool = False
    device: str = "cpu"
    use_fast_path: bool = False
    verbose: bool = False


def _as_vertices(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be Nx2/Nx3")
    return arr


def _as_triangles(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("triangles must be Tx3")
    return arr


def _uv_domain_mask(uv_coords: np.ndarray, triangles: np.ndarray, size: int) -> np.ndarray:
    uv = np.asarray(uv_coords, dtype=np.float32)[:, :2].copy()
    if uv.size == 0:
        return np.zeros((size, size), dtype=bool)
    if float(np.nanmax(np.abs(uv))) > 1.5:
        uv = uv / max(float(np.nanmax(np.abs(uv))), 1e-6)
    px = np.empty((uv.shape[0], 2), dtype=np.int32)
    px[:, 0] = np.rint(np.clip(uv[:, 0], 0, 1) * (size - 1)).astype(np.int32)
    px[:, 1] = np.rint((1.0 - np.clip(uv[:, 1], 0, 1)) * (size - 1)).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    for tri in triangles:
        cv2.fillConvexPoly(mask, px[tri].reshape(-1, 1, 2), 255)
    return mask > 0


def _hole_free_beauty(texture: np.ndarray, confidence: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill UV display/morphing holes without claiming they are observed data."""
    tex = np.clip(texture, 0, 255).astype(np.uint8).copy()
    conf = np.asarray(confidence, dtype=np.float32)
    domain = np.asarray(domain, dtype=bool)
    observed = domain & (conf > 0.0) & np.any(tex > 0, axis=2)
    method = np.zeros(domain.shape, dtype=np.uint8)  # 0 observed/outside, 1 symmetry, 2 inpaint, 3 fallback

    flipped_tex = cv2.flip(tex, 1)
    flipped_observed = cv2.flip(observed.astype(np.uint8), 1).astype(bool)
    mirror_fill = domain & ~observed & flipped_observed
    if np.any(mirror_fill):
        tex[mirror_fill] = flipped_tex[mirror_fill]
        method[mirror_fill] = 1

    filled = observed | mirror_fill
    remaining = domain & ~filled
    if np.any(remaining) and np.any(filled):
        tex = cv2.inpaint(tex, remaining.astype(np.uint8) * 255, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        method[remaining] = np.where(method[remaining] == 0, 2, method[remaining])

    still_empty = domain & ~np.any(tex > 0, axis=2)
    if np.any(still_empty):
        samples = tex[filled]
        fallback = np.median(samples, axis=0).astype(np.uint8) if samples.size else np.array([128, 128, 128], np.uint8)
        tex[still_empty] = fallback
        method[still_empty] = 3

    tex[~domain] = 0
    return tex, method


class HDUVTextureGenerator:
    """Production UV generator kept inside 3ddfa_v3/uv_module.

    Returns:
        analysis texture, beauty texture, observed mask, continuous confidence, aux.

    Scientific rule: observed/confidence masks describe only real projected pixels.
    Beauty texture may be symmetry/inpaint completed for display and morphing.
    """

    def __init__(self, config: Optional[HDUVConfig] = None) -> None:
        self.config = config or HDUVConfig()
        self.baker = UVBaker(uv_size=self.config.uv_size, super_sample=self.config.super_sample)
        self.detail = UVDetailEnhancer(
            detail_strength=self.config.detail_strength,
            unsharp_amount=self.config.unsharp_amount,
            base_sigma_s_ratio=self.config.detail_base_sigma_s_ratio,
            base_sigma_r=self.config.detail_base_sigma_r,
        )
        self.beauty_post = UVBeautyPostprocessor()

    def generate(
        self,
        image: np.ndarray,
        recon_dict: Dict[str, Any],
        debug_output_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("image must be HxWx3")

        vertices_2d = recon_dict.get("vertices_2d")
        if vertices_2d is None:
            raise ValueError("vertices_2d is mandatory but not found in recon_dict")
        vertices_2d = _as_vertices(vertices_2d, "vertices_2d")[:, :2]
        vertices_3d = _as_vertices(recon_dict.get("vertices_3d", recon_dict.get("vertices")), "vertices_3d")[:, :3]
        uv_coords = _as_vertices(recon_dict["uv_coords"], "uv_coords")[:, :2]
        triangles = _as_triangles(recon_dict["triangles"])

        if self.config.force_all_triangles_visible:
            tri_analysis = np.ones(triangles.shape[0], dtype=np.float32)
        else:
            tri_analysis = compute_triangle_visibility(
                vertices_3d=vertices_3d,
                triangles=triangles,
                vertices_2d=vertices_2d,
                image_size=img.shape[:2],
                mode="analysis",
                use_zbuffer=True,
                angle_threshold_deg=75.0,
                gamma=1.5,
                min_weight_floor=0.0,
            ).astype(np.float32)

        bake = self.baker.bake_via_barycentric if self.config.use_barycentric_bake else self.baker.bake
        tex_f, observed, confidence, is_original = bake(
            img,
            vertices_2d,
            uv_coords,
            triangles,
            tri_analysis,
        )
        tex_analysis = np.clip(tex_f, 0, 255).astype(np.uint8)
        confidence = np.clip(np.asarray(confidence, np.float32), 0.0, 1.0)
        observed = np.asarray(observed, bool)
        is_original = np.asarray(is_original, bool)

        # Optional display-only enhancement. Never changes observed masks.
        beauty_input = tex_analysis
        detail_strength_map = np.zeros(confidence.shape, dtype=np.float32)
        if self.config.enable_delighting and compute_shading_uv is not None and albedo_from_texture is not None:
            try:
                normals = recon_dict.get("normals_3d")
                alpha_sh = recon_dict.get("alpha_sh")
                if normals is not None and alpha_sh is not None:
                    normals = _as_vertices(normals, "normals_3d")[:, :3]
                    n = np.linalg.norm(normals, axis=1, keepdims=True)
                    normals_01 = np.where(n > 1e-8, normals / np.maximum(n, 1e-8), 0.0)
                    normals_uv = self.baker.bake_vertex_colors(uv_coords, triangles, (normals_01 + 1.0) * 0.5, size=self.config.uv_size)
                    shading_uv = compute_shading_uv(normals_uv, np.asarray(alpha_sh, np.float32))
                    beauty_input, _ = albedo_from_texture(beauty_input, shading_uv)
            except Exception as exc:  # display-only path; keep analysis safe
                logger.warning("[HDUV] de-lighting skipped: %s", exc)

        if self.config.enable_detail_boost:
            try:
                beauty_input, detail_strength_map = self.detail.enhance(
                    beauty_input,
                    observed,
                    confidence,
                )
            except Exception as exc:
                logger.warning("[HDUV] detail boost skipped: %s", exc)

        uv_domain = _uv_domain_mask(uv_coords, triangles, self.config.uv_size)
        beauty = self.beauty_post.process(
            beauty_input,
            observed,
            confidence,
            uv_coords=uv_coords,
            enable_symmetry=self.config.enable_symmetry_fill,
        )
        # Keep only the beauty postprocessor result (symmetry + color-correct +
        # seam smoothing, matching the v2 reference). Do NOT re-run the crude
        # mirror fill from _hole_free_beauty: it overwrites the color-corrected
        # symmetry with an uncorrected mirror and drops seam smoothing.
        observed_pixels = uv_domain & observed & np.any(beauty_input > 0, axis=2)
        completion_method = np.where(
            uv_domain & ~observed_pixels & np.any(beauty > 0, axis=2), 1, 0
        ).astype(np.uint8)

        aux: Dict[str, Any] = {
            "uv_is_original": is_original,
            "uv_domain_mask": uv_domain,
            "uv_completion_method": completion_method,
            "uv_synthetic_texture_mask": completion_method > 0,
            "tri_visibility": tri_analysis.astype(np.float32),
            "tri_visibility_analysis": tri_analysis.astype(np.float32),
            "tri_visibility_beauty": tri_analysis.astype(np.float32),
            "visibility_mode": "analysis_hard_for_observed__beauty_completed_for_display",
            "detail_strength_map": detail_strength_map.astype(np.float32),
        }

        if self.config.verbose:
            logger.info(
                "[HDUV] observed=%.3f synthetic=%.3f",
                float(np.mean(observed)),
                float(np.mean((completion_method > 0) & uv_domain)) if np.any(uv_domain) else 0.0,
            )
        return tex_analysis, beauty, observed, confidence, aux
