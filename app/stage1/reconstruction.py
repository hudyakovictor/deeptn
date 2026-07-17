from __future__ import annotations

import gc
import os
import platform
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import classify_pose, normalize_mesh, reprojection_stats, row_rotation_matrix, to_original_image
from .paths import install_runtime_paths, resolve_project_layout


@dataclass
class ReconstructionBundle:
    trans_params: np.ndarray
    angles_rad: np.ndarray
    angles_deg: np.ndarray
    pose_bin: str
    canonical_yaw: float
    rotation: np.ndarray
    translation: np.ndarray
    vertices_object: np.ndarray
    vertices_identity_only: np.ndarray
    vertices_object_normalized: np.ndarray
    vertices_bin_canonical: np.ndarray
    vertices_camera: np.ndarray
    vertices_image_224: np.ndarray
    normals_object: np.ndarray
    normals_posed: np.ndarray
    triangles: np.ndarray
    uv_coords: np.ndarray
    ldm106_indices: np.ndarray
    ldm134_indices: np.ndarray
    front_facing: np.ndarray
    renderer_visible: np.ndarray
    combined_visible: np.ndarray
    semantic_channels_224: np.ndarray
    alpha_full: np.ndarray
    alpha_id: np.ndarray
    alpha_exp: np.ndarray
    alpha_alb: np.ndarray
    alpha_sh: np.ndarray
    normalization_center: np.ndarray
    normalization_scale: float
    canonical_rotation: np.ndarray
    reprojection: dict[str, dict[str, float]]
    raw_results: dict[str, Any]

    def landmark_arrays(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for count, idx in ((106, self.ldm106_indices), (134, self.ldm134_indices)):
            key = f"ldm{count}"
            out[f"{key}_object"] = self.vertices_object[idx]
            out[f"{key}_object_normalized"] = self.vertices_object_normalized[idx]
            out[f"{key}_bin_canonical"] = self.vertices_bin_canonical[idx]
            out[f"{key}_camera"] = self.vertices_camera[idx]
            out[f"{key}_image_224"] = self.vertices_image_224[idx]
            out[f"{key}_front_facing"] = self.front_facing[idx].astype(np.uint8)
            out[f"{key}_renderer_visible"] = self.renderer_visible[idx].astype(np.uint8)
            out[f"{key}_visible"] = self.combined_visible[idx].astype(np.uint8)
        return out


class ReconstructionEngine:
    """One network inference; official renderer output is captured without patching 3DDFA."""

    def __init__(self, project_root: Path, device: str = "auto", detector: str = "retinaface", backbone: str = "resnet50"):
        self.layout = resolve_project_layout(project_root)
        install_runtime_paths(self.layout)
        self.project_root = self.layout.project_root
        self.three_ddfa_root = self.layout.three_ddfa_root
        self.device = self._resolve_device(device)
        self.detector_name = detector
        self.backbone = backbone
        self._check_assets()
        cwd = Path.cwd()
        try:
            os.chdir(self.three_ddfa_root)
            from face_box import face_box
            from model.recon import face_model
            args = Namespace(
                device=self.device, detector=detector, backbone=backbone,
                iscrop=True, ldm68=False, ldm106=True, ldm106_2d=False,
                ldm134=True, seg=True, seg_visible=True,
                useTex=True, extractTex=False, use_hd_uv=False,
            )
            self.model = face_model(args)
            self.detector = face_box(args).detector
        finally:
            os.chdir(cwd)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch
        if requested == "auto":
            # The bundled renderer only has reliable CPU or CUDA paths. MPS would
            # enter the nvdiffrast branch, so Apple Silicon intentionally uses CPU.
            if platform.system() == "Darwin":
                return "cpu"
            if torch.cuda.is_available():
                try:
                    import nvdiffrast.torch  # noqa: F401
                    return "cuda"
                except Exception:
                    return "cpu"
            return "cpu"
        if requested == "mps":
            raise ValueError("bundled 3DDFA renderer does not support MPS; use cpu")
        return requested

    def _check_assets(self) -> None:
        assets = self.layout.assets_dir
        weight = "net_recon.pth" if self.backbone == "resnet50" else "net_recon_mbnet.pth"
        required = [assets / "face_model.npy", assets / weight, assets / "large_base_net.pth"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing 3DDFA assets: " + ", ".join(missing))

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def process(self, path: Path) -> ReconstructionBundle:
        import torch
        from PIL import Image

        image = Image.open(path).convert("RGB")
        trans, tensor = self.detector(image)
        if tensor is None or trans is None:
            raise RuntimeError("face detector returned no aligned crop")
        self.model.input_img = tensor.to(self.device)

        captured_alpha: dict[str, Any] = {}
        captured_renderer: dict[str, Any] = {}
        def capture_alpha(_module: Any, _inputs: Any, output: Any) -> None:
            captured_alpha["count"] = int(captured_alpha.get("count", 0)) + 1
            captured_alpha["alpha"] = output

        alpha_hook = self.model.net_recon.register_forward_hook(capture_alpha)
        original_renderer_forward = self.model.renderer.forward

        def renderer_forward(*args: Any, **kwargs: Any) -> Any:
            output = original_renderer_forward(*args, **kwargs)
            if kwargs.get("visible_vertice") and isinstance(output, (tuple, list)) and len(output) >= 4:
                captured_renderer["indices"] = output[3]
            return output

        self.model.renderer.forward = renderer_forward
        try:
            with torch.inference_mode():
                results = self.model.forward()
        finally:
            alpha_hook.remove()
            self.model.renderer.forward = original_renderer_forward

        alpha_t = captured_alpha.get("alpha")
        if alpha_t is None:
            raise RuntimeError("failed to capture net_recon output; refusing a second inference")
        if captured_alpha.get("count") != 1:
            raise RuntimeError(f"expected exactly one net_recon call, got {captured_alpha.get('count')}")
        renderer_indices = captured_renderer.get("indices")
        if renderer_indices is None:
            raise RuntimeError("failed to capture renderer visibility")

        with torch.inference_mode():
            alpha = self.model.split_alpha(alpha_t)
            object_t = self.model.compute_shape(alpha["id"], alpha["exp"])
            identity_t = self.model.compute_shape(alpha["id"], torch.zeros_like(alpha["exp"]))
            rotation_t = self.model.compute_rotation(alpha["angle"])
            posed_t = self.model.transform(object_t, rotation_t, alpha["trans"])
            camera_t = self.model.to_camera(posed_t.clone())
            image_t = self.model.to_image(camera_t.clone())
            normal_t = self.model.compute_norm(object_t)
            posed_normal_t = normal_t @ rotation_t

        vertices_object = self._np(object_t)[0].astype(np.float32)
        vertices_identity = self._np(identity_t)[0].astype(np.float32)
        vertices_camera = self._np(camera_t)[0].astype(np.float32)
        vertices_image = self._np(image_t)[0].astype(np.float32)
        normals_object = self._np(normal_t)[0].astype(np.float32)
        normals_posed = self._np(posed_normal_t)[0].astype(np.float32)
        rotation = self._np(rotation_t)[0].astype(np.float32)
        angles_rad = self._np(alpha["angle"])[0].astype(np.float32)
        angles_deg = np.degrees(angles_rad).astype(np.float32)
        translation = self._np(alpha["trans"])[0].astype(np.float32)

        normalized, center, scale = normalize_mesh(vertices_object)
        pose_bin, canonical_yaw = classify_pose(float(angles_deg[1]))
        canonical_rotation = row_rotation_matrix(0.0, canonical_yaw, 0.0)
        canonical = (normalized @ canonical_rotation).astype(np.float32)

        count = len(vertices_object)
        front = normals_posed[:, 2] >= 0.0
        renderer = np.zeros(count, dtype=bool)
        raw_indices = self._np(renderer_indices).reshape(-1).astype(np.int64)
        raw_indices = raw_indices[(raw_indices >= 0) & (raw_indices < count)]
        renderer[np.unique(raw_indices)] = True
        combined = front & renderer

        idx106 = self._np(self.model.ldm106).reshape(-1).astype(np.int64)
        idx134 = self._np(self.model.ldm134).reshape(-1).astype(np.int64)
        expected106 = np.asarray(results["ldm106"])[0].astype(np.float32)
        expected134 = np.asarray(results["ldm134"])[0].astype(np.float32)
        reprojection = {
            "ldm106_224": reprojection_stats(vertices_image[idx106], expected106),
            "ldm134_224": reprojection_stats(vertices_image[idx134], expected134),
        }
        seg = np.asarray(results.get("seg_visible"))
        while seg.ndim > 3:
            seg = seg[0]
        if seg.ndim == 3 and seg.shape[0] == 8 and seg.shape[-1] != 8:
            seg = np.moveaxis(seg, 0, -1)
        if seg.shape != (224, 224, 8):
            raise RuntimeError(f"unexpected seg_visible shape: {seg.shape}")

        bundle = ReconstructionBundle(
            trans_params=np.asarray(trans, np.float32), angles_rad=angles_rad, angles_deg=angles_deg,
            pose_bin=pose_bin, canonical_yaw=float(canonical_yaw), rotation=rotation,
            translation=translation, vertices_object=vertices_object,
            vertices_identity_only=vertices_identity, vertices_object_normalized=normalized,
            vertices_bin_canonical=canonical, vertices_camera=vertices_camera,
            vertices_image_224=vertices_image, normals_object=normals_object,
            normals_posed=normals_posed, triangles=np.asarray(results["tri"], np.int64),
            uv_coords=np.asarray(results["uv_coords"], np.float32),
            ldm106_indices=idx106, ldm134_indices=idx134, front_facing=front,
            renderer_visible=renderer, combined_visible=combined,
            semantic_channels_224=seg.astype(np.float16), alpha_full=self._np(alpha_t)[0].astype(np.float32),
            alpha_id=self._np(alpha["id"])[0].astype(np.float32),
            alpha_exp=self._np(alpha["exp"])[0].astype(np.float32),
            alpha_alb=self._np(alpha["alb"])[0].astype(np.float32),
            alpha_sh=self._np(alpha["sh"])[0].astype(np.float32),
            normalization_center=center, normalization_scale=scale,
            canonical_rotation=canonical_rotation, reprojection=reprojection, raw_results=results,
        )
        return bundle

    def cleanup(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            gc.collect()
