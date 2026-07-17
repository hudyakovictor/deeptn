"""ITER1 high-level reconstruction API for 3DDFA-V3.

Wraps face_model.forward() into ReconstructionResult with:
  - full alpha export
  - identity / soft-neutral meshes
  - topology/basis hashes when assets are available
  - crop/camera metadata

Does not depend on project pipeline modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from util.types import (
    SCHEMA_VERSION,
    ReconstructionResult,
    compute_basis_hash,
    compute_topology_hash,
)

PathLike = Union[str, Path]


def result_dict_to_reconstruction(
    result_dict: Dict[str, Any],
    *,
    image_path: Optional[PathLike] = None,
    topology_hash: Optional[str] = None,
    basis_hash: Optional[str] = None,
    squeeze: bool = True,
) -> ReconstructionResult:
    """Convert face_model.forward() output to ReconstructionResult."""
    return ReconstructionResult.from_result_dict(
        result_dict,
        image_path=None if image_path is None else str(image_path),
        topology_hash=topology_hash,
        basis_hash=basis_hash,
        squeeze=squeeze,
    )


def attach_model_hashes(
    result: ReconstructionResult,
    face_model: Any,
) -> ReconstructionResult:
    """Fill topology_hash / basis_hash from a loaded face_model instance."""
    try:
        tri = face_model.tri.detach().cpu().numpy()
        n_verts = int(face_model.u.numel() // 3)
        result.topology_hash = compute_topology_hash(tri, n_vertices=n_verts)
    except Exception as exc:  # noqa: BLE001 — optional provenance
        result.payload.setdefault("hash_errors", []).append(f"topology:{exc}")

    try:
        u = face_model.u.detach().cpu().numpy()
        idb = face_model.id.detach().cpu().numpy()
        expb = face_model.exp.detach().cpu().numpy()
        alb = face_model.alb.detach().cpu().numpy() if hasattr(face_model, "alb") else None
        result.basis_hash = compute_basis_hash(u, idb, expb, alb)
    except Exception as exc:  # noqa: BLE001
        result.payload.setdefault("hash_errors", []).append(f"basis:{exc}")
    return result


def run_reconstruction(
    face_model: Any,
    *,
    identity_only: bool = False,
    neutral_expression: bool = False,
    neutral_scale: float = 0.1,
    image_path: Optional[PathLike] = None,
    attach_hashes: bool = True,
) -> ReconstructionResult:
    """Run face_model.forward with ITER1 contract and return ReconstructionResult.

    Prerequisites: face_model.input_img (and optionally trans_params) already set
    by the detector / preprocess path (same as demo.py).
    """
    result_dict = face_model.forward(
        identity_only=identity_only,
        neutral_expression=neutral_expression,
        neutral_scale=neutral_scale,
    )
    result = result_dict_to_reconstruction(
        result_dict,
        image_path=image_path,
        squeeze=True,
    )
    if attach_hashes:
        attach_model_hashes(result, face_model)
    result.payload["schema_version"] = SCHEMA_VERSION
    return result


def save_reconstruction_artifact(
    result: ReconstructionResult,
    out_path: PathLike,
    *,
    include_meshes: bool = True,
    include_alpha: bool = True,
) -> Path:
    """Save a compact .npz artifact for forensic pipelines."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "schema_version": result.schema_version,
        "expression_mode": result.expression_mode,
        "image_path": result.image_path,
        "topology_hash": result.topology_hash,
        "basis_hash": result.basis_hash,
        "camera": result.camera.to_dict(),
        "trans_params": result.trans_params,
        "coordinate_spaces": result.coordinate_spaces,
        "visible_idx": result.visible_idx,
        "rotation_matrix": result.rotation_matrix,
        "triangles": result.triangles if include_meshes else None,
        "uv_coords": result.uv_coords if include_meshes else None,
    }
    if include_alpha:
        payload.update(
            {
                "alpha_raw": result.alpha_raw,
                "alpha_id": result.alpha_id,
                "alpha_exp": result.alpha_exp,
                "alpha_exp_used": result.alpha_exp_used,
                "alpha_alb": result.alpha_alb,
                "alpha_angle": result.alpha_angle,
                "alpha_angle_deg": result.alpha_angle_deg,
                "alpha_sh": result.alpha_sh,
                "alpha_trans": result.alpha_trans,
            }
        )
    if include_meshes:
        payload.update(
            {
                "vertices_camera": result.vertices_camera,
                "vertices_model": result.vertices_model,
                "vertices_identity": result.vertices_identity,
                "vertices_transformed": result.vertices_transformed,
                "vertices_image": result.vertices_image,
                "normals_model": result.normals_model,
                "normals_identity": result.normals_identity,
                "normals_camera": result.normals_camera,
            }
        )
    # Drop Nones for smaller files
    payload = {k: v for k, v in payload.items() if v is not None}
    np.savez_compressed(out, **payload)
    return out


def load_reconstruction_artifact(path: PathLike) -> ReconstructionResult:
    """Load artifact saved by save_reconstruction_artifact."""
    data = np.load(path, allow_pickle=True)
    d = {k: data[k] for k in data.files}
    # unwrap 0-d object arrays
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            d[k] = v.item()
    # map artifact keys → result_dict keys
    mapped = {
        "schema_version": d.get("schema_version", SCHEMA_VERSION),
        "expression_mode": d.get("expression_mode", "full"),
        "v3d": d.get("vertices_camera"),
        "v2d": d.get("vertices_image"),
        "v3d_model": d.get("vertices_model"),
        "v3d_identity": d.get("vertices_identity"),
        "v3d_transformed": d.get("vertices_transformed"),
        "normals_model": d.get("normals_model"),
        "normals_identity": d.get("normals_identity"),
        "normals_camera": d.get("normals_camera"),
        "rotation_matrix": d.get("rotation_matrix"),
        "tri": d.get("triangles"),
        "uv_coords": d.get("uv_coords"),
        "visible_idx": d.get("visible_idx"),
        "alpha_raw": d.get("alpha_raw"),
        "alpha_id": d.get("alpha_id"),
        "alpha_exp": d.get("alpha_exp"),
        "alpha_exp_used": d.get("alpha_exp_used"),
        "alpha_alb": d.get("alpha_alb"),
        "alpha_angle": d.get("alpha_angle"),
        "alpha_angle_deg": d.get("alpha_angle_deg"),
        "alpha_sh": d.get("alpha_sh"),
        "alpha_trans": d.get("alpha_trans"),
        "camera": d.get("camera") if not isinstance(d.get("camera"), np.ndarray) else d.get("camera").item()
        if hasattr(d.get("camera"), "item")
        else d.get("camera"),
        "trans_params": d.get("trans_params"),
        "coordinate_spaces": d.get("coordinate_spaces")
        if not isinstance(d.get("coordinate_spaces"), np.ndarray)
        else d.get("coordinate_spaces").item()
        if hasattr(d.get("coordinate_spaces"), "item")
        else d.get("coordinate_spaces"),
        "topology_hash": _maybe_str(d.get("topology_hash")),
        "basis_hash": _maybe_str(d.get("basis_hash")),
        "image_path": _maybe_str(d.get("image_path")),
    }
    return ReconstructionResult.from_result_dict(mapped, squeeze=True)


def _maybe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        return str(x)
    return str(x)




def attach_visibility(
    result: ReconstructionResult,
    *,
    angle_threshold_deg: float = 75.0,
    triangles: Optional[Any] = None,
    vertices_2d: Optional[Any] = None,
    image_size: Optional[tuple] = None,
) -> ReconstructionResult:
    """ITER2: compute visibility and store on result.payload."""
    from util.visibility import compute_visibility

    verts = result.vertices_camera
    norms = result.normals_camera
    if verts is None or norms is None:
        result.payload["visibility_error"] = "missing vertices_camera or normals_camera"
        return result
    angles = result.alpha_angle_deg
    vis = compute_visibility(
        vertices_camera=verts,
        normals_camera=norms,
        angle_threshold_deg=angle_threshold_deg,
        triangles=triangles if triangles is not None else result.triangles,
        vertices_2d=vertices_2d if vertices_2d is not None else result.vertices_image,
        image_size=image_size,
        angles_deg=None if angles is None else np.asarray(angles).reshape(-1),
        renderer_visible=result.visible_idx,
    )
    result.payload["visibility"] = {
        "binary_mask": vis.binary_mask,
        "cosine_weights": vis.cosine_weights,
        "beauty_weights": vis.beauty_weights,
        "facing_cosines": vis.facing_cosines,
        "visible_count": vis.visible_count,
        "angle_threshold_deg": vis.angle_threshold_deg,
        "triangle_weights_analysis": vis.triangle_weights_analysis,
        "triangle_weights_beauty": vis.triangle_weights_beauty,
    }
    # convenience: overwrite visible_idx with hard analysis mask if present
    if vis.binary_mask is not None:
        result.visible_idx = vis.binary_mask.astype(np.int64)
    return result


def recon_dict_for_uv(result: ReconstructionResult) -> dict:
    """Build recon_dict keys expected by uv_module.HDUVTextureGenerator."""
    verts = result.vertices_camera
    if verts is None:
        raise ValueError("vertices_camera required for UV")
    d = {
        "vertices": np.asarray(verts, dtype=np.float32),
        "vertices_3d": np.asarray(verts, dtype=np.float32),
        "vertices_2d": None if result.vertices_image is None else np.asarray(result.vertices_image, dtype=np.float32),
        "triangles": None if result.triangles is None else np.asarray(result.triangles, dtype=np.int64),
        "tri": None if result.triangles is None else np.asarray(result.triangles, dtype=np.int64),
        "uv_coords": None if result.uv_coords is None else np.asarray(result.uv_coords, dtype=np.float32),
        "normals_3d": None if result.normals_camera is None else np.asarray(result.normals_camera, dtype=np.float32),
        "alpha_sh": result.alpha_sh,
    }
    if d["vertices_2d"] is None:
        raise ValueError("vertices_image/v2d required for UV bake")
    return d

def assert_iter1_contract(result: ReconstructionResult) -> None:
    """Gate checks for ITER1 completion."""
    missing = []
    for key in (
        "alpha_id",
        "alpha_exp",
        "alpha_alb",
        "alpha_angle",
        "alpha_sh",
        "alpha_trans",
        "vertices_camera",
        "vertices_identity",
        "vertices_model",
        "camera",
    ):
        if getattr(result, key) is None:
            missing.append(key)
    if missing:
        raise AssertionError(f"ITER1 contract missing fields: {missing}")
    if result.alpha_id is not None and np.asarray(result.alpha_id).shape[-1] != 80:
        raise AssertionError(f"alpha_id last dim != 80: {np.asarray(result.alpha_id).shape}")
    if result.alpha_exp is not None and np.asarray(result.alpha_exp).shape[-1] != 64:
        raise AssertionError(f"alpha_exp last dim != 64: {np.asarray(result.alpha_exp).shape}")
    if result.schema_version != SCHEMA_VERSION:
        raise AssertionError(f"schema_version {result.schema_version} != {SCHEMA_VERSION}")
