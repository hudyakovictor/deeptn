from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PHOTO_SCHEMA_VERSION, VALIDATION_SCHEMA_VERSION
from .geometry import unpack_mask
from .utils import atomic_json

MESH_COUNT = 35709
TRIANGLE_COUNT = 70789
NPZ_REQUIRED = {
    "vertices_object": (MESH_COUNT, 3),
    "vertices_identity_only": (MESH_COUNT, 3),
    "vertices_object_normalized": (MESH_COUNT, 3),
    "vertices_bin_canonical": (MESH_COUNT, 3),
    "vertices_camera": (MESH_COUNT, 3),
    "vertices_image_224": (MESH_COUNT, 2),
    "normals_object": (MESH_COUNT, 3),
    "normals_posed": (MESH_COUNT, 3),
    "triangles": (TRIANGLE_COUNT, 3),
    "uv_coords": (MESH_COUNT, 2),
    "ldm106_vertex_indices": (106,),
    "ldm134_vertex_indices": (134,),
    "alpha_full": (257,), "alpha_id": (80,), "alpha_exp": (64,), "alpha_alb": (80,), "alpha_sh": (27,),
    "angle_rad": (3,), "angle_deg_pitch_yaw_roll": (3,), "rotation_matrix": (3, 3),
    "translation": (3,), "trans_params": (5,), "normalization_center": (3,),
    "normalization_scale": (1,), "canonical_rotation_row_matrix": (3, 3), "canonical_yaw": (1,),
    "ldm106_object": (106, 3), "ldm106_object_normalized": (106, 3),
    "ldm106_bin_canonical": (106, 3), "ldm106_camera": (106, 3), "ldm106_image_224": (106, 2),
    "ldm106_identity_only": (106, 3),
    "ldm106_front_facing": (106,), "ldm106_renderer_visible": (106,), "ldm106_visible": (106,),
    "ldm134_object": (134, 3), "ldm134_object_normalized": (134, 3),
    "ldm134_bin_canonical": (134, 3), "ldm134_camera": (134, 3), "ldm134_image_224": (134, 2),
    "ldm134_identity_only": (134, 3),
    "ldm134_front_facing": (134,), "ldm134_renderer_visible": (134,), "ldm134_visible": (134,),
    "full_mesh_front_facing_packbits": (4464,),
    "full_mesh_renderer_visible_packbits": (4464,),
    "full_mesh_visible_packbits": (4464,),
    "analysis_mesh_vertex_mask_packbits": (4464,),
    "analysis_mesh_triangle_mask_packbits": (8849,),
}


class ValidationError(RuntimeError):
    pass


def _csv_check(path: Path, expected: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected:
        raise ValidationError(f"{path.name}: expected {expected} rows, got {len(rows)}")
    ids = [int(row["landmark_id"]) for row in rows]
    if ids != list(range(expected)):
        raise ValidationError(f"{path.name}: landmark_id sequence invalid")
    points = np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], np.float32)
    indices = np.asarray([int(r["vertex_index"]) for r in rows], np.int64)
    return points, indices


def validate_photo(directory: Path, write_result: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    try:
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        if info.get("schema_version") != PHOTO_SCHEMA_VERSION:
            raise ValidationError(f"photo schema mismatch: {info.get('schema_version')}")
        files = info.get("files") or {}
        optional_files = {"face_mask", "face_mask_data"}
        for name, relative in files.items():
            if relative is None:
                if name in optional_files:
                    warnings.append(f"optional file absent: {name}")
                    continue
                raise ValidationError(f"required file is None: {name}")
            path = directory / str(relative)
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValidationError(f"missing/empty file {name}: {relative}")
        csv_data = {
            "ldm106_raw": _csv_check(directory / "ldm106_raw.csv", 106),
            "ldm106_aligned": _csv_check(directory / "ldm106_aligned.csv", 106),
            "ldm134_raw": _csv_check(directory / "ldm134_raw.csv", 134),
            "ldm134_aligned": _csv_check(directory / "ldm134_aligned.csv", 134),
        }
        with np.load(directory / "reconstruction.npz", allow_pickle=False) as z:
            for key, shape in NPZ_REQUIRED.items():
                if key not in z:
                    raise ValidationError(f"NPZ missing {key}")
                if z[key].shape != shape:
                    raise ValidationError(f"NPZ {key} shape {z[key].shape}, expected {shape}")
                if np.issubdtype(z[key].dtype, np.number) and not np.isfinite(z[key]).all():
                    raise ValidationError(f"NPZ {key} contains NaN/Inf")
            triangles = z["triangles"]
            if triangles.min() < 0 or triangles.max() >= MESH_COUNT:
                raise ValidationError("triangle index out of bounds")
            for key in ("ldm106_vertex_indices", "ldm134_vertex_indices"):
                if z[key].min() < 0 or z[key].max() >= MESH_COUNT:
                    raise ValidationError(f"{key} out of bounds")
            front = unpack_mask(z["full_mesh_front_facing_packbits"], MESH_COUNT).astype(bool)
            renderer = unpack_mask(z["full_mesh_renderer_visible_packbits"], MESH_COUNT).astype(bool)
            combined = unpack_mask(z["full_mesh_visible_packbits"], MESH_COUNT).astype(bool)
            if not np.array_equal(combined, front & renderer):
                raise ValidationError("combined visibility differs from front & renderer")
            analysis_vertex_mask = unpack_mask(z["analysis_mesh_vertex_mask_packbits"], MESH_COUNT).astype(bool)
            analysis_triangle_mask = unpack_mask(z["analysis_mesh_triangle_mask_packbits"], TRIANGLE_COUNT).astype(bool)
            expected_triangle_mask = np.all(combined[triangles], axis=1)
            if not np.array_equal(analysis_triangle_mask, expected_triangle_mask):
                raise ValidationError("analysis mesh triangle mask differs from combined visibility")
            expected_full_triangle_ids = np.flatnonzero(expected_triangle_mask).astype(np.int64)
            expected_used_vertices = np.unique(triangles[expected_triangle_mask].reshape(-1)).astype(np.int64) if np.any(expected_triangle_mask) else np.empty(0, np.int64)
            expected_vertex_mask = np.zeros(MESH_COUNT, bool)
            expected_vertex_mask[expected_used_vertices] = True
            if not np.array_equal(analysis_vertex_mask, expected_vertex_mask):
                raise ValidationError("analysis mesh vertex mask differs from visible triangles")
            variable_keys = (
                "analysis_mesh_vertex_indices_full", "analysis_mesh_triangle_indices_full",
                "analysis_mesh_vertices_object", "analysis_mesh_vertices_normalized",
                "analysis_mesh_normals_object", "analysis_mesh_uv_coords", "analysis_mesh_triangles",
            )
            for key in variable_keys:
                if key not in z:
                    raise ValidationError(f"NPZ missing {key}")
            if not np.array_equal(z["analysis_mesh_vertex_indices_full"], expected_used_vertices):
                raise ValidationError("analysis mesh full vertex indices invalid")
            if not np.array_equal(z["analysis_mesh_triangle_indices_full"], expected_full_triangle_ids):
                raise ValidationError("analysis mesh full triangle indices invalid")
            vertex_count = len(expected_used_vertices)
            triangle_count = len(expected_full_triangle_ids)
            for key, width in (("analysis_mesh_vertices_object", 3), ("analysis_mesh_vertices_normalized", 3),
                               ("analysis_mesh_normals_object", 3), ("analysis_mesh_uv_coords", 2)):
                if z[key].shape != (vertex_count, width):
                    raise ValidationError(f"{key} shape invalid")
            if z["analysis_mesh_triangles"].shape != (triangle_count, 3):
                raise ValidationError("analysis_mesh_triangles shape invalid")
            if triangle_count and (z["analysis_mesh_triangles"].min() < 0 or z["analysis_mesh_triangles"].max() >= vertex_count):
                raise ValidationError("analysis mesh compact triangle index out of bounds")
            if z["alpha_alb"].size == 0 or z["alpha_sh"].size == 0:
                raise ValidationError("albedo/SH coefficients are empty")
            uv_shape = tuple(int(x) for x in z["uv_shape"])
            if len(uv_shape) != 2 or min(uv_shape) <= 0:
                raise ValidationError(f"invalid uv_shape: {uv_shape}")
            if z["uv_confidence"].shape != uv_shape:
                raise ValidationError("uv_confidence shape differs from uv_shape")
            required_uv_bytes = (uv_shape[0] * uv_shape[1] + 7) // 8
            for key in ("uv_observed_mask_packbits", "uv_is_original_packbits"):
                if z[key].shape != (required_uv_bytes,):
                    raise ValidationError(f"{key} has invalid packed length")
            if z["tri_visibility"].shape != (TRIANGLE_COUNT,):
                raise ValidationError("tri_visibility shape invalid")
            if float(z["normalization_scale"][0]) <= 0:
                raise ValidationError("normalization scale must be positive")
            rotation = z["rotation_matrix"].astype(np.float64)
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
                raise ValidationError("rotation_matrix is not a proper rotation")
            mapping = {
                "ldm106_raw": ("ldm106_object", "ldm106_vertex_indices"),
                "ldm106_aligned": ("ldm106_bin_canonical", "ldm106_vertex_indices"),
                "ldm134_raw": ("ldm134_object", "ldm134_vertex_indices"),
                "ldm134_aligned": ("ldm134_bin_canonical", "ldm134_vertex_indices"),
            }
            for name, (array_key, index_key) in mapping.items():
                points, indices = csv_data[name]
                if not np.allclose(points, z[array_key], atol=1e-6):
                    raise ValidationError(f"{name}.csv coordinates differ from NPZ")
                if not np.array_equal(indices, z[index_key]):
                    raise ValidationError(f"{name}.csv vertex indices differ from NPZ")
        semantic = np.load(directory / "semantic_channels.npz", allow_pickle=False)
        if semantic["channels_224"].shape != (224, 224, 8):
            raise ValidationError("semantic channels shape invalid")

        if files.get("uv_data"):
            with np.load(directory / str(files["uv_data"]), allow_pickle=False) as uvz:
                for key in ("texture_bgr", "texture_observed_bgr", "confidence", "observed_mask", "is_original_mask", "valid_mask",
                            "uv_domain_mask", "completion_method", "synthetic_texture_mask", "uv_shape", "uv_coords"):
                    if key not in uvz:
                        raise ValidationError(f"uv.npz missing {key}")
                uv_shape2 = tuple(int(x) for x in uvz["uv_shape"])
                for key in ("confidence", "observed_mask", "is_original_mask", "valid_mask", "uv_domain_mask", "completion_method", "synthetic_texture_mask"):
                    if uvz[key].shape != uv_shape2:
                        raise ValidationError(f"uv.npz {key} shape differs from uv_shape")
                if uvz["texture_bgr"].shape[:2] != uv_shape2:
                    raise ValidationError("uv.npz texture_bgr shape differs from uv_shape")
                if uvz["texture_observed_bgr"].shape != uvz["texture_bgr"].shape:
                    raise ValidationError("uv.npz observed/completed texture shapes differ")
                if uvz["uv_coords"].shape != (MESH_COUNT, 2):
                    raise ValidationError("uv.npz uv_coords shape invalid")

        if files.get("face_mask_data"):
            with np.load(directory / str(files["face_mask_data"]), allow_pickle=False) as fmz:
                for key in ("mask_original", "mask_crop", "mask_face", "mask_alpha_u8", "bbox_original"):
                    if key not in fmz:
                        raise ValidationError(f"face_mask.npz missing {key}")
                if fmz["mask_face"].shape != (500, 424) or fmz["mask_alpha_u8"].shape != (500, 424):
                    raise ValidationError("face_mask.npz face mask shape invalid")

        if files.get("quality"):
            q = json.loads((directory / str(files["quality"])).read_text(encoding="utf-8"))
            if q.get("status") not in {"complete", "no_face_mask"}:
                raise ValidationError(f"quality.json bad status: {q.get('status')}")
        if files.get("quality_zones"):
            with np.load(directory / str(files["quality_zones"]), allow_pickle=False) as qz:
                for key in ("zone_names", "zone_texture_roi_original_packbits", "zone_texture_pixels", "zone_visible_fraction"):
                    if key not in qz:
                        raise ValidationError(f"quality_zones.npz missing {key}")
                if qz["zone_texture_pixels"].shape[0] != qz["zone_names"].shape[0]:
                    raise ValidationError("quality_zones zone count mismatch")

        for key in ("face_crop", "uv_texture", "uv_observed_texture", "uv_confidence"):
            p = directory / str(files[key])
            if cv2.imread(str(p), cv2.IMREAD_UNCHANGED) is None:
                raise ValidationError(f"cannot decode {key}")
        if files.get("morphing_mesh"):
            mesh_path = directory / str(files["morphing_mesh"])
            with mesh_path.open("r", encoding="utf-8") as f:
                face_count = sum(1 for line in f if line.startswith("f "))
            if face_count != TRIANGLE_COUNT:
                raise ValidationError(f"morphing mesh must retain full topology: {face_count} faces")
    except Exception as exc:
        errors.append(str(exc))
    status = "complete" if not errors else "invalid"
    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "photo_id": info.get("photo_id"),
    }
    if write_result:
        atomic_json(directory / "validation.json", result)
    return result


def is_resumable(directory: Path, source_sha256: str, code_hash: str, config_hash: str, model_hash: str) -> tuple[bool, dict[str, Any] | None]:
    if not directory.is_dir():
        return False, None
    try:
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        expected = {
            "source_sha256": source_sha256, "code_hash": code_hash,
            "config_hash": config_hash, "model_hash": model_hash,
        }
        if any(info.get(k) != v for k, v in expected.items()):
            return False, None
        result = validate_photo(directory, write_result=False)
        return result["status"] == "complete", info
    except Exception:
        return False, None
