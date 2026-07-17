"""ITER1 reconstruction contract for 3DDFA-V3 forensic sensor.

Stable schema for downstream pipeline modules. Pure data types only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

SCHEMA_VERSION = "3ddfa_v3_iter1_v1"


@dataclass
class CameraContract:
    model: str = "perspective_weak"
    focal: float = 1015.0
    principal_point: tuple[float, float] = (112.0, 112.0)
    image_size: tuple[int, int] = (224, 224)
    camera_distance: float = 10.0
    projection_matrix_3x3: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.projection_matrix_3x3 is not None:
            d["projection_matrix_3x3"] = np.asarray(self.projection_matrix_3x3)
        return d


@dataclass
class ReconstructionResult:
    """Canonical forensic reconstruction package.

    Coordinate spaces (see coordinate_spaces):
      - vertices_camera: after to_camera (v3d)
      - vertices_model: shape with expression, model space
      - vertices_identity: id-only (zero expression), model space
      - vertices_transformed: posed + translated model space
      - vertices_image: 2D crop plane (y-up as in 3DDFA)
    """

    schema_version: str = SCHEMA_VERSION
    expression_mode: str = "full"  # full | identity_only | neutral_soft

    # Geometry
    vertices_camera: Optional[np.ndarray] = None  # (N,3) or (1,N,3)
    vertices_model: Optional[np.ndarray] = None
    vertices_identity: Optional[np.ndarray] = None
    vertices_transformed: Optional[np.ndarray] = None
    vertices_image: Optional[np.ndarray] = None
    triangles: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None

    normals_model: Optional[np.ndarray] = None
    normals_identity: Optional[np.ndarray] = None
    normals_camera: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None

    visible_idx: Optional[np.ndarray] = None

    # Alpha coefficients (always present after ITER1)
    alpha_raw: Optional[np.ndarray] = None  # (256,)
    alpha_id: Optional[np.ndarray] = None  # (80,)
    alpha_exp: Optional[np.ndarray] = None  # (64,) raw network exp
    alpha_exp_used: Optional[np.ndarray] = None  # after identity/neutral
    alpha_alb: Optional[np.ndarray] = None  # (80,)
    alpha_angle: Optional[np.ndarray] = None  # radians
    alpha_angle_deg: Optional[np.ndarray] = None
    alpha_sh: Optional[np.ndarray] = None  # (27,)
    alpha_trans: Optional[np.ndarray] = None  # (3,)

    # Camera / crop
    camera: CameraContract = field(default_factory=CameraContract)
    trans_params: Optional[np.ndarray] = None

    # Optional landmarks / texture
    landmarks_68: Optional[np.ndarray] = None
    landmarks_106: Optional[np.ndarray] = None
    face_texture: Optional[np.ndarray] = None

    # Provenance
    topology_hash: Optional[str] = None
    basis_hash: Optional[str] = None
    image_path: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    coordinate_spaces: Dict[str, str] = field(
        default_factory=lambda: {
            "vertices_camera": "camera",
            "vertices_model": "model_with_expression",
            "vertices_identity": "model_identity_zero_expression",
            "vertices_transformed": "model_posed_translated",
            "vertices_image": "crop_224_y_up_image_plane",
            "units": "bfm_model_units",
        }
    )

    def squeeze_batch(self) -> "ReconstructionResult":
        """Drop leading batch dim=1 on array fields when present."""

        def _sq(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if x is None:
                return None
            a = np.asarray(x)
            if a.ndim >= 2 and a.shape[0] == 1:
                return a[0]
            return a

        for name in (
            "vertices_camera",
            "vertices_model",
            "vertices_identity",
            "vertices_transformed",
            "vertices_image",
            "normals_model",
            "normals_identity",
            "normals_camera",
            "rotation_matrix",
            "alpha_raw",
            "alpha_id",
            "alpha_exp",
            "alpha_exp_used",
            "alpha_alb",
            "alpha_angle",
            "alpha_angle_deg",
            "alpha_sh",
            "alpha_trans",
            "landmarks_68",
            "landmarks_106",
            "face_texture",
        ):
            setattr(self, name, _sq(getattr(self, name)))
        return self

    def to_result_dict(self) -> Dict[str, Any]:
        """Bridge to legacy 3DDFA result_dict key names."""
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "expression_mode": self.expression_mode,
            "v3d": self.vertices_camera,
            "v2d": self.vertices_image,
            "v3d_model": self.vertices_model,
            "v3d_identity": self.vertices_identity,
            "v3d_transformed": self.vertices_transformed,
            "normals_model": self.normals_model,
            "normals_identity": self.normals_identity,
            "normals_camera": self.normals_camera,
            "rotation_matrix": self.rotation_matrix,
            "tri": self.triangles,
            "uv_coords": self.uv_coords,
            "visible_idx": self.visible_idx,
            "alpha_raw": self.alpha_raw,
            "alpha_id": self.alpha_id,
            "alpha_exp": self.alpha_exp,
            "alpha_exp_used": self.alpha_exp_used,
            "alpha_alb": self.alpha_alb,
            "alpha_angle": self.alpha_angle,
            "alpha_angle_deg": self.alpha_angle_deg,
            "alpha_sh": self.alpha_sh,
            "alpha_trans": self.alpha_trans,
            "camera": self.camera.to_dict(),
            "trans_params": self.trans_params,
            "face_texture": self.face_texture,
            "coordinate_spaces": self.coordinate_spaces,
            "topology_hash": self.topology_hash,
            "basis_hash": self.basis_hash,
            "image_path": self.image_path,
        }
        if self.landmarks_68 is not None:
            d["ldm68"] = self.landmarks_68
        if self.landmarks_106 is not None:
            d["ldm106"] = self.landmarks_106
        d.update(self.payload)
        return d

    @classmethod
    def from_result_dict(
        cls,
        result_dict: Dict[str, Any],
        *,
        image_path: Optional[str] = None,
        topology_hash: Optional[str] = None,
        basis_hash: Optional[str] = None,
        squeeze: bool = True,
    ) -> "ReconstructionResult":
        cam_raw = result_dict.get("camera") or {}
        if isinstance(cam_raw, CameraContract):
            camera = cam_raw
        else:
            camera = CameraContract(
                model=str(cam_raw.get("model", "perspective_weak")),
                focal=float(cam_raw.get("focal", 1015.0)),
                principal_point=tuple(cam_raw.get("principal_point", (112.0, 112.0))),  # type: ignore[arg-type]
                image_size=tuple(cam_raw.get("image_size", (224, 224))),  # type: ignore[arg-type]
                camera_distance=float(cam_raw.get("camera_distance", 10.0)),
                projection_matrix_3x3=cam_raw.get("projection_matrix_3x3"),
            )

        known = {
            "schema_version",
            "expression_mode",
            "v3d",
            "v2d",
            "v3d_model",
            "v3d_identity",
            "v3d_transformed",
            "normals_model",
            "normals_identity",
            "normals_camera",
            "rotation_matrix",
            "tri",
            "uv_coords",
            "visible_idx",
            "alpha_raw",
            "alpha_id",
            "alpha_exp",
            "alpha_exp_used",
            "alpha_alb",
            "alpha_angle",
            "alpha_angle_deg",
            "alpha_sh",
            "alpha_trans",
            "camera",
            "trans_params",
            "face_texture",
            "coordinate_spaces",
            "ldm68",
            "ldm106",
            "topology_hash",
            "basis_hash",
            "image_path",
            "neutral_scale",
        }
        payload = {k: v for k, v in result_dict.items() if k not in known}

        obj = cls(
            schema_version=str(result_dict.get("schema_version", SCHEMA_VERSION)),
            expression_mode=str(result_dict.get("expression_mode", "full")),
            vertices_camera=_as_np(result_dict.get("v3d")),
            vertices_model=_as_np(result_dict.get("v3d_model")),
            vertices_identity=_as_np(result_dict.get("v3d_identity")),
            vertices_transformed=_as_np(result_dict.get("v3d_transformed")),
            vertices_image=_as_np(result_dict.get("v2d")),
            triangles=_as_np(result_dict.get("tri")),
            uv_coords=_as_np(result_dict.get("uv_coords")),
            normals_model=_as_np(result_dict.get("normals_model")),
            normals_identity=_as_np(result_dict.get("normals_identity")),
            normals_camera=_as_np(result_dict.get("normals_camera")),
            rotation_matrix=_as_np(result_dict.get("rotation_matrix")),
            visible_idx=_as_np(result_dict.get("visible_idx")),
            alpha_raw=_as_np(result_dict.get("alpha_raw")),
            alpha_id=_as_np(result_dict.get("alpha_id")),
            alpha_exp=_as_np(result_dict.get("alpha_exp")),
            alpha_exp_used=_as_np(result_dict.get("alpha_exp_used")),
            alpha_alb=_as_np(result_dict.get("alpha_alb")),
            alpha_angle=_as_np(result_dict.get("alpha_angle")),
            alpha_angle_deg=_as_np(result_dict.get("alpha_angle_deg")),
            alpha_sh=_as_np(result_dict.get("alpha_sh")),
            alpha_trans=_as_np(result_dict.get("alpha_trans")),
            camera=camera,
            trans_params=_as_np(result_dict.get("trans_params")),
            landmarks_68=_as_np(result_dict.get("ldm68")),
            landmarks_106=_as_np(result_dict.get("ldm106")),
            face_texture=_as_np(result_dict.get("face_texture")),
            topology_hash=topology_hash or result_dict.get("topology_hash"),
            basis_hash=basis_hash or result_dict.get("basis_hash"),
            image_path=image_path or result_dict.get("image_path"),
            payload=payload,
            coordinate_spaces=dict(
                result_dict.get("coordinate_spaces")
                or {
                    "vertices_camera": "camera",
                    "units": "bfm_model_units",
                }
            ),
        )
        if squeeze:
            obj.squeeze_batch()
        return obj


def _as_np(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(x)


def hash_array(arr: np.ndarray, *, digest_size: int = 16) -> str:
    import hashlib

    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def compute_topology_hash(triangles: np.ndarray, n_vertices: int = 35709) -> str:
    import hashlib

    tri = np.ascontiguousarray(np.asarray(triangles, dtype=np.int64))
    h = hashlib.blake2b(digest_size=16)
    h.update(b"topology_v1")
    h.update(int(n_vertices).to_bytes(8, "little"))
    h.update(tri.tobytes())
    return h.hexdigest()


def compute_basis_hash(
    u: np.ndarray,
    id_basis: np.ndarray,
    exp_basis: np.ndarray,
    alb_basis: Optional[np.ndarray] = None,
) -> str:
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(b"basis_v1")
    for arr in (u, id_basis, exp_basis):
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(a.astype(np.float32).tobytes())
    if alb_basis is not None:
        a = np.ascontiguousarray(alb_basis)
        h.update(str(a.shape).encode())
        h.update(a.astype(np.float32).tobytes())
    return h.hexdigest()
