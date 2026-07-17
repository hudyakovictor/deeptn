"""ITER11: Photo snapshot stage (extract once, compute metrics later).

Heavy extraction (3DDFA) runs once per photo and writes a self-contained
snapshot to disk. All downstream stages (single-photo metrics, pair compare,
calibration dataset, main analysis) read only snapshots and never touch the
model again.

Design rules:
- Canonicalization is stored as an explicit transform (R, t, scale), never
  baked into the vertices. Raw vertices are the source of truth; canon-space
  data is derived on load. If the canon definition changes, snapshots stay valid.
- Landmarks (106/134) are stored as a full table in raw space; canon-space
  landmarks are derived via the same transform.
- Dense geometry (all vertices, normals, triangles) is stored so that the
  full legacy catalog (zones / pair-zone / dense residuals) remains computable.
- Arrays only, no pickle. Metadata travels as a JSON string inside the npz.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_SCHEMA_VERSION = "snapshot_v1"

_META_KEY = "__meta_json__"
_FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class CanonTransform:
    """Similarity transform mapping raw space to canonical space."""

    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    scale: float = 1.0

    def __post_init__(self) -> None:
        R = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64).reshape(-1)
        if R.shape != (3, 3) or not np.isfinite(R).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
            raise ValueError("rotation must be orthonormal")
        if t.shape != (3,) or not np.isfinite(t).all():
            raise ValueError("translation must be a finite 3-vector")
        s = float(self.scale)
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"scale must be finite and positive, got {s}")
        object.__setattr__(self, "rotation", R)
        object.__setattr__(self, "translation", t)
        object.__setattr__(self, "scale", s)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return self.scale * (pts @ self.rotation.T) + self.translation

    def apply_directions(self, directions: np.ndarray) -> np.ndarray:
        """Rotate direction vectors (normals); renormalize to unit length."""
        d = np.asarray(directions, dtype=np.float64) @ self.rotation.T
        norms = np.linalg.norm(d, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return d / norms


def canon_transform_from_pose(
    pitch_deg: float, yaw_deg: float, roll_deg: float, pose_bucket: str
) -> CanonTransform:
    """Rotation-only transform from observed pose to the bucket's canonical pose."""
    from util.alignment import canonical_angles_deg_for_bucket, euler_to_rotation_matrix

    for v in (pitch_deg, yaw_deg, roll_deg):
        if not np.isfinite(float(v)):
            raise ValueError("pose angles must be finite")
    R_pose = euler_to_rotation_matrix(
        np.deg2rad(np.array([pitch_deg, yaw_deg, roll_deg], dtype=np.float64))
    )
    canon_angles = np.asarray(canonical_angles_deg_for_bucket(pose_bucket), dtype=np.float64)
    R_canon = euler_to_rotation_matrix(np.deg2rad(canon_angles))
    return CanonTransform(rotation=R_canon @ R_pose.T, translation=np.zeros(3), scale=1.0)


@dataclass
class PhotoSnapshot:
    """Self-contained per-photo extraction artifact."""

    photo_id: str
    image_path: str
    pose_bucket: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    vertices_raw: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) int
    canon: CanonTransform
    normals_raw: np.ndarray | None = None
    alpha_id: np.ndarray | None = None
    exp_params: np.ndarray | None = None
    landmarks_106_raw: np.ndarray | None = None  # (K, 3) full table, raw space
    landmarks_134_raw: np.ndarray | None = None  # (134, 3) full table, raw space
    visibility_weights: np.ndarray | None = None  # (N,) per-vertex analysis weights
    seg_visible: np.ndarray | None = None  # (224, 224, 8) 3DDFA segmentation (skin/eyes/eyebrows/nose/lips)
    quality: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices_raw, dtype=_FLOAT_DTYPE)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
            raise ValueError(f"vertices_raw must be (N,3), got {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("vertices_raw contains non-finite values")
        self.vertices_raw = v
        tris = np.asarray(self.triangles, dtype=np.int64)
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError(f"triangles must be (M,3), got {tris.shape}")
        if tris.size and (tris.min() < 0 or tris.max() >= v.shape[0]):
            raise ValueError("triangles reference out-of-range vertex indices")
        self.triangles = tris
        for name in ("normals_raw", "landmarks_106_raw", "landmarks_134_raw"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=_FLOAT_DTYPE)
                if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
                    raise ValueError(f"{name} must be a finite (K,3) array")
                setattr(self, name, arr)
        if self.normals_raw is not None and self.normals_raw.shape[0] != v.shape[0]:
            raise ValueError("normals_raw length must match vertices_raw")
        if self.visibility_weights is not None:
            w = np.asarray(self.visibility_weights, dtype=_FLOAT_DTYPE).reshape(-1)
            if w.shape[0] != v.shape[0] or not np.isfinite(w).all():
                raise ValueError("visibility_weights must be finite (N,)")
            self.visibility_weights = w
        for name in ("alpha_id", "exp_params"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.asarray(arr, dtype=_FLOAT_DTYPE).reshape(-1)
                if not np.isfinite(arr).all():
                    raise ValueError(f"{name} contains non-finite values")
                setattr(self, name, arr)
        if self.seg_visible is not None:
            seg = np.asarray(self.seg_visible, dtype=_FLOAT_DTYPE)
            if seg.ndim != 3 or seg.shape[2] != 8:
                raise ValueError(f"seg_visible must be (H,W,8), got {seg.shape}")
            self.seg_visible = seg

    # ---- derived canon-space views (never stored) ----

    @property
    def vertices_canon(self) -> np.ndarray:
        return self.canon.apply_points(self.vertices_raw)

    @property
    def normals_canon(self) -> np.ndarray | None:
        if self.normals_raw is None:
            return None
        return self.canon.apply_directions(self.normals_raw)

    @property
    def landmarks_106_canon(self) -> np.ndarray | None:
        if self.landmarks_106_raw is None:
            return None
        return self.canon.apply_points(self.landmarks_106_raw)

    @property
    def landmarks_134_canon(self) -> np.ndarray | None:
        if self.landmarks_134_raw is None:
            return None
        return self.canon.apply_points(self.landmarks_134_raw)

def save_snapshot(snapshot: PhotoSnapshot, path: str | Path) -> Path:
    """Write one snapshot to a compressed .npz (arrays + JSON metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "vertices_raw": snapshot.vertices_raw,
        "triangles": snapshot.triangles,
        "canon_rotation": np.asarray(snapshot.canon.rotation, dtype=np.float64),
        "canon_translation": np.asarray(snapshot.canon.translation, dtype=np.float64),
    }
    for name in ("normals_raw", "alpha_id", "exp_params", "landmarks_106_raw", "landmarks_134_raw", "visibility_weights", "seg_visible"):
        arr = getattr(snapshot, name)
        if arr is not None:
            arrays[name] = arr
    meta = {
        "schema_version": snapshot.schema_version,
        "photo_id": snapshot.photo_id,
        "image_path": str(snapshot.image_path),
        "pose_bucket": snapshot.pose_bucket,
        "yaw_deg": float(snapshot.yaw_deg),
        "pitch_deg": float(snapshot.pitch_deg),
        "roll_deg": float(snapshot.roll_deg),
        "canon_scale": float(snapshot.canon.scale),
        "quality": snapshot.quality,
        "extras": snapshot.extras,
    }
    arrays[_META_KEY] = np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)
    return path


def load_snapshot(path: str | Path) -> PhotoSnapshot:
    """Load a snapshot; fail closed on schema mismatch or corruption."""
    with np.load(Path(path), allow_pickle=False) as data:
        if _META_KEY not in data:
            raise ValueError(f"{path}: missing snapshot metadata")
        meta = json.loads(bytes(data[_META_KEY].tobytes()).decode("utf-8"))
        version = meta.get("schema_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema mismatch: {version!r} != {SNAPSHOT_SCHEMA_VERSION!r}"
            )
        canon = CanonTransform(
            rotation=data["canon_rotation"],
            translation=data["canon_translation"],
            scale=float(meta["canon_scale"]),
        )
        optional = {
            name: (np.array(data[name]) if name in data else None)
            for name in (
                "normals_raw", "alpha_id", "exp_params",
                "landmarks_106_raw", "landmarks_134_raw",
                "visibility_weights", "seg_visible",
            )
        }
        return PhotoSnapshot(
            photo_id=str(meta["photo_id"]),
            image_path=str(meta["image_path"]),
            pose_bucket=str(meta["pose_bucket"]),
            yaw_deg=float(meta["yaw_deg"]),
            pitch_deg=float(meta["pitch_deg"]),
            roll_deg=float(meta["roll_deg"]),
            vertices_raw=np.array(data["vertices_raw"]),
            triangles=np.array(data["triangles"]),
            canon=canon,
            quality=dict(meta.get("quality") or {}),
            extras=dict(meta.get("extras") or {}),
            **optional,
        )


def to_metric_context(
    snapshot: PhotoSnapshot,
    annotation_groups: list | None = None,
    macro_indices: dict[str, np.ndarray] | None = None,
):
    """Build a legacy MetricContext from a snapshot (no model access needed)."""
    from util.legacy_metrics.types import MetricContext

    if macro_indices is None:
        from util.zones import MACRO_BONE_INDICES

        macro_indices = {
            k: np.asarray(sorted(v), dtype=np.int64) for k, v in MACRO_BONE_INDICES.items()
        }
    return MetricContext(
        photo_id=snapshot.photo_id,
        image_path=Path(snapshot.image_path),
        pose_bucket=snapshot.pose_bucket,
        yaw_deg=float(snapshot.yaw_deg),
        pitch_deg=float(snapshot.pitch_deg),
        roll_deg=float(snapshot.roll_deg),
        recon=None,
        vertices_raw=np.asarray(snapshot.vertices_raw, dtype=np.float64),
        vertices_canon=snapshot.vertices_canon,
        vertices_shape_neutral=None,
        normals_raw=(
            None if snapshot.normals_raw is None
            else np.asarray(snapshot.normals_raw, dtype=np.float64)
        ),
        normals_canon=snapshot.normals_canon,
        normals_shape_neutral=None,
        triangles=snapshot.triangles,
        annotation_groups=list(annotation_groups or []),
        macro_indices=macro_indices,
        landmarks_106=snapshot.landmarks_106_raw,
    )


def landmarks_table(snapshot: PhotoSnapshot) -> dict[str, np.ndarray]:
    """Full landmark table in both spaces (for calibration + main analysis)."""
    out: dict[str, np.ndarray] = {}
    if snapshot.landmarks_106_raw is not None:
        out["ldm106_raw"] = snapshot.landmarks_106_raw
        out["ldm106_canon"] = snapshot.landmarks_106_canon
    if snapshot.landmarks_134_raw is not None:
        out["ldm134_raw"] = snapshot.landmarks_134_raw
        out["ldm134_canon"] = snapshot.landmarks_134_canon
    return out
