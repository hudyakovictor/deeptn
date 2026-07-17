from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "deeputin-stage1-v2.2"
PHOTO_SCHEMA_VERSION = "deeputin-photo-v2.2"
VALIDATION_SCHEMA_VERSION = "deeputin-validation-v2.2"
SEMANTIC_POLICY = "3ddfa-semantic-skin-plus-nose-v1"
POSE_BINS = (
    ("left_profile", -95.0, -50.0, -70.0),
    ("left_deep", -50.0, -40.0, -45.0),
    ("left_mid", -40.0, -25.0, -32.5),
    ("left_light", -25.0, -10.0, -17.5),
    ("frontal", -10.0, 10.0, 0.0),
    ("right_light", 10.0, 25.0, 17.5),
    ("right_mid", 25.0, 40.0, 32.5),
    ("right_deep", 40.0, 50.0, 45.0),
    ("right_profile", 50.0, 95.000001, 70.0),
)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class Stage1Config:
    project_root: Path
    input_dir: Path
    output_dir: Path
    device: str = "auto"
    detector: str = "retinaface"
    backbone: str = "resnet50"
    uv_size: int = 768
    limit: int = 0
    overwrite: bool = False
    continue_on_error: bool = True
    save_original: bool = True

    def extraction_payload(self) -> dict[str, Any]:
        """Only settings that can change scientific output."""
        return {
            "schema_version": SCHEMA_VERSION,
            "detector": self.detector,
            "backbone": self.backbone,
            "uv_size": int(self.uv_size),
            "semantic_policy": SEMANTIC_POLICY,
            "pose_bins": POSE_BINS,
        }

    def public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
