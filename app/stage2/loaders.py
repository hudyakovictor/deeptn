from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .core import Record
from .quality_integration import load_quality_zone_summary


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_main(stage1_root: Path) -> list[Record]:
    index = stage1_root / "main_timeline.csv"
    if not index.is_file():
        raise FileNotFoundError(index)
    out: list[Record] = []
    for row in _rows(index):
        if not row.get("photo_id"):
            continue
        directory = stage1_root / row["photo_id"]
        validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "complete":
            continue
        info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
        relative_source = str(info.get("source_relative_path") or "")
        source_parts = Path(relative_source).parts
        source_group = source_parts[0] if len(source_parts) > 1 else "unknown"
        qsum = info.get("quality_summary") or {}
        gtq = qsum.get("global_texture_quality") or {}
        qzones = load_quality_zone_summary(directory)
        with np.load(directory / "reconstruction.npz", allow_pickle=False) as z:
            idx106 = z["ldm106_vertex_indices"].astype(np.int64); idx134 = z["ldm134_vertex_indices"].astype(np.int64)
            out.append(Record(
                record_id=row["photo_id"], dataset_id="main", date=row["date"], sequence=int(row["same_date_sequence"]),
                pose_bin=row["pose_bin"], angles=z["angle_deg_pitch_yaw_roll"].astype(np.float32),
                ldm106=z["ldm106_object_normalized"].astype(np.float32), ldm134=z["ldm134_object_normalized"].astype(np.float32),
                visible106=z["ldm106_visible"].astype(bool), visible134=z["ldm134_visible"].astype(bool),
                alpha_id=z["alpha_id"].astype(np.float32), alpha_exp=z["alpha_exp"].astype(np.float32),
                identity_only106=(z["ldm106_identity_only"] if "ldm106_identity_only" in z else z["vertices_identity_only"][idx106]).astype(np.float32),
                identity_only134=(z["ldm134_identity_only"] if "ldm134_identity_only" in z else z["vertices_identity_only"][idx134]).astype(np.float32),
                quality_status=str(gtq.get("status", qsum.get("status", "unknown"))),
                quality_texture_score=float(gtq.get("texture_score_0_1", 0.0) or 0.0),
                forehead_wrinkle_supported=bool(qsum.get("supported_forehead_wrinkle_pose_v1", False)),
                quality_zones=qzones,
                record_dir=str(directory),
                source_group=source_group,
                source_sha256=info.get("source_sha256"),
            ))
    return sorted(out, key=lambda r: (r.date or "9999", r.sequence, r.record_id))


def load_calibration(calibration_root: Path) -> list[Record]:
    root = calibration_root
    if (root / "calibration_datasets").is_dir():
        root = root / "calibration_datasets"
    out: list[Record] = []
    for npz_path in sorted(root.glob("*/*/record.npz")):
        directory = npz_path.parent
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as z:
            out.append(Record(
                record_id=meta["record_id"], dataset_id=meta["dataset_id"], date=None, sequence=int(meta.get("frame_index", 0)),
                pose_bin=meta["pose_bin"], angles=z["angle_deg_pitch_yaw_roll"].astype(np.float32),
                ldm106=z["ldm106_object_norm"].astype(np.float32), ldm134=z["ldm134_object_norm"].astype(np.float32),
                visible106=z["ldm106_visible_original"].astype(bool), visible134=z["ldm134_visible_original"].astype(bool),
                alpha_id=z["alpha_id"].astype(np.float32), alpha_exp=z["alpha_exp"].astype(np.float32),
                record_dir=str(directory),
            ))
    if not out:
        raise FileNotFoundError(f"no calibration records under {root}")
    return out
