#!/usr/bin/env python3
"""
DEEPUTIN Forensic Pipeline (app-final версия)
=============================================
Stages:
  S1 extract    — 3DDFA reconstruction + pose + quality + snapshot + landmark CSVs
  S2 calibrate  — noise model from person_01..05
  S3 compare    — pairwise comparison
  S4 verdict    — Bayesian H0/H1/H2
  S5 report     — final forensic report

Usage:
  python run_pipeline.py extract --input <path> --output <dir> [--export-obj] [--export-uv]
  python run_pipeline.py calibrate --input <cal_dir> --output <dir>
  python run_pipeline.py compare --snapshots <dir> --output <dir>
  python run_pipeline.py report --input <dir> --output <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, TextIO

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_box import face_box
from model.recon import face_model
from util.pose_buckets import classify_pose_bucket, normalize_bucket_name, ALL_BUCKETS
from util.quality_gate import evaluate_image_array
from util.snapshot import (
    PhotoSnapshot,
    CanonTransform,
    canon_transform_from_pose,
    save_snapshot,
    load_snapshot,
    landmarks_table,
)

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
LANDMARK_NAMES_106 = [f"lm_{i}" for i in range(106)]
LANDMARK_NAMES_134 = [f"lm_{i}" for i in range(134)]


def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO", no_progress: bool = False) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if no_progress:
        root.setLevel(logging.WARNING)


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def resolve_data_root(args_root: Optional[str]) -> Path | None:
    root = args_root or os.environ.get("DPTN_DATA_ROOT")
    if root:
        return Path(root).resolve()
    return None


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--detector", default="retinaface", choices=["retinaface", "mtcnn"])
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    p.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING / ERROR")
    p.add_argument("--log-file", default=None, help="Path to log file")
    p.add_argument("--no-progress", action="store_true", help="Disable progress output")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeputin", description="DEEPUTIN forensic pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="S1: reconstruction + snapshot + CSV")
    _add_common_args(p_extract)
    p_extract.add_argument("--input", required=True, help="Image file or directory")
    p_extract.add_argument("--output", required=True, help="Output directory")
    p_extract.add_argument("--limit", type=int, default=None, help="Max photos to process")
    p_extract.add_argument("--export-obj", action="store_true", help="Export mesh_raw.obj + mesh_aligned.obj")
    p_extract.add_argument("--export-uv", action="store_true", help="Export UV analysis PNGs")
    p_extract.add_argument("--enable-texture", action="store_true", help="Enable texture channel")

    p_cal = sub.add_parser("calibrate", help="S2: calibration from person_01..05")
    _add_common_args(p_cal)
    p_cal.add_argument("--input", required=True, help="Calibration dataset directory")
    p_cal.add_argument("--output", required=True, help="Output directory")
    p_cal.add_argument("--limit", type=int, default=None, help="Max photos per person")

    p_comp = sub.add_parser("compare", help="S3: pairwise comparison (TODO)")
    p_comp.add_argument("--snapshots", required=True, help="Snapshots directory")
    p_comp.add_argument("--output", required=True, help="Output directory")

    p_verdict = sub.add_parser("verdict", help="S4: Bayesian verdict (TODO)")
    p_verdict.add_argument("--input", required=True)
    p_verdict.add_argument("--output", required=True)

    p_report = sub.add_parser("report", help="S5: forensic report (TODO)")
    p_report.add_argument("--input", required=True)
    p_report.add_argument("--output", required=True)

    p_rexport = sub.add_parser("export-csv", help="Re-export CSVs from existing snapshots")
    p_rexport.add_argument("--snapshots", required=True, help="Directory with snapshot.npz files")
    p_rexport.add_argument("--output", required=True, help="Output directory")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def list_photo_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted([f for f in path.iterdir() if f.suffix.lower() in PHOTO_EXTS and not f.name.startswith(".")])
    return files


def _write_landmark_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
    lm_source: str,
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _landmark_rows(
    points: np.ndarray,
    photo_id: str,
    person_id: str,
    lm_index: np.ndarray | None,
    visible_idx: np.ndarray | None,
    point_names: list[str] | None,
    source_space: str,
) -> list[dict]:
    rows = []
    for i in range(points.shape[0]):
        vi = int(lm_index[i]) if lm_index is not None else None
        vw = float(visible_idx[vi]) if vi is not None and visible_idx is not None else 1.0
        rows.append({
            "point_index": i,
            "point_name": point_names[i] if point_names else f"lm_{i}",
            "x": round(float(points[i, 0]), 4),
            "y": round(float(points[i, 1]), 4),
            "z": round(float(points[i, 2]), 4),
            "visibility_weight": vw,
            "source_space": source_space,
            "valid": 1,
        })
    return rows


def process_single_photo(
    img_path: Path,
    recon_model: face_model,
    detector: Any,
    args: argparse.Namespace,
    photo_id: str = "",
    person_id: str = "",
) -> tuple[dict[str, Any], PhotoSnapshot, np.ndarray, dict, dict]:
    """S1: детекция → реконструкция → снапшот + CSV лендмарков."""
    start_t = time.time()
    logger = logging.getLogger()

    im = Image.open(img_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    trans_params, im_tensor = detector(im)
    recon_model.input_img = im_tensor.to(recon_model.device)
    result_dict = recon_model.forward(identity_only=False)

    angles_deg = result_dict["alpha_angle_deg"][0]
    yaw, pitch, roll = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
    bucket = classify_pose_bucket(yaw, pitch, roll)
    bucket = normalize_bucket_name(bucket)

    quality = evaluate_image_array(img_bgr)

    v3d = result_dict["v3d"][0]
    ldm106_idx = recon_model.ldm106.detach().cpu().numpy()
    ldm134_idx = recon_model.ldm134.detach().cpu().numpy()
    ldm106_3d = v3d[ldm106_idx].astype(np.float32)
    ldm134_3d = v3d[ldm134_idx].astype(np.float32)

    visible_idx = result_dict.get("visible_idx")
    if visible_idx is not None:
        visible_idx = visible_idx.reshape(-1)
    seg_visible = result_dict.get("seg_visible")
    if seg_visible is not None:
        seg_visible = seg_visible[0].astype(np.float32) if seg_visible.ndim == 4 else seg_visible.astype(np.float32)

    canon = canon_transform_from_pose(pitch, yaw, roll, bucket)
    alpha_id = result_dict["alpha_id"][0].astype(np.float32)
    alpha_exp = result_dict["alpha_exp"][0].astype(np.float32)

    snapshot = PhotoSnapshot(
        photo_id=photo_id or img_path.stem,
        image_path=str(img_path),
        pose_bucket=bucket,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        vertices_raw=v3d.astype(np.float32),
        triangles=np.asarray(result_dict["tri"], dtype=np.int64),
        canon=canon,
        normals_raw=result_dict["normals_camera"][0].astype(np.float32) if result_dict.get("normals_camera") is not None else None,
        alpha_id=alpha_id,
        exp_params=alpha_exp,
        landmarks_106_raw=ldm106_3d,
        landmarks_134_raw=ldm134_3d,
        visibility_weights=visible_idx.astype(np.float32) if visible_idx is not None else None,
        seg_visible=seg_visible,
        quality=quality,
        extras={
            "trans_params": None if trans_params is None else trans_params.tolist(),
        },
    )

    # ── Вычисление aligned лендмарков ──
    ldm106_aligned = snapshot.landmarks_106_canon
    ldm134_aligned = snapshot.landmarks_134_canon

    # ── CSV лендмарков (raw + aligned) ──
    lm_fields = ["point_index", "point_name", "x", "y", "z", "visibility_weight", "source_space", "valid"]
    csv_artifacts = {}

    if ldm106_3d is not None:
        raw_rows = _landmark_rows(ldm106_3d, snapshot.photo_id, person_id,
                                  ldm106_idx, visible_idx, LANDMARK_NAMES_106, "camera")
        csv_artifacts["ldm106_raw.csv"] = (lm_fields, raw_rows)
        if ldm106_aligned is not None:
            aligned_rows = _landmark_rows(ldm106_aligned, snapshot.photo_id, person_id,
                                          ldm106_idx, visible_idx, LANDMARK_NAMES_106, "canonical")
            csv_artifacts["ldm106_aligned.csv"] = (lm_fields, aligned_rows)

    if ldm134_3d is not None:
        raw_rows = _landmark_rows(ldm134_3d, snapshot.photo_id, person_id,
                                  ldm134_idx, visible_idx, LANDMARK_NAMES_134, "camera")
        csv_artifacts["ldm134_raw.csv"] = (lm_fields, raw_rows)
        if ldm134_aligned is not None:
            aligned_rows = _landmark_rows(ldm134_aligned, snapshot.photo_id, person_id,
                                          ldm134_idx, visible_idx, LANDMARK_NAMES_134, "canonical")
            csv_artifacts["ldm134_aligned.csv"] = (lm_fields, aligned_rows)

    # ── metadata.json ──
    image_hash = hashlib.sha256()
    try:
        image_hash.update(img_path.read_bytes())
    except Exception:
        pass

    meta = {
        "schema_version": "snapshot_v1",
        "photo_id": snapshot.photo_id,
        "person_id": person_id or None,
        "image_file": img_path.name,
        "image_hash": image_hash.hexdigest(),
        "pose": {
            "pitch_deg": float(pitch),
            "yaw_deg": float(yaw),
            "roll_deg": float(roll),
            "bucket": bucket,
        },
        "canonical_transform": {
            "rotation": canon.rotation.tolist(),
            "translation": canon.translation.tolist(),
            "scale": float(canon.scale),
        },
        "landmarks": {
            "ldm106_source": "3ddfa_v3",
            "ldm106_count": int(ldm106_3d.shape[0]) if ldm106_3d is not None else 0,
            "ldm134_source": "3ddfa_v3",
            "ldm134_count": int(ldm134_3d.shape[0]) if ldm134_3d is not None else 0,
        },
        "quality_score": float(quality.get("overall_score", 0)),
        "topology_hash": "bfm_2020",
        "elapsed_seconds": round(time.time() - start_t, 3),
    }

    record = {
        "photo_id": snapshot.photo_id,
        "person_id": person_id or None,
        "file_path": str(img_path),
        "file_name": img_path.name,
        "status": "ready",
        "pose_bucket": bucket,
        "pose": {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll},
        "quality": quality,
        "elapsed_seconds": round(time.time() - start_t, 3),
    }

    return record, snapshot, img_bgr, meta, csv_artifacts


def _write_photo_outputs(
    photo_dir: Path,
    record: dict,
    snapshot: PhotoSnapshot,
    img_bgr: np.ndarray,
    meta: dict,
    csv_artifacts: dict[str, tuple[list[str], list[dict]]],
    export_obj: bool = False,
    export_uv: bool = False,
) -> None:
    photo_dir.mkdir(parents=True, exist_ok=True)

    # ── face crop ──
    rgb_crop = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    thumb_img = Image.fromarray(rgb_crop)
    thumb_img.thumbnail((224, 224))
    thumb_img.save(photo_dir / "face_crop.jpg", "JPEG", quality=85)

    # ── snapshot.npz ──
    save_snapshot(snapshot, photo_dir / "snapshot.npz")

    # ── CSV лендмарков ──
    for fname, (fields, rows) in csv_artifacts.items():
        _write_landmark_csv(photo_dir / fname, fields, rows, meta["landmarks"].get("ldm106_source", ""))

    # ── metadata.json ──
    with open(photo_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    # ── quality.json ──
    with open(photo_dir / "quality.json", "w", encoding="utf-8") as f:
        json.dump(record["quality"], f, indent=2, ensure_ascii=False, default=str)

    # ── face_mask.png (crop+letterbox + skin-only alpha, как в старой версии) ──
    if snapshot.seg_visible is not None:
        try:
            trans_params = snapshot.extras.get("trans_params") if snapshot.extras else None
            _write_face_mask_png(photo_dir, img_bgr, np.asarray(snapshot.seg_visible), trans_params)
        except Exception as e:
            logging.getLogger().warning("face_mask.png SKIP: %s", e)

    # ── OBJ (опционально) ──
    if export_obj:
        _write_obj(photo_dir / "mesh_raw.obj", snapshot.vertices_raw, snapshot.triangles,
                   normals=snapshot.normals_raw)
        _write_obj(photo_dir / "mesh_aligned.obj", snapshot.vertices_canon, snapshot.triangles,
                   normals=snapshot.normals_canon)

    # ── UV (опционально) ──
    if export_uv:
        _write_uv_outputs(photo_dir)


FACE_CROP_WIDTH = 424
FACE_CROP_HEIGHT = 500


def _resize_letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(int(w * scale), 1)
    nh = max(int(h * scale), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=img.dtype) if img.ndim == 3 else np.zeros((target_h, target_w), dtype=img.dtype)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _resize_letterbox_gray(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw = max(int(w * scale), 1)
    nh = max(int(h * scale), 1)
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=mask.dtype)
    y0 = (target_h - nh) // 2
    x0 = (target_w - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def build_skin_mask_from_seg(seg_visible: np.ndarray) -> np.ndarray:
    """Build skin-only mask from 3DDFA segmentation (224x224x8).

    Returns (224,224) float mask: skin ∪ nose, excluding eyes/brows/lips.
    Matches old deeputin's _save_face_assets logic.
    """
    skin = seg_visible[:, :, 7].copy()
    nose = seg_visible[:, :, 4].copy()
    skin_mask = np.maximum(skin, nose)

    excluded = np.maximum.reduce([
        seg_visible[:, :, 0],
        seg_visible[:, :, 1],
        seg_visible[:, :, 2],
        seg_visible[:, :, 3],
        seg_visible[:, :, 5],
        seg_visible[:, :, 6],
    ])
    exclusion_weight = 1.0 / (1.0 + np.exp(-10 * (excluded - 0.5)))
    skin_mask *= 1.0 - exclusion_weight
    skin_mask = np.clip(skin_mask, 0, 1)
    return skin_mask


def _write_face_mask_png(photo_dir: Path, img_bgr: np.ndarray, seg_visible: np.ndarray, trans_params: Any = None) -> Path:
    """Write face_mask.png: cropped+letterboxed face RGBA with skin-only alpha, как в старой версии."""
    h, w = img_bgr.shape[:2]

    # 1. Skin mask from seg_visible (224x224)
    skin_224 = build_skin_mask_from_seg(seg_visible)
    skin_224_uint8 = np.clip(skin_224 * 255, 0, 255).astype(np.uint8)

    # 2. Project from 224x224 to original image using back_resize_crop_img
    if trans_params is not None and len(trans_params) >= 5:
        try:
            from util.io import back_resize_crop_img
            from PIL import Image as PILImage
            tp = np.asarray(trans_params, dtype=np.float64)
            mask_rgb = np.stack((skin_224_uint8, skin_224_uint8, skin_224_uint8), axis=-1)
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            full_mask_rgb = back_resize_crop_img(mask_rgb, tp, blank, resample_method=PILImage.BILINEAR)
            skin_mask_full = full_mask_rgb[:, :, 0].astype(np.float32) / 255.0
        except Exception as exc:
            logging.getLogger().warning("back_resize_crop_img failed (%s), fallback to cv2.resize", exc)
            skin_mask_full = cv2.resize(skin_224_uint8, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    else:
        skin_mask_full = cv2.resize(skin_224_uint8, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask_full = cv2.morphologyEx(skin_mask_full, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 2. Find bounding box from mask
    mask_uint8 = np.clip(skin_mask_full * 255, 0, 255).astype(np.uint8)
    coords = cv2.findNonZero((mask_uint8 > 10).astype(np.uint8))
    if coords is None:
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 0
        p = photo_dir / "face_mask.png"
        cv2.imwrite(str(p), rgba)
        return p

    x, y, bw, bh = cv2.boundingRect(coords)

    # 3. Expand bbox (25% margin, preserve aspect)
    target_aspect = FACE_CROP_WIDTH / FACE_CROP_HEIGHT
    crop_w = int(max(bw * 1.25, bh * 1.25 * target_aspect, 1))
    crop_h = int(max(bh * 1.25, crop_w / target_aspect, 1))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x1 = int(round(cx - crop_w / 2.0))
    y1 = int(round(cy - crop_h / 2.0))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp to image bounds
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > w:
        x1 = max(0, x1 - (x2 - w)); x2 = w
    if y2 > h:
        y1 = max(0, y1 - (y2 - h)); y2 = h

    # 4. Crop
    face_crop_bgr = img_bgr[y1:y2, x1:x2].copy()
    face_crop_mask = skin_mask_full[y1:y2, x1:x2]

    # 5. Letterbox to target size
    face_crop_bgr = _resize_letterbox(face_crop_bgr, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)
    face_crop_mask = _resize_letterbox_gray(face_crop_mask, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)

    # 6. RGBA
    rgba = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.clip(face_crop_mask * 255, 0, 255).astype(np.uint8)
    p = photo_dir / "face_mask.png"
    cv2.imwrite(str(p), rgba)
    return p


_OBJ_CAMERA_DISTANCE = 10.0  # 3DDFA-V3 camera distance


def _write_obj(obj_path: Path, vertices: np.ndarray, triangles: np.ndarray, normals: np.ndarray | None = None) -> None:
    with open(obj_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None:
            # Normals from model use model-space Z (toward viewer).
            # Camera space inverts Z, so reflect normals Z to match.
            norms = normals.copy()
            norms[:, 2] = -norms[:, 2]
            nlen = np.linalg.norm(norms, axis=1, keepdims=True)
            nlen = np.where(nlen > 0, nlen, 1.0)
            norms /= nlen
            for n in norms:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        # BFM triangles are CW in camera space; reverse to CCW for OBJ standard
        for t in triangles:
            i0, i1, i2 = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
            if normals is not None:
                f.write(f"f {i0}//{i0} {i2}//{i2} {i1}//{i1}\n")
            else:
                f.write(f"f {i0} {i2} {i1}\n")


def _write_uv_outputs(photo_dir: Path) -> None:
    """Placeholder for UV export — requires texture channel integration."""
    logger = logging.getLogger()
    logger.warning("UV export not yet implemented — requires texture channel")


def _image_hash_bytes(path: Path) -> str:
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        pass
    return h.hexdigest()


def detect_person_id(img_path: Path, input_path: Path) -> str:
    """Derive person_id from directory structure (e.g. dataset/person_01/photo.jpg → person_01)."""
    try:
        rel = img_path.resolve().parent.relative_to(input_path.resolve())
    except ValueError:
        return ""
    for p in rel.parts:
        if p.startswith("person_"):
            return p
    return rel.parts[0] if rel.parts else ""


def run_s1(
    input_paths: list[Path],
    output_dir: Path,
    recon_model: face_model,
    detector: Any,
    args: argparse.Namespace,
    dataset_label: str = "main",
    input_root: Path | None = None,
) -> list[dict]:
    """S1: Реконструкция → снапшот → CSV лендмарков → metadata."""
    logger = logging.getLogger()
    records = []
    total = len(input_paths)
    errors = 0

    logger.info("S1 %s started (%d photos)", dataset_label, total)
    t_start = time.time()

    for i, img_path in enumerate(input_paths, 1):
        pid = detect_person_id(img_path, input_root) if input_root else ""
        photo_id = img_path.stem

        if not args.no_progress:
            elapsed = time.time() - t_start
            avg = elapsed / max(i - 1, 1)
            remaining = avg * (total - i) if i > 1 else 0
            logger.info("S1 [%d/%d] %s  (~%ds remaining)", i, total, img_path.name, int(remaining))

        try:
            record, snapshot, img_bgr, meta, csv_artifacts = process_single_photo(
                img_path, recon_model, detector, args,
                photo_id=photo_id, person_id=pid,
            )
            records.append(record)

            # Per-photo directory: output/person_id/photo_id/ or output/photo_id/
            photo_rel = Path(pid) / photo_id if pid else Path(photo_id)
            photo_dir = output_dir / photo_rel

            _write_photo_outputs(
                photo_dir, record, snapshot, img_bgr, meta, csv_artifacts,
                export_obj=getattr(args, "export_obj", False),
                export_uv=getattr(args, "export_uv", False),
            )

            yaw = record["pose"]["yaw_deg"]
            pitch = record["pose"]["pitch_deg"]
            score = record["quality"].get("overall_score", 0)
            ldm106_n = meta["landmarks"]["ldm106_count"]
            ldm134_n = meta["landmarks"]["ldm134_count"]
            logger.info("S1 %s bucket=%s yaw=%.1f pitch=%.1f quality=%.3f ldm106=%d ldm134=%d",
                        photo_id, record["pose_bucket"], yaw, pitch, score, ldm106_n, ldm134_n)

        except Exception as e:
            import traceback
            logger.error("S1 %s %s", img_path.name, traceback.format_exc())
            errors += 1
            records.append({
                "photo_id": photo_id,
                "file_name": img_path.name,
                "status": "error",
                "error": str(e),
            })
            logger.error("S1 %s %s", photo_id, str(e))

    elapsed = time.time() - t_start
    ok = len(records) - errors
    logger.info("S1 %s done: %d ok, %d errors in %.1fs", dataset_label, ok, errors, elapsed)
    return records


def re_export_csv(snapshots_dir: Path, output_dir: Path) -> None:
    """Re-export per-photo landmark CSVs from existing snapshots."""
    logger = logging.getLogger()
    snapshots_dir = Path(snapshots_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(snapshots_dir.rglob("snapshot.npz"))
    if not npz_files:
        logger.warning("No snapshot.npz files found under %s", snapshots_dir)
        return

    logger.info("Exporting CSVs from %d snapshots...", len(npz_files))
    count = 0
    for npz_path in npz_files:
        try:
            snap = load_snapshot(npz_path)
        except Exception as e:
            logger.warning("SKIP %s: %s", npz_path, e)
            continue

        # Reconstruct the relative path structure
        rel = npz_path.parent.relative_to(snapshots_dir)
        photo_dir = output_dir / rel
        photo_dir.mkdir(parents=True, exist_ok=True)

        ldm106_idx = None
        ldm134_idx = None

        for fname, pts_raw, pts_aligned, n_pts, names in [
            ("ldm106_raw.csv", snap.landmarks_106_raw, snap.landmarks_106_canon, 106, LANDMARK_NAMES_106),
            ("ldm134_raw.csv", snap.landmarks_134_raw, snap.landmarks_134_canon, 134, LANDMARK_NAMES_134),
        ]:
            if pts_raw is not None:
                rows = _landmark_rows(np.asarray(pts_raw), snap.photo_id, "",
                                       ldm106_idx if "134" not in fname else ldm134_idx,
                                       snap.visibility_weights, names, "camera")
                fields = ["point_index", "point_name", "x", "y", "z", "visibility_weight", "source_space", "valid"]
                _write_landmark_csv(photo_dir / fname, fields, rows, "")

        # aligned versions
        for fname, pts in [
            ("ldm106_aligned.csv", snap.landmarks_106_canon),
            ("ldm134_aligned.csv", snap.landmarks_134_canon),
        ]:
            if pts is not None:
                rows = _landmark_rows(np.asarray(pts), snap.photo_id, "",
                                       ldm106_idx if "106" in fname else ldm134_idx,
                                       snap.visibility_weights,
                                       LANDMARK_NAMES_106 if "106" in fname else LANDMARK_NAMES_134,
                                       "canonical")
                fields = ["point_index", "point_name", "x", "y", "z", "visibility_weight", "source_space", "valid"]
                _write_landmark_csv(photo_dir / fname, fields, rows, "")

        count += 1

    logger.info("CSV export complete: %d photos -> %s", count, output_dir)


def run_s2(
    calibration_dir: Path,
    output_cal_dir: Path,
    recon_model: face_model,
    detector: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """S2: Калибровка — обработка person_01..05, построение таблицы углов."""
    logger = logging.getLogger()
    logger.info("S2 calibrate started: %s", calibration_dir)

    person_dirs = sorted([d for d in calibration_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    all_records = []
    person_summary = {}

    for person_dir in person_dirs:
        person_id = person_dir.name
        photos = list_photo_files(person_dir)
        if args.limit:
            photos = photos[:args.limit]
        logger.info("S2 calibrate %s: %d photos", person_id, len(photos))

        person_records = run_s1(photos, output_cal_dir / person_id, recon_model, detector,
                                args, dataset_label=person_id)
        all_records.extend(person_records)

        yaws = [r["pose"]["yaw_deg"] for r in person_records if r.get("status") == "ready"]
        pitches = [r["pose"]["pitch_deg"] for r in person_records if r.get("status") == "ready"]
        buckets = [r["pose_bucket"] for r in person_records if r.get("status") == "ready"]

        person_summary[person_id] = {
            "total": len(person_records),
            "ok": sum(1 for r in person_records if r.get("status") == "ready"),
            "errors": sum(1 for r in person_records if r.get("status") == "error"),
            "yaw_min": round(min(yaws), 1) if yaws else None,
            "yaw_max": round(max(yaws), 1) if yaws else None,
            "pitch_min": round(min(pitches), 1) if pitches else None,
            "pitch_max": round(max(pitches), 1) if pitches else None,
            "buckets": dict(sorted({b: buckets.count(b) for b in set(buckets)}.items())),
        }

    cal_summary = {
        "type": "calibration_summary",
        "n_persons": len(person_dirs),
        "n_photos": len(all_records),
        "n_ok": sum(1 for r in all_records if r.get("status") == "ready"),
        "persons": person_summary,
    }

    cal_summary_path = output_cal_dir / "calibration_summary.json"
    output_cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_summary_path, "w", encoding="utf-8") as f:
        json.dump(cal_summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("S2 calibrate done: %d/%d ok", cal_summary["n_ok"], cal_summary["n_photos"])
    return cal_summary


def _load_models(args: argparse.Namespace) -> tuple[face_model, Any]:
    logger = logging.getLogger()
    logger.info("Loading models...")
    t0 = time.time()

    # Force CPU — MPS issues with nvdiffrast and int64 ops
    eff_device = "cpu" if args.device in ("auto", "mps") else args.device

    class MockArgs:
        device = eff_device
        detector = args.detector
        backbone = args.backbone
        iscrop = True
        ldm68 = True
        ldm106 = True
        ldm106_2d = True
        ldm134 = True
        seg = True
        seg_visible = True
        useTex = False
        extractTex = getattr(args, "enable_texture", False) or getattr(args, "export_uv", False)

    model_args = MockArgs()
    recon_model = face_model(model_args)
    logger.info("3DDFA model loaded (%.1fs)", time.time() - t0)

    t0 = time.time()
    detector_obj = face_box(model_args)
    detector_fn = detector_obj.detector
    logger.info("Detector loaded (%.1fs)", time.time() - t0)
    return recon_model, detector_fn


def cmd_extract(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args.device = resolve_device(args.device)
    photos = list_photo_files(input_path)
    if args.limit:
        photos = photos[:args.limit]

    logger.info("Extract: %d photos from %s", len(photos), input_path)
    logger.info("Device: %s | Detector: %s | Backbone: %s", args.device, args.detector, args.backbone)

    recon_model, detector_fn = _load_models(args)
    records = run_s1(photos, output_dir, recon_model, detector_fn, args,
                     dataset_label="extract", input_root=input_path)

    # ── Manifest ──
    manifest = {
        "command": "extract",
        "input": str(input_path),
        "output": str(output_dir),
        "n_photos": len(photos),
        "n_ok": sum(1 for r in records if r.get("status") == "ready"),
        "n_errors": sum(1 for r in records if r.get("status") == "error"),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Extract done: %s", output_dir)


def cmd_calibrate(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    cal_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    args.device = resolve_device(args.device)
    logger.info("Calibrate from: %s", cal_dir)

    recon_model, detector_fn = _load_models(args)
    run_s2(cal_dir, output_dir, recon_model, detector_fn, args)
    logger.info("Calibrate done: %s", output_dir)


def cmd_compare(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    logger.info("Compare: TODO — %s -> %s", args.snapshots, args.output)
    print("Compare stage not yet implemented")


def cmd_verdict(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    logger.info("Verdict: TODO — %s -> %s", args.input, args.output)
    print("Verdict stage not yet implemented")


def cmd_report(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    logger.info("Report: TODO — %s -> %s", args.input, args.output)
    print("Report stage not yet implemented")


def cmd_export_csv(args: argparse.Namespace) -> None:
    re_export_csv(Path(args.snapshots), Path(args.output))


def main() -> None:
    args = parse_args()
    setup_logging(log_file=getattr(args, "log_file", None),
                  log_level=getattr(args, "log_level", "INFO"),
                  no_progress=getattr(args, "no_progress", False))
    if hasattr(args, "device"):
        args.device = resolve_device(args.device)

    cmds = {
        "extract": cmd_extract,
        "calibrate": cmd_calibrate,
        "compare": cmd_compare,
        "verdict": cmd_verdict,
        "report": cmd_report,
        "export-csv": cmd_export_csv,
    }
    cmd_fn = cmds.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        logging.getLogger().error("Unknown command: %s", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main()
