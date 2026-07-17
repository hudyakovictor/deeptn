#!/usr/bin/env python3
"""
Калибровка — обработка person_01..05, построение таблиц углов.
=================================================================
Для каждого человека из calibration_dataset:
  1. Детекция лица
  2. 3DDFA реконструкция
  3. Извлечение yaw/pitch/roll
  4. Классификация pose bucket
  5. Сохранение таблицы

Usage:
  python run_calibration.py \\
      --input dataset/calibration_dataset \\
      --output /Volumes/SDCARD/storage/calibration

  # Тест на 5 фото с каждого человека
  python run_calibration.py \\
      --input dataset/calibration_dataset \\
      --output /Volumes/SDCARD/storage/calibration --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_box import face_box
from model.recon import face_model
from util.pose_buckets import ALL_BUCKETS, classify_pose_bucket, normalize_bucket_name
from util.quality_gate import evaluate_image_array


PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration: build pose tables for person_01..05")
    parser.add_argument("--input", required=True, help="Путь к calibration_dataset (с папками person_01..05)")
    parser.add_argument("--output", required=True, help="Выходная папка (на SD-карту)")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить фото на человека")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--detector", default="retinaface")
    parser.add_argument("--backbone", default="resnet50")
    return parser.parse_args()


def list_photos(path: Path) -> list[Path]:
    return sorted([f for f in path.iterdir() if f.suffix.lower() in PHOTO_EXTS and not f.name.startswith(".")])


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Calibration: pose tables for person_01..05")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    if args.limit:
        print(f"Limit:  {args.limit} photos per person")
    print()

    person_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Found {len(person_dirs)} persons: {[d.name for d in person_dirs]}")
    print()

    print("Loading models...")
    t0 = time.time()

    class MockArgs:
        device = args.device
        detector = args.detector
        backbone = args.backbone
        iscrop = True
        ldm68 = True
        ldm106 = True
        ldm106_2d = True
        ldm134 = True
        seg = True
        seg_visible = True
        useTex = True
        extractTex = True

    model_args = MockArgs()
    recon_model = face_model(model_args)
    print(f"  3DDFA loaded ({time.time() - t0:.1f}s)")

    t0 = time.time()
    detector_obj = face_box(model_args)
    detector_fn = detector_obj.detector
    print(f"  Detector loaded ({time.time() - t0:.1f}s)")
    print()

    all_persons_data: dict[str, list[dict[str, Any]]] = {}
    bucket_stats: dict[str, dict[str, int]] = {}

    total_start = time.time()
    total_photos = 0
    total_ok = 0
    total_errors = 0

    for person_dir in person_dirs:
        person_id = person_dir.name
        photos = list_photos(person_dir)
        if args.limit:
            photos = photos[:args.limit]

        print(f"── {person_id} ({len(photos)} photos) ──")

        person_data: list[dict[str, Any]] = []
        t_person = time.time()

        for i, img_path in enumerate(photos, 1):
            elapsed = time.time() - t_person
            remaining = (elapsed / max(i - 1, 1)) * (len(photos) - i) if i > 1 else 0
            print(f"  [{i}/{len(photos)}] {img_path.name}  (remaining: ~{remaining:.0f}s)", flush=True)

            try:
                im = Image.open(img_path).convert("RGB")
                img_bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

                trans_params, im_tensor = detector_fn(im)

                recon_model.input_img = im_tensor.to(args.device)
                result_dict = recon_model.forward(identity_only=False)

                angles_deg = result_dict["alpha_angle_deg"][0]
                yaw, pitch, roll = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
                bucket = classify_pose_bucket(yaw, pitch, roll)
                bucket = normalize_bucket_name(bucket)

                quality = evaluate_image_array(img_bgr)

                entry = {
                    "file_name": img_path.name,
                    "person_id": person_id,
                    "yaw_deg": round(yaw, 2),
                    "pitch_deg": round(pitch, 2),
                    "roll_deg": round(roll, 2),
                    "pose_bucket": bucket,
                    "quality_score": round(quality.get("overall_score", 0), 4),
                    "quality_rejected": quality.get("is_rejected", False),
                    "quality_flags": quality.get("blocking_issues", []),
                }

                person_data.append(entry)
                print(f"    -> bucket={bucket} yaw={yaw:.1f} pitch={pitch:.1f} quality={entry['quality_score']:.3f}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  ERROR {img_path.name}: {e}")
                person_data.append({
                    "file_name": img_path.name,
                    "person_id": person_id,
                    "error": str(e),
                    "status": "error",
                })

        t_person_elapsed = time.time() - t_person
        n_ok = sum(1 for d in person_data if "error" not in d)
        n_err = sum(1 for d in person_data if "error" in d)

        all_persons_data[person_id] = person_data
        total_photos += len(person_data)
        total_ok += n_ok
        total_errors += n_err

        # Bucket statistics per person
        buckets = [d["pose_bucket"] for d in person_data if "pose_bucket" in d]
        bucket_stats[person_id] = dict(sorted({b: buckets.count(b) for b in set(buckets)}.items()))

        print(f"  {person_id}: {n_ok} ok, {n_err} errors ({t_person_elapsed:.1f}s)")
        print()

    # Build pose table — фото каждого человека сгруппированы по bucket
    pose_table: dict[str, dict[str, Any]] = {}
    for person_id, data in all_persons_data.items():
        ok_data = [d for d in data if "error" not in d]
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for d in ok_data:
            by_bucket[d["pose_bucket"]].append(d)

        pose_table[person_id] = {
            "n_photos": len(ok_data),
            "n_errors": len(data) - len(ok_data),
            "buckets": {
                bucket: {
                    "n": len(entries),
                    "yaw_min": round(min(e["yaw_deg"] for e in entries), 1),
                    "yaw_max": round(max(e["yaw_deg"] for e in entries), 1),
                    "pitch_min": round(min(e["pitch_deg"] for e in entries), 1),
                    "pitch_max": round(max(e["pitch_deg"] for e in entries), 1),
                    "photos": [{
                        "file": e["file_name"],
                        "yaw": e["yaw_deg"],
                        "pitch": e["pitch_deg"],
                        "roll": e["roll_deg"],
                        "quality": e["quality_score"],
                    } for e in entries],
                }
                for bucket, entries in by_bucket.items()
            },
        }

    # Overall coverage
    all_buckets_coverage: dict[str, int] = {}
    for person_data in all_persons_data.values():
        for d in person_data:
            if "pose_bucket" in d:
                all_buckets_coverage[d["pose_bucket"]] = all_buckets_coverage.get(d["pose_bucket"], 0) + 1

    cal_result = {
        "calibration_version": "app_final_v1",
        "n_persons": len(person_dirs),
        "n_photos_total": total_photos,
        "n_photos_ok": total_ok,
        "n_photos_errors": total_errors,
        "elapsed_seconds": round(time.time() - total_start, 1),
        "device": args.device,
        "detector": args.detector,
        "backbone": args.backbone,
        "bucket_coverage": all_buckets_coverage,
        "bucket_stats": bucket_stats,
        "pose_table": pose_table,
    }

    cal_json_path = output_dir / "calibration_pose_table.json"
    with open(cal_json_path, "w", encoding="utf-8") as f:
        json.dump(cal_result, f, indent=2, ensure_ascii=False, default=str)

    # Also save per-person CSVs
    for person_id, data in all_persons_data.items():
        csv_path = output_dir / f"{person_id}_poses.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            ok_data = [d for d in data if "error" not in d]
            f.write("file_name,yaw_deg,pitch_deg,roll_deg,pose_bucket,quality_score\n")
            for d in ok_data:
                f.write(f"{d['file_name']},{d['yaw_deg']},{d['pitch_deg']},{d['roll_deg']},{d['pose_bucket']},{d['quality_score']}\n")

    total_elapsed = time.time() - total_start
    mins = int(total_elapsed // 60)
    secs = int(total_elapsed % 60)

    print("=" * 60)
    print(f"Calibration COMPLETE in {mins}m{secs:02d}s")
    print(f"  Persons: {len(person_dirs)}")
    print(f"  Photos:  {total_ok} ok, {total_errors} errors")
    print(f"  Output:  {cal_json_path}")
    print(f"  CSV:     {output_dir}/*_poses.csv")
    print(f"  Buckets: {dict(sorted(all_buckets_coverage.items()))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
