from __future__ import annotations

import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .assets import save_image_assets, save_uv_and_mesh, technical_quality, save_face_mask, save_semantic_channels
from .config import IMAGE_EXTENSIONS, PHOTO_SCHEMA_VERSION, SCHEMA_VERSION, Stage1Config
from .geometry import pack_mask, to_original_image
from .masks import build_mask_bundle
from .naming import make_photo_id, parse_photo_name
from .paths import install_runtime_paths, resolve_project_layout
from .quality_zones import build_quality_files
from .reconstruction import ReconstructionBundle, ReconstructionEngine
from .storage import atomic_photo_directory, clean_incomplete, write_failure
from .utils import atomic_json, runtime_versions, sha256_file, sha256_json, sha256_paths, write_csv
from .validator import is_resumable, validate_photo


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _landmark_rows(points: np.ndarray, visible: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"landmark_id": i, "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
         "visible": int(visible[i]), "vertex_index": int(indices[i])}
        for i, p in enumerate(points)
    ]


class Stage1Engine:
    def __init__(self, config: Stage1Config):
        self.cfg = config
        self.layout = resolve_project_layout(config.project_root)
        install_runtime_paths(self.layout)
        self.root = self.layout.project_root
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        clean_incomplete(self.cfg.output_dir)
        self.config_hash = sha256_json(config.extraction_payload())
        package_dir = Path(__file__).resolve().parent
        code_files = (
            list(package_dir.rglob("*.py"))
            + [self.layout.model_dir / "recon.py"]
            + list(self.layout.uv_module_dir.rglob("*.py"))
        )
        self.code_hash = sha256_paths(code_files, self.root)
        weight = "net_recon.pth" if config.backbone == "resnet50" else "net_recon_mbnet.pth"
        assets_dir = self.layout.assets_dir
        model_files = [assets_dir / "face_model.npy", assets_dir / weight, assets_dir / "large_base_net.pth"]
        missing = [p for p in model_files if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing required model assets: " + ", ".join(map(str, missing)))
        self.model_hash = sha256_paths(model_files, self.root)
        self.recon = ReconstructionEngine(self.root, config.device, config.detector, config.backbone)

    def run(self) -> dict[str, Any]:
        photos = sorted(
            p for p in self.cfg.input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")
        )
        if self.cfg.limit:
            photos = photos[: self.cfg.limit]
        started = time.time(); rows: list[dict[str, Any]] = []; errors: list[dict[str, Any]] = []
        skipped = 0
        for number, path in enumerate(photos, 1):
            print(f"[{number}/{len(photos)}] {path.name}", flush=True)
            try:
                info, was_skipped = self._one(path)
                rows.append(self._index_row(info)); skipped += int(was_skipped)
            except Exception as exc:
                payload = {
                    "source_relative_path": self._relative(path), "source_filename": path.name,
                    "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "timestamp": _utc(),
                }
                errors.append(payload)
                try:
                    parsed = parse_photo_name(path); source_hash = sha256_file(path)
                    write_failure(self.cfg.output_dir, make_photo_id(parsed, source_hash), payload)
                except Exception:
                    pass
                print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
                self.recon.cleanup()
                if not self.cfg.continue_on_error:
                    raise
        rows.sort(key=lambda r: (r["date"], int(r["same_date_sequence"]), r["photo_id"]))
        pose_counts: dict[str, int] = {}
        for i, row in enumerate(rows, 1):
            row["chronology_index_global"] = i
            pose = str(row["pose_bin"])
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
            row["chronology_index_in_pose"] = pose_counts[pose]
        write_csv(self.cfg.output_dir / "main_index.csv", rows or [{"status": "no_successes"}])
        write_csv(self.cfg.output_dir / "main_timeline.csv", rows or [{"status": "no_successes"}])
        if errors:
            write_csv(self.cfg.output_dir / "errors.csv", errors)
        else:
            (self.cfg.output_dir / "errors.csv").write_text("", encoding="utf-8")
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "complete" if not errors else "complete_with_errors",
            "created_at_utc": _utc(), "input_count": len(photos), "success_count": len(rows),
            "error_count": len(errors), "skipped_valid_count": skipped, "elapsed_seconds": time.time() - started,
            "input_dir": str(self.cfg.input_dir.resolve()), "output_dir": str(self.cfg.output_dir.resolve()),
            "device": self.recon.device, "detector": self.cfg.detector, "backbone": self.cfg.backbone,
            "uv_size": self.cfg.uv_size, "code_hash": self.code_hash, "config_hash": self.config_hash,
            "model_hash": self.model_hash, "runtime": runtime_versions(),
        }
        atomic_json(self.cfg.output_dir / "stage1_manifest.json", manifest)
        print(f"DONE success={len(rows)} errors={len(errors)} skipped={skipped}", flush=True)
        return manifest

    def _one(self, path: Path) -> tuple[dict[str, Any], bool]:
        parsed = parse_photo_name(path)
        source_hash = sha256_file(path)
        photo_id = make_photo_id(parsed, source_hash)
        final = self.cfg.output_dir / photo_id
        if not self.cfg.overwrite:
            okay, info = is_resumable(final, source_hash, self.code_hash, self.config_hash, self.model_hash)
            if okay and info is not None:
                return info, True
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cannot decode image")
        rec = self.recon.process(path)
        ldm = rec.landmark_arrays()
        ldm106_original = to_original_image(ldm["ldm106_image_224"], rec.trans_params)
        ldm134_original = to_original_image(ldm["ldm134_image_224"], rec.trans_params)
        mask = build_mask_bundle(rec.semantic_channels_224, rec.trans_params, bgr.shape)

        with atomic_photo_directory(self.cfg.output_dir, photo_id, overwrite=final.exists()) as out:
            files, crop_meta = save_image_assets(path, bgr, ldm106_original, out, self.cfg.save_original)
            try:
                face_mask_files = save_face_mask(bgr, mask.hard_original, crop_meta["bbox_original"], out)
                if face_mask_files:
                    files.update(face_mask_files)
                else:
                    files["face_mask"] = None
                    files["face_mask_data"] = None
            except Exception:
                files["face_mask"] = None
                files["face_mask_data"] = None
            files["semantic_channels"] = save_semantic_channels(mask, out)
            uv_files, uv_arrays, uv_meta = save_uv_and_mesh(bgr, rec, out, self.cfg.uv_size)
            files.update(uv_files)
            quality = technical_quality(bgr, crop_meta["bbox_original"], mask.hard_original, rec.combined_visible)
            quality["uv_observed_coverage"] = uv_meta["observed_coverage"]

            pose_payload = {"pitch": float(rec.angles_deg[0]), "yaw": float(rec.angles_deg[1]), "roll": float(rec.angles_deg[2]),
                            "pose_bin": rec.pose_bin, "canonical_yaw": rec.canonical_yaw}
            try:
                quality_files, quality_summary = build_quality_files(
                    bgr=bgr,
                    hard_mask_original=mask.hard_original,
                    crop_meta=crop_meta,
                    pose=pose_payload,
                    photo_id=photo_id,
                    out=out,
                )
                files.update(quality_files)
            except Exception as exc:
                quality_summary = {"status": "failed", "error": str(exc)}
                files["quality"] = None
                files["quality_zones"] = None

            write_csv(out / "ldm106_raw.csv", _landmark_rows(ldm["ldm106_object"], ldm["ldm106_visible"], rec.ldm106_indices))
            write_csv(out / "ldm106_aligned.csv", _landmark_rows(ldm["ldm106_bin_canonical"], ldm["ldm106_visible"], rec.ldm106_indices))
            write_csv(out / "ldm134_raw.csv", _landmark_rows(ldm["ldm134_object"], ldm["ldm134_visible"], rec.ldm134_indices))
            write_csv(out / "ldm134_aligned.csv", _landmark_rows(ldm["ldm134_bin_canonical"], ldm["ldm134_visible"], rec.ldm134_indices))
            files.update({
                "ldm106_raw": "ldm106_raw.csv", "ldm106_aligned": "ldm106_aligned.csv",
                "ldm134_raw": "ldm134_raw.csv", "ldm134_aligned": "ldm134_aligned.csv",
            })

            arrays: dict[str, np.ndarray] = {
                "vertices_object": rec.vertices_object, "vertices_identity_only": rec.vertices_identity_only,
                "vertices_object_normalized": rec.vertices_object_normalized, "vertices_bin_canonical": rec.vertices_bin_canonical,
                "vertices_camera": rec.vertices_camera, "vertices_image_224": rec.vertices_image_224,
                "normals_object": rec.normals_object, "normals_posed": rec.normals_posed,
                "triangles": rec.triangles, "uv_coords": rec.uv_coords,
                "ldm106_vertex_indices": rec.ldm106_indices, "ldm134_vertex_indices": rec.ldm134_indices,
                "ldm106_identity_only": rec.vertices_identity_only[rec.ldm106_indices].astype(np.float32),
                "ldm134_identity_only": rec.vertices_identity_only[rec.ldm134_indices].astype(np.float32),
                "full_mesh_front_facing_packbits": pack_mask(rec.front_facing),
                "full_mesh_renderer_visible_packbits": pack_mask(rec.renderer_visible),
                "full_mesh_visible_packbits": pack_mask(rec.combined_visible),
                "alpha_full": rec.alpha_full, "alpha_id": rec.alpha_id, "alpha_exp": rec.alpha_exp,
                "alpha_alb": rec.alpha_alb, "alpha_sh": rec.alpha_sh,
                "angle_rad": rec.angles_rad, "angle_deg_pitch_yaw_roll": rec.angles_deg,
                "rotation_matrix": rec.rotation, "translation": rec.translation, "trans_params": rec.trans_params,
                "normalization_center": rec.normalization_center,
                "normalization_scale": np.asarray([rec.normalization_scale], np.float32),
                "canonical_rotation_row_matrix": rec.canonical_rotation,
                "canonical_yaw": np.asarray([rec.canonical_yaw], np.float32),
                **ldm, **uv_arrays,
            }
            np.savez_compressed(out / "reconstruction.npz", **arrays)
            files["reconstruction"] = "reconstruction.npz"

            info = {
                "schema_version": PHOTO_SCHEMA_VERSION, "photo_id": photo_id,
                "source_filename": path.name, "source_relative_path": self._relative(path), "source_sha256": source_hash,
                "date": parsed.date_iso, "date_year": parsed.year, "date_month": parsed.month,
                "date_day": parsed.day, "same_date_sequence": parsed.sequence,
                "extraction_timestamp": _utc(), "code_hash": self.code_hash,
                "config_hash": self.config_hash, "model_hash": self.model_hash,
                "image": {"width": int(bgr.shape[1]), "height": int(bgr.shape[0]), "extension": path.suffix.lower()},
                "pose": pose_payload,
                "camera": {"projection": "perspective", "focal": 1015.0, "principal_point": [112.0, 112.0],
                           "camera_distance": 10.0, "render_size": [224, 224]},
                "normalization": {"method": "full_mesh_rms_v1", "center": rec.normalization_center,
                                  "scale": rec.normalization_scale},
                "landmark_contract": {"raw": "object identity+expression", "aligned": "full-mesh RMS normalized then pose-bin canonical yaw"},
                "mesh_contract": {
                    "morphing": "mesh.obj keeps the complete fixed topology and uses a hole-free visual UV texture",
                    "analysis_npz": "reconstruction.npz analysis_mesh_* arrays contain the visibility-cut compact mesh; full arrays remain for correspondence compatibility",
                    "synthetic_texture_policy": "completed UV pixels are display/morphing only and never count as observed evidence",
                },
                "mask": {"status": mask.status, "error": mask.error, **mask.metadata},
                "uv": {"status": "valid", **uv_meta}, "quality_inputs": quality,
                "quality_summary": quality_summary,
                "reprojection": rec.reprojection, "crop": crop_meta, "files": files,
            }
            atomic_json(out / "info.json", info)
            result = validate_photo(out, write_result=True)
            if result["status"] != "complete":
                raise RuntimeError("validation failed: " + "; ".join(result["errors"]))
        return info, False

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.cfg.input_dir.resolve()))
        except ValueError:
            return path.name

    @staticmethod
    def _index_row(info: dict[str, Any]) -> dict[str, Any]:
        pose = info["pose"]; quality = info["quality_inputs"]
        return {
            "photo_id": info["photo_id"], "date": info["date"], "same_date_sequence": info["same_date_sequence"],
            "pose_bin": pose["pose_bin"], "pitch": pose["pitch"], "yaw": pose["yaw"], "roll": pose["roll"],
            "source_filename": info["source_filename"], "source_relative_path": info["source_relative_path"],
            "geometry_status": "valid", "segmentation_status": info["mask"]["status"], "uv_status": info["uv"]["status"],
            "combined_visible_fraction": quality["combined_visible_fraction"],
            "skin_mask_coverage": quality["skin_mask_coverage"], "uv_observed_coverage": quality["uv_observed_coverage"],
        }
