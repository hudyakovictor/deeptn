from __future__ import annotations

import math
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from ..shared.logging import setup_logger
from ..shared.progress import create_progress
from ..shared.validation import validate_info_json, validate_quality_metrics
from ..shared.schemas import PipelineDataset, PoseEstimate, QualityMetrics, Stage1Record, PoseBucket, Stage2Record
from ..shared.utils import (
    clamp_bbox,
    create_face_mask_rgba,
    detect_face_bbox,
    ensure_dir,
    expand_bbox,
    image_quality_metrics,
    list_images,
    load_json,
    parse_date_from_name,
    save_json,
    save_pickle,
    save_face_mask_png,
    stable_photo_id,
    subject_age_years_at,
    fallback_face_bbox,
    classify_pose_bucket,
)
from .modules.reconstruction import ReconstructionAdapter, resolve_reconstruction

logger = setup_logger("s1_extraction")

FACE_CROP_WIDTH = 424
FACE_CROP_HEIGHT = 500
FACE_MASK_FILENAME = "face_mask.png"
FACE_CROP_FILENAME = "face_crop.jpg"
THUMB_FILENAME = "thumb.jpg"
UV_TEXTURE_FILENAME = "uv_texture.png"
UV_CONFIDENCE_FILENAME = "uv_confidence.png"
MESH_OBJ_FILENAME = "mesh.obj"
MESH_MTL_FILENAME = "mesh.mtl"


def _resize_letterbox(bgr: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0: y0 + nh, x0: x0 + nw] = resized
    return canvas


def _resize_letterbox_gray(gray: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0: y0 + nh, x0: x0 + nw] = resized
    return canvas


class InlineMetricsExtractor:
    """
    Извлекает геометрические и текстурные метрики сразу после реконструкции.
    Используется внутри ExtractionEngine._process_one.
    """

    def __init__(self, output_dir: Path, dataset: PipelineDataset, config: dict | None = None):
        self.output_dir = output_dir
        self.dataset = dataset
        self.config = config or {}

        data_root = Path(sys.modules['deeputin.shared.utils'].__file__).parent.parent.parent / "data"
        # Try env var first
        import os
        env_root = Path(os.environ.get("DPTN_DATA_ROOT", ""))
        if env_root.exists():
            data_root = env_root

        geometry_table = self.config.get(
            "geometry_evidence_table",
            data_root / "imgtest" / "metrics_test" / "METRIC_EVIDENCE_TABLE.csv",
        )
        texture_leaderboard = self.config.get(
            "texture_leaderboard",
            data_root / "imgtest" / "unified_test" / "clean_feature_leaderboard.csv",
        )

        # Fallback to relative paths
        if not geometry_table.exists():
            geometry_table = Path(__file__).resolve().parents[2] / "data" / "imgtest" / "metrics_test" / "METRIC_EVIDENCE_TABLE.csv"
        if not texture_leaderboard.exists():
            texture_leaderboard = Path(__file__).resolve().parents[2] / "data" / "imgtest" / "unified_test" / "clean_feature_leaderboard.csv"

        # Lazy imports to avoid circular deps
        from .metrics.modules.geometry_extractor import GeometryExtractor
        from .metrics.modules.texture.texture_extractor import TextureExtractor
        from .metrics.modules.geometry.resolver import GeometryIdentityResolver
        from .metrics.modules.texture.classifier_v5 import TextureSkinClassifierV5 as TextureSkinClassifier
        from .metrics.modules.geometry.catalog import load_geometry_metric_catalog
        from .metrics.modules.texture.catalog import load_texture_metric_catalog
        from .metrics.texture_anomaly import CohortTextureAnomalyDetectorV2
        from .metrics.physical_features import PhysicalTextureExtractor

        self.geometry_resolver = GeometryIdentityResolver(geometry_table)
        self.texture_classifier = TextureSkinClassifier(quality_compensation=False)
        self.geometry_catalog = load_geometry_metric_catalog()
        self.texture_catalog = load_texture_metric_catalog(texture_leaderboard)
        self.texture_extractor = TextureExtractor()
        self.geometry_extractor = GeometryExtractor()
        self.cohort_detector = CohortTextureAnomalyDetectorV2()
        self.physical_extractor = PhysicalTextureExtractor()

        self._cohort_groups: dict[str, list[dict]] = {}

    def extract(self, record: Stage1Record, reconstruction: dict, rgba: np.ndarray) -> Stage2Record:
        """Extract metrics for a single photo."""
        photo_id = record.photo_id
        photo_dir = Path(record.face_mask_path).parent

        # ── Geometry metrics ──
        try:
            geometry = self.geometry_extractor.extract(reconstruction)
        except Exception as exc:
            logger.warning(f"[{photo_id}] Geometry extraction failed: {exc}")
            geometry = {}

        # ── Texture metrics ──
        class TextureCtx:
            image_rgb = rgba[:, :, :3]
            face_bbox = record.face_bbox
            face_mask_path = photo_dir / "face_mask.png"
            pp_iod = record.pose.iod if hasattr(record.pose, 'iod') else None
            face_min_dim = min(record.face_bbox[2], record.face_bbox[3]) if record.face_bbox else None

        texture_ctx = TextureCtx()
        texture = self.texture_extractor.extract(texture_ctx, exclude_sensitive=False)

        # Pop non-float fields
        texture_assessability = texture.pop("texture_assessability", "eligible")
        q_valid_patches = texture.pop("q_valid_patches", 0)

        # ── Geometry identity hint ──
        geometry_hint = self.geometry_resolver.resolve(geometry)

        # ── Physical features ──
        physical_features = {}
        try:
            landmarks_68 = reconstruction.get("landmarks_68")
            if not landmarks_68 or len(landmarks_68) == 0:
                landmarks_68 = reconstruction.get("landmarks_106")
            if landmarks_68 is not None and len(landmarks_68) > 0 and rgba is not None:
                landmarks = np.array(landmarks_68, dtype=np.float32)
                if landmarks.ndim == 2 and landmarks.shape[1] >= 2:
                    image_rgb = rgba[:, :, :3]
                    seg_mask = rgba[:, :, 3] > 10 if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=bool)
                    overall_q = float(record.quality.overall_quality) if record.quality else 1.0
                    pf = self.physical_extractor.extract(image_rgb, landmarks, seg_mask, overall_q)
                    physical_features = {
                        "seam_score": pf.seam_score,
                        "specular_sharpness": pf.specular_sharpness,
                        "specular_dispersion": pf.specular_dispersion,
                        "sss_index": pf.sss_index,
                        "melanin_hemo_slope": pf.melanin_hemo_slope,
                    }
                    for k, v in physical_features.items():
                        if isinstance(v, float) and not math.isfinite(v):
                            logger.attention(f"[{photo_id}] physical_{k} = {v} (NaN/Inf)")
                            physical_features[k] = 0.0
        except Exception as exc:
            logger.debug(f"[{photo_id}] Physical features failed: {exc}")

        texture.update(physical_features)

        # ── Texture classification ──
        texture_hint = self.texture_classifier.classify(
            texture, record.quality,
            pose={"yaw": float(record.pose.yaw), "pitch": float(record.pose.pitch), "roll": float(record.pose.roll)},
            year=record.date.year if record.date else None,
        )

        posterior = texture_hint.get("posterior", {}) if isinstance(texture_hint, dict) else {}
        try:
            texture["texture_silicone_prob"] = float(posterior.get("silicone", 0.5))
            texture["texture_real_prob"] = float(posterior.get("real", 0.5))
            texture["texture_skin_confidence"] = float(texture_hint.get("texture_skin_confidence", 0.0))
        except Exception:
            texture["texture_silicone_prob"] = 0.5
            texture["texture_real_prob"] = 0.5
            texture["texture_skin_confidence"] = 0.0

        # ── Metric notes ──
        texture_weights_json = texture.pop("texture_feature_weights_json", None)
        metric_notes = {
            "geometry_space": "3ddfa_v3_canonical",
            "texture_source": "face_mask.png (native)",
            "geometry_identity_hint": geometry_hint.get("identity_hint", "UNCERTAIN"),
            "texture_skin_hint": texture_hint.get("texture_skin_hint", "unknown"),
            "geometry_catalog_size": str(len(self.geometry_catalog)),
            "texture_catalog_size": str(len(self.texture_catalog)),
            "texture_classifier_model_loaded": str(texture_hint.get("model_loaded", False)).lower(),
            "texture_classifier_heuristic_fallback": str(texture_hint.get("heuristic_fallback", False)).lower(),
            "texture_silicone_prob": str(texture.get("texture_silicone_prob", 0.5)),
            "texture_real_prob": str(texture.get("texture_real_prob", 0.5)),
            "texture_quality_reason": str(texture_hint.get("quality_reason", "ok")),
        }
        if texture_hint.get("heuristic_top_rules"):
            metric_notes["texture_heuristic_top_rules"] = str(texture_hint.get("heuristic_top_rules"))
        if texture_weights_json:
            metric_notes["texture_feature_weights_json"] = texture_weights_json
        for k, v in physical_features.items():
            metric_notes[f"physical_{k}"] = str(v)

        # ── Cohort grouping for anomaly detection ──
        year = record.date.year if record.date else 2000
        quality = float(record.quality.overall_quality) if record.quality else 0.5
        cohort_key = self.cohort_detector.get_cohort_key(year, quality)
        metric_notes["cohort_key"] = cohort_key
        if cohort_key not in self._cohort_groups:
            self._cohort_groups[cohort_key] = []
        self._cohort_groups[cohort_key].append(texture.copy())

        # ── Build Stage2Record ──
        selected_keys = sorted(
            set(geometry) | set(texture)
            | set(geometry_hint.get("selected_metric_keys", []))
            | set(texture_hint.get("used_metrics", []))
        )
        stage2 = Stage2Record(
            photo_id=record.photo_id,
            dataset=record.dataset,
            bucket=record.pose.bucket,
            quality=record.quality,
            geometry=geometry,
            texture=texture,
            selected_metric_keys=selected_keys,
            metric_notes=metric_notes,
            geometry_identity_hint=str(geometry_hint.get("identity_hint", "UNCERTAIN")),
            geometry_identity_confidence=float(geometry_hint.get("identity_confidence", 0.0)),
            texture_skin_hint=str(texture_hint.get("texture_skin_hint", "unknown")),
            texture_skin_confidence=float(texture_hint.get("texture_skin_confidence", 0.0)),
            texture_assessability=texture_assessability,
            quality_summary={
                "overall_quality": float(record.quality.overall_quality),
                "blur_value": float(record.quality.blur_value),
                "noise_level": float(record.quality.noise_level),
                "jpeg_blockiness": float(record.quality.jpeg_blockiness),
                "sharpness_score": float(record.quality.sharpness_score),
                "quality_sensitive_excluded": False,
            },
        )

        # ── Save CLEAN JSON output ──
        from ..shared.validation import clean_texture_metrics, clean_geometry_metrics
        clean_tex = clean_texture_metrics(stage2.texture)
        clean_geo = clean_geometry_metrics(stage2.geometry)

        save_json(clean_geo, photo_dir / "geometry_metrics.json")
        save_json(clean_tex, photo_dir / "texture_metrics.json")

        return stage2

    def fit_cohorts(self):
        """Fit cohort anomaly models after all photos processed."""
        logger.info("Fitting cohort anomaly models...")
        for cohort_key, cohort_textures in self._cohort_groups.items():
            if len(cohort_textures) >= 3:
                try:
                    self.cohort_detector.fit_cohort(cohort_textures, cohort_key)
                    logger.debug(f"Cohort '{cohort_key}': fitted on {len(cohort_textures)} samples")
                except Exception as exc:
                    logger.warning(f"Cohort '{cohort_key}' fit failed: {exc}")

    def score_anomalies(self, stage2_records: list[Stage2Record]) -> list[Stage2Record]:
        """Score texture anomalies using fitted cohorts."""
        for record in stage2_records:
            cohort_key = record.metric_notes.get("cohort_key")
            quality = float(record.quality.overall_quality) if record.quality else 0.5
            if cohort_key is None:
                year = 2000  # fallback
                cohort_key = self.cohort_detector.get_cohort_key(year, quality)
            try:
                anomaly_result = self.cohort_detector.score(record.texture, cohort_key, quality)
                record.metric_notes["texture_anomaly_score"] = str(anomaly_result.anomaly_score)
                record.metric_notes["texture_anomaly_interpretation"] = anomaly_result.interpretation
                record.metric_notes["texture_anomaly_max_z"] = str(anomaly_result.max_z)
                if anomaly_result.feature_flags:
                    record.metric_notes["texture_anomaly_flags"] = ",".join(anomaly_result.feature_flags.keys())
            except Exception:
                record.metric_notes["texture_anomaly_score"] = "0.0"
                record.metric_notes["texture_anomaly_interpretation"] = "computation_error"
        return stage2_records


class ExtractionEngine:
    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        dataset: PipelineDataset,
        limit: int | None = None,
        config: dict | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = ensure_dir(output_dir)
        self.dataset = dataset
        self.limit = limit
        self.config = config or {}

        # Initialize 3DDFA-V3 adapter
        s1_config = self.config.get("s1", {})
        self.reconstruction_adapter = ReconstructionAdapter(
            device=s1_config.get("device", "auto"),
            detector_device=s1_config.get("detector_device", "auto"),
            backbone=s1_config.get("backbone", "resnet50"),
        )
        self.neutral_expression = s1_config.get("neutral_expression", False)
        self.identity_only = s1_config.get("identity_only", False)

        # Inline metrics extractor (replaces S2)
        self._metrics_extractor = InlineMetricsExtractor(
            output_dir=self.output_dir,
            dataset=self.dataset,
            config=self.config.get("s2", {}),
        )
        self._stage2_records: list[Stage2Record] = []
        self._error_count = 0
        self._warning_count = 0

    def run(self) -> tuple[list[Stage1Record], int, int]:
        photos = list_images(self.input_dir)
        if self.limit is not None:
            photos = photos[: self.limit]
        records: list[Stage1Record] = []
        if not photos:
            logger.warning(f"No photos found for S1 in {self.input_dir}")
            return records, 0, 0

        total = len(photos)
        logger.info(f"Processing {total} photos for {self.dataset.value} dataset (S1+metrics)")

        error_count = 0
        warning_count = 0

        progress = create_progress(total=total, description=f"S1+Metrics {self.dataset.value}")
        if hasattr(progress, '_progress') and progress._progress:
            progress._progress.start()

        for index, photo_path in enumerate(photos, start=1):
            try:
                progress.update(photo_id=photo_path.stem, status="reconstructing")
                record = self._process_one(photo_path)
                records.append(record)

                # Soft validation of info.json
                info_issues = validate_info_json(record.model_dump(), photo_path.stem)
                for issue in info_issues:
                    if issue.level == "warning":
                        logger.warning(str(issue))
                        warning_count += 1
                    elif issue.level == "attention":
                        logger.attention(str(issue))

                # Validate quality metrics
                if record.quality:
                    q_issues = validate_quality_metrics(record.quality.model_dump(), photo_path.stem)
                    for issue in q_issues:
                        if issue.level == "warning":
                            logger.warning(str(issue))
                            warning_count += 1
                        else:
                            logger.attention(str(issue))

                progress.advance()

            except Exception as exc:
                logger.error(f"[{photo_path.stem}] Failed: {exc}")
                error_count += 1
                progress.update(photo_id=photo_path.stem, status=f"FAILED", error=True)
                progress.advance()

        if hasattr(progress, '_progress') and progress._progress:
            progress._progress.stop()

        # Fit cohorts and score anomalies
        self._metrics_extractor.fit_cohorts()
        self._stage2_records = self._metrics_extractor.score_anomalies(self._stage2_records)

        # Save stage2 manifest
        save_json([r.model_dump() for r in self._stage2_records], self.output_dir / "stage2_manifest.json")

        save_json([r.model_dump() for r in records], self.output_dir / "stage1_manifest.json")

        # Summary
        if error_count > 0:
            logger.error(f"S1+Metrics complete: {len(records)}/{total} processed, {error_count} errors, {len(self._stage2_records)} metrics")
        elif warning_count > 0:
            logger.success(f"S1+Metrics complete: {len(records)}/{total} processed, {warning_count} warnings, {len(self._stage2_records)} metrics")
        else:
            logger.success(f"S1+Metrics complete: {len(records)}/{total} processed ✓, {len(self._stage2_records)} metrics")

        self._error_count = error_count
        self._warning_count = warning_count
        return records, error_count, warning_count

    def _process_one(self, photo_path: Path) -> Stage1Record:
        photo_id = stable_photo_id(photo_path)
        photo_dir = ensure_dir(self.output_dir / photo_id)
        image_bgr = cv2.imread(str(photo_path))
        if image_bgr is None:
            raise RuntimeError(f"Не удалось прочитать изображение: {photo_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run full 3DDFA-V3 reconstruction with disk caching
        reconstruction_result = resolve_reconstruction(
            adapter=self.reconstruction_adapter,
            image_path=photo_path,
            entry_dir=photo_dir,
            neutral_expression=self.neutral_expression,
            identity_only=self.identity_only,
        )

        # Save face mask (424x500 letterboxed RGBA crop with skin alpha mask) + face_crop.jpg + thumb.jpg
        mask_path, crop_path, thumb_path = self._save_face_assets(image_bgr, reconstruction_result, photo_dir)

        # Save full reconstruction to pickle
        reconstruction_dict = self._reconstruction_to_dict(reconstruction_result)
        reconstruction_path = save_pickle(reconstruction_dict, photo_dir / "reconstruction.pkl")

        # Save UV texture + confidence map
        uv_paths = self._save_uv_assets(image_bgr, reconstruction_dict, photo_dir)

        # Save 3D mesh (OBJ + MTL)
        mesh_paths = self._save_mesh_assets(reconstruction_dict, photo_dir)

        # Copy original photo to output directory
        orig_copy_path = photo_dir / photo_path.name
        if not orig_copy_path.exists():
            shutil.copy2(str(photo_path), str(orig_copy_path))

        # Extract pose from 3DDFA (not from filename!)
        angles_deg = reconstruction_result.angles_deg
        pitch, yaw, roll = float(angles_deg[0]), float(angles_deg[1]), float(angles_deg[2])
        bucket = reconstruction_result.pose_bucket

        photo_date = parse_date_from_name(photo_path.stem)
        age_years = subject_age_years_at(photo_date)

        quality = QualityMetrics(**image_quality_metrics(image_bgr))

        pose = PoseEstimate(
            photo_id=photo_id,
            date=photo_date,
            age_years=age_years,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            bucket=PoseBucket(bucket),
            pose_source="3ddfa_v3",
            confidence=0.8 if bucket != "unknown" else 0.5,
        )

        expression_flags = self._expression_flags(reconstruction_result)

        record = Stage1Record(
            photo_id=photo_id,
            dataset=self.dataset,
            source_path=str(photo_path),
            date=photo_date,
            age_years=age_years,
            pose=pose,
            quality=quality,
            face_bbox=list(map(int, self._estimate_bbox_from_landmarks(reconstruction_result))),
            face_mask_path=str(mask_path),
            reconstruction_path=str(reconstruction_path),
            image_size=[int(image_rgb.shape[1]), int(image_rgb.shape[0])],
            expression_flags=expression_flags,
            readiness={
                "geometry": "available",
                "texture": "available",
            },
        )

        # ── INLINE METRICS EXTRACTION (replaces S2) ──
        # Load face_mask.png as RGBA for texture extraction
        rgba = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning(f"[{photo_id}] Failed to load face_mask.png for metrics")
            rgba = np.zeros((FACE_CROP_HEIGHT, FACE_CROP_WIDTH, 4), dtype=np.uint8)

        try:
            stage2 = self._metrics_extractor.extract(record, reconstruction_dict, rgba)
            self._stage2_records.append(stage2)
            from ..shared.validation import clean_geometry_metrics, clean_texture_metrics
            photo_dir = Path(record.face_mask_path).parent
            clean_geo = clean_geometry_metrics(stage2.geometry)
            clean_tex = clean_texture_metrics(stage2.texture)
            save_json(clean_geo, photo_dir / "geometry_metrics.json")
            save_json(clean_tex, photo_dir / "texture_metrics.json")
        except Exception as exc:
            logger.error(f"[{photo_id}] Inline metrics extraction failed: {exc}")

        save_json(record.model_dump(), photo_dir / "info.json")
        return record

    def _save_face_assets(self, image_bgr: np.ndarray, recon, photo_dir: Path) -> tuple[Path, Path, Path]:
        seg_visible = recon.payload.get("seg_visible")
        trans_params = recon.trans_params
        h, w = image_bgr.shape[:2]

        mask = None
        if seg_visible is not None and seg_visible.ndim == 3 and seg_visible.shape[2] >= 8:
            skin_224 = np.maximum(seg_visible[:, :, 7], seg_visible[:, :, 4]).copy()
            excluded_224 = np.maximum.reduce([
                seg_visible[:, :, 0], seg_visible[:, :, 1], seg_visible[:, :, 2], seg_visible[:, :, 3],
                seg_visible[:, :, 5], seg_visible[:, :, 6],
            ])
            exclusion_weight = 1.0 / (1.0 + np.exp(-10 * (excluded_224 - 0.5)))
            skin_224 *= (1.0 - exclusion_weight)
            skin_224_uint8 = np.clip(skin_224 * 255, 0, 255).astype(np.uint8)

            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "3ddfa_v3"))
                from util.io import back_resize_crop_img
                from PIL import Image as PILImage
                mask_rgb = np.stack((skin_224_uint8, skin_224_uint8, skin_224_uint8), axis=-1)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
                full_mask_rgb = back_resize_crop_img(mask_rgb, trans_params, blank, resample_method=PILImage.BILINEAR)
                mask = full_mask_rgb[:, :, 0]
            except Exception:
                mask = cv2.resize(skin_224_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask is None:
            landmarks = recon.landmarks_106
            if landmarks is not None and len(landmarks) > 0:
                x_min, y_min = landmarks[:, 0].min(), landmarks[:, 1].min()
                x_max, y_max = landmarks[:, 0].max(), landmarks[:, 1].max()
                bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                bbox = expand_bbox(clamp_bbox(bbox, image_bgr.shape), image_bgr.shape, margin=0.15)
                _, face_rgba = create_face_mask_rgba(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), bbox)
                mask = face_rgba[:, :, 3]
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        coords = cv2.findNonZero((mask > 10).astype(np.uint8))
        if coords is None:
            mask_path = save_face_mask_png(np.dstack([cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), np.zeros((h, w), dtype=np.uint8)]), photo_dir / FACE_MASK_FILENAME)
            return mask_path, Path(""), Path("")

        x, y, bw, bh = cv2.boundingRect(coords)
        target_aspect = FACE_CROP_WIDTH / FACE_CROP_HEIGHT
        crop_w = int(max(bw * 1.25, bh * 1.25 * target_aspect, 1))
        crop_h = int(max(bh * 1.25, crop_w / target_aspect, 1))
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 = max(0, x1 - (x2 - w))
            x2 = w
        if y2 > h:
            y1 = max(0, y1 - (y2 - h))
            y2 = h

        face_crop_bgr = image_bgr[y1:y2, x1:x2].copy()
        face_crop_mask = mask[y1:y2, x1:x2]
        face_crop_bgr = _resize_letterbox(face_crop_bgr, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)
        face_crop_mask = _resize_letterbox_gray(face_crop_mask, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)

        face_crop_rgba = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2BGRA)
        face_crop_rgba[:, :, 3] = face_crop_mask
        mask_path = photo_dir / FACE_MASK_FILENAME
        cv2.imwrite(str(mask_path), face_crop_rgba)

        crop_path = photo_dir / FACE_CROP_FILENAME
        cv2.imwrite(str(crop_path), face_crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        thumb_path = photo_dir / THUMB_FILENAME
        h_t, w_t = face_crop_bgr.shape[:2]
        side = min(w_t, h_t)
        left = (w_t - side) // 2
        top = (h_t - side) // 2
        thumb_crop = face_crop_bgr[top:top+side, left:left+side]
        thumb = cv2.resize(thumb_crop, (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return mask_path, crop_path, thumb_path

    def _save_uv_assets(self, image_bgr: np.ndarray, recon_dict: dict, photo_dir: Path) -> dict[str, Path]:
        paths = {}
        try:
            core_repo = Path(__file__).resolve().parents[2] / "core"
            sys.path.insert(0, str(core_repo))
            from uv_module.hd_uv_generator import HDUVConfig, HDUVTextureGenerator

            v2d_224 = np.asarray(recon_dict.get("vertices_2d", []), dtype=np.float64)
            tp = recon_dict.get("trans_params")
            if v2d_224.size > 0 and tp is not None:
                v2d_orig = self._transform_vertices_2d_to_original(v2d_224, np.asarray(tp))
                recon_dict["vertices_2d"] = v2d_orig.astype(np.float32)

            uv_gen = HDUVTextureGenerator(HDUVConfig(uv_size=768))
            uv_tex_analysis, uv_tex_beauty, uv_mask_visible, uv_confidence, aux = uv_gen.generate(image_bgr, recon_dict)

            uv_texture_path = photo_dir / UV_TEXTURE_FILENAME
            uv_rgb = cv2.cvtColor(uv_tex_beauty, cv2.COLOR_BGR2RGB)
            from PIL import Image as PILImage
            PILImage.fromarray(uv_rgb.astype(np.uint8), mode="RGB").save(str(uv_texture_path))
            paths["uv_texture"] = uv_texture_path

            uv_confidence_path = photo_dir / UV_CONFIDENCE_FILENAME
            if uv_confidence.ndim == 2:
                conf_uint8 = np.clip(uv_confidence * 255.0, 0, 255).astype(np.uint8)
            else:
                conf_uint8 = np.clip(uv_confidence, 0, 255).astype(np.uint8)
            cv2.imwrite(str(uv_confidence_path), conf_uint8)
            paths["uv_confidence"] = uv_confidence_path
        except Exception as exc:
            logger.warning("UV texture generation not available: %s", exc)
        return paths

    def _transform_vertices_2d_to_original(self, vertices_2d_224: np.ndarray, trans_params: np.ndarray) -> np.ndarray:
        v2d = vertices_2d_224.copy()
        target_size = 224
        v2d[:, 1] = target_size - 1 - v2d[:, 1]
        w0, h0, s = float(trans_params[0]), float(trans_params[1]), float(trans_params[2])
        cx, cy = float(trans_params[3]), float(trans_params[4])
        w = int(w0 * s)
        h = int(h0 * s)
        left = int(w / 2 - target_size / 2 + (cx - w0 / 2) * s)
        up = int(h / 2 - target_size / 2 + (h0 / 2 - cy) * s)
        v2d[:, 0] = (v2d[:, 0] + left) / w * w0
        v2d[:, 1] = (v2d[:, 1] + up) / h * h0
        return v2d

    def _save_mesh_assets(self, recon_dict: dict, photo_dir: Path) -> dict[str, Path]:
        paths = {}
        try:
            vertices = np.asarray(recon_dict.get("vertices", []), dtype=np.float32)
            triangles = np.asarray(recon_dict.get("triangles", []), dtype=np.int32)
            normals = np.asarray(recon_dict.get("normals", []), dtype=np.float32)
            if len(vertices) == 0 or len(triangles) == 0:
                return paths

            mtl_path = photo_dir / MESH_MTL_FILENAME
            mtl_content = f"""# DEEPUTIN 3DDFA-V3 mesh
newmtl face_material
Ka 0.2 0.2 0.2
Kd 0.8 0.8 0.8
Ks 0.0 0.0 0.0
d 1.0
illum 2
map_Kd {UV_TEXTURE_FILENAME}
"""
            mtl_path.write_text(mtl_content)
            paths["mesh_mtl"] = mtl_path

            obj_path = photo_dir / MESH_OBJ_FILENAME
            lines = [f"mtllib {MESH_MTL_FILENAME}\n"]
            for v in vertices:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for v in vertices:
                lines.append(f"vt 0.0 0.0\n")
            for n in normals:
                lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            lines.append("usemtl face_material\n")
            for t in triangles:
                lines.append(f"f {t[0]+1}/{t[0]+1}/{t[0]+1} {t[1]+1}/{t[1]+1}/{t[1]+1} {t[2]+1}/{t[2]+1}/{t[2]+1}\n")
            obj_path.write_text("".join(lines))
            paths["mesh_obj"] = obj_path
        except Exception as exc:
            logger.warning("Mesh export error: %s", exc)
        return paths

    def _estimate_bbox_from_landmarks(self, recon) -> tuple[int, int, int, int]:
        landmarks_106 = recon.landmarks_106
        if landmarks_106 is not None and len(landmarks_106) > 0:
            x_min, y_min = landmarks_106[:, 0].min(), landmarks_106[:, 1].min()
            x_max, y_max = landmarks_106[:, 0].max(), landmarks_106[:, 1].max()
            bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            return expand_bbox(clamp_bbox(bbox, (recon.image_size[1], recon.image_size[0]) if hasattr(recon, 'image_size') else (512, 512)), (512, 512), margin=0.15)
        return (0, 0, 512, 512)

    def _reconstruction_to_dict(self, recon) -> dict:
        return {
            "space": "3ddfa_v3_canonical",
            "image_shape": [int(recon.vertices_image.shape[0]), int(recon.vertices_image.shape[1])] if recon.vertices_image is not None else [512, 512],
            "vertices": recon.vertices_world.tolist() if recon.vertices_world is not None else [],
            "vertices_camera": recon.vertices_camera.tolist() if recon.vertices_camera is not None else [],
            "vertices_2d": recon.vertices_image.tolist() if recon.vertices_image is not None else [],
            "vertices_3d": recon.vertices_world.tolist() if recon.vertices_world is not None else [],
            "triangles": recon.triangles.tolist() if recon.triangles is not None else [],
            "normals": recon.normals_world.tolist() if recon.normals_world is not None else [],
            "landmarks_106": recon.landmarks_106.tolist() if recon.landmarks_106 is not None else [],
            "landmarks_68": [],
            "uv_coords": recon.uv_coords.tolist() if recon.uv_coords is not None else None,
            "pose": {
                "yaw": float(recon.angles_deg[1]),
                "pitch": float(recon.angles_deg[0]),
                "roll": float(recon.angles_deg[2]),
            },
            "angles_deg": [float(recon.angles_deg[0]), float(recon.angles_deg[1]), float(recon.angles_deg[2])],
            "bucket": recon.pose_bucket,
            "rotation_matrix": recon.rotation_matrix.tolist() if recon.rotation_matrix is not None else [],
            "translation": recon.translation.tolist() if recon.translation is not None else [],
            "mesh_quality": {
                "vertex_count": int(recon.vertices_world.shape[0]) if recon.vertices_world is not None else 0,
                "face_count": int(recon.triangles.shape[0]) if recon.triangles is not None else 0,
                "visible_vertices": int(np.count_nonzero(recon.visible_idx_renderer)) if recon.visible_idx_renderer is not None else 0,
            },
            "annotation_groups": [g.tolist() for g in recon.annotation_groups] if recon.annotation_groups else [],
            "visible_idx_renderer": recon.visible_idx_renderer.tolist() if recon.visible_idx_renderer is not None else None,
            "seg_visible": recon.payload.get("seg_visible"),
            "trans_params": recon.trans_params.tolist() if recon.trans_params is not None else None,
            "id_params": recon.payload.get("id_params", []).tolist() if isinstance(recon.payload.get("id_params"), np.ndarray) else [],
            "exp_params": recon.payload.get("exp_params", []).tolist() if isinstance(recon.payload.get("exp_params"), np.ndarray) else [],
        }

    def _expression_flags(self, recon) -> dict[str, bool]:
        exp_params = recon.payload.get("exp_params")
        if exp_params is None or len(exp_params) == 0:
            return {"smile_excluded": False, "jaw_excluded": False, "neutralized": False}

        exp_np = np.asarray(exp_params, dtype=float)
        jaw_open = float(abs(exp_np[0])) if len(exp_np) > 0 else 0.0
        smile_intensity = float(max(abs(exp_np[1]), abs(exp_np[2]))) if len(exp_np) > 2 else 0.0

        smile_excluded = smile_intensity > 2.0
        jaw_excluded = jaw_open > 0.8

        return {
            "smile_excluded": bool(smile_excluded),
            "jaw_excluded": bool(jaw_excluded),
            "neutralized": bool(smile_excluded or jaw_excluded),
        }