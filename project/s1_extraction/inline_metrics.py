"""
Inline Metrics Extractor for S1
===============================
Extracts geometry + texture metrics immediately after 3DDFA reconstruction.
Replaces separate S2 stage.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..shared.logging import setup_logger
from ..shared.schemas import PipelineDataset, Stage1Record, Stage2Record
from ..shared.utils import (
    load_json,
    load_pickle,
    load_rgba_png,
    save_json,
)
from .metrics.modules import (
    GeometryExtractor,
    TextureExtractor,
    GeometryIdentityResolver,
    TextureSkinClassifier,
    load_geometry_metric_catalog,
    load_texture_metric_catalog,
)
from .metrics.physical_features import PhysicalTextureExtractor
from .metrics.texture_anomaly import CohortTextureAnomalyDetectorV2
from ..shared.validation import (
    validate_texture_metrics,
    validate_geometry_metrics,
    clean_texture_metrics,
    clean_geometry_metrics,
)

log = setup_logger("s1_metrics")


class InlineMetricsExtractor:
    """
    Извлекает геометрические и текстурные метрики для одного фото сразу после реконструкции.
    
    Используется внутри ExtractionEngine._process_one() вместо отдельного этапа S2.
    """

    def __init__(
        self,
        output_dir: str | Path,
        dataset: PipelineDataset,
        config: dict | None = None,
        reconstruction_adapter: Any = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dataset = dataset
        self.config = config or {}
        self.reconstruction_adapter = reconstruction_adapter

        # Paths for reference tables
        import os
        data_root = Path(os.environ.get("DPTN_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))

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

        self.geometry_resolver = GeometryIdentityResolver(geometry_table)
        self.texture_classifier = TextureSkinClassifier(texture_leaderboard)
        self.geometry_catalog = load_geometry_metric_catalog()
        self.texture_catalog = load_texture_metric_catalog(texture_leaderboard)
        self.geometry_extractor = GeometryExtractor()
        self.texture_extractor = TextureExtractor()
        self.cohort_detector = CohortTextureAnomalyDetectorV2()
        self.physical_extractor = PhysicalTextureExtractor()

        # Cohort groups for anomaly detection (accumulated across photos)
        self._cohort_groups: dict[str, list[dict]] = {}

    def extract(
        self,
        record: Stage1Record,
        reconstruction: Dict[str, Any],
        rgba: np.ndarray,
    ) -> Stage2Record:
        """
        Извлекает метрики для одного фото.
        
        Args:
            record: Stage1Record с данными S1
            reconstruction: Dict из reconstruction.pkl
            rgba: Face mask RGBA (424x500)
            
        Returns:
            Stage2Record с geometry, texture, классификацией
        """
        photo_id = record.photo_id
        photo_dir = Path(record.face_mask_path).parent

        # ── Geometry metrics ──
        try:
            geometry = self.geometry_extractor.extract(reconstruction)
        except Exception as exc:
            log.warning(f"[{photo_id}] Geometry extraction failed: {exc}")
            geometry = {}

        # ── Legacy full geometry metrics (1000+ metrics) ──
        try:
            from .metrics.modules.geometry.legacy_metrics.context import build_metric_context
            from .metrics.modules.geometry.legacy_metrics.runner import compute_single_photo_metrics

            ctx = build_metric_context(
                photo_id=photo_id,
                image_path=Path(record.source_path),
                reconstruction=reconstruction,
                adapter=self.reconstruction_adapter,
                pose_bucket=record.pose.bucket.value if hasattr(record.pose, 'bucket') else "unknown",
                quality=record.quality.model_dump() if record.quality else None,
                geometry_metrics=geometry,
            )
            legacy_values, legacy_errors = compute_single_photo_metrics(ctx)
            for mv in legacy_values:
                geometry[mv.spec.name] = mv.value
            if legacy_errors:
                log.warning(f"[{photo_id}] {len(legacy_errors)} legacy metric errors")
        except Exception as exc:
            log.debug(f"[{photo_id}] Legacy metrics not available: {exc}")

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
                            log.attention(f"[{photo_id}] physical_{k} = {v} (NaN/Inf)")
                            physical_features[k] = 0.0
        except Exception as exc:
            log.debug(f"[{photo_id}] Physical features failed: {exc}")

        texture.update(physical_features)

        # ── Texture classification ──
        texture_hint = self.texture_classifier.classify(texture, record.quality)

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

        # ── Soft validation ──
        tex_issues = validate_texture_metrics(texture, photo_id)
        geo_issues = validate_geometry_metrics(geometry, photo_id)
        for issue in tex_issues + geo_issues:
            if issue.level == "warning":
                log.warning(str(issue))
            elif issue.level == "attention":
                log.attention(str(issue))

        # ── Cohort grouping for anomaly detection ──
        year = record.date.year if record.date else 2000
        quality = float(record.quality.overall_quality) if record.quality else 0.5
        cohort_key = self.cohort_detector.get_cohort_key(year, quality)
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
        clean_tex = clean_texture_metrics(stage2.texture)
        clean_geo = clean_geometry_metrics(stage2.geometry)

        save_json(clean_geo, photo_dir / "geometry_metrics.json")
        save_json(clean_tex, photo_dir / "texture_metrics.json")

        return stage2

    def finalize_cohorts(self) -> None:
        """Fit cohort anomaly models after all photos processed."""
        log.info("Fitting cohort anomaly models...")
        for cohort_key, cohort_textures in self._cohort_groups.items():
            if len(cohort_textures) >= 3:
                try:
                    self.cohort_detector.fit_cohort(cohort_textures, cohort_key)
                    log.debug(f"Cohort '{cohort_key}': fitted on {len(cohort_textures)} samples")
                except Exception as exc:
                    log.warning(f"Cohort '{cohort_key}' fit failed: {exc}")

    def score_anomalies(self, records: list[Stage2Record]) -> list[Stage2Record]:
        """Score texture anomalies for all records using fitted cohorts."""
        for record in records:
            year = record.date.year if record.date else 2000
            quality = float(record.quality.overall_quality) if record.quality else 0.5
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
        return records