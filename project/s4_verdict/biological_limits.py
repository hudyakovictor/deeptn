from __future__ import annotations

from datetime import timedelta
from typing import List, Dict, Tuple
import numpy as np


# Маппинг: старые (несуществующие) имена → реальные имена из geometry catalog.
#Geometry catalog использует zone_* и другие конвенции.
METRIC_ALIASES: Dict[str, str] = {
    # Nose
    "bone_nasion_depth": "zone_nose_bridge_tip_depth_std_ratio",
    "landmark_nose_chin_distance": "nose_bridge_tip_bbox_area_ratio",
    "nose_width": "zone_nose_wing_L_span_lateral_ratio",
    "nose_tip_projection": "zone_nose_bridge_tip_span_depth_ratio",
    # Soft tissue / cheeks
    "soft_cheek_volume": "cheekbone_L_bbox_volume_ratio",
    "soft_nasolabial_depth": "zone_nose_wing_L_depth_std_ratio",
    "jawline_definition": "zone_jaw_L_depth_std_ratio",
    "marionette_depth": "zone_jaw_L_span_depth_ratio",
    # Eyes
    "landmark_eye_width_L": "orbit_L_bbox_area_ratio",
    "landmark_eye_width_R": "orbit_R_bbox_area_ratio",
    "eyelid_height_L": "zone_orbit_L_span_vertical_ratio",
    "eyelid_height_R": "zone_orbit_R_span_vertical_ratio",
    # Zygomatic / skull width
    "bone_zygomatic_width": "zone_cheekbone_L_span_lateral_ratio",
    "face_scale": "forehead_bbox_area_ratio",
    "zygomatic_arch_height_L": "zone_cheekbone_L_span_vertical_ratio",
    "zygomatic_arch_height_R": "zone_cheekbone_R_span_vertical_ratio",
    # Chin
    "bone_chin_projection": "zone_chin_span_vertical_ratio",
    "chin_soft_volume": "chin_bbox_volume_ratio",
    "menton_position": "zone_chin_normal_mean_z",
    # Bone growth
    "bone_gonial_angle": "zone_jaw_angle_L_normal_mean_x",
    "bone_mandible_width": "zone_jaw_L_span_lateral_ratio",
    "bone_interorbital_distance": "zone_orbit_L_span_lateral_ratio",
    # Asymmetry
    "bone_asymmetry_x": "zone_rel_brow_ridge_L_to_jaw_angle_L_distance_ratio",
    "bone_asymmetry_y": "zone_rel_brow_ridge_R_to_jaw_angle_R_distance_ratio",
    "bone_asymmetry_z": "zone_rel_cheekbone_L_to_temporal_L_distance_ratio",
}


def _resolve_metric(metric_name: str) -> str:
    """Resolve a metric name through the alias map."""
    return METRIC_ALIASES.get(metric_name, metric_name)


class BiologicalConstraintChecker:
    """Проверяет биологическую возможность изменений."""
    
    CONSTRAINTS = {
        # Хирургические лимиты (минимальное время на заживление + видимость)
        "rhinoplasty": {
            "min_gap": timedelta(days=180),
            "affected_metrics": [
                "bone_nasion_depth",
                "landmark_nose_chin_distance",
                "nose_width",
                "nose_tip_projection",
            ],
            "max_change_ratio": 0.30,  # 30% relative change (metrics are ratios, not mm)
            "description": "Ринопластика требует минимум 6 месяцев на заживление. Форма носа не может измениться кардинально за меньший срок.",
        },
        "facelift": {
            "min_gap": timedelta(days=90),
            "affected_metrics": [
                "soft_cheek_volume",
                "soft_nasolabial_depth",
                "jawline_definition",
                "marionette_depth",
            ],
            "max_change_ratio": 0.40,
            "description": "Подтяжка лица (SMAS) требует 3+ месяцев на спад отека. Резкие изменения мягких тканей раньше — невозможны.",
        },
        "blepharoplasty": {
            "min_gap": timedelta(days=60),
            "affected_metrics": [
                "landmark_eye_width_L",
                "landmark_eye_width_R",
                "eyelid_height_L",
                "eyelid_height_R",
            ],
            "max_change_ratio": 0.25,
            "description": "Блефаропластика: минимум 2 месяца на заживление век.",
        },
        "zygomatic_implant": {
            "min_gap": timedelta(days=365),
            "affected_metrics": [
                "bone_zygomatic_width",
                "face_scale",
                "zygomatic_arch_height_L",
                "zygomatic_arch_height_R",
            ],
            "max_change_ratio": 0.20,
            "description": "Импланты скул требуют до года на оссеоинтеграцию и спад отека.",
        },
        "chin_implant": {
            "min_gap": timedelta(days=180),
            "affected_metrics": [
                "bone_chin_projection",
                "chin_soft_volume",
                "menton_position",
            ],
            "max_change_ratio": 0.30,
            "description": "Имплант подбородка: 6+ месяцев на финальную форму.",
        },
        
        # Естественные лимиты
        "bone_growth_adult": {
            "min_gap": timedelta(days=365),
            "affected_metrics": [
                "bone_zygomatic_width",
                "bone_nasion_depth",
                "bone_gonial_angle",
                "bone_mandible_width",
                "bone_interorbital_distance",
            ],
            "max_change_ratio": 0.05,  # 5% — кости взрослого не меняются
            "description": "Костные структуры взрослого человека (старше 25 лет) практически не изменяются. Изменение >5% за год невозможно без травмы/операции.",
        },
        "asymmetry_inversion": {
            "min_gap": timedelta(days=365 * 5),
            "affected_metrics": [
                "bone_asymmetry_x",
                "bone_asymmetry_y",
                "bone_asymmetry_z",
            ],
            "max_change_ratio": 999,  # Любое изменение знака — подозрительно
            "description": "Инверсия асимметрии (левая скуловая кость вдруг 'длиннее' правой) невозможна естественным путём.",
        },
    }
    
    def check(self, photo_a: Dict, photo_b: Dict, 
              metrics_a: Dict, metrics_b: Dict) -> List[Dict]:
        """
        photo: {date, age_years, ...}
        metrics: {metric_name: value}
        
        Metric names are resolved through METRIC_ALIASES so both
        old names (bone_nasion_depth) and new names (zone_nose_bridge_tip_depth_std_ratio)
        are supported.
        """
        flags = []
        gap = abs((photo_a["date"] - photo_b["date"]).days)
        
        for constraint_name, constraint in self.CONSTRAINTS.items():
            for metric_name in constraint["affected_metrics"]:
                # Resolve alias to actual geometry catalog name
                resolved_name = _resolve_metric(metric_name)
                
                # Try both original and resolved name
                val_a = metrics_a.get(resolved_name, metrics_a.get(metric_name))
                val_b = metrics_b.get(resolved_name, metrics_b.get(metric_name))
                
                if val_a is None or val_b is None:
                    continue
                
                try:
                    val_a = float(val_a)
                    val_b = float(val_b)
                except (TypeError, ValueError):
                    continue
                
                if not (np.isfinite(val_a) and np.isfinite(val_b)):
                    continue
                
                # Relative change (since metrics are ratios, not mm)
                scale = max(abs(val_a), abs(val_b), 1e-6)
                delta = abs(val_a - val_b)
                delta_ratio = delta / scale
                
                max_change = constraint.get("max_change_ratio", 0.30)
                
                # Проверка 1: Слишком быстрое изменение
                if gap < constraint["min_gap"].days and delta_ratio > max_change:
                    flags.append({
                        "type": "BIOLOGICALLY_IMPOSSIBLE",
                        "constraint": constraint_name,
                        "metric": resolved_name,
                        "original_metric": metric_name,
                        "delta_ratio": float(delta_ratio),
                        "delta_abs": float(delta),
                        "gap_days": gap,
                        "min_required_days": constraint["min_gap"].days,
                        "description": constraint["description"],
                        "severity": "CRITICAL" if delta_ratio > max_change * 2 else "HIGH",
                    })
                
                # Проверка 2: Инверсия асимметрии
                if constraint_name == "asymmetry_inversion":
                    if (val_a > 0.1 and val_b < -0.1) or (val_a < -0.1 and val_b > 0.1):
                        flags.append({
                            "type": "ASYMMETRY_INVERSION",
                            "constraint": constraint_name,
                            "metric": resolved_name,
                            "original_metric": metric_name,
                            "val_a": float(val_a),
                            "val_b": float(val_b),
                            "description": constraint["description"],
                            "severity": "CRITICAL",
                        })
        
        return flags
