"""Метрики S1 (бывшие S2) — извлечение геометрии и текстуры."""

from .geometry_extractor import GeometryExtractor
from .texture_extractor import TextureExtractor
from .zone_analyzer import ZoneAnalyzer
from .geometry import (
    GEOMETRY_CORE_METRICS,
    GeometryIdentityResolver,
    load_geometry_metric_catalog,
)
from .texture.catalog import TEXTURE_CORE_METRICS, load_texture_metric_catalog
from .texture.classifier_v5 import TextureSkinClassifierV5 as TextureSkinClassifier

__all__ = [
    "GeometryExtractor",
    "TextureExtractor",
    "ZoneAnalyzer",
    "GEOMETRY_CORE_METRICS",
    "GeometryIdentityResolver",
    "load_geometry_metric_catalog",
    "TEXTURE_CORE_METRICS",
    "load_texture_metric_catalog",
    "TextureSkinClassifier",
]