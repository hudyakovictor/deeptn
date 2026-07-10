"""Texture modules."""

from .texture_extractor import TextureExtractor
from .classifier import TextureSkinClassifierV2
from .catalog import TEXTURE_CORE_METRICS, PHYSICAL_AUX_METRICS, load_texture_metric_catalog

__all__ = [
    "TextureExtractor",
    "TextureSkinClassifierV2",
    "TEXTURE_CORE_METRICS",
    "PHYSICAL_AUX_METRICS",
    "load_texture_metric_catalog",
]

TextureSkinClassifier = TextureSkinClassifierV2