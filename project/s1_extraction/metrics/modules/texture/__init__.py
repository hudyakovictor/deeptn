"""Texture modules."""

from .texture_extractor import TextureExtractor
# FIX #16: Use V5 as default (was V2). V2 kept for backward compatibility.
from .classifier_v5 import TextureSkinClassifierV5
from .classifier import TextureSkinClassifierV2  # Legacy, kept for backward compat
from .catalog import TEXTURE_CORE_METRICS, PHYSICAL_AUX_METRICS, load_texture_metric_catalog

__all__ = [
    "TextureExtractor",
    "TextureSkinClassifierV5",
    "TextureSkinClassifierV2",
    "TEXTURE_CORE_METRICS",
    "PHYSICAL_AUX_METRICS",
    "load_texture_metric_catalog",
]

# Default alias points to V5
TextureSkinClassifier = TextureSkinClassifierV5