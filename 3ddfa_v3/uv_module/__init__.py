"""
uv_module — HD UV textures for 3DDFA_V3 (ITER2 packaged inside library tree).

Exports analysis UV (real photo only) and beauty UV (fill/inpaint for render).
"""
from .hd_uv_generator import HDUVTextureGenerator, HDUVConfig
from .visibility import compute_triangle_visibility, compute_vertex_visibility

try:
    from .uvio import UVIOExporter, ObjData
except Exception:  # optional heavy deps
    UVIOExporter = None  # type: ignore
    ObjData = None  # type: ignore

__all__ = [
    "HDUVTextureGenerator",
    "HDUVConfig",
    "compute_triangle_visibility",
    "compute_vertex_visibility",
    "UVIOExporter",
    "ObjData",
]

__version__ = "iter2_v1"
