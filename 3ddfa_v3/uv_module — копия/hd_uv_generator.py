"""
Production HD UV — загрузка из dutin/uv_module без shadowing backend-stub.

Упрощённый baker в backend ломал uv_texture.png / uv_confidence.png.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_DUTIN_ROOT = Path(__file__).resolve().parents[3]
_PKG_DIR = _DUTIN_ROOT / "uv_module"
_PKG_NAME = "dutin_uv_module"


def _load_production_uv() -> object:
    if _PKG_NAME not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            _PKG_NAME,
            _PKG_DIR / "__init__.py",
            submodule_search_locations=[str(_PKG_DIR)],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[_PKG_NAME] = pkg
        assert pkg_spec.loader is not None
        pkg_spec.loader.exec_module(pkg)

    mod_name = f"{_PKG_NAME}.hd_uv_generator"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    mod_spec = importlib.util.spec_from_file_location(
        mod_name,
        _PKG_DIR / "hd_uv_generator.py",
        submodule_search_locations=[str(_PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(mod_spec)
    mod.__package__ = _PKG_NAME
    sys.modules[mod_name] = mod
    assert mod_spec.loader is not None
    mod_spec.loader.exec_module(mod)
    return mod


_production = _load_production_uv()
HDUVConfig = _production.HDUVConfig
HDUVTextureGenerator = _production.HDUVTextureGenerator
