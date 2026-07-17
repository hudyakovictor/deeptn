#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = APP_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DEEPUTIN app6 — deterministic 3DDFA_V3 stage-1 extraction")
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--input", type=Path, required=True, help="Directory of YYYY_MM_DD[_N].ext photos")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--detector", default="retinaface")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "mbnetv3"])
    p.add_argument("--uv-size", type=int, default=768)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--no-original-copy", action="store_true")
    return p


def main() -> int:
    a = build_parser().parse_args()
    root = a.project_root.resolve()
    # The app and 3ddfa_v3 directories are siblings under project_root.
    # Add project_root first so the `app` package can be imported.
    if root.name == "app" and (root / "stage1").is_dir():
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.stage1.paths import install_runtime_paths, resolve_project_layout
    layout = resolve_project_layout(root)
    install_runtime_paths(layout)
    os.chdir(layout.project_root)
    from app.stage1.config import Stage1Config
    from app.stage1.engine import Stage1Engine
    cfg = Stage1Config(
        project_root=layout.project_root, input_dir=a.input.resolve(), output_dir=a.output.resolve(),
        device=a.device, detector=a.detector, backbone=a.backbone, uv_size=a.uv_size,
        limit=a.limit, overwrite=a.overwrite, continue_on_error=not a.fail_fast,
        save_original=not a.no_original_copy,
    )
    Stage1Engine(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
