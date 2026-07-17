from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


class ProjectLayoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProjectLayout:
    project_root: Path
    app_dir: Path
    three_ddfa_root: Path
    uv_module_dir: Path
    assets_dir: Path
    model_dir: Path


def resolve_project_layout(project_root: Path, *, require_runtime: bool = True) -> ProjectLayout:
    """Resolve the fixed sibling layout used on the user's Mac.

    Expected layout:
        <project_root>/app
        <project_root>/3ddfa_v3
        <project_root>/3ddfa_v3/uv_module

    Passing the `app` directory itself is accepted and normalized to its parent
    to prevent the former app/3ddfa_v3 path mistake.
    """
    root = Path(project_root).expanduser().resolve()
    if root.name == "app" and (root / "stage1").is_dir():
        root = root.parent
    app_dir = root / "app"
    three_ddfa_root = root / "3ddfa_v3"
    layout = ProjectLayout(
        project_root=root,
        app_dir=app_dir,
        three_ddfa_root=three_ddfa_root,
        uv_module_dir=three_ddfa_root / "uv_module",
        assets_dir=three_ddfa_root / "assets",
        model_dir=three_ddfa_root / "model",
    )
    missing = []
    if not layout.app_dir.is_dir():
        missing.append(layout.app_dir)
    if require_runtime:
        for path in (layout.three_ddfa_root, layout.uv_module_dir, layout.assets_dir, layout.model_dir):
            if not path.is_dir():
                missing.append(path)
    if missing:
        expected = root / "app"
        ddfa = root / "3ddfa_v3"
        raise ProjectLayoutError(
            "project folders are misplaced; expected sibling directories "
            f"{expected} and {ddfa}; missing: " + ", ".join(str(path) for path in missing)
        )
    return layout


def install_runtime_paths(layout: ProjectLayout) -> None:
    """Expose both the app package and 3DDFA_V3 top-level modules."""
    for path in (layout.project_root, layout.three_ddfa_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
