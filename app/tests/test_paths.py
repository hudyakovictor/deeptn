from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from app.stage1.paths import ProjectLayoutError, install_runtime_paths, resolve_project_layout


class ProjectLayoutTests(unittest.TestCase):
    def _layout(self, root: Path) -> None:
        for path in (root / "app" / "stage1", root / "3ddfa_v3" / "uv_module", root / "3ddfa_v3" / "assets", root / "3ddfa_v3" / "model"):
            path.mkdir(parents=True, exist_ok=True)

    def test_sibling_app_and_3ddfa_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._layout(root)
            layout = resolve_project_layout(root)
            self.assertEqual(layout.app_dir, root / "app")
            self.assertEqual(layout.uv_module_dir, root / "3ddfa_v3" / "uv_module")

    def test_app_directory_argument_is_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._layout(root)
            self.assertEqual(resolve_project_layout(root / "app").project_root, root)

    def test_missing_uv_module_has_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "app" / "stage1").mkdir(parents=True)
            (root / "3ddfa_v3" / "assets").mkdir(parents=True)
            (root / "3ddfa_v3" / "model").mkdir(parents=True)
            with self.assertRaisesRegex(ProjectLayoutError, "uv_module"):
                resolve_project_layout(root)

    def test_runtime_paths_include_root_and_3ddfa(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._layout(root)
            layout = resolve_project_layout(root)
            old = list(sys.path)
            try:
                install_runtime_paths(layout)
                self.assertIn(str(root), sys.path)
                self.assertIn(str(root / "3ddfa_v3"), sys.path)
            finally:
                sys.path[:] = old


if __name__ == "__main__":
    unittest.main()
