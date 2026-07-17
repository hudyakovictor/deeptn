from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.stage1.storage import atomic_photo_directory, clean_incomplete


class StorageTests(unittest.TestCase):
    def test_failed_write_is_not_published(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(RuntimeError):
                with atomic_photo_directory(root, "photo", False) as tmp:
                    (tmp / "partial").write_text("x")
                    raise RuntimeError("interrupted")
            self.assertFalse((root / "photo").exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_success_is_published(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with atomic_photo_directory(root, "photo", False) as tmp:
                (tmp / "ok").write_text("yes")
            self.assertEqual((root / "photo" / "ok").read_text(), "yes")

    def test_cleanup_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / ".x.incomplete-dead").mkdir()
            self.assertEqual(clean_incomplete(root), 1)
            self.assertFalse(any(root.iterdir()))


if __name__ == "__main__": unittest.main()
