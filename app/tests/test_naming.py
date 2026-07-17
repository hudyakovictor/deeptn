from __future__ import annotations

import unittest
from pathlib import Path

from app.stage1.naming import make_photo_id, parse_photo_name


class NamingTests(unittest.TestCase):
    def test_plain_date_has_sequence_one(self):
        p = parse_photo_name(Path("1999_01_11.jpg"))
        self.assertEqual(p.date_iso, "1999-01-11")
        self.assertEqual(p.sequence, 1)

    def test_explicit_sequence(self):
        self.assertEqual(parse_photo_name(Path("1999_01_11_12.png")).sequence, 12)

    def test_copy_suffix_in_parentheses(self):
        p = parse_photo_name(Path("1999_01_11 (2).jpg"))
        self.assertEqual(p.sequence, 2)
        self.assertEqual(p.canonical_stem, "1999_01_11_2")

    def test_copy_suffix_with_underscore(self):
        p = parse_photo_name(Path("1999_01_11_2.jpg"))
        self.assertEqual(p.sequence, 2)

    def test_invalid_names_rejected(self):
        for name in ("1999-1-1.jpg", "copy.jpg", "2023_02_29.jpg"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_photo_name(Path(name))

    def test_same_photo_id_for_same_filename(self):
        p = parse_photo_name(Path("1999_01_11.jpg"))
        self.assertEqual(make_photo_id(p, "a" * 64), "1999_01_11")
        self.assertEqual(make_photo_id(p, "b" * 64), "1999_01_11")

    def test_same_photo_id_for_different_copy_suffix(self):
        p1 = parse_photo_name(Path("1999_01_11 (2).jpg"))
        p2 = parse_photo_name(Path("1999_01_11_2.jpg"))
        self.assertEqual(make_photo_id(p1, "a" * 64), make_photo_id(p2, "a" * 64))


if __name__ == "__main__": unittest.main()
