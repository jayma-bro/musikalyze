"""Tests gabarits et chemins."""

import unittest

from musikalize.templates import (
    bracket_to_brace,
    build_format_mapping,
    resolve_template,
    sanitize_path_segment,
    sanitize_relative_path,
)


class TestTemplates(unittest.TestCase):
    def test_bracket_to_brace(self) -> None:
        self.assertEqual(bracket_to_brace("[meta_genre]"), "{meta_genre}")

    def test_resolve_template_safe_missing(self) -> None:
        m = build_format_mapping({"artist": "A"}, {}, ext="opus")
        self.assertEqual(resolve_template("{tag_artist}-{missing}", m), "A-")

    def test_sanitize_path(self) -> None:
        self.assertNotIn("<", sanitize_relative_path("foo/bar<bad>"))
        self.assertEqual(sanitize_path_segment("a:b"), "a_b")

    def test_build_format_mapping_track(self) -> None:
        m = build_format_mapping({"tracknumber": "3"}, {}, ext=None)
        self.assertEqual(m["tag_track_number"], 3)


if __name__ == "__main__":
    unittest.main()
