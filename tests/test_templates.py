"""Template and path helper tests."""

import unittest

from musikalyze.templates import (
    build_format_mapping,
    extract_placeholder_keys,
    resolve_template,
    sanitize_path_segment,
    sanitize_relative_path,
)


class TestTemplates(unittest.TestCase):
    def test_extract_placeholder_keys(self) -> None:
        self.assertIn("meta_genre", extract_placeholder_keys("{meta_genre}-{tag_artist:02d}"))

    def test_resolve_template_safe_missing(self) -> None:
        m = build_format_mapping({"artist": "A"}, {}, ext="opus")
        self.assertEqual(resolve_template("{tag_artist}-{missing}", m), "A-")

    def test_sanitize_path(self) -> None:
        self.assertNotIn("<", sanitize_relative_path("foo/bar<bad>"))
        self.assertEqual(sanitize_path_segment("a:b"), "a_b")

    def test_build_format_mapping_track(self) -> None:
        m = build_format_mapping({"tracknumber": "3"}, {}, ext=None)
        self.assertEqual(m["tag_track_number"], 3)

    def test_sanitize_portable_names(self) -> None:
        self.assertEqual(sanitize_path_segment("CON"), "_CON")
        self.assertEqual(sanitize_path_segment(".."), "_")
        self.assertEqual(sanitize_path_segment("folder. "), "folder")
        self.assertNotIn("\n", sanitize_path_segment("bad\nname"))

    def test_sanitize_prevents_relative_escape(self) -> None:
        self.assertEqual(sanitize_relative_path("../Artist/../../song?.opus"), "_/Artist/_/_/song_.opus")


if __name__ == "__main__":
    unittest.main()
