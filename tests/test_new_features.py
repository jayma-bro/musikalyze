"""Tests for genre deduplication, template separator, and export config."""

import json
import unittest
from pathlib import Path

from musikalyze.config import ExportConfig, TaggingConfig
from musikalyze.tagging import _deduplicate_genres, _extract_main_genre


class TestGenreDeduplication(unittest.TestCase):
    def test_extract_main_genre_simple(self) -> None:
        self.assertEqual(_extract_main_genre("Rock"), "Rock")
        self.assertEqual(_extract_main_genre("Jazz"), "Jazz")

    def test_extract_main_genre_triple_dash(self) -> None:
        self.assertEqual(_extract_main_genre("Reggae---Dub"), "Reggae")
        self.assertEqual(_extract_main_genre("Electronic---Dubstep"), "Electronic")

    def test_extract_main_genre_single_dash(self) -> None:
        self.assertEqual(_extract_main_genre("Post-Punk"), "Post")

    def test_extract_main_genre_empty(self) -> None:
        self.assertEqual(_extract_main_genre(""), "")
        self.assertEqual(_extract_main_genre(None), None)

    def test_deduplicate_genres_simple(self) -> None:
        result = _deduplicate_genres(["Rock", "Jazz", "Pop"])
        self.assertEqual(result, ["Rock", "Jazz", "Pop"])

    def test_deduplicate_genres_with_duplicates(self) -> None:
        result = _deduplicate_genres(["Reggae---Dub", "Electronic---Dub"])
        self.assertEqual(result, ["Reggae", "Electronic"])

    def test_deduplicate_genres_removes_duplicate_mains(self) -> None:
        result = _deduplicate_genres(["Rock---Classic", "Rock---Metal", "Jazz"])
        self.assertEqual(result, ["Rock", "Jazz"])

    def test_deduplicate_genres_preserves_order(self) -> None:
        result = _deduplicate_genres(["B", "A", "B", "C", "A"])
        self.assertEqual(result, ["B", "A", "C"])

    def test_deduplicate_genres_empty_list(self) -> None:
        result = _deduplicate_genres([])
        self.assertEqual(result, [])

    def test_extract_main_genre_whitespace(self) -> None:
        self.assertEqual(_extract_main_genre("  Rock  ---  Classic  "), "Rock")


class TestTemplateSeparator(unittest.TestCase):
    def test_separator_config(self) -> None:
        config = TaggingConfig(separator=" | ")
        self.assertEqual(config.separator, " | ")

    def test_default_separator(self) -> None:
        config = TaggingConfig()
        self.assertEqual(config.separator, ";")


class TestExportConfigMode(unittest.TestCase):
    def test_transcode_mode(self) -> None:
        config = ExportConfig(output_root=Path("/tmp"), mode="transcode")
        self.assertEqual(config.mode, "transcode")

    def test_retag_mode(self) -> None:
        config = ExportConfig(output_root=Path("/tmp"), mode="retag")
        self.assertEqual(config.mode, "retag")

    def test_default_mode(self) -> None:
        config = ExportConfig(output_root=Path("/tmp"))
        self.assertEqual(config.mode, "transcode")

    def test_serialization_roundtrip(self) -> None:
        config = ExportConfig(
            output_root=Path("/tmp"),
            formats="mp3",
            mode="retag",
            path_template="{artist}/{title}"
        )
        data = {
            "output_root": str(config.output_root),
            "formats": config.formats,
            "mode": config.mode,
            "path_template": config.path_template,
            "format_options": config.format_options,
            "overwrite": config.overwrite,
        }
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        self.assertEqual(loaded["mode"], "retag")


class TestTaggingConfigSeparator(unittest.TestCase):
    def test_custom_separator(self) -> None:
        config = TaggingConfig(separator=" / ")
        self.assertEqual(config.separator, " / ")

    def test_dict_field(self) -> None:
        config = TaggingConfig(extra={"custom_tag": "{meta_mood_val}"})
        self.assertEqual(config.extra["custom_tag"], "{meta_mood_val}")


class TestLabelExtractorRegression(unittest.TestCase):
    def test_label_names_dict_type(self) -> None:
        """Test that LabelExtractor can accept dict for label_names."""
        # This validates the dataclass can be instantiated with dict label_names
        from pathlib import Path

        from musikalyze.config import LabelExtractor
        
        # Test basic dict format
        extractor = LabelExtractor(
            name="test",
            embedder_name="effnet",
            graph_path=Path("./models/test.pb"),
            label_names={"label1": (0, 44), "label2": (45, 65)},
            task="regression"
        )
        self.assertEqual(extractor.label_names, {"label1": (0, 44), "label2": (45, 65)})
        
        # Test with single label
        extractor2 = LabelExtractor(
            name="test2", 
            embedder_name="effnet",
            graph_path=Path("./models/test.pb"),
            label_names={"acoustic_good": (47, 100)},
            task="regression"
        )
        self.assertEqual(extractor2.label_names, {"acoustic_good": (47, 100)})


if __name__ == "__main__":
    unittest.main()
