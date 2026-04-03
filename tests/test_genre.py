"""Tests logique genre (sans Essentia)."""

import unittest

from musikalize.analysis_ops import split_genre_label as _split_genre_label
from musikalize.config import AnalysisConfig


class TestGenre(unittest.TestCase):
    def test_split_genre_main(self) -> None:
        cfg = AnalysisConfig(genre_main=True, genre_separators=("---",))
        self.assertEqual(
            _split_genre_label(
                "Rock---Post-Punk",
                genre_main=cfg.genre_main,
                separators=cfg.genre_separators,
            ),
            ["Rock", "Post-Punk"],
        )

    def test_split_genre_sub_only(self) -> None:
        cfg = AnalysisConfig(genre_main=False, genre_separators=("---",))
        self.assertEqual(
            _split_genre_label(
                "Rock---Post-Punk",
                genre_main=cfg.genre_main,
                separators=cfg.genre_separators,
            ),
            ["Post-Punk"],
        )


if __name__ == "__main__":
    unittest.main()
