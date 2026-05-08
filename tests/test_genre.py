"""Genre string splitting tests (no Essentia)."""

import unittest
from types import SimpleNamespace

from musikalyze.analysis_ops import split_genre_label


class TestGenre(unittest.TestCase):
    def test_split_genre_main(self) -> None:
        cfg = SimpleNamespace(genre_main=True, genre_separators=("---",))
        self.assertEqual(
            split_genre_label(
                "Rock---Post-Punk",
                genre_main=cfg.genre_main,
                separators=cfg.genre_separators,
            ),
            ["Rock", "Post-Punk"],
        )

    def test_split_genre_sub_only(self) -> None:
        cfg = SimpleNamespace(genre_main=False, genre_separators=("---",))
        self.assertEqual(
            split_genre_label(
                "Rock---Post-Punk",
                genre_main=cfg.genre_main,
                separators=cfg.genre_separators,
            ),
            ["Post-Punk"],
        )


if __name__ == "__main__":
    unittest.main()
