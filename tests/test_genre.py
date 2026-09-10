"""Genre string splitting tests (no Essentia)."""

import unittest

from musikalyze.analysis_ops import main_sub_from_label


class TestGenre(unittest.TestCase):
    def test_split_genre_main(self) -> None:
        main, sub = main_sub_from_label("Rock---Post-Punk", ("---",))
        self.assertEqual(main, "Rock")
        self.assertEqual(sub, "Post-Punk")

    def test_split_genre_sub_only(self) -> None:
        main, sub = main_sub_from_label("Rock---Post-Punk", ("---",))
        self.assertEqual(main, "Rock")
        self.assertEqual(sub, "Post-Punk")


if __name__ == "__main__":
    unittest.main()
