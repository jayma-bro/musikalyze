"""Contract tests for the explicit tagging API."""

from pathlib import Path

import numpy as np

from musikalyze.analysis_ops import pct
from musikalyze.config import (
    AnalysisResult,
    ExportConfig,
    LabelExtractor,
    PredictionRecord,
    TaggingConfig,
)
from musikalyze.tagging import (
    _get_tag_name,
    _norm_text,
    apply_tagging_config,
    merge_logical_tags_for_export,
    popm_to_stars,
    stars_to_popm,
)


def test_only_explicit_templates_are_resolved_and_lists_are_deduplicated():
    config = TaggingConfig(
        tags={"genre": "{meta_genres}", "key": "{meta_key}"},
        extra={"custom": "{meta_values}"},
    )
    result = apply_tagging_config(
        {"artist": "Original", "album": "Album"},
        AnalysisResult(meta={
            "meta_genres": ["Rock", "Rock", "Pop"],
            "meta_key": "C#",
            "meta_values": ["one", "", "one", "two"],
        }),
        config,
    )
    assert result == {"genre": ["Rock", "Pop"], "key": ["C#"], "custom": ["one", "two"]}


def test_multi_entry_false_keeps_separator_joined_values():
    config = TaggingConfig(
        separator=";",
        multi_entry=False,
        tags={"mood": "{meta_scale};{meta_moods}"},
    )
    result = apply_tagging_config(
        {}, AnalysisResult(meta={"meta_scale": "major", "meta_moods": ["happy", "energetic"]}), config
    )
    assert result == {"mood": "major;happy;energetic"}


def test_multi_entry_splits_literal_separator_between_templates():
    config = TaggingConfig(tags={"mood": "{meta_scale};{meta_moods}"})
    result = apply_tagging_config(
        {}, AnalysisResult(meta={"meta_scale": "major", "meta_moods": ["happy", "energetic"]}), config
    )
    assert result["mood"] == ["major", "happy", "energetic"]


def test_classical_values_are_valid_template_inputs():
    config = TaggingConfig(tags={"bpm": "{meta_bpm}", "gain": "{meta_rgain_gain}"})
    result = apply_tagging_config(
        {}, AnalysisResult(meta={"meta_bpm": 120, "meta_rgain_gain": -5.2}), config
    )
    assert result == {"bpm": ["120"], "gain": ["-5.2"]}


def test_percentage_configuration_is_integer_based():
    extractor = LabelExtractor(
        name="mood",
        embedder_name="effnet",
        graph_path=Path("mood.pb"),
        label_names={"low": (0, 49), "high": (50, 100)},
        thold=70,
        task="multilabel",
    )
    assert extractor.thold == 70


def test_numpy_scores_and_label_lists_are_normalized():
    assert pct(np.float64(0.443)) == 44
    record = PredictionRecord(
        name="genre400",
        category="genre",
        labels=["Reggae---Dub", "Electronic---Dub"],
        scores=[0.83, 0.80],
        top_label=["Reggae---Dub", "Electronic---Dub"],
        top_score=[0.83, 0.80],
        sep=";",
    )
    meta = record.flat_meta_from_record
    assert meta["meta_genre_genre400_main"] == ["Reggae"]
    assert meta["meta_genre_genre400_sub"] == ["Dub"]


def test_popm_rating_uses_standard_discrete_star_values():
    expected = {0: 0, 1: 1, 64: 2, 128: 3, 196: 4, 255: 5}
    for raw, stars in expected.items():
        assert popm_to_stars(raw) == stars
        assert stars_to_popm(stars) == raw


def test_format_specific_tag_values_are_normalized():
    assert _norm_text((2, 0)) == "2"
    assert _get_tag_name("catalognumber", ("ID3v2",)) == "TXXX:CATALOGNUMBER"
    assert _get_tag_name("catalognumber", ("Vorbis",)) == "CATALOGNUMBER"
    assert _get_tag_name("catalognumber", ("iTunes",)) == "----:com.apple.iTunes:CATALOGNUMBER"
    assert _get_tag_name("catalognumber", ("ASF",)) == "WM/CatalogNo"
    assert _get_tag_name("replaygain_track_gain", ("iTunes",)) == (
        "----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN"
    )


def test_unconfigured_tags_can_be_dropped_from_export():
    original = {"artist": "Artist", "comment": "old", "genre": "Old"}
    resolved = {"genre": ["New"]}
    assert merge_logical_tags_for_export(original, resolved) == {
        "artist": "Artist", "comment": "old", "genre": ["New"]
    }
    assert merge_logical_tags_for_export(
        original, resolved, preserve_unconfigured=False
    ) == {"genre": ["New"]}


def test_retag_is_the_export_switch():
    config = ExportConfig(output_root=Path("out"), retag=True)
    assert config.retag is True
    assert config.mode == "retag"
