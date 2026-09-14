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
from musikalyze.tagging import apply_tagging_config


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
    assert result == {"genre": "Rock;Pop", "key": "C#", "custom": "one;two"}


def test_classical_values_are_valid_template_inputs():
    config = TaggingConfig(tags={"bpm": "{meta_bpm}", "gain": "{meta_rgain_gain}"})
    result = apply_tagging_config(
        {}, AnalysisResult(meta={"meta_bpm": 120, "meta_rgain_gain": -5.2}), config
    )
    assert result == {"bpm": "120", "gain": "-5.2"}


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


def test_retag_is_the_export_switch():
    config = ExportConfig(output_root=Path("out"), retag=True)
    assert config.retag is True
    assert config.mode == "retag"
