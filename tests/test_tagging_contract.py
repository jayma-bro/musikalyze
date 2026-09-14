"""Contract tests for the explicit tagging API."""

from pathlib import Path

from musikalyze.config import AnalysisResult, ExportConfig, LabelExtractor, TaggingConfig
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


def test_retag_is_the_export_switch():
    config = ExportConfig(output_root=Path("out"), retag=True)
    assert config.retag is True
    assert config.mode == "retag"
