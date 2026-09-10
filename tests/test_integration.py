"""Integration tests for all task categories using real ML models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from musikalyze.config import EmbeddingModel, LabelExtractor, TaggingConfig
from musikalyze.process import MusicProcess

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
_MODELS_DIR = _FIXTURE_DIR / "models"
_AUDIO_FILES = [
    _FIXTURE_DIR / "test_1.wav",
    _FIXTURE_DIR / "test_2.mp3",
    _FIXTURE_DIR / "test_3.mp3",
]

_EMBEDDER = EmbeddingModel(
    embedding_model=_MODELS_DIR / "discogs-effnet-bs64-1.pb",
    name="effnet",
)

_GENRE_EXTRACTOR = LabelExtractor(
    name="genre400",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "genre_discogs400-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "genre_discogs400-discogs-effnet-1.json",
    category="genre",
    task="multilabel",
    count=3,
    thold=0.7,
    count_thold_policy="union",
    input_tensor="serving_default_model_Placeholder",
    output_tensor="PartitionedCall:0",
)

_MOOD_HAPPY_EXTRACTOR = LabelExtractor(
    name="happy",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "mood_happy-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "mood_happy-discogs-effnet-1.json",
    category="mood",
    task="classification",
    output_tensor="model/Softmax",
)

_DANCE_EXTRACTOR = LabelExtractor(
    name="danceability",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "danceability-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "danceability-discogs-effnet-1.json",
    category="mood",
    task="regression",
    output_tensor="model/Softmax",
)

_VOICE_EXTRACTOR = LabelExtractor(
    name="voice_instrumental",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "voice_instrumental-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "voice_instrumental-discogs-effnet-1.json",
    category="mood",
    task="classification",
    output_tensor="model/Softmax",
)

_ACOUSTIC_EXTRACTOR = LabelExtractor(
    name="acoustic_electronic",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "nsynth_acoustic_electronic-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "nsynth_acoustic_electronic-discogs-effnet-1.json",
    category="mood",
    task="classification",
    output_tensor="model/Softmax",
)

_INSTRUMENT_EXTRACTOR = LabelExtractor(
    name="instrument",
    embedder_name="effnet",
    graph_path=_MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.pb",
    labels_path=_MODELS_DIR / "mtg_jamendo_instrument-discogs-effnet-1.json",
    category="classical",
    task="multilabel",
)

_TAGGING_CONFIG = TaggingConfig(
    artist="{tag_artist}",
    title="{tag_title}",
    genre="{meta_genre}",
)


def _make_process(audio_path: Path, extractors=None, tagging_config=None):
    extractors = extractors or []
    tagging = tagging_config or TaggingConfig()
    return MusicProcess(
        audio_file=audio_path,
        embedders=[_EMBEDDER],
        extractors=extractors,
        tagging_config=tagging,
    )


# ---------------------------------------------------------------------------
# Genre (multilabel)
# ---------------------------------------------------------------------------


class TestIntegrationGenre:
    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_genre_multilabel_labels(self, audio):
        proc = _make_process(audio, [_GENRE_EXTRACTOR])
        proc.analyze_file()
        labels = proc.labels
        genre_key = "meta_genre_genre400_all"
        assert genre_key in labels
        assert isinstance(labels[genre_key], dict)

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_genre_flat_meta(self, audio):
        proc = _make_process(audio, [_GENRE_EXTRACTOR])
        proc.analyze_file()
        assert "meta_genre_genre400_val" in proc.labels
        assert "meta_genre_genre400_dict" in proc.labels

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_genre_label_access(self, audio):
        proc = _make_process(audio, [_GENRE_EXTRACTOR])
        top = proc.label("genre400")
        assert isinstance(top, list)
        assert len(top) >= 1

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_genre_tag_file(self, audio):
        proc = _make_process(audio, [_GENRE_EXTRACTOR], _TAGGING_CONFIG)
        tags = proc.tag_file()
        assert isinstance(tags, dict)


# ---------------------------------------------------------------------------
# Mood (classification)
# ---------------------------------------------------------------------------


class TestIntegrationMood:
    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_mood_classification_labels(self, audio):
        proc = _make_process(audio, [_MOOD_HAPPY_EXTRACTOR])
        proc.analyze_file()
        labels = proc.labels
        assert "meta_mood_happy_val" in labels
        assert "meta_mood_happy_dict" in labels

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_mood_label_access(self, audio):
        proc = _make_process(audio, [_MOOD_HAPPY_EXTRACTOR])
        top = proc.label("happy")
        assert isinstance(top, list)
        assert len(top) >= 1

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_voice_classification(self, audio):
        proc = _make_process(audio, [_VOICE_EXTRACTOR])
        proc.analyze_file()
        assert "meta_mood_voice_instrumental_val" in proc.labels

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_acoustic_classification(self, audio):
        proc = _make_process(audio, [_ACOUSTIC_EXTRACTOR])
        proc.analyze_file()
        assert "meta_mood_acoustic_electronic_val" in proc.labels


# ---------------------------------------------------------------------------
# Classical / multilabel (mtg_jamendo_instrument)
# ---------------------------------------------------------------------------


class TestIntegrationClassical:
    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_classical_multilabel_labels(self, audio):
        proc = _make_process(audio, [_INSTRUMENT_EXTRACTOR])
        proc.analyze_file()
        assert "meta_instrument" in proc.labels
        assert isinstance(proc.labels["meta_instrument"], list)

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_classical_flat_meta(self, audio):
        proc = _make_process(audio, [_INSTRUMENT_EXTRACTOR])
        proc.analyze_file()
        assert "meta_instrument" in proc.labels


# ---------------------------------------------------------------------------
# Regression (danceability)
# ---------------------------------------------------------------------------


class TestIntegrationRegression:
    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_regression_score(self, audio):
        proc = _make_process(audio, [_DANCE_EXTRACTOR])
        proc.analyze_file()
        score = proc.labels["meta_mood_danceability_val"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_regression_dict(self, audio):
        proc = _make_process(audio, [_DANCE_EXTRACTOR])
        proc.analyze_file()
        assert "meta_mood_danceability_dict" in proc.labels

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_regression_label_access(self, audio):
        proc = _make_process(audio, [_DANCE_EXTRACTOR])
        val = proc.label("danceability")
        assert isinstance(val, (float, list))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestIntegrationClassification:
    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_classification_happy(self, audio):
        proc = _make_process(audio, [_MOOD_HAPPY_EXTRACTOR])
        proc.analyze_file()
        top_label = proc.labels["meta_mood_happy"]
        assert isinstance(top_label, list)
        assert len(top_label) >= 1

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_classification_voice(self, audio):
        proc = _make_process(audio, [_VOICE_EXTRACTOR])
        proc.analyze_file()
        top_label = proc.labels["meta_mood_voice_instrumental"]
        assert isinstance(top_label, list)

    @pytest.mark.parametrize("audio", [str(p) for p in _AUDIO_FILES if p.exists()])
    def test_classification_acoustic(self, audio):
        proc = _make_process(audio, [_ACOUSTIC_EXTRACTOR])
        proc.analyze_file()
        top_label = proc.labels["meta_mood_acoustic_electronic"]
        assert isinstance(top_label, list)


# ---------------------------------------------------------------------------
# Flat meta aggregation
# ---------------------------------------------------------------------------


class TestIntegrationFlatMeta:
    def test_multi_extractor_flat_meta(self):
        audio = _AUDIO_FILES[0]
        if not audio.exists():
            pytest.skip("test_1.wav not available")
        extractors = [_GENRE_EXTRACTOR, _MOOD_HAPPY_EXTRACTOR, _DANCE_EXTRACTOR]
        proc = _make_process(audio, extractors)
        proc.analyze_file()
        flat = proc.labels
        genre_keys = [k for k in flat if k.startswith("meta_genre_genre400_")]
        mood_keys = [k for k in flat if k.startswith("meta_mood_happy_")]
        dance_keys = [k for k in flat if k.startswith("meta_mood_danceability_")]
        assert len(genre_keys) >= 3, f"Expected genre keys, got: {genre_keys}"
        assert len(mood_keys) >= 3, f"Expected mood keys, got: {mood_keys}"
        assert len(dance_keys) >= 3, f"Expected dance keys, got: {dance_keys}"

    def test_meta_property_access(self):
        audio = _AUDIO_FILES[0]
        if not audio.exists():
            pytest.skip("test_1.wav not available")
        proc = _make_process(audio, [_MOOD_HAPPY_EXTRACTOR])
        val = proc.meta_mood_happy
        assert isinstance(val, list)

    def test_multiple_categories_flat_meta(self):
        audio = _AUDIO_FILES[0]
        if not audio.exists():
            pytest.skip("test_1.wav not available")
        extractors = [
            _GENRE_EXTRACTOR,
            _MOOD_HAPPY_EXTRACTOR,
            _INSTRUMENT_EXTRACTOR,
            _DANCE_EXTRACTOR,
        ]
        proc = _make_process(audio, extractors)
        proc.analyze_file()
        flat = proc.labels
        genre_keys = [k for k in flat if k.startswith("meta_genre_")]
        mood_keys = [k for k in flat if k.startswith("meta_mood_")]
        classical_keys = [k for k in flat if k.startswith("meta_instrument_")]
        assert len(genre_keys) >= 3, f"No genre keys: {list(flat.keys())[:10]}"
        assert len(mood_keys) >= 3, f"No mood keys: {list(flat.keys())[:10]}"
        assert len(classical_keys) >= 3, f"No classical keys: {list(flat.keys())[:10]}"


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------


class TestConfigJson:
    _CONFIG_PATH = _FIXTURE_DIR / "config_example.json"

    @classmethod
    def _load_config(cls):
        if not cls._CONFIG_PATH.exists():
            pytest.skip("config_example.json not available")
        text = cls._CONFIG_PATH.read_text()
        return json.loads(text)

    @classmethod
    def _build_from_json(cls, base_dir: Path):
        cfg = cls._load_config()
        project_root = _FIXTURE_DIR.parent
        embedders = []
        for name, data in cfg.get("embedding_models", {}).items():
            embedders.append(
                EmbeddingModel(
                    embedding_model=project_root / data["embedding_model"],
                    name=data.get("name", name),
                )
            )
        extractors = []
        for data in cfg.get("label_extractors", []):
            extractors.append(
                LabelExtractor(
                    name=data["name"],
                    embedder_name=data["embedder_name"],
                    graph_path=project_root / data["graph_path"],
                    labels_path=project_root / data.get("labels_path", ""),
                    category=data.get("category", "other"),
                    task=data.get("task", "classification"),
                    count=data.get("count", 1),
                    thold=data.get("thold", 1.0),
                    count_thold_policy=data.get("count_thold_policy", "intersection"),
                    output_tensor=data.get("output_tensor", "model/Sigmoid"),
                    input_tensor=data.get("input_tensor", "model/Placeholder"),
                )
            )
        tagging_cfg = cfg.get("tagging_config", {})
        tagging = TaggingConfig(**{k: v for k, v in tagging_cfg.items() if v is not None})
        export_cfg_data = cfg.get("export_config", {})
        from musikalyze.config import ExportConfig
        export_cfg = ExportConfig(
            output_root=Path(export_cfg_data.get("output_root", "output")),
            formats=export_cfg_data.get("formats", "opus"),
            path_template=export_cfg_data.get("path_template", "{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}"),
            format_options=export_cfg_data.get("format_options", {}),
        )
        return embedders, extractors, tagging, export_cfg

    def test_config_file_exists(self):
        assert self._CONFIG_PATH.exists()

    def test_config_structure(self):
        cfg = self._load_config()
        assert "embedding_models" in cfg
        assert "label_extractors" in cfg
        assert "tagging_config" in cfg
        assert "export_config" in cfg

    def test_config_has_effnet(self):
        cfg = self._load_config()
        assert "effnet" in cfg["embedding_models"]

    def test_config_has_genre_extractor(self):
        cfg = self._load_config()
        names = [e["name"] for e in cfg["label_extractors"]]
        assert "genre400" in names

    def test_config_has_mood_extractor(self):
        cfg = self._load_config()
        names = [e["name"] for e in cfg["label_extractors"]]
        assert "happy" in names

    def test_config_has_regression_extractor(self):
        cfg = self._load_config()
        extractors = cfg["label_extractors"]
        regression_names = [e["name"] for e in extractors if e.get("task") == "regression"]
        assert "danceability" in regression_names

    def test_config_has_multilabel_extractor(self):
        cfg = self._load_config()
        extractors = cfg["label_extractors"]
        multilabel_names = [e["name"] for e in extractors if e.get("task") == "multilabel"]
        assert "genre400" in multilabel_names

    def test_config_loads_models_and_builds_process(self):
        _embedders, extractors, tagging, _export_cfg = self._build_from_json(_MODELS_DIR)
        audio = _AUDIO_FILES[0]
        if not audio.exists():
            pytest.skip("test_1.wav not available")
        proc = _make_process(audio, extractors, tagging)
        proc.analyze_file()
        assert len(proc.labels) > 0

    def test_config_integration_full_pipeline(self):
        embedders, extractors, tagging, export_cfg = self._build_from_json(_MODELS_DIR)
        audio = _AUDIO_FILES[0]
        if not audio.exists():
            pytest.skip("test_1.wav not available")
        proc = MusicProcess(
            audio_file=audio,
            embedders=embedders,
            extractors=extractors,
            tagging_config=tagging,
            export_config=export_cfg,
        )
        proc.load_audio()
        eng = proc.analyze_file()
        assert eng is not None
        tags = proc.tag_file()
        assert isinstance(tags, dict)

    def test_config_exporters_have_correct_categories(self):
        cfg = self._load_config()
        extractors = cfg["label_extractors"]
        categories = {e["category"] for e in extractors}
        assert "genre" in categories
        assert "mood" in categories
        assert "classical" in categories

    def test_config_exporter_tasks(self):
        cfg = self._load_config()
        extractors = cfg["label_extractors"]
        tasks = {e["task"] for e in extractors}
        assert "multilabel" in tasks
        assert "classification" in tasks
        assert "regression" in tasks
