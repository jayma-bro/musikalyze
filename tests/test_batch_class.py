"""Tests for the MusicBatch class (no Essentia models required)."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import pandas as pd
import pytest

import musikalyze.batch as batch_mod
from musikalyze import MusicBatch
from musikalyze.config import EmbeddingModel, ExportConfig, LabelExtractor
from musikalyze.lazy_engine import LazyMetaEngine
from musikalyze.process import MusicProcess


def _write_wav(path: Path) -> None:
    rate = 8000
    n = rate // 10
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    (tmp_path / "artist_a").mkdir()
    (tmp_path / "artist_b").mkdir()
    _write_wav(tmp_path / "artist_a" / "01 - song.wav")
    _write_wav(tmp_path / "artist_a" / "02 - song.wav")
    _write_wav(tmp_path / "artist_b" / "03 - song.wav")
    (tmp_path / "notes.txt").write_text("not audio")
    return tmp_path


class _StubProcess:
    """Stand-in for MusicProcess: instant labels, no audio/model work."""

    fail_paths: frozenset[str] = frozenset()
    drop_key_paths: frozenset[str] = frozenset()
    last: _StubProcess | None = None

    def __init__(
        self,
        *,
        audio_file,
        embedders=(),
        extractors=(),
        tagging_config=None,
        export_config=None,
        separator=";",
    ):
        self.audio_path = Path(audio_file)
        self.export_config = export_config
        _StubProcess.last = self

    @property
    def labels(self) -> dict:
        if str(self.audio_path) in _StubProcess.fail_paths:
            raise RuntimeError("boom")
        labels = {
            "tag_artist": "artist",
            "meta_bpm": 120.0,
            "meta_genre400_all": {"Rock": 0.9, "Pop": 0.1},
        }
        if str(self.audio_path) in _StubProcess.drop_key_paths:
            labels.pop("meta_genre400_all")
        return labels

    def process_file(self):
        return None, {}, [self.audio_path]

    def preview_path(self, ext: str = "opus") -> Path:
        return self.export_config.output_root / f"{self.audio_path.stem}.{ext}"


@pytest.fixture
def stub_process(monkeypatch):
    _StubProcess.fail_paths = frozenset()
    _StubProcess.drop_key_paths = frozenset()
    _StubProcess.last = None
    monkeypatch.setattr(batch_mod, "MusicProcess", _StubProcess)
    return _StubProcess


# ---------------------------------------------------------------------------
# Discovery / pythonic access (real MusicProcess, no IO triggered)
# ---------------------------------------------------------------------------


def test_files_listing(library):
    batch = MusicBatch(library)
    expected = sorted(
        [
            str(library / "artist_a" / "01 - song.wav"),
            str(library / "artist_a" / "02 - song.wav"),
            str(library / "artist_b" / "03 - song.wav"),
        ]
    )
    assert batch.files == expected
    assert all(isinstance(f, str) for f in batch.files)


def test_files_not_recursive(library):
    batch = MusicBatch(library, recursive=False)
    assert batch.files == []


def test_extensions_filter(library):
    assert len(MusicBatch(library, extensions={".wav"}).files) == 3
    assert MusicBatch(library, extensions={".flac"}).files == []


def test_len_iter_getitem(library):
    batch = MusicBatch(library)
    assert len(batch) == 3
    procs = list(batch)
    assert len(procs) == 3
    assert all(isinstance(p, batch_mod.MusicProcess) for p in procs)
    assert isinstance(batch[0], batch_mod.MusicProcess)
    assert isinstance(batch[1:2], list)
    assert batch[-1].audio_path == library / "artist_b" / "03 - song.wav"


def test_sample(library):
    batch = MusicBatch(library)
    half = batch.sample(0.5)
    assert isinstance(half, MusicBatch)
    assert len(half) == 2
    assert len(batch) == 3  # original untouched
    assert len(batch.sample(2)) == 2


def test_summary(library):
    info = MusicBatch(library).summary()
    assert info["n_files"] == 3
    assert info["extensions"] == {".wav": 3}
    assert info["total_size_bytes"] > 0
    assert info["total_size_human"].endswith("B")
    assert info["root"] == str(library)


def test_not_a_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        MusicBatch(tmp_path / "missing")


def test_pool_flag(library):
    assert MusicBatch(library)._use_pool() is False
    assert MusicBatch(library, max_workers=4)._use_pool() is True


# ---------------------------------------------------------------------------
# analyze()
# ---------------------------------------------------------------------------


def test_analyze_explodes_dict(library, stub_process):
    df = MusicBatch(library).analyze("genre400_all")
    assert isinstance(df, pd.DataFrame)
    assert next(iter(df.columns)) == "_path"
    assert {"Rock", "Pop"} <= set(df.columns)
    assert "meta_bpm" not in df.columns  # only the exploded key's labels
    assert df["Rock"].iloc[0] == pytest.approx(0.9)
    assert df["Pop"].iloc[0] == pytest.approx(0.1)
    assert len(df) == 3
    assert df["_path"].is_monotonic_increasing


def test_analyze_scalar_key(library, stub_process):
    df = MusicBatch(library).analyze("bpm")  # normalized to meta_bpm
    assert list(df.columns) == ["_path", "meta_bpm"]
    assert df["meta_bpm"].iloc[0] == pytest.approx(120.0)


def test_analyze_explicit_meta_prefix(library, stub_process):
    df = MusicBatch(library).analyze("meta_genre400_all")
    assert "Rock" in df.columns


def test_analyze_missing_key_pads_none(library, stub_process):
    _StubProcess.drop_key_paths = frozenset({str(library / "artist_a" / "01 - song.wav")})
    df = MusicBatch(library).analyze("genre400_all")
    assert len(df) == 3
    row = df[df["_path"] == str(library / "artist_a" / "01 - song.wav")].iloc[0]
    assert pd.isna(row["Rock"])


def test_analyze_failed_file_pads_none(library, stub_process, caplog):
    _StubProcess.fail_paths = frozenset({str(library / "artist_b" / "03 - song.wav")})
    with caplog.at_level(logging.WARNING):
        df = MusicBatch(library).analyze("genre400_all")
    assert len(df) == 3
    row = df[df["_path"] == str(library / "artist_b" / "03 - song.wav")].iloc[0]
    assert pd.isna(row["Rock"])
    assert any("Analysis failed" in r.message for r in caplog.records)


def test_analyze_requires_key(library, stub_process):
    with pytest.raises(ValueError):
        MusicBatch(library).analyze("")


def test_analyze_requires_files(tmp_path, stub_process):
    with pytest.raises(ValueError):
        MusicBatch(tmp_path).analyze("genre400_all")


# ---------------------------------------------------------------------------
# export() / preview_paths()
# ---------------------------------------------------------------------------


def test_export_overrides_output_root(library, stub_process, tmp_path):
    out = tmp_path / "out"
    batch = MusicBatch(library, export_config=ExportConfig(output_root=tmp_path / "elsewhere"))
    batch.export(out)
    assert out.exists()
    assert stub_process.last.export_config.output_root == out
    assert batch._failures == []


def test_export_default_config(library, stub_process, tmp_path):
    out = tmp_path / "out"
    MusicBatch(library).export(out)
    cfg = stub_process.last.export_config
    assert cfg.output_root == out
    assert cfg.formats == "opus"


def test_export_failure_is_skipped(library, tmp_path, monkeypatch):
    class _Failing(_StubProcess):
        def process_file(self):
            raise RuntimeError("nope")

    monkeypatch.setattr(batch_mod, "MusicProcess", _Failing)
    batch = MusicBatch(library)
    batch.export(tmp_path / "out")  # must not raise
    assert len(batch._failures) == 3
    assert all("nope" in err for _, err in batch._failures)


def test_export_empty_library_warns(tmp_path, stub_process, caplog):
    with caplog.at_level(logging.WARNING):
        MusicBatch(tmp_path).export(tmp_path / "out")
    assert any("No audio files" in r.message for r in caplog.records)


def test_preview_paths(library, stub_process):
    out_root = library / "out"
    paths = MusicBatch(library, export_config=ExportConfig(output_root=out_root)).preview_paths("mp3")
    assert len(paths) == 3
    assert all(p.suffix == ".mp3" for p in paths)
    assert all(p.parent == out_root for p in paths)


def test_preview_requires_config(library):
    with pytest.raises(ValueError):
        MusicBatch(library).preview_paths()


# ---------------------------------------------------------------------------
# Parallel-worker serialization
# ---------------------------------------------------------------------------


def test_serialization_roundtrip():
    e = EmbeddingModel(embedding_model=Path("m.pb"), name="maest", patch_size=8)
    assert batch_mod._deserialize_embedding(batch_mod._serialize_embedding(e)) == e

    x = LabelExtractor(name="g", embedder_name="effnet", graph_path=Path("g.pb"), labels_path=Path("l.json"))
    assert batch_mod._deserialize_extractor(batch_mod._serialize_extractor(x)) == x


def test_worker_process_one_reports_errors(tmp_path):
    path = str(tmp_path / "missing.wav")
    ok_path, ok, error = batch_mod._worker_process_one(path, [], [], {}, {"output_root": str(tmp_path)}, ";")
    assert ok_path == path
    assert ok is False
    assert error


def test_worker_analyze_one_reports_errors(tmp_path):
    path = str(tmp_path / "missing.wav")
    got_path, labels, error = batch_mod._worker_analyze_one(path, [], [], ";")
    assert got_path == path
    assert labels is None
    assert error


# ---------------------------------------------------------------------------
# Regression: build_flat_meta / preview_path with a set of keys
# ---------------------------------------------------------------------------


def test_build_flat_meta_accepts_key_collection():
    eng = LazyMetaEngine(None, {}, {}, audio_path=Path("x.mp3"), sep=";")
    assert eng.build_flat_meta({"tag_artist", "meta_genre400_val"}) == {}
    assert eng.build_flat_meta("meta_genre400_val") == {}


def test_preview_path_end_to_end(library):
    wav = library / "artist_a" / "01 - song.wav"
    out_root = library / "out"
    proc = MusicProcess(audio_file=wav, export_config=ExportConfig(output_root=out_root))
    path = proc.preview_path("opus")
    assert isinstance(path, Path)
    assert path.suffix == ".opus"
    assert path.is_relative_to(out_root)
