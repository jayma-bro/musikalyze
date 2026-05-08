"""Main per-file pipeline: ``MusicProcess``."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence, Union

from musikalyze.audio_io import load_audio
from musikalyze.config import AnalysisResult, EmbeddingModel, ExportConfig, LabelExtractor, TaggingConfig
from musikalyze.export_ffmpeg import export_multiple_formats
from musikalyze.lazy_engine import LazyMetaEngine
from musikalyze.tagging import (
    apply_tagging_config,
    file_meta_from_tags,
    merge_logical_tags_for_export,
    read_tags_raw,
    tags_to_tag_prefix,
)
from musikalyze.templates import build_format_mapping, extract_placeholder_keys, resolve_template


def _tagging_template_strings(cfg: TaggingConfig) -> list[str]:
    out: list[str] = []
    for f in fields(cfg):
        if f.name == "extra":
            continue
        v = getattr(cfg, f.name)
        if isinstance(v, str):
            out.append(v)
    for _k, v in cfg.extra.items():
        if isinstance(v, str):
            out.append(v)
    return out


def _collect_needed_meta_keys(
    tagging_config: TaggingConfig,
    export_config: ExportConfig | None,
) -> set[str]:
    parts = _tagging_template_strings(tagging_config)
    if export_config:
        parts.append(export_config.path_template)
    return extract_placeholder_keys(*parts)


class MusicProcess:
    """
    Load audio, compute all embeddings in ``analyze_file()``, then resolve labels and classical
    descriptors lazily when templates or ``label()`` need them.
    """

    def __init__(
        self,
        *,
        audio_file: Path | str,
        embedders: Sequence[EmbeddingModel] = (),
        extractors: Sequence[LabelExtractor] = (),
        tagging_config: TaggingConfig | None = None,
        export_config: ExportConfig | None = None,
        separator: str = ";",
    ) -> None:
        self.audio_path = Path(audio_file)
        self.tagging_config = tagging_config or TaggingConfig()
        self.export_config = export_config
        self.separator = separator

        self._embedders = {e.name: e for e in embedders}
        self._extractors = {e.name: e for e in extractors}

        self._audio_mono: Any = None
        self._lazy_engine: LazyMetaEngine | None = None
        self._tags_raw: dict[str, Any] = {}
        self._tags_prefixed: dict[str, Any] = {}
        self._tags_resolved: dict[str, str] = {}
        self._meta_cache: dict[str, Any] | None = None
        self._embeddings_ready = False

    @property
    def audio_mono(self) -> Any:
        return self._audio_mono

    @property
    def tags_original(self) -> dict[str, Any]:
        return dict(self._tags_raw)

    @property
    def tags_resolved(self) -> dict[str, str]:
        return dict(self._tags_resolved)
    
    @property
    def labels(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not self._tags_raw:
            self.read_tags()
        out.update(self._tags_prefixed)
        if self._audio_mono is None:
            self.load_audio()
        if not self._embeddings_ready:
            self.analyze_file()
        out.update(self._engine().build_flat_meta())
        return out
        


    def _engine(self) -> LazyMetaEngine:
        if self._lazy_engine is None:
            if self._audio_mono is None:
                raise RuntimeError("Call load_audio() first.")
            self._lazy_engine = LazyMetaEngine(
                self._audio_mono,
                self._embedders,
                self._extractors,
                audio_path=self.audio_path,
                sep=self.separator
            )
        return self._lazy_engine

    def load_audio(self) -> Any:
        self._audio_mono, _ = load_audio(self.audio_path, track="mono", sample_rate=16000, resample_quality=4)
        self._lazy_engine = None
        self._embeddings_ready = False
        self._meta_cache = None
        return self._audio_mono

    def read_tags(self) -> dict[str, Any]:
        self._tags_raw = read_tags_raw(self.audio_path)
        self._tags_prefixed = tags_to_tag_prefix(self._tags_raw)
        return self._tags_raw

    def _meta_flat_for_tagging(self) -> dict[str, Any]:
        keys = _collect_needed_meta_keys(self.tagging_config, self.export_config)
        eng = self._engine()
        meta = eng.build_flat_meta(keys)
        meta.update(file_meta_from_tags(self._tags_raw, keys))
        self._meta_cache = meta
        return meta

    def label(self, key: Union[str, Sequence[str]]) -> Union[str, Sequence[Any]]:
        if isinstance(key, str):
            keys = [key]
            single = True
        else:
            keys = list(key)
            single = False

        if not self._tags_raw:
            self.read_tags()
        out: list[Any] = []
        for k in keys:
            if k.startswith("tag_"):
                out.append(self._tags_prefixed.get(k))
                continue
            if k.startswith("meta"):
                nk = k
            else:
                nk = f"meta_{k}"
            if self._audio_mono is None:
                self.load_audio()
            if not self._embeddings_ready:
                self.analyze_file()
            out.append(self._engine().get_one_meta(nk))
        return out[0] if single else out

    def __getattr__(self, name: str) -> Any:
        if name.startswith("meta"):
            return self.label(name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def analyze_file(self) -> LazyMetaEngine:
        """Load audio if needed, then initialize engine."""

        if self._audio_mono is None:
            self.load_audio()
        return self._engine()

    def tag_file(self) -> dict[str, str]:
        if not self._tags_raw:
            self.read_tags()
        if not self._embeddings_ready:
            self.analyze_file()
        meta_flat = self._meta_flat_for_tagging()
        ar = AnalysisResult(meta=meta_flat)
        self._tags_resolved = apply_tagging_config(self._tags_prefixed, ar, self.tagging_config)
        return self._tags_resolved

    def export_file(self) -> list[Path]:
        if self.export_config is None:
            raise ValueError("export_config is required for export_file()")
        if not self._tags_resolved:
            self.tag_file()
        if not self._embeddings_ready:
            self.analyze_file()
        keys = _collect_needed_meta_keys(self.tagging_config, self.export_config)
        meta = self._engine().build_flat_meta(keys)
        meta.update(file_meta_from_tags(self._tags_raw, keys))
        merged = merge_logical_tags_for_export(self._tags_raw, self._tags_resolved)
        return export_multiple_formats(
            self.audio_path,
            self.export_config.output_root,
            self.export_config.path_template,
            self.export_config.formats,
            merged,
            meta,
            self.export_config.format_options,
            sanitize_paths=self.export_config.sanitize_paths,
            overwrite=self.export_config.overwrite,
        )

    def process_file(self) -> tuple[LazyMetaEngine | None, dict[str, str], list[Path] | None]:
        self.read_tags()
        self.load_audio()
        eng = self.analyze_file()
        tr = self.tag_file()
        paths: list[Path] | None = None
        if self.export_config is not None:
            paths = self.export_file()
        return eng, tr, paths

    def preview_path(self, ext: str = "opus") -> Path:
        if self.export_config is None:
            raise ValueError("export_config is required")
        if self._audio_mono is None:
            self.read_tags()
            self.load_audio()
        if not self._embeddings_ready:
            self.analyze_file()
        if not self._tags_resolved:
            self.tag_file()
        from musikalyze.export_ffmpeg import build_output_path

        keys = _collect_needed_meta_keys(self.tagging_config, self.export_config)
        meta = self._engine().build_flat_meta(keys)
        meta.update(file_meta_from_tags(self._tags_raw, keys))
        return self.export_config.output_root / build_output_path(
            self.export_config.path_template,
            merge_logical_tags_for_export(self._tags_raw, self._tags_resolved),
            meta,
            ext,
            sanitize=self.export_config.sanitize_paths,
        )

    def format_preview(self, template: str) -> str:
        if not self._tags_prefixed:
            self.read_tags()
        keys = extract_placeholder_keys(template)
        if any(k.startswith("meta_") for k in keys):
            if self._audio_mono is None:
                self.load_audio()
            if not self._embeddings_ready:
                self.analyze_file()
        meta = self._engine().build_flat_meta(keys)
        meta.update(file_meta_from_tags(self._tags_raw, keys))
        m = build_format_mapping(self._tags_prefixed, meta, ext=None)
        return resolve_template(template, m)
