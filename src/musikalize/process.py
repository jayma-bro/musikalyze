"""Classe principale : analyse, tags et export."""

from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence, Union

from musikalize.analysis import analyze_audio
from musikalize.audio_io import load_mono_16k
from musikalize.config import (
    AnalysisConfig,
    AnalysisResult,
    ClassicalConfig,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    ModelPath,
    TaggingConfig,
)
from musikalize.export_ffmpeg import export_multiple_formats
from musikalize.lazy_engine import LazyMetaEngine
from musikalize.tagging import apply_tagging_config, read_tags_raw, tags_to_tag_prefix
from musikalize.templates import build_format_mapping, extract_placeholder_keys, resolve_template

log = logging.getLogger(__name__)


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
    Pipeline par fichier : chargement audio, analyse (lazy possible), tags, export.

    Nouvelle API : ``embedders`` + ``label_extractors``. Ancienne API : ``model_path`` + ``analysis_config``.
    """

    def __init__(
        self,
        *,
        audio_file: Path | str,
        embedders: Sequence[EmbeddingModel] | None = None,
        label_extractors: Sequence[LabelExtractor] | None = None,
        classical_config: ClassicalConfig | None = None,
        model_path: ModelPath | None = None,
        analysis_config: AnalysisConfig | None = None,
        tagging_config: TaggingConfig | None = None,
        export_config: ExportConfig | None = None,
    ) -> None:
        self.audio_path = Path(audio_file)
        self.tagging_config = tagging_config or TaggingConfig()
        self.export_config = export_config
        self.classical_config = classical_config or ClassicalConfig()

        self._model_path = model_path
        self.analysis_config = analysis_config or AnalysisConfig()

        if embedders is not None:
            self._embedders = {e.name: e for e in embedders}
            self._label_extractors = list(label_extractors or ())
        elif model_path is not None:
            self._embedders = None
            self._label_extractors = None
        else:
            raise ValueError(
                "Fournissez soit embedders= (liste, éventuellement vide pour classique seul), "
                "soit model_path= pour l'API historique."
            )

        self._audio_mono: Any = None
        self._analysis_legacy: AnalysisResult | None = None
        self._lazy_engine: LazyMetaEngine | None = None
        self._tags_raw: dict[str, Any] = {}
        self._tags_prefixed: dict[str, Any] = {}
        self._tags_resolved: dict[str, str] = {}
        self._meta_cache: dict[str, Any] | None = None

    @property
    def audio_mono(self) -> Any:
        return self._audio_mono

    @property
    def analysis(self) -> AnalysisResult | None:
        """API historique : résultat après ``analyze_file`` ; sinon ``None``."""

        return self._analysis_legacy

    @property
    def tags_original(self) -> dict[str, Any]:
        return dict(self._tags_raw)

    @property
    def tags_resolved(self) -> dict[str, str]:
        return dict(self._tags_resolved)

    def _use_lazy(self) -> bool:
        return self._embedders is not None and self._label_extractors is not None

    def _engine(self) -> LazyMetaEngine:
        if self._lazy_engine is None:
            if self._audio_mono is None:
                raise RuntimeError("Charger l'audio avec load_audio() d'abord.")
            assert self._embedders is not None and self._label_extractors is not None
            self._lazy_engine = LazyMetaEngine(
                self._audio_mono,
                self._embedders,
                self._label_extractors,
                self.classical_config,
                list_join_sep=self.tagging_config.separator,
            )
        return self._lazy_engine

    def load_audio(self) -> Any:
        self._audio_mono = load_mono_16k(self.audio_path)
        self._lazy_engine = None
        self._meta_cache = None
        return self._audio_mono

    def read_tags(self) -> dict[str, Any]:
        self._tags_raw = read_tags_raw(self.audio_path)
        self._tags_prefixed = tags_to_tag_prefix(self._tags_raw)
        return self._tags_raw

    def _meta_flat_for_tagging(self) -> dict[str, Any]:
        if self._use_lazy():
            keys = _collect_needed_meta_keys(self.tagging_config, self.export_config)
            eng = self._engine()
            self._meta_cache = eng.build_flat_meta(keys)
            return self._meta_cache
        if self._analysis_legacy is not None:
            return dict(self._analysis_legacy.meta)
        return {}

    def label(self, key: Union[str, Sequence[str]]) -> Any:
        """
        Accès programmatique aux métadonnées.

        - Une clé : ``"meta_bpm"`` ou ``"bpm"``.
        - Plusieurs clés : retourne une liste dans le même ordre.
        """

        if isinstance(key, str):
            keys = [key]
            single = True
        else:
            keys = list(key)
            single = False

        out: list[Any] = []
        if not self._tags_raw:
            self.read_tags()
        for k in keys:
            if k.startswith("tag_"):
                out.append(self._tags_prefixed.get(k))
                continue
            if k.startswith("meta_"):
                nk = k
            else:
                nk = f"meta_{k}"
            if self._use_lazy():
                eng = self._engine()
                v = eng.get_one_meta(nk)
                out.append(v)
            elif self._analysis_legacy is not None:
                out.append(self._analysis_legacy.meta.get(nk))
            else:
                raise RuntimeError("Appelez analyze_file() ou load_audio() puis analyze_file().")
        return out[0] if single else out

    def __getattr__(self, name: str) -> Any:
        if name.startswith("meta_"):
            return self.label(name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def analyze_file(self) -> AnalysisResult | LazyMetaEngine:
        """Charge l'audio et exécute l'analyse (historique) ou prépare le moteur lazy."""

        if self._audio_mono is None:
            self.load_audio()
        assert self._audio_mono is not None

        if self._use_lazy():
            return self._engine()

        assert self._model_path is not None
        self._analysis_legacy = analyze_audio(self._audio_mono, self._model_path, self.analysis_config)
        return self._analysis_legacy

    def tag_file(self) -> dict[str, str]:
        if not self._tags_raw:
            self.read_tags()

        if self._use_lazy():
            if self._audio_mono is None:
                self.load_audio()
            meta_flat = self._meta_flat_for_tagging()
            ar = AnalysisResult(meta=meta_flat)
        else:
            if self._analysis_legacy is None:
                self.analyze_file()
            ar = self._analysis_legacy

        self._tags_resolved = apply_tagging_config(self._tags_prefixed, ar, self.tagging_config)
        return self._tags_resolved

    def export_file(self) -> list[Path]:
        if self.export_config is None:
            raise ValueError("export_config requis pour export_file()")
        if not self._tags_resolved:
            self.tag_file()

        if self._use_lazy():
            if self._audio_mono is None:
                self.load_audio()
            keys = _collect_needed_meta_keys(self.tagging_config, self.export_config)
            meta = self._engine().build_flat_meta(keys)
        else:
            if self._analysis_legacy is None:
                raise ValueError("analyze_file() doit être appelé avant export_file()")
            meta = self._analysis_legacy.meta

        return export_multiple_formats(
            self.audio_path,
            self.export_config.output_root,
            self.export_config.path_template,
            self.export_config.formats,
            self._tags_resolved,
            meta,
            self.export_config.format_options,
            sanitize_paths=self.export_config.sanitize_paths,
            overwrite=self.export_config.overwrite,
        )

    def process_file(self) -> tuple[Any, dict[str, str], list[Path] | None]:
        self.read_tags()
        self.load_audio()
        ar = self.analyze_file()
        tr = self.tag_file()
        paths: list[Path] | None = None
        if self.export_config is not None:
            paths = self.export_file()
        return ar, tr, paths

    def preview_path(self, ext: str = "opus") -> Path:
        if self.export_config is None:
            raise ValueError("export_config requis")
        if self._audio_mono is None:
            self.read_tags()
            self.load_audio()
            self.analyze_file()
        if not self._tags_resolved:
            self.tag_file()
        from musikalize.export_ffmpeg import build_output_path

        if self._use_lazy():
            meta = self._engine().build_flat_meta(
                _collect_needed_meta_keys(self.tagging_config, self.export_config)
            )
        else:
            meta = self._analysis_legacy.meta if self._analysis_legacy else {}
        return self.export_config.output_root / build_output_path(
            self.export_config.path_template,
            self._tags_resolved,
            meta,
            ext,
            sanitize=self.export_config.sanitize_paths,
        )

    def format_preview(self, template: str) -> str:
        if not self._tags_prefixed:
            self.read_tags()
        if self._use_lazy():
            if self._audio_mono is None:
                self.load_audio()
            keys = extract_placeholder_keys(template)
            meta = self._engine().build_flat_meta(keys)
        else:
            if self._analysis_legacy is None:
                raise ValueError("analyze_file() requis pour les méta-données")
            meta = self._analysis_legacy.meta
        m = build_format_mapping(self._tags_prefixed, meta, ext=None)
        return resolve_template(template, m)
