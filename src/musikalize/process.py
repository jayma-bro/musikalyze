"""Classe principale : analyse, tags et export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from musikalize.analysis import analyze_audio
from musikalize.audio_io import load_mono_16k
from musikalize.config import (
    AnalysisConfig,
    AnalysisResult,
    ExportConfig,
    ModelPath,
    TaggingConfig,
)
from musikalize.export_ffmpeg import export_multiple_formats
from musikalize.tagging import apply_tagging_config, read_tags_raw, tags_to_tag_prefix
from musikalize.templates import build_format_mapping, resolve_template


class MusicProcess:
    """
    Pipeline par fichier : chargement audio, analyse Essentia, gabarits de tags, export.

    Les propriétés exposent l'état pour inspection ou tests unitaires dans un notebook.
    """

    def __init__(
        self,
        *,
        audio_file: Path | str,
        model_path: ModelPath,
        analysis_config: AnalysisConfig | None = None,
        tagging_config: TaggingConfig | None = None,
        export_config: ExportConfig | None = None,
    ) -> None:
        self.audio_path = Path(audio_file)
        self.model_path = model_path
        self.analysis_config = analysis_config or AnalysisConfig()
        self.tagging_config = tagging_config or TaggingConfig()
        self.export_config = export_config

        self._audio_mono: Any = None
        self._analysis: AnalysisResult | None = None
        self._tags_raw: dict[str, Any] = {}
        self._tags_prefixed: dict[str, Any] = {}
        self._tags_resolved: dict[str, str] = {}

    @property
    def audio_mono(self) -> Any:
        """Signal mono 16 kHz (numpy) après ``load_audio`` ou ``analyze_file``."""

        return self._audio_mono

    @property
    def analysis(self) -> AnalysisResult | None:
        return self._analysis

    @property
    def tags_original(self) -> dict[str, Any]:
        """Tags lus sur le fichier source (clés logiques)."""

        return dict(self._tags_raw)

    @property
    def tags_resolved(self) -> dict[str, str]:
        """Champs résolus après gabarits (clés logiques)."""

        return dict(self._tags_resolved)

    def load_audio(self) -> Any:
        """Charge le signal mono 16 kHz."""

        self._audio_mono = load_mono_16k(self.audio_path)
        return self._audio_mono

    def read_tags(self) -> dict[str, Any]:
        """Lit les tags du fichier source."""

        self._tags_raw = read_tags_raw(self.audio_path)
        self._tags_prefixed = tags_to_tag_prefix(self._tags_raw)
        return self._tags_raw

    def analyze_file(self) -> AnalysisResult:
        """Exécute l'analyse Essentia / TensorFlow et les descripteurs classiques."""

        if self._audio_mono is None:
            self.load_audio()
        assert self._audio_mono is not None
        self._analysis = analyze_audio(self._audio_mono, self.model_path, self.analysis_config)
        return self._analysis

    def tag_file(self) -> dict[str, str]:
        """Applique ``TaggingConfig`` et remplit ``tags_resolved``."""

        if not self._tags_raw:
            self.read_tags()
        self._tags_resolved = apply_tagging_config(
            self._tags_prefixed,
            self._analysis,
            self.tagging_config,
        )
        return self._tags_resolved

    def export_file(self) -> list[Path]:
        """Exporte vers ``ExportConfig.output_root`` selon formats et gabarit."""

        if self.export_config is None:
            raise ValueError("export_config requis pour export_file()")
        if not self._tags_resolved:
            self.tag_file()
        if self._analysis is None:
            raise ValueError("analyze_file() doit être appelé avant export_file()")

        meta = self._analysis.meta if self._analysis else {}
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

    def process_file(self) -> tuple[AnalysisResult, dict[str, str], list[Path] | None]:
        """Enchaîne lecture tags, chargement, analyse, tagging, et export si config fournie."""

        self.read_tags()
        self.load_audio()
        ar = self.analyze_file()
        tr = self.tag_file()
        paths: list[Path] | None = None
        if self.export_config is not None:
            paths = self.export_file()
        return ar, tr, paths

    def preview_path(self, ext: str = "opus") -> Path:
        """Chemin de sortie prévu (sans écrire), utile pour tests."""

        if self.export_config is None:
            raise ValueError("export_config requis")
        if self._analysis is None:
            self.read_tags()
            self.load_audio()
            self.analyze_file()
        if not self._tags_resolved:
            self.tag_file()
        from musikalize.export_ffmpeg import build_output_path

        meta = self._analysis.meta if self._analysis else {}
        return self.export_config.output_root / build_output_path(
            self.export_config.path_template,
            self._tags_resolved,
            meta,
            ext,
            sanitize=self.export_config.sanitize_paths,
        )

    def format_preview(self, template: str) -> str:
        """Résout un gabarit arbitraire avec l'état courant (tags + méta)."""

        if not self._tags_prefixed:
            self.read_tags()
        if self._analysis is None:
            raise ValueError("analyze_file() requis pour les méta-données")
        m = build_format_mapping(self._tags_prefixed, self._analysis.meta, ext=None)
        return resolve_template(template, m)
