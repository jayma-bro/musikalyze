"""Typed configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Embedding model (EffNet Discogs or MAEST)."""

    name: str
    embedding_model: Path
    embedding_output: str = "PartitionedCall:1"
    backend: Literal["effnet_discogs", "maest"] = "effnet_discogs"
    input_tensor: str | None = None
    """MAEST only: TensorFlow input node name (default inside Essentia: mel patch input)."""

    patch_size: int | None = None
    patch_hop_size: int | None = None
    batch_size: int | None = None
    """MAEST patch/batch settings; leave unset to let Essentia infer from the graph path."""


@dataclass(slots=True)
class LabelExtractor:
    """Classification / regression head on top of a named embedding."""

    name: str
    category: Literal["genre", "mood", "other"]
    embedder: Union[str, EmbeddingModel]
    graph_path: Path
    labels_path: Path | None = None
    label_names: Sequence[str] | None = None

    input_tensor: str = "serving_default_model_Placeholder"
    output_tensor: str = "PartitionedCall:0"
    task: Literal["classification", "regression", "multilabel"] = "classification"

    genre_main: bool = True
    genre_count: int | None = 5
    genre_thold: float | None = None
    genre_separators: tuple[str, ...] = ("---", "//")
    genre_join_separator: str = ";"
    count_thold_policy: Literal["intersection", "union"] = "intersection"

    def resolved_embedder_name(self) -> str:
        if isinstance(self.embedder, EmbeddingModel):
            return self.embedder.name
        return str(self.embedder)


@dataclass(slots=True)
class TaggingConfig:
    """Per-field templates using `{tag_*}` and `{meta_*}` (Python ``str.format`` syntax only)."""

    separator: str = ";"
    artist: str | None = "{tag_artist}"
    title: str | None = "{tag_title}"
    album: str | None = "{tag_album}"
    genre: str | None = "{meta_genre}"
    composer: str | None = None
    date: str | None = "{tag_date}"
    tracknumber: str | None = "{tag_tracknumber}"
    discnumber: str | None = "{tag_discnumber}"
    comment: str | None = None
    extra: Mapping[str, str | None] = field(default_factory=dict)


@dataclass(slots=True)
class ExportConfig:
    """Transcoded output and path template."""

    output_root: Path
    formats: str | list[str] = "opus"
    path_template: str = "{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}"
    format_options: dict[str, dict[str, str]] = field(default_factory=dict)
    sanitize_paths: bool = True
    overwrite: bool = False


@dataclass(slots=True)
class AnalysisResult:
    """Container for resolved metadata passed to tag templates."""

    meta: dict[str, Any] = field(default_factory=dict)
