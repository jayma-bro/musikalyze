"""Configurations et résultats d'analyse typés."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union

# ---------------------------------------------------------------------------
# Nouvelle API : embeddings multiples + extracteurs de labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Modèle d'embedding (EffNet Discogs, MAEST, graphe TF générique)."""

    name: str
    embedding_model: Path
    embedding_output: str = "PartitionedCall:1"
    backend: Literal["effnet_discogs", "maest", "generic_tf"] = "effnet_discogs"
    input_tensor: str | None = None
    """Pour ``generic_tf`` / MAEST : entrée du graphe (ex. ``serving_default_input_1``)."""


@dataclass(slots=True)
class LabelExtractor:
    """Tête de prédiction (genre, mood, autre) branchée sur un embedding nommé."""

    name: str
    category: Literal["genre", "mood", "other"]
    embedder: Union[str, EmbeddingModel]
    graph_path: Path
    labels_path: Path | None = None
    label_names: Sequence[str] | None = None
    """Remplace les libellés du JSON (même longueur que les classes ou tronqué)."""

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


# ---------------------------------------------------------------------------
# API historique
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassificationHeadSpec:
    """Spécification d'un classifieur 2D (embeddings → logits/probas)."""

    name: str
    graph_path: Path
    labels_path: Path | None = None
    input_tensor: str = "serving_default_model_Placeholder"
    output_tensor: str = "PartitionedCall:0"


@dataclass(frozen=True, slots=True)
class ModelPath:
    """Chemins vers le modèle d'embedding EffNet et les têtes optionnelles."""

    embedding_model: Path
    embedding_output: str = "PartitionedCall:1"
    heads: Sequence[ClassificationHeadSpec] = ()


@dataclass(slots=True)
class ClassicalConfig:
    """Descripteurs classiques Essentia (calculés à la demande)."""

    bpm: bool = True
    key: bool = True
    scale: bool = True
    danceability_classical: bool = False


@dataclass(slots=True)
class AnalysisConfig:
    """Paramètres d'analyse Essentia / TF (rétrocompat) et post-traitement genre."""

    genre_main: bool = True
    genre_count: int | None = 5
    genre_thold: float | None = None
    genre_join_separator: str = ";"
    genre_separators: tuple[str, ...] = ("---", "//")
    bpm: bool = True
    key: bool = True
    scale: bool = True
    danceability_classical: bool = False


@dataclass(slots=True)
class TaggingConfig:
    """Gabarits de tags : clés logiques → chaînes avec `{tag_*}` et `{meta_*}`."""

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
    """Export audio et gabarit de chemin."""

    output_root: Path
    formats: str | list[str] = "opus"
    path_template: str = "{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}"
    format_options: dict[str, dict[str, str]] = field(default_factory=dict)
    sanitize_paths: bool = True
    overwrite: bool = False


@dataclass(slots=True)
class AnalysisResult:
    """Résultat d'analyse : tenseurs agrégés pour gabarits et inspection."""

    meta: dict[str, Any] = field(default_factory=dict)
    genre_labels: list[str] = field(default_factory=list)
    genre_scores: dict[str, float] = field(default_factory=dict)
    head_predictions: dict[str, Any] = field(default_factory=dict)
    bpm: float | None = None
    key_edma: str | None = None
    scale_edma: str | None = None
    danceability: float | None = None
    embeddings: Any = None
