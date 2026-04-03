"""Configurations et résultats d'analyse typés."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


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
class AnalysisConfig:
    """Paramètres d'analyse Essentia / TF et post-traitement genre."""

    genre_main: bool = True
    """Si True, découpe le label principal sur `genre_separators` (ex. Rock---Post-Punk → Rock + Post-Punk)."""

    genre_count: int | None = 5
    """Nombre max de genres secondaires (hors « principaux » dédupliqués), après seuil éventuel."""

    genre_thold: float | None = None
    """Si défini, ne garde que les genres avec probabilité ≥ ce seuil (puis applique `genre_count`)."""

    genre_join_separator: str = ";"
    """Séparateur pour la chaîne `meta_genre` et listes affichées."""

    genre_separators: tuple[str, ...] = ("---", "//")
    """Séparateurs reconnus dans le label brut du modèle (ordre de test)."""

    bpm: bool = True
    key: bool = True
    scale: bool = True
    danceability_classical: bool = False
    """Danceability via algorithme classique Essentia (pas un modèle TF)."""


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
    """Autres clés logiques (albumartist, …) → gabarit."""


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
    """Clés plates pour `meta_*` : ex. meta_genre, meta_bpm, meta_key, meta_mood_happy, …"""

    genre_labels: list[str] = field(default_factory=list)
    """Liste finale de genres (principaux + secondaires selon config)."""

    genre_scores: dict[str, float] = field(default_factory=dict)
    """Label → probabilité pour les genres retenus ou top complet selon implémentation."""

    head_predictions: dict[str, Any] = field(default_factory=dict)
    """Par nom de tête : probas, label top-1, etc."""

    bpm: float | None = None
    key_edma: str | None = None
    scale_edma: str | None = None
    danceability: float | None = None

    embeddings: Any = None
    """Tableau numpy d'embeddings (optionnel, peut être volumineux)."""
