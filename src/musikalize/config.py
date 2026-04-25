"""Typed configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, Union
import json

from musikalize.analysis_ops import (
    main_sub_from_label,
    meta_key_base,
)


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
    embedder_name: str
    graph_path: Path
    labels_path: Path | None = None
    label_names: Sequence[str] | None = None
    category: Literal["genre", "mood", "other"] = "other"

    input_tensor: str = "model/Placeholder"
    output_tensor: str = "model/Sigmoid"
    kind: Literal["TfPred", "TfPred2D"] = "TfPred2D"
    task: Literal["classification", "regression", "multilabel"] = "classification"

    separator: str = ";"
    count: int = 1
    thold: float = 1.0
    count_thold_policy: Literal["intersection", "union"] = "union"

@dataclass
class PredictionRecord:
    """Information predicted by the extractor"""
    name: str
    category: Literal["genre", "mood", "classical", "other"]
    labels: list[str]
    scores: list[float]
    top_label: list[str]
    top_score: list[float]
    sep: str

    @property
    def flat_meta_from_record(self) -> dict[str, Any]:
        base = meta_key_base(self)
        if self.category == "classical":
            return {base: self.top_label}

        out: dict[str, Any] = {
            f"{base}_val": self.top_score,
            f"{base}_val_pct": self._pct(self.top_score),
            f"{base}_dict": self._dict(self.top_label, self.top_score),
            f"{base}_dict_pct": self._dict(self.top_label, self._pct(self.top_score)),
            f"{base}_all": self._dict(self.labels, self.scores),
            f"{base}_all_pct": self._dict(self.labels, self._pct(self.scores)),
            base: self.top_label
        }

        if self.category == "genre":
            mains: list[str] = []
            subs: list[str] = []
            for lab in self.top_label:
                m, s = main_sub_from_label(lab, ("---", "//"))
                if m and m not in mains:
                    mains.append(m)
                if s and s not in subs:
                    subs.append(s)
            out[f"{base}_main"] = self.sep.join(mains)
            out[f"{base}_sub"] = self.sep.join(subs)
            out[base] = mains + subs
        out.update(self._stringify(out))
        return out

    def _dict(self, labels: list[str], scores: Union[list[float], list[int]]) -> dict[str, Any]:
        return {
            labels[i]: scores[i]
            for i in range(min(len(labels), len(scores)))
        }

    def _pct(self, value: list[float])-> list[int]:
        return [int(round(n*100)) for n in value]

    def _stringify(self, dictionary: Dict[str, Any]) -> Dict[str, str]:
        out = {}
        for item in dictionary:
            if type(dictionary[item]) is str:
                out[f"{item}_str"] = dictionary[item]
            elif type(dictionary[item]) is list:
                out[f"{item}_str"] = self.sep.join([str(var) for var in dictionary[item]])
            else:
                json.dumps(dictionary[item], ensure_ascii=False)
        return(out)


@dataclass(slots=True)
class TaggingConfig:
    """Per-field templates using `{tag_*}` and `{meta_*}` (Python ``str.format`` syntax only)."""

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
