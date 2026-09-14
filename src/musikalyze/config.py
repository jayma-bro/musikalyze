"""Typed configuration objects."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from musikalyze.analysis_ops import (
    main_sub_from_label,
    meta_key_base,
    pct,
)


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """Embedding model (EffNet or MAEST)."""

    embedding_model: Path
    name: Literal["effnet", "maest"] = "effnet"

    input_tensor: str | None = None
    patch_size: int | None = None
    patch_hop_size: int | None = None
    batch_size: int | None = None
    """MAEST patch/batch settings; leave unset to let Essentia infer from the graph path."""


@dataclass(slots=True)
class LabelExtractor:
    """Classification / regression head on top of a named embedding.

    ``label_names`` can be either:

    * ``Sequence[str]`` – classification / multilabel mode (one label per output neuron).
    * ``dict[str, tuple[float, float]]`` – regression mode with **threshold mapping**.
      Each value is ``(low, high)`` defining the score range for that label.
      When a regression score falls within a range, the corresponding label is assigned.
      On overlap the *lower* label wins; exact boundary values round to the lower label.
    """

    name: str
    embedder_name: str
    graph_path: Path
    labels_path: Path | None = None
    label_names: Sequence[str] | dict[str, tuple[int, int]] | None = None
    category: Literal["genre", "mood", "other"] = "other"

    input_tensor: str = "model/Placeholder"
    output_tensor: str = "model/Sigmoid"
    task: Literal["classification", "regression", "multilabel"] = "classification"

    separator: str = ";"
    count: int = 1
    # Public thresholds are percentages, from 0 to 100.
    thold: int = 100
    count_thold_policy: Literal["intersection", "union"] = "intersection"

    def __post_init__(self) -> None:
        if not 0 <= self.thold <= 100:
            raise ValueError("thold must be an integer percentage between 0 and 100")
        if isinstance(self.label_names, dict):
            for label, bounds in self.label_names.items():
                if len(bounds) != 2 or not all(isinstance(v, int) for v in bounds):
                    raise TypeError(f"Bounds for {label!r} must be a pair of integer percentages")
                low, high = bounds
                if not 0 <= low <= high <= 100:
                    raise ValueError(f"Bounds for {label!r} must be within 0..100")

@dataclass
class PredictionRecord:
    """Information predicted by the extractor"""
    name: str
    category: Literal["genre", "mood", "other"]
    labels: list[str]
    scores: list[float]
    top_label: list[str]
    top_score: list[float]
    sep: str

    @property
    def flat_meta_from_record(self) -> dict[str, Any]:
        base = meta_key_base(self)

        out: dict[str, Any] = {
            f"{base}_val": self.top_score,
            f"{base}_val_pct": pct(self.top_score),
            f"{base}_dict": self._dict(self.top_label, self.top_score),
            f"{base}_dict_pct": self._dict(self.top_label, pct(self.top_score)),
            f"{base}_all": self._dict(self.labels, self.scores),
            f"{base}_all_pct": self._dict(self.labels, pct(self.scores)),
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

    def _dict(self, labels: list[str], scores: list[float] | list[int]) -> dict[str, Any]:
        return {
            labels[i]: scores[i]
            for i in range(min(len(labels), len(scores)))
        }

    def _stringify(self, dictionary: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for item, value in dictionary.items():
            if type(value) is str:
                out[f"{item}_str"] = value
            elif type(value) is list:
                out[f"{item}_str"] = self.sep.join([str(var) for var in value])
            else:
                out[f"{item}_str"] = json.dumps(value, ensure_ascii=False)
        return out


@dataclass(slots=True)
class TaggingConfig:
    """Explicit tag templates.

    ``tags`` contains standard logical file tags and ``extra`` contains arbitrary
    custom tags. An empty mapping means that the corresponding original tags are
    left untouched during export.
    """

    separator: str = ";"
    tags: dict[str, str | None] = field(default_factory=dict)
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
    retag: bool = False
    # Deprecated spelling retained only to load existing serialized configs.
    mode: Literal["transcode", "retag"] | None = field(default=None, repr=False)
    _template_warning: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode is not None:
            self.retag = self.mode == "retag"
        else:
            self.mode = "retag" if self.retag else "transcode"
        if not re.search(r"\{(tag_|meta_)", self.path_template):
            object.__setattr__(self, "_template_warning", True)


@dataclass(slots=True)
class AnalysisResult:
    """Container for resolved metadata passed to tag templates."""

    meta: dict[str, Any] = field(default_factory=dict)
