"""Lazy analysis: embeddings (once), per-head predictions, classical descriptors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from musikalize.analysis_ops import (
    load_label_list,
    main_sub_from_label,
    mean_pool_time,
    probs_from_raw,
    select_genre_indices,
    split_genre_label,
)
from musikalize.config import EmbeddingModel, LabelExtractor
from musikalize.exceptions import PredictionError, UnknownEmbedderError

log = logging.getLogger(__name__)

_CLASSICAL_KEYS = frozenset(
    {"meta_bpm", "meta_key", "meta_scale", "meta_danceability"}
)


@dataclass
class PredictionRecord:
    name: str
    category: Literal["genre", "mood", "other"]
    labels: list[str]
    probabilities: Any
    top_label: str
    top_score: float
    genre_strings: list[str] = field(default_factory=list)
    genre_mains: list[str] = field(default_factory=list)
    genre_subs: list[str] = field(default_factory=list)


def compute_embedding(audio: Any, emb: EmbeddingModel) -> Any:
    path = str(Path(emb.embedding_model).resolve())
    if emb.backend == "effnet_discogs":
        from essentia.standard import TensorflowPredictEffnetDiscogs

        return TensorflowPredictEffnetDiscogs(
            graphFilename=path,
            output=emb.embedding_output,
        )(audio)

    from essentia.standard import TensorflowPredictMAEST

    kwargs: dict[str, Any] = {"graphFilename": path, "output": emb.embedding_output}
    if emb.input_tensor is not None:
        kwargs["input"] = emb.input_tensor
    if emb.patch_size is not None:
        kwargs["patchSize"] = emb.patch_size
    if emb.patch_hop_size is not None:
        kwargs["patchHopSize"] = emb.patch_hop_size
    if emb.batch_size is not None:
        kwargs["batchSize"] = emb.batch_size
    return TensorflowPredictMAEST(**kwargs)(audio)


def run_label_head(embeddings: Any, ex: LabelExtractor) -> PredictionRecord:
    import numpy as np
    from essentia.standard import TensorflowPredict2D

    labels: list[str] = []
    if ex.label_names is not None:
        labels = [str(x) for x in ex.label_names]
    elif ex.labels_path is not None:
        labels = load_label_list(Path(ex.labels_path))

    try:
        raw = TensorflowPredict2D(
            graphFilename=str(Path(ex.graph_path).resolve()),
            input=ex.input_tensor,
            output=ex.output_tensor,
        )(embeddings)
    except Exception as e:
        raise PredictionError(f'Head "{ex.name}": {e}') from e

    pooled = mean_pool_time(np.asarray(raw))

    if ex.task == "regression" and pooled.size <= 2:
        val = float(np.ravel(pooled)[0])
        lab = labels[0] if labels else str(val)
        probs = np.array([1.0], dtype=np.float64)
        return PredictionRecord(
            name=ex.name,
            category=ex.category,
            labels=labels or [lab],
            probabilities=probs,
            top_label=str(val),
            top_score=1.0,
        )

    probs = probs_from_raw(pooled)
    if labels and len(labels) != probs.size:
        log.warning(
            'Label count (%d) != probability count (%d) for "%s"; truncating.',
            len(labels),
            probs.size,
            ex.name,
        )
        n = min(len(labels), probs.size)
        labels = labels[:n]
        probs = probs[:n]

    top_i = int(np.argmax(probs)) if probs.size else 0
    top_label = labels[top_i] if labels and top_i < len(labels) else str(top_i)
    top_score = float(probs[top_i]) if probs.size else 0.0

    rec = PredictionRecord(
        name=ex.name,
        category=ex.category,
        labels=labels,
        probabilities=probs,
        top_label=top_label,
        top_score=top_score,
    )

    if ex.category == "genre" and labels:
        idxs = select_genre_indices(
            probs,
            genre_count=ex.genre_count,
            genre_thold=ex.genre_thold,
            policy=ex.count_thold_policy,
        )
        seen: set[str] = set()
        mains: list[str] = []
        subs: list[str] = []
        for i in idxs:
            lab = labels[i] if i < len(labels) else str(i)
            for part in split_genre_label(
                lab,
                genre_main=ex.genre_main,
                separators=ex.genre_separators,
            ):
                if part not in seen:
                    seen.add(part)
                    rec.genre_strings.append(part)
            m, s = main_sub_from_label(lab, ex.genre_separators)
            if m and m not in mains:
                mains.append(m)
            if s and s not in subs:
                subs.append(s)
        rec.genre_mains = mains
        rec.genre_subs = subs

    return rec


def meta_key_for_extractor(ex: LabelExtractor) -> str:
    if ex.category == "mood":
        return f"meta_mood_{ex.name}"
    if ex.category == "genre":
        return f"meta_genre_{ex.name}"
    return f"meta_{ex.name}"


def flat_meta_from_record(ex: LabelExtractor, rec: PredictionRecord, list_sep: str) -> dict[str, Any]:
    base = meta_key_for_extractor(ex)
    dprob = {
        rec.labels[i]: float(rec.probabilities[i])
        for i in range(min(len(rec.labels), len(rec.probabilities)))
    }
    d_json = json.dumps(dprob, ensure_ascii=False)

    out: dict[str, Any] = {
        f"{base}_val": rec.top_score,
        f"{base}_val_str": str(rec.top_score),
        f"{base}_dict": dprob,
        f"{base}_dict_str": d_json,
        f"{base}_all": dprob,
        f"{base}_all_str": d_json,
    }

    if ex.category == "genre":
        joined = ex.genre_join_separator.join(rec.genre_strings) if rec.genre_strings else rec.top_label
        out[base] = joined
        out[f"{base}_main"] = ex.genre_join_separator.join(rec.genre_mains)
        out[f"{base}_sub"] = ex.genre_join_separator.join(rec.genre_subs)
    else:
        out[base] = rec.top_label

    return out


class LazyMetaEngine:
    """Embeddings are computed once via ``compute_all_embeddings()``; heads and classical features are lazy."""

    def __init__(
        self,
        audio: Any,
        embedders: Mapping[str, EmbeddingModel],
        extractors: Sequence[LabelExtractor],
        *,
        list_join_sep: str = ";",
    ) -> None:
        self._audio = audio
        self._embedders = dict(embedders)
        self._extractors = list(extractors)
        self._by_name = {e.name: e for e in extractors}
        self._list_join_sep = list_join_sep

        self._emb: dict[str, Any] = {}
        self._pred: dict[str, PredictionRecord] = {}
        self._classical_cache: dict[str, Any] = {}

    def _embedder_model(self, ex: LabelExtractor) -> EmbeddingModel:
        name = ex.resolved_embedder_name()
        if name not in self._embedders:
            known = ", ".join(sorted(self._embedders)) or "(none)"
            raise UnknownEmbedderError(
                f'Unknown embedding "{name}" for extractor "{ex.name}". Known: {known}'
            )
        return self._embedders[name]

    def compute_all_embeddings(self) -> None:
        """Run every registered embedding model once (call from ``MusicProcess.analyze_file()``)."""

        for name, model in self._embedders.items():
            if name not in self._emb:
                self._emb[name] = compute_embedding(self._audio, model)

    def embedding(self, embedder_name: str) -> Any:
        if embedder_name not in self._emb:
            raise RuntimeError(
                f'Embedding "{embedder_name}" is missing. Call analyze_file() first to compute embeddings.'
            )
        return self._emb[embedder_name]

    def ensure_prediction(self, extractor_name: str) -> PredictionRecord:
        if extractor_name not in self._pred:
            ex = self._by_name.get(extractor_name)
            if ex is None:
                raise UnknownEmbedderError(f'Unknown extractor "{extractor_name}".')
            emb_mod = self._embedder_model(ex)
            emb_tensor = self.embedding(emb_mod.name)
            self._pred[extractor_name] = run_label_head(emb_tensor, ex)
        return self._pred[extractor_name]

    def _ensure_classical_key(self, key: str) -> Any:
        if key in self._classical_cache:
            return self._classical_cache[key]
        audio = self._audio
        if key == "meta_bpm":
            from essentia.standard import PercivalBpmEstimator

            v = int(round(float(PercivalBpmEstimator()(audio))))
        elif key == "meta_key":
            from essentia.standard import KeyExtractor

            k, _scale, _ = KeyExtractor()(audio)
            v = str(k)
        elif key == "meta_scale":
            from essentia.standard import KeyExtractor

            _k, scale, _ = KeyExtractor()(audio)
            v = str(scale)
        elif key == "meta_danceability":
            from essentia.standard import Danceability

            d, _ = Danceability()(audio)
            v = float(d)
        else:
            return None
        self._classical_cache[key] = v
        return v

    def _needs_extractor(self, ex: LabelExtractor, keys: set[str] | None) -> bool:
        if keys is None:
            return True
        base = meta_key_for_extractor(ex)
        if "meta_genre" in keys or "meta_genre_main" in keys or "meta_genre_sub" in keys:
            if ex.category == "genre":
                return True
        if "meta_mood" in keys and ex.category == "mood":
            return True
        for k in keys:
            if k == base or k.startswith(base + "_"):
                return True
        return False

    def build_flat_meta(self, keys_needed: set[str] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = keys_needed

        if keys is None:
            for ck in _CLASSICAL_KEYS:
                val = self._ensure_classical_key(ck)
                if val is not None:
                    out[ck] = val
        else:
            for ck in _CLASSICAL_KEYS:
                if ck in keys:
                    val = self._ensure_classical_key(ck)
                    if val is not None:
                        out[ck] = val

        for ex in self._extractors:
            if not self._needs_extractor(ex, keys):
                continue
            rec = self.ensure_prediction(ex.name)
            out.update(flat_meta_from_record(ex, rec, self._list_join_sep))

        if keys is None or any(
            k in keys for k in ("meta_genre", "meta_genre_main", "meta_genre_sub")
        ):
            genre_ex = [e for e in self._extractors if e.category == "genre"]
            if genre_ex:
                all_g: list[str] = []
                all_m: list[str] = []
                all_s: list[str] = []
                for ex in genre_ex:
                    rec = self.ensure_prediction(ex.name)
                    all_g.extend(rec.genre_strings)
                    all_m.extend(rec.genre_mains)
                    all_s.extend(rec.genre_subs)
                out["meta_genre"] = self._list_join_sep.join(dict.fromkeys(all_g))
                out["meta_genre_main"] = self._list_join_sep.join(dict.fromkeys(all_m))
                out["meta_genre_sub"] = self._list_join_sep.join(dict.fromkeys(all_s))

        if keys is None or "meta_mood" in keys:
            mood_ex = [e for e in self._extractors if e.category == "mood"]
            if mood_ex:
                tops = [self.ensure_prediction(e.name).top_label for e in mood_ex]
                out["meta_mood"] = self._list_join_sep.join(tops)

        return out

    def get_one_meta(self, key: str) -> Any:
        k = key if key.startswith("meta_") else f"meta_{key}"
        m = self.build_flat_meta({k})
        if k not in m:
            m = self.build_flat_meta(None)
        return m.get(k)
