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
    merge_values,
    stringify,
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
    top_label: list[str]
    top_score: list[float]
    genre_strings: list[str] = field(default_factory=list)
    genre_mains: list[str] = field(default_factory=list)
    genre_subs: list[str] = field(default_factory=list)


    def genre_extract(self, suffix: str | None = None, format: Literal["label", "score", "full"] = "label"):
        print("ok")
        

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
    from essentia import Pool
    from essentia.standard import TensorflowPredict2D, TensorflowPredict

    raw_labels: list[str] = []
    if ex.label_names is not None:
        raw_labels = [str(x) for x in ex.label_names]
    elif ex.labels_path is not None:
        raw_labels = load_label_list(Path(ex.labels_path))

    if ex.kind == "TfPred":
        pool = Pool()
        pool.set(ex.input_tensor, embeddings)
        try:
            raw = TensorflowPredict(
                graphFilename=str(Path(ex.graph_path).resolve()),
                inputs=[ex.input_tensor],
                outputs=[ex.output_tensor],
            )(pool)[ex.output_tensor]
        except Exception as e:
            raise PredictionError(f'Head "{ex.name}" with {ex.kind}: {e}') from e
    elif ex.kind == "TfPred2D":
        try:
            raw = TensorflowPredict2D(
                graphFilename=str(Path(ex.graph_path).resolve()),
                input=ex.input_tensor,
                output=ex.output_tensor,
            )(embeddings)
        except Exception as e:
            raise PredictionError(f'Head "{ex.name}" with {ex.kind}: {e}') from e
    else:
        raise Exception("Extractor's kind not found")

    pooled = mean_pool_time(np.asarray(raw))
    probabilities = []
    top_label = []
    top_score = []
    if ex.task == "regression" and pooled.size <= 2:
        probabilities = [float(pooled[0])]
        index = int(min(int(prob * len(raw_labels)), n - 1))
        top_label=[raw_labels[index]]
        top_score=[1.0]

    probs = probs_from_raw(pooled)
    order = np.argsort(-probs)
    if raw_labels and len(raw_labels) != probs.size:
        log.warning(
            'Label count (%d) != probability count (%d) for "%s"; truncating.',
            len(raw_labels),
            probs.size,
            ex.name,
        )
        n = min(len(raw_labels), probs.size)
        raw_labels = raw_labels[:n]
        probs = probs[:n]

    if ex.task == "classification":
        top_i = int(np.argmax(probs))
        labels=[raw_labels[i] for i in order]
        probabilities=probs[order].tolist()
        top_label=[raw_labels[top_i]]
        top_score=[float(probs[top_i])]
    elif ex.task == "multilabel":
        thold_count = len([i for i in order if probs[i] >= ex.thold])
        idxs = max(ex.count, thold_count) if ex.count_thold_policy == "union" else min(ex.count, thold_count)
        top_order = order[:idxs]
        labels=[raw_labels[i] for i in order]
        probabilities=probs[order].tolist()
        top_label=[raw_labels[i] for i in top_order]
        top_score=probs[top_order].tolist()

    return PredictionRecord(
            name=ex.name,
            category=ex.category,
            labels=labels,
            probabilities=probabilities,
            top_label=top_label,
            top_score=top_score,
        )


def meta_key_for_extractor(ex: LabelExtractor) -> str:
    if ex.category == "mood":
        return f"meta_mood_{ex.name}"
    if ex.category == "genre":
        return f"meta_genre_{ex.name}"
    return f"meta_{ex.name}"


def flat_meta_from_record(ex: LabelExtractor, rec: PredictionRecord) -> dict[str, Any]:
    base = meta_key_for_extractor(ex)
    dprob = {
        rec.top_label[i]: float(rec.top_score[i])
        for i in range(min(len(rec.top_label), len(rec.top_score)))
    }
    dprob_all = {
        rec.labels[i]: float(rec.probabilities[i])
        for i in range(min(len(rec.labels), len(rec.probabilities)))
    }
    out: dict[str, Any] = {
        f"{base}_val": rec.top_score,
        f"{base}_dict": dprob,
        f"{base}_all": dprob_all,
        base: rec.top_label
    }

    if ex.category == "genre":
        mains: list[str] = []
        subs: list[str] = []
        for lab in rec.top_label:
            m, s = main_sub_from_label(lab, ex.genre_separators)
            if m and m not in mains:
                mains.append(m)
            if s and s not in subs:
                subs.append(s)
        out[f"{base}_main"] = mains
        out[f"{base}_sub"] = subs
        out[base] = mains + subs
    out.update(stringify(out))
    return out


class LazyMetaEngine:
    """Embeddings are computed once via ``compute_all_embeddings()``; heads and classical features are lazy."""

    def __init__(
        self,
        audio: Any,
        embedders: Mapping[str, EmbeddingModel],
        extractors: Sequence[LabelExtractor],
        *,
        audio_path: Path,
    ) -> None:
        self._audio = audio
        self._embedders = dict(embedders)
        self._extractors = list(extractors)
        self._extractors_by_name = {e.name: e for e in extractors}

        self._emb: dict[str, Any] = {}
        self._extractors_pred: dict[str, PredictionRecord] = {}
        self._classical_cache: dict[str, Any] = {}
        self._audio_path: Path = audio_path

    def _embedder_model(self, ex: LabelExtractor) -> EmbeddingModel:
        name = ex.embedder_name
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
    
    def compute_embedding(self, embedder_name: str) -> None:
        """Run a signe embedding model define by the name"""
        if embedder_name not in self._emb:
            for name, model in self._embedders.items():
                if embedder_name == name:
                    self._emb[name] = compute_embedding(self._audio, model)
                    return
            raise ValueError(f"embedding model named {embedder_name} is not found")

    def embedding(self, embedder_name: str) -> Any:
        if embedder_name not in self._emb:
            self.compute_embedding(embedder_name)
        return self._emb[embedder_name]

    def ensure_prediction(self, extractor_name: str) -> PredictionRecord:
        if extractor_name not in self._extractors_pred:
            ex = self._extractors_by_name.get(extractor_name)
            if ex is None:
                raise UnknownEmbedderError(f'Unknown extractor "{extractor_name}".')
            emb_mod = self._embedder_model(ex)
            emb_tensor = self.embedding(emb_mod.name)
            self._extractors_pred[extractor_name] = run_label_head(emb_tensor, ex)
        return self._extractors_pred[extractor_name]

    def _ensure_classical_key(self, key: str) -> Any:
        if key in self._classical_cache:
            return self._classical_cache[key]
        audio = self._audio
        if key == "meta_bpm":
            from essentia.standard import RhythmExtractor2013, MonoLoader
            new_audio = MonoLoader(filename=str(self._audio_path.resolve()))()
            bpm, beats, beats_confidence, _, beats_intervals = RhythmExtractor2013(method="multifeature")(new_audio)
            v = int(round(float(bpm)))
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

    def _needs_extractor(self, ex: LabelExtractor, key: str | None) -> bool:
        if key is None:
            return True
        base = meta_key_for_extractor(ex)
        if key == base or key.startswith(base + "_"):
            return True
        return False

    def build_flat_meta(self, key: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if key is None:
            for ck in _CLASSICAL_KEYS:
                val = self._ensure_classical_key(ck)
                if val is not None:
                    out[ck] = val
        else:
            for ck in _CLASSICAL_KEYS:
                if ck in key:
                    val = self._ensure_classical_key(ck)
                    if val is not None:
                        out[ck] = val

        for ex in self._extractors:
            if self._needs_extractor(ex, key):
                rec = self.ensure_prediction(ex.name)
                out.update(flat_meta_from_record(ex, rec))
        
        genres_base = "meta_genres"
        if key is None or key.startswith(genres_base):
            genres_ex = [e for e in self._extractors if e.category == "genre"]
            if genres_ex:
                genres_dict = self._meta_extractor(genres_ex, genres_base)
                out.update(genres_dict)
                
        moods_base = "meta_moods"
        if key is None or key.startswith(moods_base):
            moods_ex = [e for e in self._extractors if e.category == "mood"]
            if moods_ex:
                moods_dict = self._meta_extractor(moods_ex, moods_base)
                out.update(moods_dict)

        return out
    
    def _meta_extractor(self, extractors: Sequence[LabelExtractor], meta_base: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ex in extractors:
            rec = self.ensure_prediction(ex.name)
            base = meta_key_for_extractor(ex)
            active_ex = flat_meta_from_record(ex, rec)
            for item in active_ex:
                if item.endswith("_str"):
                    continue
                suffix = item.replace(base, "")
                base_suffix = meta_base + suffix
                value = active_ex[item]
                if base_suffix not in out:
                    out[base_suffix] = value
                else:
                    try:
                        out[base_suffix] = merge_values(out[base_suffix], value)
                    except TypeError as e:
                        print(f"Error for {base_suffix} : {e}")
                        continue
        out.update(stringify(out))
        return out

    def get_one_meta(self, key: str) -> Any:
        m = self.build_flat_meta(key)
        if key not in m:
            m = self.build_flat_meta(None)
        return m.get(key)
