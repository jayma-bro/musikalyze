"""Lazy analysis: embeddings (once), per-head predictions, classical descriptors."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from musikalyze.analysis_ops import (
    load_label_list,
    mean_pool_time,
    merge_values,
    meta_key_base,
    pct,
)
from musikalyze.audio_io import load_audio
from musikalyze.config import EmbeddingModel, LabelExtractor, PredictionRecord
from musikalyze.exceptions import PredictionError, UnknownEmbedderError

log = logging.getLogger(__name__)

_CLASSICAL_KEYS = frozenset(
    {"meta_bpm", "meta_key", "meta_scale", "meta_rgain_gain", "meta_rgain_peak", "meta_rgain_peak_dbfs"}
)
class LazyMetaEngine:
    """Embeddings are computed once via ``compute_all_embeddings()``; heads and classical features are lazy."""

    def __init__(
        self,
        audio: Any,
        embedders: Mapping[str, EmbeddingModel],
        extractors: Mapping[str, LabelExtractor],
        *,
        audio_path: Path,
        sep: str,
    ) -> None:
        self.sep: str = sep
        self._audio: Any = audio
        self._embedders = dict(embedders)
        self._extractors = dict(extractors)

        self._emb: dict[str, Any] = {}
        self._pred: dict[str, PredictionRecord] = {}
        self._audio_path: Path = audio_path
        self._flat_meta_cache: dict[str, Any] | None = None
        self._classical_meta: dict[str, Any] = {}
        self._stereo_cache: tuple[Any, int] | None = None

    def cleanup(self) -> None:
        """Release resources and free memory."""
        import gc
        self._emb.clear()
        self._pred.clear()
        self._audio = None
        self._stereo_cache = None
        self._flat_meta_cache = None
        self._classical_meta.clear()
        gc.collect()

    def __del__(self) -> None:
        """Cleanup when object is garbage collected."""
        try:
            self.cleanup()
        except Exception:  # noqa: BLE001, S110
            pass

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

        for name in self._embedders:
            if name not in self._emb:
                self._emb[name] = self.compute_embedding(name)

    def compute_embedding(self, embedder_name: str) -> None:
        """Run a signe embedding model define by the name"""
        if embedder_name not in self._emb:
            if embedder_name in self._embedders:
                emb = self._embedders[embedder_name]
                path = str(Path(emb.embedding_model).resolve())
                if emb.name == "effnet":
                    from essentia.standard import TensorflowPredictEffnetDiscogs

                    self._emb[embedder_name] = TensorflowPredictEffnetDiscogs(
                        graphFilename=path,
                        output="PartitionedCall:1",
                    )(self._audio)
                    return
                elif emb.name == "maest":
                    from essentia.standard import TensorflowPredictMAEST

                    kwargs: dict[str, Any] = {"graphFilename": path, "output": "PartitionedCall/Identity_12"}
                    if emb.input_tensor is not None:
                        kwargs["input"] = emb.input_tensor
                    if emb.patch_size is not None:
                        kwargs["patchSize"] = emb.patch_size
                    if emb.patch_hop_size is not None:
                        kwargs["patchHopSize"] = emb.patch_hop_size
                    if emb.batch_size is not None:
                        kwargs["batchSize"] = emb.batch_size
                    self._emb[embedder_name] = TensorflowPredictMAEST(**kwargs)(self._audio)
                    return
            raise ValueError(f"embedding model named {embedder_name} is not found")

    def compute_all_extractor(self) -> None:
        _ = self._ensure_classical_key(None)
        for ex_name in self._extractors:
            _ = self.ensure_prediction(ex_name)

    def embedding(self, embedder_name: str) -> Any:
        if embedder_name not in self._emb:
            self.compute_embedding(embedder_name)
        return self._emb[embedder_name]

    def ensure_prediction(self, extractor_name: str) -> PredictionRecord:
        if extractor_name not in self._pred:
            self._pred[extractor_name] = self.run_label_head(extractor_name)
        return self._pred[extractor_name]

    def run_label_head(self, extractor_name: str) -> PredictionRecord:
        import numpy as np
        from essentia import Pool
        from essentia.standard import TensorflowPredict, TensorflowPredict2D

        ex = self._extractors.get(extractor_name)
        if ex is None:
            raise UnknownEmbedderError(f'Unknown extractor "{extractor_name}".')
        emb_mod = self._embedder_model(ex)
        embeddings = self.embedding(emb_mod.name)
        raw_labels: list[str] = []
        if ex.label_names is not None:
            if isinstance(ex.label_names, dict):
                # Regression threshold mapping: {label: (low, high)}
                raw_labels = list(ex.label_names.keys())
            else:
                raw_labels = [str(x) for x in ex.label_names]
        elif ex.labels_path is not None:
            try:
                raw_labels = load_label_list(Path(ex.labels_path), extractor_name=ex.name)
            except ValueError as e:
                raise PredictionError(
                    f'Extractor "{ex.name}" (eexmbedder={ex.embedder_name}): {e}'
                ) from e

        if ex.embedder_name == "maest":
            pool = Pool()
            pool.set(ex.input_tensor, embeddings)
            try:
                raw = TensorflowPredict(
                    graphFilename=str(Path(ex.graph_path).resolve()),
                    inputs=[ex.input_tensor],
                    outputs=[ex.output_tensor],
                )(pool)[ex.output_tensor]
            except Exception as e:
                raise PredictionError(f'Head "{ex.name}" with {ex.embedder_name}: {e}') from e
        elif ex.embedder_name == "effnet":
            outputs = [ex.output_tensor]
            if ex.task == "regression" and ex.output_tensor != "model/Identity":
                outputs.append("model/Identity")
            last_error: Exception | None = None
            for output_tensor in outputs:
                try:
                    raw = TensorflowPredict2D(
                        graphFilename=str(Path(ex.graph_path).resolve()),
                        input=ex.input_tensor,
                        output=output_tensor,
                    )(embeddings)
                    break
                except Exception as error:  # noqa: BLE001
                    last_error = error
            else:
                raise PredictionError(
                    f'Head "{ex.name}" with {ex.embedder_name}: {last_error}'
                ) from last_error
        else:
            raise UnknownEmbedderError(f"Extractor '{extractor_name}' references unknown embedder")

        pooled = mean_pool_time(np.asarray(raw))
        labels = []
        scores = []
        top_label = []
        top_score = []

        # Handle regression with threshold mapping (dict label_names)
        if isinstance(ex.label_names, dict) and pooled.size > 0:
            # Regression threshold mapping mode
            score = round(float(pooled[0]) * 100) if pooled.size > 0 else 0
            thresholds: dict[str, tuple[int, int]] = ex.label_names

            # Find the label with matching range
            matching_labels = []

            # Check each label to see if score fits in its range
            for label_name, (low, high) in sorted(thresholds.items(), key=lambda item: item[1][0]):
                if low <= score <= high:
                    matching_labels.append(label_name)

            scores = top_score = [pooled[0]]
            # Return empty list if no matches (to indicate no label assigned)
            if matching_labels:
                labels = top_label = [matching_labels[0]]  # Return first match
            else:
                # When no matching label, return empty list to indicate no label assigned
                labels = top_label = []
        elif ex.task == "regression" and pooled.size <= 2:
            scores = [float(pooled[0])]
            index = int(min(int((1.0 - scores[0]) * len(raw_labels)), len(raw_labels) - 1))
            labels=[raw_labels[index]]
            top_label=labels
            top_score=[scores[0]]

        order = np.argsort(-pooled)
        if (raw_labels and len(raw_labels) != pooled.size) and ex.task != "regression":
            log.warning(
                'Label count (%d) != probability count (%d) for "%s"; truncating.',
                len(raw_labels),
                pooled.size,
                ex.name,
            )
            n = min(len(raw_labels), pooled.size)
            raw_labels = raw_labels[:n]
            pooled = pooled[:n]

        if ex.task == "classification":
            top_i = int(np.argmax(pooled))
            labels=[raw_labels[i] for i in order]
            scores=pooled[order].tolist()
            top_label=[raw_labels[top_i]]
            top_score=[float(pooled[top_i])]
        elif ex.task == "multilabel":
            threshold = ex.thold / 100.0
            thold_count = len([i for i in order if pooled[i] >= threshold])
            idxs = max(ex.count, thold_count) if ex.count_thold_policy == "union" else min(ex.count, thold_count)
            top_order = order[:idxs]
            labels=[raw_labels[i] for i in order]
            scores=pooled[order].tolist()
            top_label=[raw_labels[i] for i in top_order]
            top_score=pooled[top_order].tolist()

        return PredictionRecord(
                name=ex.name,
                category=ex.category,
                labels=labels,
                scores=[round(n, 2) for n in scores],
                top_label=top_label,
                top_score=[round(n, 2) for n in top_score],
                sep=ex.separator,
            )

    def _ensure_classical_key(self, key: str | None) -> dict[str, Any]:
        requested = set(_CLASSICAL_KEYS) if key is None else {
            name for name in _CLASSICAL_KEYS if key == name or key.startswith(name + "_")
        }
        missing = requested - self._classical_meta.keys()
        if not missing:
            return dict(self._classical_meta)

        pred_list = []
        if key is None or key.startswith("meta_bpm"):
            from essentia.standard import RhythmExtractor2013
            new_audio, _ = load_audio(str(self._audio_path.resolve()))
            bpm, _beats, _beats_confidence, _, _beats_intervals = RhythmExtractor2013(method="multifeature")(new_audio)
            pred_list.append({
                "name": "bpm",
                "labels": round(float(bpm)),
            })
        if key is None or key == "meta_key" or key == "meta_scale":
            from essentia.standard import KeyExtractor

            k, scale, _ = KeyExtractor()(self._audio)
            pred_list.append({
                "name": "key",
                "labels": k,
            })
            pred_list.append({
                "name": "scale",
                "labels": scale,
            })
        if key is None or key in ["meta_rgain_gain", "meta_rgain_peak", "meta_rgain_peak_dbfs"]:
            from essentia.standard import LoudnessEBUR128
            if self._stereo_cache is None:
                audio_stereo, sample_rate = load_audio(str(self._audio_path.resolve()), track="stereo")
                self._stereo_cache = (audio_stereo, sample_rate)
            else:
                audio_stereo, sample_rate = self._stereo_cache
            _momentary, _short_term, integrated, _loudness_range = LoudnessEBUR128(
                sampleRate=sample_rate,
                hopSize=0.1,
                startAtZero=False
            )(audio_stereo)
            REPLAYGAIN_TARGET = -18.0
            track_gain = float(REPLAYGAIN_TARGET - integrated)
            peak_value = float(np.max(np.abs(audio_stereo)))
            peak_dbfs = float(20 * np.log10(peak_value) if peak_value > 0 else -np.inf)
            pred_list.append({
                "name": "rgain_gain",
                "labels": track_gain,
            })
            pred_list.append({
                "name": "rgain_peak",
                "labels": peak_value,
            })
            pred_list.append({
                "name": "rgain_peak_dbfs",
                "labels": peak_dbfs,
            })
        for item in pred_list:
            self._classical_meta[f"meta_{item['name']}"] = item["labels"]
        return dict(self._classical_meta)

    def _needs_extractor(self, ex: LabelExtractor, key: str | None) -> bool:
        if key is None:
            return True
        base = meta_key_base(ex)
        return key == base or key.startswith(base + "_")

    def build_flat_meta(self, key: str | Iterable[str] | None = None) -> dict[str, Any]:
        """Flat ``meta_*`` mapping for one key, a collection of keys, or everything (``None``)."""
        if key is None or isinstance(key, str):
            return self._build_flat_meta_one(key)
        out: dict[str, Any] = {}
        for k in key:
            out.update(self._build_flat_meta_one(k))
        return out

    def _build_flat_meta_one(self, key: str | None) -> dict[str, Any]:
        if key is None and self._flat_meta_cache is not None:
            return dict(self._flat_meta_cache)

        out: dict[str, Any] = {}
        if key is None:
            out.update(self._ensure_classical_key(None))
            self.compute_all_extractor()
            for pred in self._pred.values():
                out.update(pred.flat_meta_from_record)
        else:
            # Classical descriptors are scalar metadata, not predictions.
            classical = self._ensure_classical_key(key)
            for ck, value in classical.items():
                if key == ck or key.startswith(ck + "_"):
                    out[ck] = value

            # indivudual prediction
            for ex in self._extractors.values():
                if self._needs_extractor(ex, key):
                    rec = self.ensure_prediction(ex.name)
                    out.update(rec.flat_meta_from_record)

        # grouped prediction
        genres_base = "meta_genres"
        if key is None or key.startswith(genres_base):
            genres_ex = [e for e in self._extractors.values() if e.category == "genre"]
            if genres_ex:
                genres_dict = self._meta_extractor(genres_ex, genres_base)
                out.update(genres_dict)

        moods_base = "meta_moods"
        if key is None or key.startswith(moods_base):
            moods_ex = [e for e in self._extractors.values() if e.category == "mood"]
            if moods_ex:
                moods_dict = self._meta_extractor(moods_ex, moods_base)
                out.update(moods_dict)

        meta_base = "metas"
        if key is None or key.startswith(meta_base):
            full_dict = {}
            self.compute_all_extractor()
            attributes = [
                meta_base,
                f"{meta_base}_pct",
                f"{meta_base}_all",
                f"{meta_base}_all_pct",
                f"{meta_base}_label",
                f"{meta_base}_label_pct",
                f"{meta_base}_label_all",
                f"{meta_base}_label_all_pct",
            ]
            for attribute in attributes:
                if key is None or key.startswith(attribute):
                    full_dict[attribute] = {}
                    for pred in self._pred.values():
                        pred_key = None
                        pred_val = None
                        if len(pred.top_label) == len(pred.top_score) == 1:
                            pred_key = pred.top_label[0] if "_label" in attribute else pred.name
                            pred_val = pred.top_score[0]
                        else:
                            if "_all" in attribute:
                                pred_key = pred.labels
                                pred_val = pred.scores
                            else:
                                pred_key = pred.top_label
                                pred_val = pred.top_score
                        if type(pred_key) == type(pred_val) == list:
                            full_dict[attribute][pred.name] = {
                                pred_key[i]: pct(pred_val[i]) if "_pct" in attribute else pred_val[i]
                                for i in range(min(len(pred_key), len(pred_val)))
                            }
                        else:
                            full_dict[attribute][pred_key] = pct(pred_val) if "_pct" in attribute else pred_val

            out.update(full_dict)

        if key is None:
            self._flat_meta_cache = dict(out)
        return out

    def _meta_extractor(self, extractors: Sequence[LabelExtractor], meta_base: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ex in extractors:
            rec = self.ensure_prediction(ex.name)
            base = meta_key_base(ex)
            active_ex = rec.flat_meta_from_record
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
        if meta_base == "meta_genres":
            # The main genre is the main part of the single highest-scoring
            # complete genre. Subgenres are collected from the selected labels.
            ranked: list[tuple[float, str]] = []
            for ex in extractors:
                rec = self.ensure_prediction(ex.name)
                ranked.extend(zip(rec.top_score, rec.top_label))
            ranked.sort(key=lambda item: item[0], reverse=True)
            if ranked:
                main = ranked[0][1].split("---", 1)[0].strip()
                subs: list[str] = []
                for _, label in ranked:
                    parts = label.split("---", 1)
                    if len(parts) > 1 and parts[1].strip() not in subs:
                        subs.append(parts[1].strip())
                out["meta_genres_main"] = main
                out["meta_genres_sub"] = self.sep.join(subs)
                # Singular aliases are kept as the convenient template names.
                out["meta_genre_main"] = main
                out["meta_genre_sub"] = self.sep.join(subs)
        out.update(self._stringify(out))
        return out

    def get_one_meta(self, key: str) -> Any:
        meta_dict = self.build_flat_meta(key)
        if key not in meta_dict:
            meta_dict = self.build_flat_meta(None)
        return meta_dict.get(key)

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
