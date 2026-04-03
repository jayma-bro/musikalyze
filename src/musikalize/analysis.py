"""Pipeline EffNet + têtes TensorflowPredict2D (API historique ``ModelPath``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from musikalize.analysis_ops import (
    load_label_list,
    mean_pool_time,
    probs_from_raw,
    select_genre_indices,
    split_genre_label,
)
from musikalize.config import AnalysisConfig, AnalysisResult, ModelPath


def _build_genre_list_legacy(
    probs: Any,
    labels: list[str],
    cfg: AnalysisConfig,
) -> tuple[list[str], dict[str, float]]:
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    policy: Any = "intersection"
    idxs = select_genre_indices(
        probs,
        genre_count=cfg.genre_count,
        genre_thold=cfg.genre_thold,
        policy=policy,
    )
    seen: set[str] = set()
    out: list[str] = []
    scores: dict[str, float] = {}

    for i in idxs:
        lab = labels[i] if i < len(labels) else str(i)
        p = float(probs[i])
        parts = split_genre_label(lab, genre_main=cfg.genre_main, separators=cfg.genre_separators)
        for part in parts:
            if part not in seen:
                seen.add(part)
                out.append(part)
                scores[part] = max(scores.get(part, 0.0), p)
    return out, scores


def run_classical_descriptors(audio: Any, cfg: AnalysisConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if cfg.bpm:
        from essentia.standard import PercivalBpmEstimator

        bpm_algo = PercivalBpmEstimator()
        bpm = float(bpm_algo(audio))
        out["bpm"] = bpm
        out["meta_bpm"] = int(round(bpm))
    if cfg.key or cfg.scale:
        from essentia.standard import KeyExtractor

        ke = KeyExtractor()
        key, scale, _ = ke(audio)
        if cfg.key:
            out["key_edma"] = str(key)
            out["meta_key"] = str(key)
        if cfg.scale:
            out["scale_edma"] = str(scale)
            out["meta_scale"] = str(scale)
    if cfg.danceability_classical:
        from essentia.standard import Danceability

        d = Danceability()
        dance, _ = d(audio)
        out["danceability"] = float(dance)
        out["meta_danceability"] = float(dance)
    return out


def analyze_audio(
    audio: Any,
    model_path: ModelPath,
    analysis_cfg: AnalysisConfig,
) -> AnalysisResult:
    """API historique : un EffNet + têtes ``ClassificationHeadSpec``."""

    import numpy as np
    from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs

    eff = TensorflowPredictEffnetDiscogs(
        graphFilename=str(model_path.embedding_model.resolve()),
        output=model_path.embedding_output,
    )
    embeddings = eff(audio)

    result = AnalysisResult(embeddings=embeddings)
    head_preds: dict[str, Any] = {}
    meta_flat: dict[str, Any] = {}

    for spec in model_path.heads:
        labels: list[str] = []
        if spec.labels_path is not None:
            labels = load_label_list(Path(spec.labels_path))

        raw = TensorflowPredict2D(
            graphFilename=str(spec.graph_path.resolve()),
            input=spec.input_tensor,
            output=spec.output_tensor,
        )(embeddings)

        pooled = mean_pool_time(np.asarray(raw))
        probs = probs_from_raw(pooled)

        top_i = int(np.argmax(probs)) if probs.size else 0
        top_label = labels[top_i] if labels and top_i < len(labels) else str(top_i)
        top_score = float(probs[top_i]) if probs.size else 0.0

        head_preds[spec.name] = {
            "probabilities": probs,
            "labels": labels,
            "top_label": top_label,
            "top_index": top_i,
            "top_score": top_score,
        }

        if spec.name == "genre" and labels:
            glist, _gscores = _build_genre_list_legacy(probs, labels, analysis_cfg)
            result.genre_labels = glist
            joined = analysis_cfg.genre_join_separator.join(glist)
            meta_flat["meta_genre"] = joined
        else:
            meta_flat[f"meta_{spec.name}"] = top_label

    result.head_predictions = head_preds

    classical = run_classical_descriptors(audio, analysis_cfg)
    if "bpm" in classical:
        result.bpm = classical["bpm"]
    if "key_edma" in classical:
        result.key_edma = classical["key_edma"]
    if "scale_edma" in classical:
        result.scale_edma = classical["scale_edma"]
    if "danceability" in classical:
        result.danceability = classical["danceability"]

    for mk, mv in classical.items():
        if mk.startswith("meta_"):
            meta_flat[mk] = mv

    result.meta = meta_flat
    return result
