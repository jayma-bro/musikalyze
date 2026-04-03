"""Pipeline EffNet + têtes TensorflowPredict2D et descripteurs classiques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from musikalize.config import AnalysisConfig, AnalysisResult, ModelPath


def load_label_list(labels_path: Path) -> list[str]:
    """Charge la liste des classes depuis un JSON Essentia (clé `classes` ou liste racine)."""

    raw = labels_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "classes" in data:
        return [str(x) for x in data["classes"]]
    raise ValueError(f"Format JSON de labels non reconnu: {labels_path}")


def _mean_pool_time(pred: Any) -> Any:
    """Moyenne sur l'axe temps / patches."""

    import numpy as np

    x = np.asarray(pred, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=0)
    if x.ndim == 3:
        return np.mean(x, axis=(0, 1))
    return np.mean(x.reshape(-1, x.shape[-1]), axis=0)


def _split_genre_label(
    label: str,
    *,
    genre_main: bool,
    separators: tuple[str, ...],
) -> list[str]:
    """Découpe un label Discogs-style en segments."""

    s = label.strip()
    if not s:
        return []
    for sep in separators:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if not genre_main and parts:
                return [parts[-1]]
            return parts
    return [s]


def _select_genre_indices(
    probs: Any,
    _labels: list[str],
    cfg: AnalysisConfig,
) -> list[int]:
    """Indices triés par probabilité décroissante, selon seuil et nombre max."""

    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs)
    order_list = [int(i) for i in order]
    if cfg.genre_thold is not None:
        order_list = [i for i in order_list if probs[i] >= cfg.genre_thold]
    limit = cfg.genre_count if cfg.genre_count is not None else len(order_list)
    return order_list[:limit]


def _build_genre_list(
    probs: Any,
    labels: list[str],
    cfg: AnalysisConfig,
) -> tuple[list[str], dict[str, float]]:
    """Liste finale de genres et scores (label affiché → max prob)."""

    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    idxs = _select_genre_indices(probs, labels, cfg)
    seen: set[str] = set()
    out: list[str] = []
    scores: dict[str, float] = {}

    for i in idxs:
        lab = labels[i] if i < len(labels) else str(i)
        p = float(probs[i])
        parts = _split_genre_label(lab, genre_main=cfg.genre_main, separators=cfg.genre_separators)
        for part in parts:
            if part not in seen:
                seen.add(part)
                out.append(part)
                scores[part] = max(scores.get(part, 0.0), p)
    return out, scores


def run_classical_descriptors(audio: Any, cfg: AnalysisConfig) -> dict[str, Any]:
    """BPM, tonalité (EDMA), danceability classique."""

    out: dict[str, Any] = {}
    if cfg.bpm:
        from essentia.standard import PercivalBpmEstimator

        bpm_algo = PercivalBpmEstimator()
        out["bpm"] = float(bpm_algo(audio))
    if cfg.key or cfg.scale:
        from essentia.standard import KeyExtractor

        ke = KeyExtractor()
        key, scale, _ = ke(audio)
        if cfg.key:
            out["key_edma"] = str(key)
        if cfg.scale:
            out["scale_edma"] = str(scale)
    if cfg.danceability_classical:
        from essentia.standard import Danceability

        d = Danceability()
        dance, _ = d(audio)
        out["danceability"] = float(dance)
    return out


def _probs_from_raw(pooled: Any) -> Any:
    """Convertit sortie modèle en distribution de probabilités."""

    import numpy as np

    pooled = np.asarray(pooled, dtype=np.float64).ravel()
    if pooled.size and (np.max(np.abs(pooled)) > 1.5 or np.min(pooled) < -0.01):
        ex = np.exp(pooled - np.max(pooled))
        return ex / np.sum(ex)
    s = np.sum(pooled)
    if s > 0:
        return pooled / s
    return pooled


def analyze_audio(
    audio: Any,
    model_path: ModelPath,
    analysis_cfg: AnalysisConfig,
) -> AnalysisResult:
    """
    Calcule embeddings EffNet, enchaîne les têtes 2D, fusionne méta pour gabarits.

    La tête nommée ``genre`` (si présente) alimente ``genre_labels`` / ``meta_genre``.
    Les autres têtes exposent ``meta_<nom>`` = label le plus probable.
    """

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

        pooled = _mean_pool_time(np.asarray(raw))
        probs = _probs_from_raw(pooled)

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
            glist, gscores = _build_genre_list(probs, labels, analysis_cfg)
            result.genre_labels = glist
            result.genre_scores = gscores
            joined = analysis_cfg.genre_join_separator.join(glist)
            meta_flat["meta_genre"] = joined
        else:
            meta_flat[f"meta_{spec.name}"] = top_label

    result.head_predictions = head_preds

    classical = run_classical_descriptors(audio, analysis_cfg)
    if "bpm" in classical:
        result.bpm = classical["bpm"]
        meta_flat["meta_bpm"] = classical["bpm"]
    if "key_edma" in classical:
        result.key_edma = classical["key_edma"]
        meta_flat["meta_key"] = classical["key_edma"]
    if "scale_edma" in classical:
        result.scale_edma = classical["scale_edma"]
        meta_flat["meta_scale"] = classical["scale_edma"]
    if "danceability" in classical:
        result.danceability = classical["danceability"]
        meta_flat["meta_danceability"] = classical["danceability"]

    result.meta = meta_flat
    return result
