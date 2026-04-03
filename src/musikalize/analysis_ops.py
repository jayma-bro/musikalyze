"""Shared analysis helpers (pure functions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


def load_label_list(labels_path: Path) -> list[str]:
    raw = labels_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "classes" in data:
        return [str(x) for x in data["classes"]]
    raise ValueError(f"Format JSON de labels non reconnu: {labels_path}")


def mean_pool_time(pred: Any) -> Any:
    import numpy as np

    x = np.asarray(pred, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=0)
    if x.ndim == 3:
        return np.mean(x, axis=(0, 1))
    return np.mean(x.reshape(-1, x.shape[-1]), axis=0)


def probs_from_raw(pooled: Any) -> Any:
    import numpy as np

    pooled = np.asarray(pooled, dtype=np.float64).ravel()
    if pooled.size and (np.max(np.abs(pooled)) > 1.5 or np.min(pooled) < -0.01):
        ex = np.exp(pooled - np.max(pooled))
        return ex / np.sum(ex)
    s = np.sum(pooled)
    if s > 0:
        return pooled / s
    return pooled


def split_genre_label(
    label: str,
    *,
    genre_main: bool,
    separators: tuple[str, ...],
) -> list[str]:
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


def select_genre_indices(
    probs: Any,
    *,
    genre_count: int | None,
    genre_thold: float | None,
    policy: Literal["intersection", "union"],
) -> list[int]:
    import numpy as np

    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs)
    order_list = [int(i) for i in order]

    if policy == "intersection":
        if genre_thold is not None:
            order_list = [i for i in order_list if probs[i] >= genre_thold]
        limit = genre_count if genre_count is not None else len(order_list)
        return order_list[:limit]

    # union: seuil OU top-k
    by_thold = [i for i in order_list if genre_thold is None or probs[i] >= genre_thold]
    topk = order_list[: genre_count] if genre_count is not None else order_list
    merged: list[int] = []
    seen: set[int] = set()
    for i in by_thold + topk:
        if i not in seen:
            seen.add(i)
            merged.append(i)
    return merged


def main_sub_from_label(label: str, separators: tuple[str, ...]) -> tuple[str, str]:
    """Main segment (before first separator) and subgenre (last segment)."""

    s = label.strip()
    if not s:
        return "", ""
    for sep in separators:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[-1]
            if len(parts) == 1:
                return parts[0], ""
    return s, ""
