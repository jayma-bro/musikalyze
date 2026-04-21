"""Shared analysis helpers (pure functions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Dict


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

# not used
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

def merge_values(existing: Union[List, Dict, str], new: Union[List, Dict, str]) -> Union[List, Dict]:
    """Merge tow values"""
    if isinstance(new, list):
        existing_list = [existing] if not isinstance(existing, list) else existing
        return list(set(existing_list + new))
    elif isinstance(new, dict):
        if isinstance(existing, dict):
            existing.update(new)
            return existing
        else:
            raise TypeError(f"Type conflict : {type(existing)} vs dict")
    elif isinstance(new, str):
        if isinstance(existing, str):
            return list({existing, new})
        elif isinstance(existing, list):
            return list(set(existing + [new]))
        else:
            raise TypeError(f"Type conflict : {type(existing)} vs str")
    else:
        raise TypeError(f"Type not managed : {type(new)}")


def meta_key_base(obj: LabelExtractor | PredictionRecord) -> str:
    if obj.category == "mood":
        return f"meta_mood_{obj.name}"
    if obj.category == "genre":
        return f"meta_genre_{obj.name}"
    return f"meta_{obj.name}"


def stringify(dictionary: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for item in dictionary:
        out[f"{item}_str"] = dictionary[item] if type(dictionary[item]) is str else json.dumps(dictionary[item], ensure_ascii=False)
    return(out)