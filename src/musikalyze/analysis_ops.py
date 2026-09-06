"""Shared analysis helpers (pure functions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal
def load_label_list(
    labels_path: Path,
    extractor_name: str | None = None,
) -> list[str]:
    """Load label list from a JSON file (list of strings or {"classes": [...]})."""
    label = f' (extractor "{extractor_name}")' if extractor_name else ""
    for enc in ("utf-8", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            raw = labels_path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        raise ValueError(
            f"Cannot decode labels file {labels_path}{label}: "
            f"all encodings (utf-8, utf-16-le, utf-16-be, latin-1) failed"
        )
    raw = raw.lstrip("\ufeff")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {labels_path}{label}: {e} "
            f"(first 200 chars: {repr(raw[:200])})"
        ) from e
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "classes" in data:
        return [str(x) for x in data["classes"]]
    raise ValueError(
        f"Unexpected JSON structure in {labels_path}{label}: "
        f"expected a JSON list or a dict with 'classes' key, "
        f"got {type(data).__name__}"
    )


def meta_key_base(obj: object) -> str:
    category = getattr(obj, "category", "other")
    name = getattr(obj, "name", "unknown")
    if category == "mood":
        return f"meta_mood_{name}"
    if category == "genre":
        return f"meta_genre_{name}"
    return f"meta_{name}"


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


def merge_values(existing: list | dict | str, new: list | dict | str) -> list | dict:
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


def stringify(dictionary: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for item in dictionary:
        out[f"{item}_str"] = dictionary[item] if type(dictionary[item]) is str else json.dumps(dictionary[item], ensure_ascii=False)
    return(out)


def pct(value: float | list[float]) -> int | list[int]:
    return int(round(value * 100)) if type(value) == float else [int(round(n * 100)) for n in value]


# Re-export for backwards compatibility
__all__ = ["load_label_list", "mean_pool_time", "merge_values", "meta_key_base", "pct"]