"""Template resolution for `{tag_*}` / `{meta_*}` and safe path segments."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

_FORMAT_FIELDS = re.compile(r"\{([^{}:]+)(?::[^}]*)?\}")
_FORMAT_WITH_SPEC = re.compile(r"\{([^{}:]+):(.+?)\}")


def extract_placeholder_keys(*templates: str) -> set[str]:
    """Return field names used in ``str.format``-style templates."""

    keys: set[str] = set()
    for t in templates:
        if not t:
            continue
        keys.update(_FORMAT_FIELDS.findall(t))
    return keys


def build_format_mapping(
    tag_map: Mapping[str, Any],
    meta_map: Mapping[str, Any],
    *,
    ext: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the mapping for ``str.format_map`` with ``tag_*`` and ``meta_*`` keys."""

    out: dict[str, Any] = {}
    for k, v in tag_map.items():
        key = k if str(k).startswith("tag_") else f"tag_{k}"
        out[key] = v
    for k, v in meta_map.items():
        key = k if str(k).startswith("meta_") else f"meta_{k}"
        out[key] = v
    if ext is not None:
        out["ext"] = ext
    if extra:
        out.update(extra)
    tn = out.get("tag_tracknumber")
    if tn is not None and "tag_track_number" not in out:
        try:
            out["tag_track_number"] = int(str(tn).split("/")[0].strip())
        except ValueError:
            out["tag_track_number"] = 0
    elif "tag_track_number" not in out:
        out["tag_track_number"] = 0
    return out


def _non_empty_values(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with ``None``-values removed and empty strings replaced with ``""``."""
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def resolve_template(
    template: str,
    mapping: Mapping[str, Any],
    *,
    separator: str = ";",
    join_meta: bool = True,
) -> str:
    """Apply ``str.format_map``; missing keys become empty strings.

    If *join_meta* is ``True`` and the resolved value is a list, the list
    elements are joined with *separator*.  Values that are ``None`` or empty
    strings are silently omitted from templates so that consecutive separators
    (e.g. ``"{a};{b};{c}"`` when *b* is empty) do not appear.
    """

    clean = _non_empty_values(mapping)

    class _SafeFormat(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return ""

        def __getitem__(self, key: str) -> Any:
            val = super().__getitem__(key)
            if val is None:
                return ""
            if join_meta and isinstance(val, list) and len(val) > 0:
                parts = []
                for item in val:
                    s = str(item).strip() if item is not None else ""
                    if s:
                        parts.append(s)
                if parts:
                    return separator.join(parts)
                return ""
            # Return int/float as-is to support format specifiers like :02d
            if isinstance(val, (int, float)):
                return val
            return str(val).strip() if val is not None else ""

    safe = _SafeFormat((k, v) for k, v in clean.items())
    return template.format_map(safe)


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def sanitize_path_segment(segment: str, max_len: int = 200) -> str:
    """Make one filename or directory component portable and safe.

    The result is valid on POSIX and Windows, cannot be ``.``/``..`` or a
    Windows device name, and never contains control characters or trailing
    spaces/dots. Unicode letters are preserved.
    """
    bad = '<>:"/\\|?*'
    text = unicodedata.normalize("NFKC", str(segment))
    text = "".join("_" if ord(char) < 32 or ord(char) == 127 else char for char in text)
    for char in bad:
        text = text.replace(char, "_")
    text = text.strip().rstrip(" .")
    if not text or text in {".", ".."}:
        return "_"
    stem = text.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        text = f"_{text}"
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip(" .") + "..."
    return text or "_"


def sanitize_relative_path(path_str: str) -> str:
    """Sanitize every component and force a relative, portable path."""
    parts = str(path_str).replace("\\", "/").split("/")
    safe_parts = [sanitize_path_segment(part) for part in parts if part]
    return "/".join(safe_parts) or "_"
