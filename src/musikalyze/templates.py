"""Template resolution for `{tag_*}` / `{meta_*}` and safe path segments."""

from __future__ import annotations

import re
from typing import Any, Mapping

_FORMAT_FIELDS = re.compile(r"\{([^{}:]+)(?::[^}]*)?\}")


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


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def resolve_template(template: str, mapping: Mapping[str, Any]) -> str:
    """Apply ``str.format_map``; missing keys become empty strings."""

    safe = _SafeFormat((k, (v if v is not None else "")) for k, v in mapping.items())
    return template.format_map(safe)


def sanitize_path_segment(segment: str, max_len: int = 200) -> str:
    bad = '<>:"/\\|?*'
    s = segment.strip()
    for c in bad:
        s = s.replace(c, "_")
    s = s.replace("\x00", "")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s or "_"


def sanitize_relative_path(path_str: str) -> str:
    parts = path_str.replace("\\", "/").split("/")
    return "/".join(sanitize_path_segment(p) for p in parts if p)
