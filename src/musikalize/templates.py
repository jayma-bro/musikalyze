"""Résolution de gabarits `{tag_*}` / `{meta_*}` et chemins sûrs."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Crochets optionnels : [meta_genre] → {meta_genre}
_BRACKET = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*)\]")


def bracket_to_brace(template: str) -> str:
    """Convertit les placeholders `[name]` en `{name}` pour str.format."""

    def repl(m: re.Match[str]) -> str:
        return "{" + m.group(1) + "}"

    return _BRACKET.sub(repl, template)


def build_format_mapping(
    tag_map: Mapping[str, Any],
    meta_map: Mapping[str, Any],
    *,
    ext: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit le dict pour `str.format(**kw)` avec préfixes tag_ et meta_."""

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
    # Raccourcis numériques pour padding
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
    """Valeurs manquantes → chaîne vide (gabarits partiels)."""

    def __missing__(self, key: str) -> str:
        return ""


def resolve_template(template: str, mapping: Mapping[str, Any]) -> str:
    """Applique ``str.format_map`` après conversion des crochets."""

    t = bracket_to_brace(template)
    safe = _SafeFormat((k, (v if v is not None else "")) for k, v in mapping.items())
    return t.format_map(safe)


def sanitize_path_segment(segment: str, max_len: int = 200) -> str:
    """Retire ou remplace les caractères problématiques pour un segment de chemin."""

    bad = '<>:"/\\|?*'
    s = segment.strip()
    for c in bad:
        s = s.replace(c, "_")
    s = s.replace("\x00", "")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s or "_"


def sanitize_relative_path(path_str: str) -> str:
    """Sanitise chaque composante d'un chemin relatif."""

    parts = path_str.replace("\\", "/").split("/")
    return "/".join(sanitize_path_segment(p) for p in parts if p)
