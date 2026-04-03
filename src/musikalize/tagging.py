"""Lecture / écriture de tags et application des gabarits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mutagen import File as MutagenFile

from musikalize.config import AnalysisResult, TaggingConfig
from musikalize.templates import build_format_mapping, resolve_template

# Clés logiques → noms de tags mutagen / ffmpeg (minuscules)
_LOGICAL_KEYS = (
    "artist",
    "title",
    "album",
    "genre",
    "date",
    "tracknumber",
    "discnumber",
    "composer",
    "albumartist",
    "comment",
)


def read_tags_raw(path: Path) -> dict[str, Any]:
    """
    Lit les tags courants dans un dict de clés logiques (minuscules).

    S'appuie sur l'API Mutagen quand les clés unifiées existent ; sinon mapping ID3 partiel.
    """

    f = MutagenFile(path)
    if f is None:
        return {}

    out: dict[str, Any] = {}
    if f.tags is None:
        return {}

    def set_if(k: str, val: Any) -> None:
        if val is None:
            return
        if isinstance(val, list) and val:
            out[k] = val[0] if not hasattr(val[0], "text") else val[0].text[0]
        elif hasattr(val, "text"):
            out[k] = val.text[0] if val.text else ""
        else:
            out[k] = str(val)

    for key in _LOGICAL_KEYS:
        try:
            set_if(key, f.get(key))
        except (KeyError, TypeError, ValueError):
            pass

    # ID3 bruts si pas de clés faciles
    if not out and hasattr(f.tags, "get"):
        id3_map = {
            "TIT2": "title",
            "TPE1": "artist",
            "TALB": "album",
            "TCON": "genre",
            "TDRC": "date",
            "TRCK": "tracknumber",
            "TPOS": "discnumber",
            "TCOM": "composer",
            "TPE2": "albumartist",
            "COMM": "comment",
        }
        for raw, logical in id3_map.items():
            try:
                fr = f.tags.get(raw)
                if fr is not None:
                    set_if(logical, fr)
            except (KeyError, TypeError, ValueError):
                pass

    return {k: v for k, v in out.items() if v is not None and str(v).strip() != ""}


def tags_to_tag_prefix(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Préfixe `tag_` pour les clés logiques."""

    m: dict[str, Any] = {}
    for k, v in flat.items():
        key = k if str(k).startswith("tag_") else f"tag_{k}"
        m[key] = v
    return m


def apply_tagging_config(
    tag_map: Mapping[str, Any],
    analysis: AnalysisResult | None,
    cfg: TaggingConfig,
) -> dict[str, str]:
    """
    Produit un dict de champs résolus (clés logiques) à partir des gabarits.

    ``tag_map`` : clés logiques (artist, …) ou déjà préfixées ``tag_*``.
    Les méta viennent de ``analysis.meta`` (clés ``meta_*``).
    """

    meta = (analysis.meta if analysis is not None else {}) or {}
    base = build_format_mapping(tag_map, meta, ext=None)
    resolved: dict[str, str] = {}

    def one(template: str | None, key: str) -> None:
        if template is None:
            return
        resolved[key] = resolve_template(template, base)

    one(cfg.artist, "artist")
    one(cfg.title, "title")
    one(cfg.album, "album")
    one(cfg.genre, "genre")
    one(cfg.composer, "composer")
    one(cfg.date, "date")
    one(cfg.tracknumber, "tracknumber")
    one(cfg.discnumber, "discnumber")
    one(cfg.comment, "comment")
    for k, tmpl in cfg.extra.items():
        one(tmpl, k)

    return {k: v for k, v in resolved.items() if v is not None}


def write_tags_to_file(path: Path, tags: Mapping[str, str]) -> None:
    """Écrit les tags (clés logiques) sur un fichier audio via Mutagen."""

    f = MutagenFile(path, easy=True)
    if f is None:
        raise ValueError(f"Format non pris en charge pour écriture: {path}")
    for k, v in tags.items():
        if v is None or str(v).strip() == "":
            continue
        try:
            f[k] = str(v)
        except (KeyError, TypeError, ValueError):
            try:
                f.tags[k] = str(v)
            except (KeyError, TypeError, AttributeError):
                pass
    f.save()


def write_tags_to_file_safe(path: Path, tags: Mapping[str, str]) -> None:
    """Écriture best-effort (ignore les clés non supportées)."""

    f = MutagenFile(path, easy=True)
    if f is None:
        return
    for k, v in tags.items():
        if v is None or str(v).strip() == "":
            continue
        try:
            f[k] = str(v)
        except (KeyError, TypeError, ValueError):
            continue
    try:
        f.save()
    except (OSError, ValueError):
        pass
