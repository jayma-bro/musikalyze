"""Read/write tags and apply templates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile

from musikalyze.config import AnalysisResult, TaggingConfig
from musikalyze.templates import build_format_mapping, resolve_template


def _extract_main_genre(genre_string: str) -> str:
    """Return the first-level category from a hyphen-separated genre string.
    
    e.g. "Reggae---Dub" -> "Reggae"
    """
    if not genre_string:
        return genre_string
    return genre_string.split("---")[0].split("-")[0].strip()


def _deduplicate_genres(genres: list[str]) -> list[str]:
    """Remove duplicate genres while preserving order.
    
    For genre strings like {"Reggae---Dub": 83, "Electronic---Dub": 80}:
    - Extract main genres: ["Reggae", "Electronic"]
    - Deduplicate and return
    """
    seen = set()
    result = []
    for g in genres:
        main = _extract_main_genre(g) if "---" in g or "-" in g else g
        if main and main not in seen:
            seen.add(main)
            result.append(main)
    return result

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
    "lyrics",
    "copyright",
    "publisher",
    "encodedby",
    "encoder",
    "isrc",
    "language",
    "albumsort",
    "artistsort",
    "titlesort",
    "website",
    "bpm",
    "mood",
    "grouping",
    "key",
    "tcop",  # Track Commercial Orientation (MP3 tag)
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "popularity",
    "speechiness",
    "valence",
    "tempo",
)

_REPLAYGAIN_TAGS = (
    "replaygain_track_gain",
    "replaygain_track_peak",
    "replaygain_album_gain",
    "replaygain_album_peak",
    "REPLAYGAIN_TRACK_GAIN",
    "REPLAYGAIN_TRACK_PEAK",
    "REPLAYGAIN_ALBUM_GAIN",
    "REPLAYGAIN_ALBUM_PEAK",
)


def _norm_text(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, list) and val:
        v0 = val[0]
        if hasattr(v0, "text"):
            t = v0.text
            return str(t[0]).strip() if t else None
        return str(v0).strip()
    if hasattr(val, "text"):
        t = val.text
        return str(t[0]).strip() if t else None
    s = str(val).strip()
    return s or None


def read_tags_raw(path: Path) -> dict[str, Any]:
    """Read common and extended logical tags from the audio file (best effort)."""

    f = MutagenFile(path)
    if f is None:
        return {}

    out: dict[str, Any] = {}
    if f.tags is None:
        return {}

    for key in _LOGICAL_KEYS:
        try:
            v = _norm_text(f.get(key))
            if v:
                out[key] = v
        except (KeyError, TypeError, ValueError, AttributeError):
            pass

    for key in _REPLAYGAIN_TAGS:
        try:
            v = _norm_text(f.get(key))
            if v:
                lk = key.lower()
                out[lk] = v
        except (KeyError, TypeError, ValueError, AttributeError):
            pass

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
            "TCOP": "copyright",
            "TPUB": "publisher",
            "TENC": "encodedby",
            "TSRC": "isrc",
            "TBPM": "bpm",
            "TMOO": "mood",
        }
        for raw, logical in id3_map.items():
            try:
                fr = f.tags.get(raw)
                v = _norm_text(fr)
                if v:
                    out[logical] = v
            except (KeyError, TypeError, ValueError, AttributeError):
                pass

    return {k: v for k, v in out.items() if v is not None and str(v).strip() != ""}


def tags_to_tag_prefix(flat: Mapping[str, Any]) -> dict[str, Any]:
    m: dict[str, Any] = {}
    for k, v in flat.items():
        key = k if str(k).startswith("tag_") else f"tag_{k}"
        m[key] = v
    for k in ["tracknumber", "discnumber"]:
        if f"tag_{k}" in m:
            m[f"tag_{k}_f"] = format_nbr(flat[k])
    return m

def format_nbr(s: Any) -> str:
    if type(s) == int:
        s = str(s)
    if "/" in s:
        return s.split("/")[0]
    s = s.lstrip("0")
    if not s:
        return ""
    return s if len(s) > 1 else f"0{s}"

def merge_logical_tags_for_export(
    original: Mapping[str, Any],
    resolved: Mapping[str, str],
) -> dict[str, str]:
    """
    Start from original file tags, then overlay fields produced by ``TaggingConfig``.

    Unmentioned fields keep their original values; only keys present in ``resolved`` override.
    """

    out: dict[str, str] = {}
    for k, v in original.items():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out[k] = s
    for k, v in resolved.items():
        if v is not None and str(v).strip():
            out[k] = str(v).strip()
    return out


def file_meta_from_tags(tags_logical: Mapping[str, Any], keys_needed: set[str]) -> dict[str, Any]:
    """Expose selected on-file values as ``meta_*`` for templates (e.g. ReplayGain)."""

    out: dict[str, Any] = {}
    if not keys_needed:
        return out
    rg_track = tags_logical.get("replaygain_track_gain")
    if rg_track and any(k.startswith("meta_replaygain") for k in keys_needed):
        out["meta_replaygain_track"] = str(rg_track)
    rg_album = tags_logical.get("replaygain_album_gain")
    if rg_album and any(k == "meta_replaygain_album" or k.startswith("meta_replaygain_album") for k in keys_needed):
        out["meta_replaygain_album"] = str(rg_album)
    return out


def apply_tagging_config(
    tag_map: Mapping[str, Any],
    analysis: AnalysisResult | None,
    cfg: TaggingConfig,
) -> dict[str, str]:
    meta = (analysis.meta if analysis is not None else {}) or {}
    base = build_format_mapping(tag_map, meta, ext=None)
    resolved: dict[str, str] = {}

    def one(template: str | None, key: str) -> None:
        if template is None:
            return
        resolved[key] = resolve_template(template, base, separator=cfg.separator)

    one(cfg.artist, "artist")
    one(cfg.title, "title")
    one(cfg.album, "album")
    one(cfg.genre, "genre")
    
    # Apply genre deduplication
    if "genre" in resolved and resolved["genre"]:
        genre_val = resolved["genre"]
        # Try to extract main genre and deduplicate
        main_genre = _extract_main_genre(genre_val)
        if main_genre:
            resolved["genre"] = main_genre
    else:
        # Try to get genre from meta and deduplicate
        genre_meta = meta.get("meta_genre") or meta.get("meta_mood_genre")
        if genre_meta:
            if isinstance(genre_meta, dict):
                # It's a dict of {genre: score}
                scores = sorted(genre_meta.items(), key=lambda x: x[1], reverse=True)
                top_genres = [_extract_main_genre(g) for g, s in scores if s > 0]
                deduped = _deduplicate_genres(top_genres)
                if deduped:
                    resolved["genre"] = cfg.separator.join(deduped)
            elif isinstance(genre_meta, list):
                genres = [_extract_main_genre(str(g)) for g in genre_meta if g]
                deduped = _deduplicate_genres(genres)
                if deduped:
                    resolved["genre"] = cfg.separator.join(deduped)
    
    one(cfg.composer, "composer")
    one(cfg.date, "date")
    one(cfg.tracknumber, "tracknumber")
    one(cfg.discnumber, "discnumber")
    one(cfg.comment, "comment")
    
    # Handle audio feature tags (e.g., acousticness, danceability, etc.)
    for key in ["key", "tcop", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "popularity", "speechiness", "valence", "tempo"]:
        if key in meta:
            # If we have a meta value, resolve it with template
            template = cfg.extra.get(key) or f"{{meta_{key}}}"
            if template:
                resolved[key] = resolve_template(template, base, separator=cfg.separator)
    
    # Handle other extra tags
    for k, tmpl in cfg.extra.items():
        one(tmpl, k)

    return {k: v for k, v in resolved.items() if v is not None}


def write_tags_to_file(path: Path, tags: Mapping[str, str]) -> None:
    f = MutagenFile(path, easy=True)
    if f is None:
        raise ValueError(f"Unsupported format for writing tags: {path}")
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
