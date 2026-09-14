"""Read, resolve and write logical audio metadata tags."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.id3 import ID3, POPM, TXXX
from mutagen.mp4 import MP4FreeForm

from musikalyze._id3_tag_map import _ID3_TAG_MAP
from musikalyze.config import AnalysisResult, TaggingConfig
from musikalyze.templates import build_format_mapping, resolve_template

_EXTENSION_TO_TAG_KEYS = {
    ".mp3": ("ID3v2",),
    ".flac": ("Vorbis",),
    ".ogg": ("Vorbis",),
    ".opus": ("Vorbis",),
    ".m4a": ("iTunes",),
    ".wma": ("ASF",),
}

# Logical names are deliberately independent from codec-specific names.
_CORE_LOGICAL_KEYS = (
    "artist", "title", "album", "genre", "date", "tracknumber", "discnumber",
    "composer", "albumartist", "comment", "lyrics", "copyright", "publisher",
    "encodedby", "encoder", "isrc", "language", "albumsort", "artistsort",
    "titlesort", "website", "bpm", "mood", "grouping", "key", "rating", "tcop",
)
_AUDIO_FEATURE_KEYS = (
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "popularity", "speechiness", "valence", "tempo",
)
_LOGICAL_KEYS = _CORE_LOGICAL_KEYS + _AUDIO_FEATURE_KEYS
_REPLAYGAIN_TAGS = (
    "replaygain_track_gain", "replaygain_track_peak", "replaygain_album_gain",
    "replaygain_album_peak", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_PEAK",
    "REPLAYGAIN_ALBUM_GAIN", "REPLAYGAIN_ALBUM_PEAK",
)


def _detect_tag_keys(path: Path) -> tuple[str, ...]:
    return _EXTENSION_TO_TAG_KEYS.get(path.suffix.lower(), ("Vorbis",))


def _get_tag_name(logical_key: str, tag_keys: Sequence[str]) -> str:
    """Return a usable codec tag name, ignoring Picard explanatory suffixes."""
    lookup_key = "_rating" if logical_key == "rating" else logical_key
    entry = _ID3_TAG_MAP.get(logical_key) or _ID3_TAG_MAP.get(lookup_key) or _ID3_TAG_MAP.get("tcop" if logical_key == "copyright" else logical_key)
    if not entry:
        return logical_key
    for family in tag_keys:
        raw = entry.get(family)
        if raw:
            return str(raw).splitlines()[0].split(" + ")[0].strip().split(" ")[0]
    return logical_key


def _norm_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    if hasattr(value, "text"):
        text = value.text
        value = text[0] if text else None
    result = str(value).strip() if value is not None else ""
    return result or None


def read_tags_raw(path: Path) -> dict[str, Any]:
    """Read supported file tags into logical names without altering the file."""
    audio = MutagenFile(path)
    if audio is None or audio.tags is None:
        return {}
    out: dict[str, Any] = {}
    for logical in _CORE_LOGICAL_KEYS:
        for candidate in (logical, _get_tag_name(logical, _detect_tag_keys(path))):
            try:
                value = _norm_text(audio.get(candidate) if hasattr(audio, "get") else None)
                if value:
                    out[logical] = value
                    break
                value = _norm_text(audio.tags.get(candidate))
                if value:
                    out[logical] = value
                    break
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
    for raw in _REPLAYGAIN_TAGS:
        try:
            value = _norm_text(audio.get(raw))
            if value:
                out[raw.lower()] = value
        except (KeyError, TypeError, ValueError, AttributeError):
            continue

    # POPM stores ratings as 0..255. Expose the public logical value as stars
    # in the 0..5 range; the original email/counter are preserved during export.
    if path.suffix.lower() == ".mp3" and isinstance(audio.tags, ID3):
        popm_frames = audio.tags.getall("POPM")
        if popm_frames:
            out["rating"] = round(float(popm_frames[0].rating) * 5.0 / 255.0, 2)
    elif path.suffix.lower() in {".opus", ".ogg", ".flac"}:
        for key, value in audio.tags.items():
            if str(key).upper().startswith("RATING"):
                try:
                    raw_rating = float(_norm_text(value) or "0")
                    out["rating"] = round(raw_rating * 5.0 if raw_rating <= 1 else raw_rating, 2)
                except ValueError:
                    pass
                break
    elif path.suffix.lower() == ".m4a":
        value = audio.tags.get("----:com.apple.iTunes:rating")
        if value:
            try:
                out["rating"] = round(float(_norm_text(value) or "0"), 2)
            except ValueError:
                pass
    return out


def tags_to_tag_prefix(flat: Mapping[str, Any]) -> dict[str, Any]:
    out = {k if str(k).startswith("tag_") else f"tag_{k}": v for k, v in flat.items()}
    for key in ("tracknumber", "discnumber"):
        if key in flat:
            out[f"tag_{key}_f"] = format_nbr(flat[key])
    return out


def format_nbr(value: Any) -> str:
    text = str(value)
    if "/" in text:
        text = text.split("/", 1)[0]
    text = text.lstrip("0")
    return "" if not text else text if len(text) > 1 else f"0{text}"


def _dedupe(value: Any, separator: str) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    seen: set[str] = set()
    result: list[Any] = []
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(item)
    return separator.join(str(item) for item in result)


def _extract_main_genre(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split("---", 1)[0].split("-", 1)[0].strip()


def _deduplicate_genres(genres: list[str]) -> list[str]:
    result: list[str] = []
    for genre in genres:
        main = _extract_main_genre(genre)
        if main and main not in result:
            result.append(main)
    return result


def _genre_parts(value: Any, separator: str) -> tuple[str, str]:
    values = value if isinstance(value, (list, tuple)) else str(value).split(separator)
    mains: list[str] = []
    subs: list[str] = []
    for item in values:
        parts = [part.strip() for part in str(item).split("---") if part.strip()]
        if not parts:
            continue
        if parts[0] not in mains:
            mains.append(parts[0])
        if len(parts) > 1 and parts[1] not in subs:
            subs.append(parts[1])
    return separator.join(mains), separator.join(subs)


def apply_tagging_config(tag_map: Mapping[str, Any], analysis: AnalysisResult | None, cfg: TaggingConfig) -> dict[str, str]:
    meta = (analysis.meta if analysis else {}) or {}
    mapping = build_format_mapping(tag_map, meta, ext=None)
    resolved: dict[str, str] = {}
    for logical, template in {**cfg.tags, **cfg.extra}.items():
        if template is None:
            continue
        value = resolve_template(template, mapping, separator=cfg.separator)
        if logical == "genre":
            value = _dedupe(value.split(cfg.separator), cfg.separator)
        else:
            value = _dedupe(value.split(cfg.separator), cfg.separator)
        if value:
            resolved[logical] = str(value).strip()
    return resolved


def merge_logical_tags_for_export(original: Mapping[str, Any], resolved: Mapping[str, str]) -> dict[str, str]:
    out = {str(k): str(v).strip() for k, v in original.items() if v is not None and str(v).strip()}
    out.update({str(k): str(v).strip() for k, v in resolved.items() if v is not None and str(v).strip()})
    return out


def file_meta_from_tags(tags_logical: Mapping[str, Any], keys_needed: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if any(key.startswith("meta_tag_replaygain") for key in keys_needed):
        for name in ("replaygain_track_gain", "replaygain_album_gain"):
            if name in tags_logical:
                out[f"meta_tag_{name}"] = tags_logical[name]
    return out


def _write_one(audio: Any, path: Path, logical: str, value: str) -> None:
    family = path.suffix.lower()
    if logical == "rating":
        try:
            stars = min(5.0, max(0.0, float(value)))
        except ValueError:
            return
        if family == ".mp3":
            raw = ID3(path)
            raw.delall("POPM")
            raw.add(POPM(email="user@email", rating=round(stars * 255 / 5), count=0))
            raw.save()
        elif family in {".opus", ".ogg", ".flac"}:
            audio.tags["RATING:user@email"] = [str(stars / 5.0)]
        elif family == ".m4a":
            audio.tags["----:com.apple.iTunes:rating"] = [
                MP4FreeForm(str(round(stars, 2)).encode("utf-8"))
            ]
        else:
            audio["rating"] = [str(stars)]
        return
    if logical == "tcop":
        logical = "copyright"
    if family == ".mp3":
        easy_names = {"artist", "title", "album", "genre", "date", "tracknumber", "discnumber", "composer", "albumartist", "comment", "lyrics", "copyright", "publisher", "encodedby", "encoder", "isrc", "bpm", "mood", "grouping", "key"}
        if logical in easy_names:
            try:
                audio[logical] = [value]
                return
            except (KeyError, TypeError, ValueError):
                pass
        raw = ID3(path)
        raw.delall(f"TXXX:{logical}")
        raw.add(TXXX(encoding=3, desc=logical, text=[value]))
        raw.save()
        return
    try:
        audio[logical] = [value]
        return
    except (KeyError, TypeError, ValueError):
        pass
    raw = _get_tag_name(logical, _detect_tag_keys(path))
    audio.tags[raw] = [value]


def write_tags_to_file(path: Path, tags: Mapping[str, str]) -> None:
    audio = MutagenFile(path, easy=True)
    if audio is None:
        raise ValueError(f"Unsupported format for writing tags: {path}")
    for logical, value in tags.items():
        if value is not None and str(value).strip():
            _write_one(audio, path, logical, str(value))
    audio.save()


def write_tags_to_file_safe(path: Path, tags: Mapping[str, str]) -> None:
    write_tags_to_file(path, tags)


def copy_and_write_tags(source: Path, destination: Path, tags: Mapping[str, str]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    write_tags_to_file(destination, tags)
    return destination
 