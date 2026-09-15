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
    "titlesort", "website", "bpm", "mood", "grouping", "key", "rating", "catalognumber", "tcop",
)
_AUDIO_FEATURE_KEYS = (
    "acousticness", "danceability", "energy", "instrumentalness", "liveness",
    "popularity", "speechiness", "valence", "tempo",
)
_LOGICAL_KEYS = _CORE_LOGICAL_KEYS + _AUDIO_FEATURE_KEYS

# ID3 POPM uses discrete rating bytes, not a linear 0..255 star scale.
_POPM_RATING_VALUES = (0, 1, 64, 128, 196, 255)


def popm_to_stars(value: int) -> int:
    """Convert an ID3 POPM byte to the nearest standard 0..5 star rating."""
    raw = min(255, max(0, int(value)))
    return min(range(6), key=lambda stars: abs(_POPM_RATING_VALUES[stars] - raw))


def stars_to_popm(stars: float) -> int:
    """Convert a 0..5 star rating to the canonical ID3 POPM byte."""
    normalized = min(5, max(0, round(float(stars))))
    return _POPM_RATING_VALUES[normalized]


_REPLAYGAIN_TAGS = (
    "replaygain_track_gain", "replaygain_track_peak", "replaygain_album_gain",
    "replaygain_album_peak", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_TRACK_PEAK",
    "REPLAYGAIN_ALBUM_GAIN", "REPLAYGAIN_ALBUM_PEAK",
)


def _detect_tag_keys(path: Path) -> tuple[str, ...]:
    return _EXTENSION_TO_TAG_KEYS.get(path.suffix.lower(), ("Vorbis",))


def _get_tag_name(logical_key: str, tag_keys: Sequence[str]) -> str:
    """Return a usable codec tag name, including Picard suffixed entries."""
    lookup_key = "_rating" if logical_key == "rating" else logical_key
    entry = _ID3_TAG_MAP.get(logical_key) or _ID3_TAG_MAP.get(lookup_key)
    if entry is None and logical_key == "copyright":
        entry = _ID3_TAG_MAP.get("tcop")
    if entry is None:
        prefix = f"{lookup_key} ("
        entry = next((value for key, value in _ID3_TAG_MAP.items() if key.startswith(prefix)), None)
    if not entry:
        return logical_key
    for family in tag_keys:
        raw = entry.get(family)
        if raw:
            return str(raw).splitlines()[0].split(" + ")[0].strip().split(" ")[0]
    return logical_key


def _norm_text(value: Any) -> str | None:
    """Normalize Mutagen scalar values without leaking container details."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    if hasattr(value, "text"):
        text = value.text
        value = text[0] if text else None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    result = str(value).strip() if value is not None else ""
    return result or None


def read_tags_raw(path: Path) -> dict[str, Any]:
    """Read supported file tags into logical names without altering the file.

    A few ASF/WMA files contain malformed UTF-16 tag descriptors. Mutagen then
    raises ``UnicodeError`` while parsing the file; metadata reading must not
    prevent audio analysis/export, so FFmpeg remains the preservation path.
    """
    try:
        audio = MutagenFile(path)
    except (OSError, UnicodeError, ValueError):
        return {}
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
    # ReplayGain is stored as TXXX in ID3, ordinary comments in Vorbis, and
    # iTunes freeform atoms in MP4. Match all three representations by their
    # logical suffix rather than relying on Mutagen's format-specific aliases.
    for raw_key, raw_value in audio.tags.items():
        key = str(raw_key).lower()
        logical = next(
            (name for name in _REPLAYGAIN_TAGS if key == name.lower() or key.endswith(f":{name.lower()}")),
            None,
        )
        if logical is None and key.startswith("txxx:"):
            logical = key.removeprefix("txxx:")
        if logical in {name.lower() for name in _REPLAYGAIN_TAGS}:
            value = _norm_text(raw_value)
            if value:
                out[logical] = value

    if path.suffix.lower() == ".mp3" and isinstance(audio.tags, ID3):
        # Some Picard/ID3 files use a COMM frame instead of the mapped TXXX
        # frame for Catalog Number. Normalize both forms to one logical key.
        for frame in audio.tags.getall("COMM"):
            if frame.desc and frame.desc.casefold() == "catalog number":
                value = _norm_text(frame.text)
                if value:
                    out["catalognumber"] = value
                    break

    # POPM stores ratings as 0..255. Expose the public logical value as stars
    # in the 0..5 range; the original email/counter are preserved during export.
    if path.suffix.lower() == ".m4a":
        # Mutagen represents the iTunes track atom as [(track, total)].
        track = audio.tags.get("trkn")
        if track:
            value = track[0] if isinstance(track, (list, tuple)) else track
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            normalized = _norm_text(value)
            if normalized:
                out["tracknumber"] = normalized
    if path.suffix.lower() == ".mp3" and isinstance(audio.tags, ID3):
        popm_frames = audio.tags.getall("POPM")
        if popm_frames:
            out["rating"] = popm_to_stars(popm_frames[0].rating)
    elif path.suffix.lower() in {".opus", ".ogg", ".flac"}:
        rating_candidates: list[tuple[str, float]] = []
        for key, value in audio.tags.items():
            if str(key).lower() in {"rating", "rating:user@email"} or str(key).upper().startswith("RATING:"):
                try:
                    rating_candidates.append((str(key), float(_norm_text(value) or "0")))
                except ValueError:
                    continue
        if rating_candidates:
            # Prefer the generic 0..100 convention used by AIMP when present;
            # otherwise use the Picard RATING:<email> 0..1 convention.
            generic = next((value for key, value in rating_candidates if key.lower() == "rating"), None)
            raw_rating = generic / 20.0 if generic is not None else rating_candidates[0][1] * 5.0
            out["rating"] = round(min(5.0, max(0.0, raw_rating)), 2)
    elif path.suffix.lower() == ".m4a":
        value = audio.tags.get("----:com.apple.iTunes:rating")
        if value:
            try:
                out["rating"] = round(float(_norm_text(value) or "0"), 2)
            except ValueError:
                pass
    return out


def tags_to_tag_prefix(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix logical source tags with ``tag_`` for template resolution."""
    out = {k if str(k).startswith("tag_") else f"tag_{k}": v for k, v in flat.items()}
    for key in ("tracknumber", "discnumber"):
        if key in flat:
            out[f"tag_{key}_f"] = format_nbr(flat[key])
    return out


def format_nbr(value: Any) -> str:
    """Format a track/disc number as a two-digit value.

    The first component of values such as ``"2/12"`` is used, and leading
    zeroes are normalized before padding single-digit numbers. For a multi-entry
    value, the first entry is used because a filename can contain one number.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = str(value)
    if "/" in text:
        text = text.split("/", 1)[0]
    text = text.lstrip("0")
    return "" if not text else text if len(text) > 1 else f"0{text}"


def _split_and_dedupe(value: Any, separator: str) -> list[str]:
    """Flatten template results into distinct entries.

    Splitting each item is intentional: ``{meta_scale};{meta_moods}`` must
    produce one scale entry followed by the mood entries, not a scale joined
    to the first mood.
    """
    values = value if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        for part in str(item).split(separator):
            text = part.strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


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


def apply_tagging_config(
    tag_map: Mapping[str, Any],
    analysis: AnalysisResult | None,
    cfg: TaggingConfig,
) -> dict[str, str | list[str]]:
    """Resolve configured tag templates against source tags and analysis meta.

    Empty values and duplicates are removed. Depending on ``cfg.multi_entry``,
    each resolved field is returned as a list of entries or as one separator-
    joined string; no file is modified by this function.

    Parameters are the ``tag_*`` source mapping, optional analysis result, and
    the :class:`TaggingConfig` that defines templates and separator behavior.
    """
    meta = (analysis.meta if analysis else {}) or {}
    mapping = build_format_mapping(tag_map, meta, ext=None)
    resolved: dict[str, str | list[str]] = {}
    for logical, template in {**cfg.tags, **cfg.extra}.items():
        if template is None:
            continue
        value = resolve_template(template, mapping, separator=cfg.separator)
        entries = _split_and_dedupe(value, cfg.separator)
        if entries:
            resolved[logical] = entries if cfg.multi_entry else cfg.separator.join(entries)
    return resolved


def merge_logical_tags_for_export(
    original: Mapping[str, Any],
    resolved: Mapping[str, str | list[str]],
    *,
    preserve_unconfigured: bool = True,
) -> dict[str, str | list[str]]:
    """Combine original and resolved tags according to preservation policy."""
    out: dict[str, str | list[str]] = {}
    if preserve_unconfigured:
        out.update({str(k): v for k, v in original.items() if v is not None and str(v).strip()})
    out.update({str(k): v for k, v in resolved.items() if v is not None and str(v).strip()})
    return out


def file_meta_from_tags(tags_logical: Mapping[str, Any], keys_needed: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if any(key.startswith("meta_tag_replaygain") for key in keys_needed):
        for name in ("replaygain_track_gain", "replaygain_album_gain"):
            if name in tags_logical:
                out[f"meta_tag_{name}"] = tags_logical[name]
    return out


def _write_one(audio: Any, path: Path, logical: str, value: str | list[str]) -> None:
    values = value if isinstance(value, list) else [value]
    value = values[0] if values else ""
    family = path.suffix.lower()
    if logical == "rating":
        try:
            stars = min(5.0, max(0.0, float(value)))
        except ValueError:
            return
        if family == ".mp3":
            raw = ID3(path)
            raw.delall("POPM")
            raw.add(POPM(email="user@email", rating=stars_to_popm(stars), count=0))
            raw.save()
        elif family in {".opus", ".ogg", ".flac"}:
            audio.tags["Rating"] = [str(round(stars * 20))]
        elif family == ".m4a":
            audio.tags["----:com.apple.iTunes:rating"] = [
                MP4FreeForm(str(round(stars, 2)).encode("utf-8"))
            ]
        else:
            audio["rating"] = [str(stars)]
        return
    if logical == "tcop":
        logical = "copyright"
    if logical.startswith("replaygain_") and family == ".m4a":
        audio.tags[f"----:com.apple.iTunes:{logical}"] = [
            MP4FreeForm(value.encode("utf-8"))
        ]
        return
    if logical == "tracknumber" and family == ".m4a":
        number = format_nbr(value).lstrip("0") or "0"
        audio.tags["trkn"] = [(int(number), 0)]
        return
    if family == ".mp3":
        if logical == "catalognumber":
            raw = ID3(path)
            raw.delall("TXXX:CATALOGNUMBER")
            for frame in list(raw.getall("COMM")):
                if frame.desc and frame.desc.casefold() == "catalog number":
                    raw.delall(f"COMM:{frame.desc}")
            for item in values:
                raw.add(TXXX(encoding=3, desc="CATALOGNUMBER", text=[item]))
            raw.save()
            return
        easy_names = {"artist", "title", "album", "genre", "date", "tracknumber", "discnumber", "composer", "albumartist", "comment", "lyrics", "copyright", "publisher", "encodedby", "encoder", "isrc", "bpm", "mood", "grouping", "key"}
        if logical in easy_names:
            try:
                audio[logical] = values
                return
            except (KeyError, TypeError, ValueError):
                pass
        raw = ID3(path)
        raw.delall(f"TXXX:{logical}")
        for item in values:
            raw.add(TXXX(encoding=3, desc=logical, text=[item]))
        raw.save()
        return
    try:
        audio[logical] = values
        return
    except (KeyError, TypeError, ValueError):
        pass
    raw = _get_tag_name(logical, _detect_tag_keys(path))
    audio.tags[raw] = values


def _clear_tags_keep_artwork(path: Path) -> None:
    """Remove metadata fields while retaining embedded artwork."""
    audio = MutagenFile(path)
    if audio is None or audio.tags is None:
        return
    suffix = path.suffix.lower()
    if isinstance(audio.tags, ID3):
        artwork = list(audio.tags.getall("APIC"))
        audio.tags.clear()
        for frame in artwork:
            audio.tags.add(frame)
    elif suffix == ".m4a":
        artwork = audio.tags.get("covr")
        audio.tags.clear()
        if artwork:
            audio.tags["covr"] = artwork
    elif suffix in {".flac", ".ogg", ".opus"}:
        artwork = []
        if suffix == ".flac" and hasattr(audio, "pictures"):
            artwork = list(audio.pictures)
        elif "metadata_block_picture" in audio.tags:
            artwork = audio.tags.get("metadata_block_picture")
        audio.tags.clear()
        if artwork:
            if suffix == ".flac":
                for picture in artwork:
                    audio.add_picture(picture)
            else:
                audio.tags["metadata_block_picture"] = artwork
    else:
        audio.tags.clear()
    audio.save()


def write_tags_to_file(
    path: Path,
    tags: Mapping[str, str | list[str]],
    *,
    preserve_unconfigured: bool = True,
) -> None:
    """Write logical tags with Mutagen, optionally clearing unconfigured fields.

    ``path`` is modified in place. List values become multiple native entries
    where supported, and ``preserve_unconfigured=False`` removes metadata while
    retaining embedded artwork.
    """
    if not preserve_unconfigured:
        _clear_tags_keep_artwork(path)
    audio = MutagenFile(path, easy=True)
    if audio is None:
        raise ValueError(f"Unsupported format for writing tags: {path}")
    for logical, value in tags.items():
        if value is not None and str(value).strip():
            _write_one(audio, path, logical, value)
    audio.save()


def write_tags_to_file_safe(
    path: Path,
    tags: Mapping[str, str | list[str]],
    *,
    preserve_unconfigured: bool = True,
) -> None:
    write_tags_to_file(path, tags, preserve_unconfigured=preserve_unconfigured)


def copy_and_write_tags(
    source: Path,
    destination: Path,
    tags: Mapping[str, str | list[str]],
    *,
    preserve_unconfigured: bool = True,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    write_tags_to_file(destination, tags, preserve_unconfigured=preserve_unconfigured)
    return destination
 