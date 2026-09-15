"""Transcode with ffmpeg, metadata, and path templates."""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.flac import Picture
from mutagen.id3 import APIC, ID3, POPM, TBPM, TXXX
from mutagen.mp4 import MP4Cover, MP4FreeForm

from musikalyze.audio_io import FFMPEG_TIMEOUT
from musikalyze.tagging import popm_to_stars, stars_to_popm
from musikalyze.templates import (
    build_format_mapping,
    resolve_template,
    sanitize_path_segment,
)

_FORMAT_DEFAULTS: dict[str, dict[str, str]] = {
    "opus": {"acodec": "libopus", "audio_bitrate": "160k"},
    "ogg": {"acodec": "libvorbis", "audio_bitrate": "192k"},
    "mp3": {"acodec": "libmp3lame", "audio_bitrate": "320k"},
    "flac": {"acodec": "flac"},
    "wav": {"acodec": "pcm_s16le"},
    "m4a": {"acodec": "aac", "audio_bitrate": "256k"},
    "wma": {"acodec": "wmav2", "audio_bitrate": "192k"},
}


def _merge_options(fmt: str, user: Mapping[str, dict[str, str]]) -> dict[str, str]:
    """Merge format-specific defaults with user-provided options."""
    base = dict(_FORMAT_DEFAULTS.get(fmt, {"acodec": "libopus", "audio_bitrate": "160k"}))
    base.update(user.get(fmt, {}))
    return base


def logical_tags_to_tag_prefix(resolved: Mapping[str, str]) -> dict[str, Any]:
    """Map logical tags to template variables, including formatted track numbers."""

    out = {f"tag_{k}": v for k, v in resolved.items() if v is not None}
    for key in ("tracknumber", "discnumber"):
        if key in resolved:
            value = resolved[key]
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            text = str(value).split("/", 1)[0].lstrip("0") or "0"
            out[f"tag_{key}_f"] = text if len(text) > 1 else f"0{text}"
    return out


def build_output_path(
    path_template: str,
    resolved_tags: Mapping[str, str],
    meta_map: Mapping[str, Any],
    ext: str,
) -> Path:
    """Build a relative output path from the template, including the file extension.
    
    Resolves ``{tag_*}`` and ``{meta_*}`` placeholders in *path_template*, then
    sanitises each path segment (removing characters like ``< > : " / \\ | ? *``).
    """

    tag_pref = logical_tags_to_tag_prefix(resolved_tags)
    ext_name = str(ext).replace("\\", "/").rsplit("/", 1)[-1].lstrip(".")
    safe_ext = sanitize_path_segment(ext_name, max_len=16)
    mapping = build_format_mapping(tag_pref, meta_map, ext=safe_ext)
    # Split the template before interpolation. This preserves separators that
    # the user explicitly placed in the template, while preventing a value such
    # as ``"music/test"`` from creating an unintended subdirectory.
    template_parts = path_template.replace("\\", "/").split("/")
    resolved_parts: list[str] = []
    for part in template_parts:
        resolved = resolve_template(part, mapping)
        resolved_parts.append(sanitize_path_segment(resolved))
    return Path("/".join(part for part in resolved_parts if part))


def export_audio(
    source: Path,
    dest: Path,
    fmt: str,
    options: dict[str, dict[str, str]],
    metadata: Mapping[str, str | list[str]],
    *,
    overwrite: bool = False,
    multi_entry: bool = True,
    preserve_metadata: bool = True,
) -> None:
    """Transcode one audio file with FFmpeg and write its metadata.

    ``source`` is decoded and written to ``dest`` in ``fmt``. ``options`` holds
    format-specific codec settings, while ``metadata`` contains logical tags.
    ``multi_entry`` controls whether list values become repeated native fields;
    ``preserve_metadata`` controls copying of unconfigured source metadata.
    Artwork is restored separately and is not re-encoded as video. Raises
    :class:`FileExistsError` when the destination exists without ``overwrite``
    and :class:`RuntimeError` when FFmpeg fails.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)

    opts = _merge_options(fmt.lower(), options)
    cmd = ["ffmpeg", "-nostdin"]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd.extend(["-i", str(source.resolve())])
    # Export only the audio stream. Artwork is copied explicitly below so that
    # attached-picture streams are not accidentally re-encoded or duplicated.
    cmd.extend(["-map", "0:a:0", "-map_metadata", "0" if preserve_metadata else "-1"])
    for k, value in _ffmpeg_metadata_args(metadata).items():
        values = value if isinstance(value, list) and multi_entry else [value]
        for item in values:
            cmd.extend(["-metadata", f"{k}={item}"])
    acodec = opts.get("acodec", "libopus")
    if fmt.lower() in {"mp3", "m4a"}:
        cmd.extend(["-c:v", "copy"])
    else:
        cmd.append("-vn")
    cmd.extend(["-c:a", acodec])
    if "audio_bitrate" in opts:
        cmd.extend(["-b:a", opts["audio_bitrate"]])
    if fmt.lower() == "wav":
        cmd.extend(["-ar", "44100"])
    cmd.append(str(dest.resolve()))

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode(errors="replace") if error.stderr else ""
        raise RuntimeError(f"ffmpeg export failed for {dest}:\n{stderr}") from error
    _copy_artwork(source, dest)
    _copy_rating(source, dest, metadata, preserve_metadata=preserve_metadata)
    _write_format_specific_overrides(dest, fmt, metadata, multi_entry=multi_entry)
    _remove_redundant_codec_aliases(dest, fmt, metadata)


def _rating_stars(value: Any) -> float | None:
    """Parse the public rating representation, expressed as 0..5 stars."""
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        stars = float(str(value).strip().lower().removesuffix(" stars"))
    except (TypeError, ValueError):
        return None
    return min(5.0, max(0.0, stars))


def _source_popm(source: Path) -> list[POPM]:
    try:
        audio = MutagenFile(source)
        if audio is not None and isinstance(audio.tags, ID3):
            return [frame for frame in audio.tags.getall("POPM") if isinstance(frame, POPM)]
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return []


def _copy_rating(
    source: Path,
    destination: Path,
    metadata: Mapping[str, str | list[str]],
    *,
    preserve_metadata: bool = True,
) -> None:
    """Map MP3 POPM ratings to the target container's rating convention.

    MP3 uses POPM's 0..255 byte scale. Vorbis-family containers use the
    generic ``Rating`` field with a 0..100 score. Legacy ``RATING:<email>``
    fields are removed from the output to avoid conflicting ratings.
    """
    source_frames = _source_popm(source) if preserve_metadata else []
    configured = _rating_stars(metadata.get("rating")) if "rating" in metadata else None
    if not source_frames and configured is None:
        return

    stars = configured if configured is not None else popm_to_stars(source_frames[0].rating)
    stars = min(5.0, max(0.0, stars))
    destination_format = destination.suffix.lower()
    try:
        target = MutagenFile(destination)
        if target is None:
            return
        if destination_format == ".mp3":
            tags = target.tags if isinstance(target.tags, ID3) else ID3()
            if target.tags is None:
                target.add_tags()
                tags = target.tags
            for frame in tags.getall("POPM"):
                tags.delall(f"POPM:{frame.email}")
            frames = source_frames or [POPM(email="user@email", rating=stars_to_popm(stars), count=0)]
            for frame in frames:
                tags.add(POPM(email=frame.email, rating=stars_to_popm(stars), count=getattr(frame, "count", 0)))
        elif destination_format in {".opus", ".ogg", ".flac"}:
            tags = target.tags
            if tags is None:
                target.add_tags()
                tags = target.tags
            for key in list(tags.keys()):
                if str(key).upper().startswith("RATING"):
                    del tags[key]
            # AIMP and several players use the generic Vorbis Rating field
            # with a 0..100 score (four stars = 80). Do not also write the
            # legacy RATING:<email> form: two values could diverge later.
            tags["Rating"] = [str(round(stars * 20))]
        elif destination_format == ".m4a":
            if target.tags is None:
                target.add_tags()
            target.tags["----:com.apple.iTunes:rating"] = [
                MP4FreeForm(str(round(stars, 2)).encode("utf-8"))
            ]
        else:
            # ASF and less common containers have no uniform Mutagen API. Keep
            # a readable custom value rather than dropping the user's stars.
            if target.tags is None:
                target.add_tags()
            target.tags["rating"] = [str(round(stars, 2))]
        target.save()
    except (OSError, KeyError, TypeError, ValueError):
        return


def _remove_redundant_codec_aliases(destination: Path, fmt: str, metadata: Mapping[str, str | list[str]]) -> None:
    """Remove copied aliases only where they are not canonical for the target."""
    if "bpm" not in metadata or fmt.lower() == "mp3":
        return
    try:
        audio = MutagenFile(destination)
        if audio is None or audio.tags is None:
            return
        for alias in ("TBPM", "tbpm"):
            if alias in audio.tags:
                del audio.tags[alias]
        audio.save()
    except (OSError, KeyError, TypeError, ValueError):
        return


def _picture_from_source(source: Path) -> list[Picture]:
    """Extract all embedded images into the common FLAC Picture representation."""
    audio = MutagenFile(source)
    if audio is None or audio.tags is None:
        return []
    if isinstance(audio.tags, ID3):
        pictures: list[Picture] = []
        for frame in audio.tags.getall("APIC"):
            picture = Picture()
            picture.type = frame.type
            picture.mime = frame.mime
            picture.desc = frame.desc
            picture.data = frame.data
            pictures.append(picture)
        return pictures
    if hasattr(audio, "pictures") and audio.pictures:
        return list(audio.pictures)
    raw = audio.tags.get("metadata_block_picture")
    if raw:
        pictures: list[Picture] = []
        values = raw if isinstance(raw, list) else [raw]
        for encoded in values:
            try:
                pictures.append(Picture(base64.b64decode(encoded)))
            except (TypeError, ValueError):
                continue
        return pictures
    covers = audio.tags.get("covr")
    if covers:
        pictures = []
        for cover in covers:
            try:
                data = bytes(cover)
                mime = "image/png" if getattr(cover, "imageformat", 0) == 14 else "image/jpeg"
                picture = Picture()
                picture.type = 3
                picture.mime = mime
                picture.desc = "Cover (front)"
                picture.data = data
                pictures.append(picture)
            except (TypeError, ValueError):
                continue
        return pictures
    return []


def _copy_artwork(source: Path, destination: Path) -> None:
    """Copy artwork explicitly because FFmpeg does not map picture metadata reliably."""
    try:
        pictures = _picture_from_source(source)
        if not pictures:
            return
        target = MutagenFile(destination)
        if target is None:
            return
        suffix = destination.suffix.lower()
        if suffix == ".mp3":
            if target.tags is None:
                target.add_tags()
            for picture in pictures:
                target.tags.add(APIC(encoding=3, mime=picture.mime, type=picture.type, desc=picture.desc, data=picture.data))
        elif suffix == ".m4a":
            if target.tags is None:
                target.add_tags()
            target.tags["covr"] = [MP4Cover(picture.data, imageformat=MP4Cover.FORMAT_PNG if picture.mime == "image/png" else MP4Cover.FORMAT_JPEG) for picture in pictures]
        elif suffix == ".flac":
            # FFmpeg may already have copied the source pictures as Vorbis
            # comments. Replace them so every image is preserved exactly once.
            target.clear_pictures()
            for picture in pictures:
                target.add_picture(picture)
        elif suffix in {".ogg", ".opus"}:
            if target.tags is None:
                target.add_tags()
            for key in list(target.tags):
                if str(key).casefold() == "metadata_block_picture":
                    del target.tags[key]
            target.tags["metadata_block_picture"] = [base64.b64encode(picture.write()).decode("ascii") for picture in pictures]
        else:
            return
        target.save()
    except (OSError, KeyError, TypeError, ValueError):
        # Unsupported artwork conversion must not make an otherwise valid export fail.
        return


def _write_format_specific_overrides(
    destination: Path,
    fmt: str,
    metadata: Mapping[str, str | list[str]],
    *,
    multi_entry: bool = True,
) -> None:
    """Write fields whose container spelling differs from FFmpeg's generic name."""
    try:
        target = MutagenFile(destination)
        if target is None or target.tags is None:
            return
        suffix = fmt.lower().lstrip(".")
        if suffix == "m4a":
            catalog = metadata.get("catalognumber")
            if catalog:
                if isinstance(catalog, list):
                    catalog = catalog[0]
                target.tags["----:com.apple.iTunes:CATALOGNUMBER"] = [
                    MP4FreeForm(str(catalog).encode("utf-8"))
                ]
            for logical in ("replaygain_track_gain", "replaygain_track_peak"):
                value = metadata.get(logical)
                if value:
                    value = value[0] if isinstance(value, list) else value
                    target.tags[f"----:com.apple.iTunes:{logical}"] = [MP4FreeForm(str(value).encode("utf-8"))]
            if metadata.get("tracknumber"):
                track_value = metadata["tracknumber"]
                if isinstance(track_value, list):
                    track_value = track_value[0]
                number = str(track_value).split("/", 1)[0].lstrip("0") or "0"
                target.tags["trkn"] = [(int(number), 0)]
        elif suffix == "mp3":
            id3 = target.tags if isinstance(target.tags, ID3) else ID3(destination)
            catalog = metadata.get("catalognumber")
            if catalog:
                if isinstance(catalog, list):
                    catalog = catalog[0]
                id3.delall("TXXX:CATALOGNUMBER")
                id3.add(TXXX(encoding=3, desc="CATALOGNUMBER", text=[str(catalog)]))
            if metadata.get("bpm"):
                bpm_value = metadata["bpm"]
                if isinstance(bpm_value, list):
                    bpm_value = bpm_value[0]
                # TBPM is the canonical MP3 representation. Replace copied
                # values rather than leaving a stale frame beside the update.
                id3.delall("TBPM")
                id3.add(TBPM(encoding=3, text=[str(bpm_value)]))
            for logical in ("replaygain_track_gain", "replaygain_track_peak"):
                value = metadata.get(logical)
                if not value:
                    continue
                if isinstance(value, list):
                    value = value[0]
                for frame in list(id3.getall("TXXX")):
                    if frame.desc and frame.desc.casefold() == logical.casefold():
                        id3.delall(f"TXXX:{frame.desc}")
                id3.add(TXXX(encoding=3, desc=logical, text=[str(value)]))
            target.tags = id3
        if suffix in {"ogg", "opus", "flac"}:
            # FFmpeg can leave the source value as a second Vorbis comment.
            # Remove case variants and write the conventional uppercase names;
            # some readers (notably MediaInfo) only display those spellings.
            vorbis_names = {
                "genre": "GENRE",
                "bpm": "BPM",
                "tracknumber": "TRACKNUMBER",
                "replaygain_track_gain": "REPLAYGAIN_TRACK_GAIN",
                "replaygain_track_peak": "REPLAYGAIN_TRACK_PEAK",
                "catalognumber": "CATALOGNUMBER",
            }
            for logical, canonical in vorbis_names.items():
                value = metadata.get(logical)
                if not value:
                    continue
                values = value if isinstance(value, list) and multi_entry else [value]
                for existing in list(target.tags):
                    if str(existing).casefold() == logical.casefold():
                        del target.tags[existing]
                target.tags[canonical] = [str(item) for item in values]
            if multi_entry:
                # FFmpeg's repeated -metadata options are not consistently
                # preserved by every Vorbis muxer. Re-apply every configured
                # list as one multi-valued comment after muxing.
                for logical, value in metadata.items():
                    if not isinstance(value, list):
                        continue
                    canonical = vorbis_names.get(logical, logical)
                    for existing in list(target.tags):
                        if str(existing).casefold() == str(logical).casefold():
                            del target.tags[existing]
                    target.tags[canonical] = [str(item) for item in value]
        target.save()
    except (OSError, KeyError, TypeError, ValueError):
        return


def _ffmpeg_metadata_args(meta: Mapping[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Map logical tag keys to ffmpeg metadata keys.
    
    Known keys (``artist``, ``title``, ``album``, etc.) are translated to
    their ffmpeg equivalents (e.g. ``tracknumber`` → ``track``).
    """
    key_map = {
        "artist": "artist",
        "title": "title",
        "album": "album",
        "genre": "genre",
        "date": "date",
        "tracknumber": "track",
        "discnumber": "disc",
        "composer": "composer",
        "albumartist": "album_artist",
        "comment": "comment",
        "lyrics": "lyrics",
        "copyright": "copyright",
        "publisher": "publisher",
        "encodedby": "encoded_by",
        "encoder": "encoder",
        "isrc": "isrc",
        "bpm": "bpm",
        "mood": "mood",
        "replaygain_track_gain": "REPLAYGAIN_TRACK_GAIN",
        "replaygain_album_gain": "REPLAYGAIN_ALBUM_GAIN",
    }
    out: dict[str, str | list[str]] = {}
    for logical, ff in key_map.items():
        v = meta.get(logical)
        if v is not None and str(v).strip() != "":
            out[ff] = v
    # Custom tags and less common standard tags are passed through unchanged.
    for logical, value in meta.items():
        if (
            logical not in key_map
            and not str(logical).startswith("tag_")
            and value is not None
            and str(value).strip()
            and logical != "rating"
            and not isinstance(value, dict)
        ):
            out[str(logical)] = value
    return out


def export_multiple_formats(
    source: Path,
    output_root: Path,
    path_template: str,
    formats: str | list[str],
    resolved_tags: Mapping[str, str | list[str]],
    meta_map: Mapping[str, Any],
    format_options: dict[str, dict[str, str]],
    *,
    overwrite: bool = False,
    multi_entry: bool = True,
    preserve_metadata: bool = True,
) -> list[Path]:
    """Export one source to each requested format.

    The path template is resolved independently for every extension. ``resolved_tags``
    supplies both filename values and metadata; ``format_options`` supplies
    codec settings. ``multi_entry`` and ``preserve_metadata`` are forwarded to
    :func:`export_audio`. Returns the destination paths in format order.
    """

    fmts = [formats] if isinstance(formats, str) else list(formats)
    out_paths: list[Path] = []
    for fmt in fmts:
        ext = fmt.lower().lstrip(".")
        rel = build_output_path(
            path_template,
            resolved_tags,
            meta_map,
            ext,
        )
        dest = output_root / rel
        export_audio(
            source,
            dest,
            ext,
            format_options,
            resolved_tags,
            overwrite=overwrite,
            multi_entry=multi_entry,
            preserve_metadata=preserve_metadata,
        )
        out_paths.append(dest)
    return out_paths
