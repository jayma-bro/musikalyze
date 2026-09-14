"""Transcode with ffmpeg, metadata, and path templates."""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.flac import Picture
from mutagen.id3 import APIC, ID3, POPM
from mutagen.mp4 import MP4FreeForm

from musikalyze.audio_io import FFMPEG_TIMEOUT
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
            text = str(resolved[key]).split("/", 1)[0].lstrip("0") or "0"
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
    metadata: Mapping[str, str],
    *,
    overwrite: bool = False,
) -> None:
    """Transcode *source* audio to *dest* using ffmpeg.
    
    Writes metadata tags and applies codec settings from *options*.
    Raises :exc:`FileExistsError` if *dest* already exists and *overwrite* is ``False``.
    Uses a timeout of :data:`~musikalyze.audio_io.FFMPEG_TIMEOUT` seconds.
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
    # Opus/Ogg/FLAC store artwork as metadata, not as a copied video stream.
    # MP3 and MP4 can keep an attached picture stream during transcoding.
    if fmt.lower() in {"mp3", "m4a"}:
        cmd.extend(["-map", "0", "-map_metadata", "0"])
    else:
        cmd.extend(["-map", "0:a:0", "-map_metadata", "0"])
    for k, v in _ffmpeg_metadata_args(metadata).items():
        cmd.extend(["-metadata", f"{k}={v}"])
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
    _copy_rating(source, dest, metadata)
    _remove_redundant_codec_aliases(dest, fmt, metadata)


def _rating_stars(value: Any) -> float | None:
    """Parse the public rating representation, expressed as 0..5 stars."""
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


def _copy_rating(source: Path, destination: Path, metadata: Mapping[str, str]) -> None:
    """Map MP3 POPM ratings to the target container's rating convention.

    MP3 uses POPM's 0..255 byte scale. Vorbis-family containers use
    ``RATING:<email>`` with a 0..1 value. For formats without an established
    portable equivalent, a freeform ``rating`` value in stars is retained.
    """
    source_frames = _source_popm(source)
    configured = _rating_stars(metadata.get("rating")) if "rating" in metadata else None
    if not source_frames and configured is None:
        return

    stars = configured if configured is not None else source_frames[0].rating * 5.0 / 255.0
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
            frames = source_frames or [POPM(email="user@email", rating=round(stars * 255 / 5), count=0)]
            for frame in frames:
                tags.add(POPM(email=frame.email, rating=round(stars * 255 / 5), count=frame.count))
        elif destination_format in {".opus", ".ogg", ".flac"}:
            tags = target.tags
            if tags is None:
                target.add_tags()
                tags = target.tags
            for key in list(tags.keys()):
                if str(key).upper().startswith("RATING"):
                    del tags[key]
            frames = source_frames or [POPM(email="user@email", rating=0, count=0)]
            for frame in frames:
                email = frame.email or "user@email"
                tags[f"RATING:{email}"] = [str(stars / 5.0)]
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


def _remove_redundant_codec_aliases(destination: Path, fmt: str, metadata: Mapping[str, str]) -> None:
    """Remove raw ID3 aliases that ffmpeg may copy beside canonical Vorbis tags.

    For example, an MP3 ``TBPM`` frame can survive metadata mapping as a literal
    ``TBPM`` Vorbis comment while the canonical ``bpm`` comment is also written.
    ``TBPM`` and ``bpm`` represent the same logical field, so keeping both would
    expose stale values to tag readers. Other unknown tags remain untouched.
    """
    if fmt.lower() in {"mp3", "m4a"} or "bpm" not in metadata:
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


def _copy_artwork(source: Path, destination: Path) -> None:
    """Copy embedded artwork through the target container's metadata system."""
    try:
        source_file = MutagenFile(source)
        destination_file = MutagenFile(destination)
        if source_file is None or destination_file is None or source_file.tags is None:
            return

        source_tags = source_file.tags
        destination_tags = destination_file.tags
        if destination_tags is None:
            destination_file.add_tags()
            destination_tags = destination_file.tags

        apic_frames = [frame for frame in source_tags.values() if isinstance(frame, APIC)]
        if apic_frames and hasattr(destination_tags, "add"):
            for frame in apic_frames:
                destination_tags.add(frame)
        elif apic_frames and destination_tags is not None:
            frame = apic_frames[0]
            picture = Picture()
            picture.type = 3
            picture.mime = frame.mime
            picture.desc = frame.desc or "Cover (front)"
            picture.data = frame.data
            encoded = base64.b64encode(picture.write()).decode("ascii")
            destination_tags["metadata_block_picture"] = [encoded]
        elif "covr" in source_tags and "covr" in destination_tags:
            destination_tags["covr"] = source_tags["covr"]
        elif "covr" in source_tags and hasattr(destination_file, "__setitem__"):
            destination_file["covr"] = source_tags["covr"]
        elif hasattr(source_file, "pictures") and source_file.pictures:
            picture = source_file.pictures[0]
            if hasattr(destination_file, "add_picture"):
                destination_file.add_picture(picture)
            elif destination_tags is not None:
                encoded = base64.b64encode(picture.write()).decode("ascii")
                destination_tags["metadata_block_picture"] = [encoded]
        destination_file.save()
    except (OSError, KeyError, TypeError, ValueError):
        # Artwork support varies by container; audio export must still succeed.
        return


def _ffmpeg_metadata_args(meta: Mapping[str, str]) -> dict[str, str]:
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
    out: dict[str, str] = {}
    for logical, ff in key_map.items():
        v = meta.get(logical)
        if v is not None and str(v).strip() != "":
            out[ff] = str(v)
    # Custom tags and less common standard tags are passed through unchanged.
    for logical, value in meta.items():
        if (
            logical not in key_map
            and not str(logical).startswith("tag_")
            and value is not None
            and str(value).strip()
            and not isinstance(value, (dict, list))
        ):
            out[str(logical)] = str(value)
    return out


def export_multiple_formats(
    source: Path,
    output_root: Path,
    path_template: str,
    formats: str | list[str],
    resolved_tags: Mapping[str, str],
    meta_map: Mapping[str, Any],
    format_options: dict[str, dict[str, str]],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Transcode *source* to one or more formats under *output_root*.
    
    Resolves the path template for each format, creates destination paths, and
    calls :func:`export_audio` for each.  Returns a list of the created paths.
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
        )
        out_paths.append(dest)
    return out_paths
