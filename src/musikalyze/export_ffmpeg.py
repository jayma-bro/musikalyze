"""Transcode with ffmpeg, metadata, and path templates."""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.flac import Picture
from mutagen.id3 import APIC

from musikalyze.audio_io import FFMPEG_TIMEOUT
from musikalyze.templates import (
    build_format_mapping,
    resolve_template,
    sanitize_path_segment,
    sanitize_relative_path,
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
    *,
    sanitize: bool = True,
) -> Path:
    """Build a relative output path from the template, including the file extension.
    
    Resolves ``{tag_*}`` and ``{meta_*}`` placeholders in *path_template*, then
    sanitises each path segment (removing characters like ``< > : " / \\ | ? *``).
    """

    tag_pref = logical_tags_to_tag_prefix(resolved_tags)
    ext_name = str(ext).replace("\\", "/").rsplit("/", 1)[-1].lstrip(".")
    safe_ext = sanitize_path_segment(ext_name, max_len=16)
    mapping = build_format_mapping(tag_pref, meta_map, ext=safe_ext)
    raw = resolve_template(path_template, mapping)
    if sanitize:
        raw = sanitize_relative_path(raw)
    return Path(raw)


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
    sanitize_paths: bool = True,
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
            sanitize=sanitize_paths,
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
