"""Export transcodage via ffmpeg, métadonnées et gabarits de chemin."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from musikalize.templates import build_format_mapping, resolve_template, sanitize_relative_path

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
    base = dict(_FORMAT_DEFAULTS.get(fmt, {"acodec": "libopus", "audio_bitrate": "160k"}))
    base.update(user.get(fmt, {}))
    return base


def logical_tags_to_tag_prefix(resolved: Mapping[str, str]) -> dict[str, Any]:
    """``artist`` → ``tag_artist`` pour gabarits de chemin."""

    return {f"tag_{k}": v for k, v in resolved.items() if v is not None}


def build_output_path(
    path_template: str,
    resolved_tags: Mapping[str, str],
    meta_map: Mapping[str, Any],
    ext: str,
    *,
    sanitize: bool = True,
) -> Path:
    """Construit un chemin relatif depuis le gabarit (avec ``{ext}``)."""

    tag_pref = logical_tags_to_tag_prefix(resolved_tags)
    mapping = build_format_mapping(tag_pref, meta_map, ext=ext)
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
    """Transcode ``source`` vers ``dest`` avec ffmpeg."""

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
    for k, v in _ffmpeg_metadata_args(metadata).items():
        cmd.extend(["-metadata", f"{k}={v}"])
    acodec = opts.get("acodec", "libopus")
    cmd.extend(["-c:a", acodec])
    if "audio_bitrate" in opts:
        cmd.extend(["-b:a", opts["audio_bitrate"]])
    if fmt.lower() == "wav":
        cmd.extend(["-ar", "44100"])
    cmd.append(str(dest.resolve()))

    subprocess.run(cmd, check=True, capture_output=True)


def _ffmpeg_metadata_args(meta: Mapping[str, str]) -> dict[str, str]:
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
    }
    out: dict[str, str] = {}
    for logical, ff in key_map.items():
        v = meta.get(logical)
        if v is not None and str(v).strip() != "":
            out[ff] = str(v)
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
    """Exporte ``source`` vers plusieurs formats sous ``output_root``."""

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
