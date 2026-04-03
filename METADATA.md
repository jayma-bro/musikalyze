# Reference: `MusicProcess` API and metadata keys

## `MusicProcess` — workflow methods (typical order)

Call these in order for a full pass; you can also stop after any step for debugging.

| Method | Description |
|--------|-------------|
| `read_tags()` | Read metadata from the source file (Mutagen). Fills `tags_original` / internal `tag_*` map. |
| `load_audio()` | Decode to mono float32 at 16 kHz for Essentia. |
| `analyze_file()` | **Computes every registered embedding once** (`EmbeddingModel` list). Safe to call multiple times only after `load_audio()` invalidates caches. |
| `tag_file()` | Resolves `TaggingConfig` templates. Runs `analyze_file()` if needed, then computes only the model heads and classical features referenced in templates (lazy). |
| `export_file()` | Transcode with ffmpeg and write metadata. Requires `export_config`. Merges original file tags with resolved fields (see README). |
| `process_file()` | `read_tags` → `load_audio` → `analyze_file` → `tag_file` → `export_file` (if export is configured). |
| `preview_path(ext=...)` | Resolved output path without writing. |
| `format_preview(template)` | Resolve an arbitrary template string with current tags + metadata. |

## `MusicProcess` — accessors

| Member | Description |
|--------|-------------|
| `audio_path` | `Path` to the input audio file. |
| `audio_mono` | Mono 16 kHz signal after `load_audio()`. |
| `tags_original` | Copy of logical tags read from the file (`artist`, `title`, …). |
| `tags_resolved` | Logical fields after `tag_file()` (used for export / ffmpeg). |
| `label(key)` | Programmatic access: one key (`"meta_bpm"` or `"bpm"`) or a **list** of keys → list of values. Supports `tag_*` keys. |
| `meta_*` (attributes) | e.g. `music.meta_bpm` → same as `music.label("meta_bpm")`. |

There is no `analysis` object: metadata for templates lives in a flat `meta_*` namespace built on demand.

## File tags: `tag_*` prefix

Templates use `{tag_artist}`, `{tag_title}`, etc. Values come from `read_tags_raw()` (logical keys), then prefixed for formatting.

**Commonly read keys** (best effort; container and format dependent):

`artist`, `title`, `album`, `genre`, `date`, `tracknumber`, `discnumber`, `composer`, `albumartist`, `comment`, `lyrics`, `copyright`, `publisher`, `encodedby`, `encoder`, `isrc`, `language`, `albumsort`, `artistsort`, `titlesort`, `website`, `bpm`, `mood`, `grouping`, plus ReplayGain fields such as `replaygain_track_gain` / `replaygain_album_gain` when present.

This is **not** an exhaustive list of every frame every format can store; rare or vendor-specific frames may require extending `read_tags_raw()` in `tagging.py`.

## Model metadata: `meta_*` prefix

### Classical (Essentia, computed when referenced)

| Key | Description |
|-----|-------------|
| `meta_bpm` | Integer BPM (Percival estimator, rounded). |
| `meta_key` | Key from `KeyExtractor`. |
| `meta_scale` | Scale/mode from `KeyExtractor`. |
| `meta_danceability` | Scalar in ~[0, 1] from Essentia `Danceability`. |

### ReplayGain from **existing file tags**

If the template references them, values are copied from the file (not recomputed):

| Key | Source |
|-----|--------|
| `meta_replaygain_track` | e.g. `replaygain_track_gain` / `REPLAYGAIN_TRACK_GAIN` when present. |
| `meta_replaygain_album` | Album ReplayGain field when present. |

**Note:** Computing ReplayGain from audio (EBU R128 / scan) is not implemented yet; only passthrough from tags.

### Aggregates (multiple `LabelExtractor` with `category="genre"` / `"mood"`)

| Key | Description |
|-----|-------------|
| `meta_genre` | Combined processed genre segments from all genre extractors (deduplicated order). |
| `meta_genre_main` | “Main” segments (text before `---` / `//`, etc.). |
| `meta_genre_sub` | “Subgenre” segments (last segment after a separator). |
| `meta_mood` | Top-1 label from each mood extractor, joined with `TaggingConfig.separator`. |

### Per `LabelExtractor`

Naming:

- **Genre** → `meta_genre_<name>`, plus `meta_genre_<name>_main`, `meta_genre_<name>_sub`.
- **Mood** → `meta_mood_<name>`.
- **Other** → `meta_<name>` (e.g. `meta_instrument`).

Suffixes on the **base key** (e.g. `meta_mood_happy`):

| Suffix | Meaning |
|--------|---------|
| *(base)* | Top-1 label string (for genre extractors: joined selected genres). |
| `_val` | Top-1 score (probability or `1.0` for tiny regression outputs). |
| `_val_str` | Same as string. |
| `_dict` | `dict` mapping label → probability. |
| `_dict_str` | JSON string of that dict. |
| `_all` | Same as `_dict` (full distribution). |
| `_all_str` | Same as `_dict_str`. |

## Errors

- **`UnknownEmbedderError`**: An extractor references an embedding name not listed in `embedders`.
- **`PredictionError`**: TensorFlow / Essentia failure for a head.
- Missing template keys resolve to **empty strings**; `label()` may still return `None` for unknown keys after a full meta build.
