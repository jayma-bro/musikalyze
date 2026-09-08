# musikalyze

**Analyse audio** (Essentia + TensorFlow → EffNet Discogs, MAEST, classifier heads), **lit les tags** (Mutagen), et **exporte en multi-formats** avec ffmpeg en utilisant des gabarits de chemins.

Pre-trained **Essentia model weights** are licensed under **CC BY-NC-SA 4.0** (non-commercial). See [Essentia models](https://essentia.upf.edu/documentation/models.html).

Full **API and metadata key reference**: [METADATA.md](METADATA.md).

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH`
- `essentia-tensorflow` (ML backend)

## Install

```bash
pip install -e .
pip install -e ".[dev]"  # optional: dev dependencies (pytest, ruff)
```

## Models

Download `.pb` / `.json` from [Essentia models](https://essentia.upf.edu/documentation/models.html):

- **Embedding models**: Discogs EffNet, MAEST (for feature extraction)
- **Classifier heads**: Genre/mood classification/regression models (`.pb` + label lists in `.json`)

For **MAEST embeddings**, the `EmbeddingModel` uses Essentia's `TensorflowPredictMAEST` with optional `patch_size`, `patch_hop_size`, and `batch_size`. Classifier heads use the generic `TensorflowPredict2D` API.

> **Note**: `input_tensor` / `output_tensor` names vary per `.pb` file. Adjust them on each `LabelExtractor` and `EmbeddingModel` to match your graph.

## Quick Start

```python
from pathlib import Path

from musikalyze import (
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    MusicProcess,
    TaggingConfig,
)

# Define embedding models (feature extractors)
effnet = EmbeddingModel(
    name="effnet",
    embedding_model=Path("./models/discogs-effnet-bs64-1.pb"),
)
maest = EmbeddingModel(
    name="maest",
    embedding_model=Path("./models/discogs-maest-30s-pw-519l-2.pb"),
)

# Define classifier heads on top of embeddings
genre400 = LabelExtractor(
    name="genre400",
    embedder_name="effnet",
    graph_path=Path("./models/genre_discogs400-discogs-effnet-1.pb"),
    labels_path=Path("./models/genre_discogs400-discogs-effnet-1.json"),
    category="genre",
)

mood_happy = LabelExtractor(
    name="happy",
    embedder_name="effnet",
    graph_path=Path("./models/mood_happy-discogs-effnet-1.pb"),
    labels_path=Path("./models/mood_happy-discogs-effnet-1.json"),
    category="mood",
)

# Configure tagging templates and export settings
music = MusicProcess(
    audio_file=Path("./data/track.mp3"),
    embedders=[effnet, maest],
    extractors=[genre400, mood_happy],
    tagging_config=TaggingConfig(
        artist="{tag_artist}",
        title="{tag_title}",
        genre="{meta_genre_main}",
    ),
    export_config=ExportConfig(
        output_root=Path("./output"),
        formats="opus",
        path_template="{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}",
        format_options={"opus": {"audio_bitrate": "160k"}},
    ),
)

# Run the full pipeline
music.process_file()
# Or step by step: read_tags(), load_audio(), analyze_file(), tag_file(), export_file()

# Access predictions programmatically
labels = music.labels  # dict of all computed labels
bpm = music.meta_bpm  # shortcut: same as music.label("meta_bpm")
```

## Workflow

`MusicProcess` provides both a full pipeline and step-by-step methods:

| Method | Description |
|---|---|
| `process_file()` | Full pipeline: read tags → load audio → analyze → tag → export |
| `read_tags()` | Read metadata from the source file (Mutagen) |
| `load_audio()` | Decode to mono float32 at 16 kHz for Essentia analysis |
| `analyze_file()` | Compute all registered embedding models (EffNet, MAEST) once |
| `tag_file()` | Resolve `TaggingConfig` templates into resolved tag values |
| `export_file()` | Transcode with ffmpeg and write metadata |
| `labels` | Property — returns all computed labels as a `dict[str, Any]` |
| `label(key)` | Programmatic access to a single key or a list of keys |
| `preview_path(ext)` | Resolved output path without writing to disk |
| `format_preview(template)` | Resolve an arbitrary template string with current tags + metadata |
| `audio_mono` | Property — mono audio signal after `load_audio()` |
| `tags_original` | Property — original tags read from the file |
| `tags_resolved` | Property — resolved tags after `tag_file()` |

### Lazy Evaluation

- **Embeddings** (`EffNet`, `MAEST`) are computed **once** via `analyze_file()` and cached.
- **Classifier heads** and **classical descriptors** (`meta_bpm`, `meta_key`, etc.) run **lazily** — only when a template or `label()` needs them.
- Calling `load_audio()` invalidates the embedding cache, so subsequent `analyze_file()` recomputes everything fresh.

### Meta Access via Attributes

Any attribute starting with `meta_` resolves automatically:

```python
music.meta_bpm      # → same as music.label("meta_bpm")
music.meta_genre    # → top genre labels
music.meta_mood_happy_val  # → confidence score for mood_happy
```

## Templates

Templates use Python `str.format` syntax: `{tag_artist}`, `{meta_genre}`, `{tag_track_number:02d}`, etc.

- `{tag_*}` values come from the source file's metadata tags.
- `{meta_*}` values come from audio analysis (classical Essentia descriptors + ML model predictions).
- Missing keys resolve to empty strings — no errors.

### Export and Tags

Unless you map a field in `TaggingConfig`, its value is **not recomputed**: export metadata starts from the original file tags and **overrides** only the logical keys produced by `tag_file()`.

## Parallel Batch Processing

`process_files_parallel` processes multiple audio files concurrently using separate processes:

```python
from pathlib import Path
from musikalyze import list_audio_files, process_files_parallel

paths = list_audio_files(Path("./library"))
results = process_files_parallel(
    paths,
    embedders=[effnet],
    extractors=[genre400],
    tagging_config=TaggingConfig(),
    export_config=ExportConfig(output_root=Path("./out"), formats="opus"),
    max_workers=4,
)
# results: list of (audio_path_str, success: bool, error_message_or_None)
```

Use a `if __name__ == "__main__":` guard on platforms that require it (e.g., Windows).

```python
from musikalyze import sample_audio_files

paths = sample_audio_files(Path("./library"), sample=0.1)  # 10% sample
```

## Error Classes

| Exception | Meaning |
|---|---|
| `UnknownEmbedderError` | An extractor references an embedding name not in the `embedders` list |
| `PredictionError` | TensorFlow / Essentia inference failure for a classifier head |
| `UnknownMetaKeyError` | Requested metadata key is unavailable |

## Embedding Models

| `EmbeddingModel.name` | Essentia Algorithm | Notes |
|---|---|---|
| `"effnet"` | `TensorflowPredictEffnetDiscogs` | Discogs EffNet embedding model |
| `"maest"` | `TensorflowPredictMAEST` | MAEST with optional `patch_size`, `patch_hop_size`, `batch_size` |

## Supported Audio Formats

Essentia handles: `.wav`, `.mp3`, `.flac`, `.aiff`, `.ogg`. Other formats (`.m4a`, `.aac`, `.wma`, etc.) are converted via ffmpeg on the fly.
