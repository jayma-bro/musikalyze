# musikalyze

Python library to **analyze** audio with **Essentia + TensorFlow** (EffNet Discogs, MAEST, classifier heads), **read/write tags** (Mutagen), and **transcode** with **ffmpeg** using path templates.

Pre-trained **Essentia model weights** are licensed under **CC BY-NC-SA 4.0** (non-commercial). See [Essentia models](https://essentia.upf.edu/documentation/models.html).

Full **API and metadata key reference**: [METADATA.md](METADATA.md).

## Requirements

- Python 3.10+
- `ffmpeg` on `PATH`
- ML stack: `pip install -e ".[tensorflow]"` (`essentia-tensorflow`)

## Install

```bash
cd music_organizer
pip install -e .
pip install -e ".[tensorflow]"
```

## Models

Download `.pb` / `.json` from [Essentia models](https://essentia.upf.edu/documentation/models.html) (e.g. [discogs-effnet](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/), MAEST, genre/mood heads). Set `input_tensor` / `output_tensor` on each `LabelExtractor` to match the graph. For **MAEST embeddings**, use `EmbeddingModel(backend="maest", …)`; musikalyze calls `**TensorflowPredictMAEST`** (not the generic `TensorflowPredict` pool API).

## Usage

### Embeddings + label heads

1. `**analyze_file()**` runs **every** `EmbeddingModel` once and caches the tensors.
2. **Label heads** and **classical** descriptors (`meta_bpm`, etc.) run **only when** a template or `label()` needs them.

```python
from pathlib import Path

from musikalyze import (
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    MusicProcess,
    TaggingConfig,
)

data = Path("./data")

effnet = EmbeddingModel(
    name="effnet",
    embedding_model=data / "discogs-effnet-bs64-1.pb",
    embedding_output="PartitionedCall:1",
    backend="effnet_discogs",
)
maest = EmbeddingModel(
    name="maest",
    embedding_model=data / "discogs-maest-30s-pw-519l-2.pb",
    embedding_output="PartitionedCall/Identity_12",  # adjust to your graph
    backend="maest",
)

genre400 = LabelExtractor(
    name="genre400",
    category="genre",
    embedder="effnet",
    graph_path=data / "genre_discogs400-discogs-effnet-1.pb",
    labels_path=data / "genre_discogs400-discogs-effnet-1.json",
    genre_main=True,
    genre_count=5,
)

music = MusicProcess(
    audio_file=data / "track.mp3",
    embedders=[effnet, maest],
    label_extractors=[genre400],
    tagging_config=TaggingConfig(
        genre="{meta_genre_main}",
        artist="{tag_artist}",
        title="{tag_title}",
    ),
    export_config=ExportConfig(
        output_root=Path("./output"),
        formats="opus",
        path_template="{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}",
        format_options={"opus": {"audio_bitrate": "160k"}},
    ),
)

music.process_file()
# Or step by step: read_tags, load_audio, analyze_file, tag_file, export_file
# music.meta_bpm, music.label("meta_mood_happy"), music.label(["meta_key", "meta_bpm"])
```

### Templates

Only Python `str.format` syntax: `{tag_artist}`, `{meta_genre}`, `{tag_track_number:02d}`, etc.

### Export and tags

Unless you map a field in `TaggingConfig`, its value is **not** recomputed: **export metadata starts from the original file tags** and **overrides** only the logical keys produced by `tag_file()` (e.g. if you only set the `genre` template, other tags stay as on disk). ffmpeg receives the merged map (see `merge_logical_tags_for_export` in `tagging.py`).

### Parallel batch

`process_files_parallel` takes the same `embedders` / `label_extractors` / configs and pickles them into workers. Use a `if __name__ == "__main__":` guard on some platforms.

```python
from pathlib import Path

from musikalyze import ExportConfig, MusicProcess, TaggingConfig, list_audio_files, process_files_parallel

# build embedders / extractors as above, then:
paths = list_audio_files(Path("./library"))
process_files_parallel(
    paths,
    embedders=[effnet],
    label_extractors=[genre400],
    tagging_config=TaggingConfig(),
    export_config=ExportConfig(output_root=Path("./out"), formats="opus"),
    max_workers=4,
)
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Embedding backends


| `EmbeddingModel.backend` | Essentia algorithm                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| `effnet_discogs`         | `TensorflowPredictEffnetDiscogs`                                                                  |
| `maest`                  | `TensorflowPredictMAEST` (optional: `patch_size`, `patch_hop_size`, `batch_size`, `input_tensor`) |


## Limitations

- **WMA** and some containers: limited tag support.
- **ReplayGain in `meta_*`**: read from tags only; loudness **computation** is future work.
- **Tensor names** vary per `.pb`; adjust `LabelExtractor` / `EmbeddingModel` accordingly.