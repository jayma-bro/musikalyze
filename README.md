# musikalyze

`musikalyze` is a Python toolkit for analysing, tagging and exporting music
libraries. It combines Essentia/TensorFlow models with Mutagen and FFmpeg to
provide:

- EffNet and MAEST embeddings;
- genre, mood and other label extractors;
- classical audio descriptors such as BPM, key and ReplayGain;
- template-based metadata tagging;
- metadata-preserving transcoding to common audio formats;
- retagging without re-encoding;
- batch processing, a DataFrame API and a command-line interface;
- optional Plotly-based visualisation through `MusicEDA`.

The current release is **1.2.0**.

## Installation

Install the package in the Python environment used for analysis:

```bash
python -m pip install -e .
```

Development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Optional visualisation dependencies:

```bash
python -m pip install -e ".[viz]"
```

FFmpeg is required for transcoding and for decoding formats that Essentia does
not read directly. Check that it is available with:

```bash
ffmpeg -version
```

Analysis also requires the Essentia/TensorFlow model files used by the chosen
embedding models and label extractors. Model files are not bundled with the
package.

## Core concepts

- `EmbeddingModel` loads an Essentia embedding model such as EffNet or MAEST.
- `LabelExtractor` applies a classification, multilabel or regression head to
  an embedding.
- `MusicProcess` analyses and exports one audio file.
- `MusicBatch` applies the same pipeline to a directory, sequentially and with
  a `tqdm` progress bar.
- `TaggingConfig` declares the tags that should be written.
- `ExportConfig` controls transcoding, retagging and output paths.
- `MusicEDA` provides optional exploratory visualisation for analysis data.

Internal model scores are normally floats in `0..1`. User-facing thresholds
and percentage metadata use integer percentages in `0..100`.

## Python example

The following is a compact version of the workflow used in the demonstration
notebook. Paths are examples and must point to the models available on the
local machine.

```python
from pathlib import Path

from musikalyze import (
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    MusicBatch,
    TaggingConfig,
)

models = Path("./models")

effnet = EmbeddingModel(
    embedding_model=models / "discogs-effnet-bs64-1.pb",
    name="effnet",
)

# A multilabel genre model. ``thold`` is a percentage, not a 0..1 float.
genre = LabelExtractor(
    name="genre512",
    embedder_name="effnet",
    graph_path=models / "genre_discogs400-discogs-effnet-1.pb",
    labels_path=models / "genre_discogs400-discogs-effnet-1.json",
    category="genre",
    task="multilabel",
    count=3,
    thold=40,
    count_thold_policy="union",
)

# A regression model can map a score to user-defined percentage intervals.
approachability = LabelExtractor(
    name="approachability",
    embedder_name="effnet",
    graph_path=models / "approachability_regression-discogs-effnet-1.pb",
    labels_path=models / "approachability_regression-discogs-effnet-1.json",
    category="mood",
    label_names={
        "approachable_low": (0, 35),
        "approachable_mid": (36, 70),
        "approachable_high": (71, 100),
    },
    output_tensor="model/Identity",
    task="regression",
)

tagging = TaggingConfig(
    separator=";",
    multi_entry=True,
    tags={
        "genre": "{meta_genres}",
        "mood": "{meta_moods}",
        "key": "{meta_key}",
        "bpm": "{meta_bpm}",
        "copyright": "{meta_genres_main};{meta_scale}",
    },
    extra={
        "approachability": "{meta_mood_approachability_val_pct}",
    },
)

export = ExportConfig(
    output_root=Path("./output"),
    formats="opus",
    path_template="{tag_artist}/{tag_tracknumber_f} - {tag_title}.{ext}",
    format_options={"opus": {"audio_bitrate": "256k"}},
)

batch = MusicBatch(
    audio_path=Path("./library"),
    embedders=[effnet],
    extractors=[genre, approachability],
    tagging_config=tagging,
    export_config=export,
    tempo_model_path=models / "deeptemp-k16-3.pb",  # optional
)

batch.export()
```

The same configuration can be used for one file with `MusicProcess`. The
batch API uses `ExportConfig.output_root`; an optional argument to
`batch.export(path)` temporarily overrides that output directory.

## Analysis and DataFrames

A single file can expose one value or all available metadata:

```python
from musikalyze import MusicProcess

process = MusicProcess(
    audio_file=Path("./library/song.mp3"),
    embedders=[effnet],
    extractors=[genre],
)

process.label("meta_genres")
process.label("meta_bpm")
process.labels
```

Batch analysis returns one row per file:

```python
df = batch.analyze("analyze")
```

Useful targeted forms include:

```python
df = batch.analyze("meta_genres")
df = batch.analyze(["meta_genres", "meta_bpm", "tag_artist"])
```

`analyze("analyze")` includes file information and grouped metadata such as
`metas_all_pct`. Nested dictionaries can be expanded with:

```python
df = batch.explode_metas(df, "metas_all_pct")
```

## Metadata and tagging model

Metadata has two namespaces:

- `tag_*` values are read from the original file;
- `meta_*` values are computed by musikalyze.

For example:

```text
{tag_artist}
{tag_title}
{tag_tracknumber_f}
{meta_genres}
{meta_genres_main}
{meta_mood_happy_val_pct}
{meta_bpm}
```

`TaggingConfig.tags` contains standard logical tags. `TaggingConfig.extra`
contains custom tags such as model scores. Values are split on the configured
separator, duplicate or empty entries are removed, and `multi_entry=True`
(default) writes them as separate tag entries when the target format supports
multiple values. Set `multi_entry=False` to write one separator-joined value.

Exports preserve original metadata by default. Only tags explicitly declared
in `tags` or `extra` are replaced or added. Artwork is preserved when the
container supports it. Logical tag names are translated to the appropriate
ID3, Vorbis, MP4 or ASF representation. Ratings written to Vorbis-family
containers include the generic `Rating=0..100` convention used by players
such as AIMP. Legacy `RATING:<email>` values are read but not written, so two
rating fields cannot diverge.

Genre metadata follows the selected model scores. `meta_genres_main` contains
the main part of the highest-scoring complete genre, while
`meta_genres_sub` contains selected subgenres without duplicates.

See [`METADATA.md`](METADATA.md) for the compact metadata reference.

## Transcoding and retagging

Normal export transcodes audio with FFmpeg:

```python
ExportConfig(
    output_root=Path("./output"),
    formats="opus",
    format_options={"opus": {"audio_bitrate": "256k"}},
)
```

A bitrate such as `256k` is an encoder target, not an exact measured bitrate.
Opus is a lossy codec; 256 kbps is generally considered very high quality, but
it cannot improve a lossy source such as an MP3.

To change tags without re-encoding:

```python
ExportConfig(
    output_root=Path("./retagged"),
    retag=True,
)
```

In retag mode, the source file is copied and only the metadata is modified.
Format, bitrate and FFmpeg codec options are ignored. The original encoding,
audio stream and artwork are preserved.

`delete_after=True` can be set in `ExportConfig` when the source should be
removed only after a successful export.

Output path templates are always sanitized. Separators written in the template
create directories, while separators coming from a metadata value are escaped:

```text
{tag_artist}/{tag_title}.{ext}
```

A title such as `music/test` becomes `music_test.opus`, not an unintended
nested directory.

## Optional TempoCNN model

`meta_bpm` uses Essentia's `RhythmExtractor2013` by default. An external
TempoCNN model can be selected with:

```python
MusicProcess(
    audio_file=Path("song.opus"),
    tempo_model_path=Path("models/deeptemp-k16-3.pb"),
)
```

The model is loaded only when BPM metadata is requested. Opus and other
unsupported source containers are decoded through the FFmpeg fallback before
TempoCNN receives 11025 Hz mono audio.

## Command-line interface

The package installs the `musikalyze` command:

```bash
musikalyze ./library --config ./config.json export ./output
```

A single file is also accepted:

```bash
musikalyze ./song.mp3 --config ./config.json export ./output
```

Other commands:

```bash
musikalyze ./library --config ./config.json analyze
musikalyze ./library --config ./config.json analyze --key meta_genres
musikalyze ./library --config ./config.json analyze --output analysis.json
musikalyze ./library --config ./config.json preview
```

The command requires a JSON configuration. Relative paths inside the JSON are
resolved relative to the configuration file. See [`config.schema.json`](config.schema.json)
and [`config.example.json`](config.example.json).

At the start of an export, musikalyze reports TensorFlow device visibility.
This is a runtime visibility check, not a guarantee that every operation in an
Essentia graph is placed on the GPU. TensorFlow and Essentia logs are quiet by
default; set `TF_CPP_MIN_LOG_LEVEL=0` before launching if detailed diagnostics
are needed.

## Visualisation

Install the optional visualisation dependencies:

```bash
python -m pip install -e ".[viz]"
```

Then:

```python
from musikalyze import MusicEDA

eda = MusicEDA(df, genre_cols="genre400_all")
figure = eda.plot_genre_threshold_analysis()
figure.show()
```

## Testing

Fast tests:

```bash
pytest -q -m 'not integration'
```

Integration tests use real audio and model fixtures and can take several
minutes:

```bash
pytest -q -m integration
```

## Project files

- `src/musikalyze/`: package source;
- `tests/`: unit and integration tests;
- `demo/`: notebooks, sample configuration and demonstration resources;
- `config.schema.json`: JSON Schema for CLI configuration;
- `config.example.json`: small standalone configuration example.

## License

musikalyze is released under the MIT license.
