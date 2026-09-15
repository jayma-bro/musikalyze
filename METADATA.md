# Metadata reference

This document summarizes the metadata contract used by musikalyze. It is
intended as a reference for configuring templates, not as a description of
container-specific internals.

## Two metadata namespaces

musikalyze keeps source tags and analysis results separate.

### `tag_*`: source-file metadata

`tag_*` values are read from the original file without audio analysis. Common
keys include:

```text
tag_artist
tag_title
tag_album
tag_genre
tag_date
tag_tracknumber
tag_discnumber
tag_albumartist
tag_composer
tag_comment
tag_lyrics
tag_copyright
tag_publisher
tag_encodedby
tag_encoder
tag_isrc
tag_language
tag_albumsort
tag_artistsort
tag_titlesort
tag_website
tag_bpm
tag_mood
tag_grouping
tag_key
tag_rating
tag_replaygain_track_gain
tag_replaygain_track_peak
tag_replaygain_album_gain
tag_replaygain_album_peak
```

These values can be used in output paths and tag templates. Unconfigured source
tags are preserved during export whenever the target format has a compatible
representation.

### `meta_*`: musikalyze results

Classical descriptors are exposed as scalar metadata:

```text
meta_bpm
meta_key
meta_scale
meta_rgain_gain
meta_rgain_peak
meta_rgain_peak_dbfs
```

Extractor metadata uses the extractor category and name. For an extractor
called `genre400` in category `genre`:

```text
meta_genre_genre400
meta_genre_genre400_val
meta_genre_genre400_val_pct
meta_genre_genre400_dict
meta_genre_genre400_dict_pct
meta_genre_genre400_all
meta_genre_genre400_all_pct
```

For an extractor called `happy` in category `mood`:

```text
meta_mood_happy
meta_mood_happy_val
meta_mood_happy_val_pct
meta_mood_happy_dict
meta_mood_happy_dict_pct
meta_mood_happy_all
meta_mood_happy_all_pct
```

Values with `_pct` are integer percentages in `0..100`. Other score values are
model values, normally floats in `0..1`.

## Tagging configuration

```python
TaggingConfig(
    separator=";",
    multi_entry=True,
    tags={
        "genre": "{meta_genres}",
        "key": "{meta_key}",
        "bpm": "{meta_bpm}",
        "copyright": "{meta_genres_main};{meta_scale}",
    },
    extra={
        "energy": "{meta_mood_energy_val_pct}",
        "source": "{tag_artist} - {tag_title}",
    },
)
```

- `tags` contains standard logical audio tags;
- `extra` contains arbitrary custom tags;
- `separator` separates values inside templates and removes empty elements and duplicates;
- `multi_entry=True` (the default) writes list values as separate tag entries when
  the target format supports them; `False` writes one separator-joined value;
- `preserve_unconfigured=True` (the default) keeps every original tag that is not
  mentioned in `tags` or `extra`; set it to `False` to export only configured
  tags, while embedded artwork is still retained;
- an empty or absent configured value does not automatically erase the source
  tag;
- only keys explicitly configured in `tags` or `extra` are overwritten.

### Editable logical tags

The following standard logical tags can be configured in `TaggingConfig.tags`:

| Logical key | Meaning |
| --- | --- |
| `artist` | Track artist |
| `title` | Track title |
| `album` | Album title |
| `genre` | One or more genres |
| `date` | Release or recording date |
| `tracknumber` | Track number |
| `discnumber` | Disc number |
| `composer` | Composer |
| `albumartist` | Album artist |
| `comment` | Comment |
| `lyrics` | Lyrics |
| `copyright` | Copyright notice |
| `publisher` | Publisher/label |
| `encodedby` | Encoding application/user |
| `encoder` | Encoder name |
| `isrc` | International Standard Recording Code |
| `language` | Language |
| `albumsort`, `artistsort`, `titlesort` | Sort-order fields |
| `website` | Related website |
| `bpm` | Beats per minute |
| `mood` | One or more mood labels |
| `grouping` | Grouping/work field |
| `key` | Musical key |
| `rating` | Rating from 0 to 5 stars |
| `replaygain_track_gain` | ReplayGain track gain |
| `replaygain_track_peak` | ReplayGain track peak |
| `replaygain_album_gain` | ReplayGain album gain |
| `replaygain_album_peak` | ReplayGain album peak |

`TaggingConfig.extra` can edit any additional custom tag, including model
features such as `energy`, `danceability`, `acousticness` and
`instrumentalness`. These names are written using the target container's
custom metadata mechanism.

The logical names are translated per format: for example, MP3 uses ID3
frames such as `TPE1`, `TIT2`, `TCON`, `TBPM` and `TXXX`, Vorbis-family files
use comments such as `ARTIST`, `TITLE`, `GENRE` and `REPLAYGAIN_TRACK_GAIN`,
and M4A uses iTunes atoms such as `©ART`, `©nam`, `©gen` and freeform
ReplayGain atoms. The original codec-specific spelling is preserved for tags
that are not explicitly overwritten whenever the format supports it.

Standard logical keys include `artist`, `title`, `album`, `genre`, `date`,
`tracknumber`, `discnumber`, `composer`, `albumartist`, `comment`, `lyrics`,
`copyright`, `publisher`, `encodedby`, `encoder`, `isrc`, `language`, sort
fields, `website`, `bpm`, `mood`, `grouping`, `key` and `rating`.

Custom model features such as `energy`, `danceability` or
`instrumentalness` belong in `extra`. They are written using the target
container's available metadata mechanism.

## Genres and moods

Genre labels can contain a main genre and a subgenre separated by `---`:

```text
Reggae---Dub
Electronic---Dub
```

The selected labels are ordered by score and limited by the extractor's
`count`, `thold` and `count_thold_policy` settings.

For example, if the selected scores are:

```text
Reggae---Dub       83
Electronic---Dub   80
```

then:

```text
meta_genres_main = ["Reggae"]
meta_genres_sub  = ["Dub"]
```

The main genre comes only from the highest-scoring complete label. Subgenres
are deduplicated while preserving their first-seen order. Grouped mood and
genre values are returned as lists; `TaggingConfig.separator` controls their
serialized form in tags.

## Thresholds and regression labels

User-facing thresholds are integer percentages:

```python
LabelExtractor(
    name="aggressive",
    embedder_name="effnet",
    graph_path=Path("models/aggressive.pb"),
    labels_path=Path("models/aggressive.json"),
    category="mood",
    task="regression",
    label_names={
        "aggressive_low": (0, 16),
        "aggressive_mid": (17, 44),
        "aggressive_high": (45, 100),
    },
)
```

A model value of `0.443` is rounded to `44` before interval matching. If
intervals overlap, the interval with the lowest lower bound wins. If no
interval matches, no label is emitted.

For multilabel extractors, `thold` is also in `0..100`; `count` and
`count_thold_policy` control how many labels are selected.

## Classical descriptors

Classical values are calculated by Essentia when requested:

| Key | Meaning |
|---|---|
| `meta_bpm` | detected or TempoCNN BPM, rounded to an integer |
| `meta_key` | estimated key, such as `C#` |
| `meta_scale` | estimated scale, such as `minor` |
| `meta_rgain_gain` | calculated track gain |
| `meta_rgain_peak` | calculated linear peak |
| `meta_rgain_peak_dbfs` | calculated peak in dBFS |

Existing ReplayGain tags remain source metadata under `tag_replaygain_*` and
are distinct from calculated `meta_rgain_*` values.

An external TempoCNN model can be selected with `tempo_model_path` on
`MusicProcess` or `MusicBatch`. Without it, `RhythmExtractor2013` is used for
BPM.

## Track and disc numbers

Source values may contain a number and a total:

```text
tracknumber = "02/12"
discnumber = "1/2"
```

`format_nbr()` extracts the first number and formats it with two digits:

```python
from musikalyze.tagging import format_nbr

format_nbr("02/12")  # "02"
format_nbr("1/2")    # "01"
format_nbr(3)         # "03"
```

Templates can use:

```text
{tag_tracknumber}
{tag_tracknumber_f}
{tag_discnumber}
{tag_discnumber_f}
{tag_track_number:02d}
```

The `_f` values are already formatted strings. The numeric
`tag_track_number` form is useful with a format specification such as `:02d`.

## Ratings

The logical `rating` value is expressed as `0..5` stars:

```python
TaggingConfig(tags={"rating": "5"})
```

musikalyze maps it to the target container when possible. For Vorbis-family
containers it writes only the generic player form; for example, four stars
becomes `Rating=80`. Legacy `RATING:<email>` fields are read for compatibility
but are removed rather than written, preventing two ratings from diverging.


| Format | Representation |
|---|---|
| MP3 | ID3 `POPM`, canonical values `0, 1, 64, 128, 196, 255` for `0..5` stars |
| Opus/Ogg/FLAC | generic `Rating`, score `0..100` |
| M4A | iTunes freeform rating, score `0..5` |
| Other formats | readable text fallback where supported |

When converting an MP3, the `POPM` email and counter are retained where the
target format supports an equivalent representation.

## Export behavior

```python
ExportConfig(
    output_root=Path("output"),
    formats="opus",
    path_template="{tag_artist}/{tag_title}.{ext}",
    format_options={"opus": {"audio_bitrate": "256k"}},
)
```

Normal exports use FFmpeg and preserve source tags and artwork as far as the
target container allows. `256k` is an Opus bitrate target, not an exact
measured bitrate.

For no-reencode tagging:

```python
ExportConfig(
    output_root=Path("retagged"),
    retag=True,
)
```

The file is copied first, then tags are modified on the copy. The audio stream
is not re-encoded and codec settings are ignored. `delete_after=True` removes
the source only after successful output creation.

Output paths are always sanitized. Explicit separators in the template create
folders, while separators inside metadata values are escaped. For example,
`{tag_title}.opus` with a title `music/test` produces `music_test.opus`.

## Container mapping

Logical keys are translated to container-specific metadata:

| Format | Metadata family |
|---|---|
| MP3 | ID3v2 / EasyID3 / TXXX |
| FLAC | Vorbis comments |
| Ogg | Vorbis comments |
| Opus | Vorbis comments |
| M4A | MP4/iTunes atoms and freeform fields |
| WMA | ASF metadata where supported |

`copyright` is the logical name for copyright. `TCOP` is the ID3 frame name,
not a separate business field. `TRACKNUMBER` is the normal Vorbis/Opus name
for logical `tracknumber`.

## Batch analysis

```python
df = batch.analyze("analyze")
df = batch.explode_metas(df, "metas_all_pct")
```

The analysis DataFrame can contain file metadata, grouped predictions and
percentage score dictionaries. Targeted analysis can request a list such as:

```python
batch.analyze(["meta_genres", "meta_bpm", "tag_artist"])
```
