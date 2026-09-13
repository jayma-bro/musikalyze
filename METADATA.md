# Metadata Tags

This document lists all editable tags supported by musikalyze, including both file tags (`tag_*`) and analysis metadata (`meta_*`).

## File Tags (tag_*)

These tags are read from the audio file's existing metadata or written via templates:

| Tag | Description | Example |
|-----|-------------|---------|
| `tag_artist` | Artist name | "Radiohead" |
| `tag_title` | Track title | "Creep" |
| `tag_album` | Album name | "Pablo Honey" |
| `tag_date` | Release date | "1993" |
| `tag_tracknumber` | Track number | "1" or "1/12" |
| `tag_discnumber` | Disc number | "1" or "1/2" |
| `tag_albumartist` | Album artist | "Radiohead" |
| `tag_composer` | Composer | "Thom Yorke" |
| `tag_comment` | Comment | "Remastered 2009" |
| `tag_lyrics` | Lyrics | "Sometimes a gift..." |
| `tag_copyright` | Copyright | "© 1993 EMI" |
| `tag_publisher` | Publisher | "EMI Music Publishing" |
| `tag_encodedby` | Encoding software | "FFmpeg 4.4" |
| `tag_encoder` | Encoder | "opusenc" |
| `tag_isrc` | ISRC code | "GBAAA9300123" |
| `tag_language` | Primary language | "en" |
| `tag_albumsort` | Album sort order | "Pablo Honey" |
| `tag_artistsort` | Artist sort order | "Radiohead" |
| `tag_titlesort` | Title sort order | "Creep" |
| `tag_website` | Artist website | "https://radiohead.com" |
| `tag_bpm` | Beats per minute | "140" |
| `tag_mood` | Mood descriptor | "Melancholy" |
| `tag_grouping` | Grouping/series | "Albums" |
| `tag_key` | Musical key | "C#" |
| `tag_tcop` | Track Commercial Orientation | "Commercial" |

## Analysis Metadata (meta_*)

These tags are computed by musikalyze's analysis engines:

### Genre Metadata

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_genre` | Primary genre | "Rock" |
| `meta_genres` | All genres with scores | `{"Rock": 0.95, "Alternative": 0.87}` |
| `meta_genres_str` | Genres as string | "Rock;Alternative" |
| `meta_genre_dancehall` | Dancehall genre score | `0.87` |
| `meta_genre_dub` | Dub genre score | `0.75` |

### Mood Metadata

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_mood` | Primary mood | "Happy" |
| `meta_moods` | All moods with scores | `{"Happy": 0.82, "Energetic": 0.76}` |
| `meta_mood_str` | Moods as string | "Happy;Energetic" |
| `meta_mood_happy` | Happy mood score | `0.82` |
| `meta_mood_sad` | Sad mood score | `0.15` |

### Classical/Technical Metadata

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_bpm` | Beats per minute | `120` |
| `meta_key` | Musical key | "C#" |
| `meta_scale` | Scale type | "Minor" |
| `meta_rgain_gain` | ReplayGain track gain (dB) | `-5.2` |
| `meta_rgain_peak` | ReplayGain peak value | `0.85` |
| `meta_rgain_peak_dbfs` | ReplayGain peak in dBFS | `-1.4` |

### Audio Feature Metadata

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_acousticness` | How acoustic a track is (0.0-1.0) | `0.85` |
| `meta_danceability` | How suitable for dancing (0.0-1.0) | `0.75` |
| `meta_energy` | Energy level (0.0-1.0) | `0.90` |
| `meta_instrumentalness` | Instrumentalness score (0.0-1.0) | `0.30` |
| `meta_liveness` | Detection of live performance (0.0-1.0) | `0.20` |
| `meta_popularity` | Popularity score (0-100) | `75` |
| `meta_speechiness` | Speechiness score (0.0-1.0) | `0.10` |
| `meta_valence` | Musical positiveness (0.0-1.0) | `0.60` |
| `meta_tempo` | Tempo in BPM | `120` |

### Comprehensive Metadata

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_all` | All analysis results as dict | `{"meta_genre": {...}, ...}` |
| `meta_all_pct` | All analysis results with percentages | `{"meta_genre": {...}, ...}` |
| `meta_all_str` | All analysis results as JSON string | `"{...}"` |
| `meta_label` | Primary label per extractor | `{"genre400": "Rock", ...}` |
| `meta_label_pct` | Primary label with scores | `{"genre400": {"Rock": 0.95}, ...}` |
| `meta_label_all` | All labels per extractor | `{"genre400": ["Rock", "Alternative"], ...}` |
| `meta_label_all_pct` | All labels with scores | `{"genre400": {"Rock": 0.95, "Alternative": 0.87}, ...}` |
| `metas` | All metadata values | `{"meta_genre": "Rock", ...}` |
| `metas_pct` | All metadata with percentages | `{"meta_genre": {"Rock": 0.95}, ...}` |
| `metas_all` | All metadata as nested dict | `{"genre": {...}, "mood": {...}, ...}` |
| `metas_all_pct` | All metadata with percentages | `{"genre": {...}, "mood": {...}, ...}` |
| `metas_label` | All labels as nested dict | `{"genre": {...}, "mood": {...}, ...}` |
| `metas_label_pct` | All labels with percentages | `{"genre": {...}, "mood": {...}, ...}` |
| `metas_label_all` | All labels (all scores) nested | `{"genre": {...}, "mood": {...}, ...}` |
| `metas_label_all_pct` | All labels with all scores | `{"genre": {...}, "mood": {...}, ...}` |

## ReplayGain Metadata from File Tags

These are read from existing ReplayGain tags in the file:

| Tag | Description | Example |
|-----|-------------|---------|
| `meta_replaygain_track` | Track ReplayGain | "-5.2 dB" |
| `meta_replaygain_album` | Album ReplayGain | "-3.8 dB" |

## Template Variables

Templates use `str.format()` syntax with `tag_*` and `meta_*` prefixes:

```
# Path template
{tag_artist}/{tag_album}/{tag_tracknumber:02d} - {tag_title}.{ext}

# Tagging template
{tag_artist} - {tag_title} ({meta_genre})
```

### Special Variables

| Variable | Description |
|----------|-------------|
| `{ext}` | File extension (e.g., "opus", "mp3") |
| `{tag_track_number}` | Track number as integer (extracted from `tag_tracknumber`) |

### Separator

Templates support a configurable separator (default: `;`) for joining multiple values:

```
# With separator=";":
{meta_genre} → "Rock;Alternative;Indie"

# With separator=", ":
{meta_genre} → "Rock, Alternative, Indie"
```

### Empty Value Handling

When a template variable resolves to an empty value, it is silently omitted to avoid consecutive separators:

```
# Template: "{tag_artist};{tag_composer};{tag_title}"
# If tag_composer is empty:
# Result: "Radiohead;Creep" (not "Radiohead;;Creep")
```

## Genre Handling

### Main Genre Extraction

When a genre contains sub-genres (e.g., "Reggae---Dub"), only the first level is used as the main genre:

```
"Reggae---Dub" → "Reggae"
"Electronic---Dub" → "Electronic"
```

### Genre Deduplication

Duplicate genres are automatically removed. For example:

```
{"Reggae---Dub": 83, "Electronic---Dub": 80}
→ Extract main genres: ["Reggae", "Electronic"]
→ Deduplicate: ["Reggae", "Electronic"]
→ Result: "Reggae;Electronic"
```

## Custom Tags

You can define custom tags in the `extra` section of `TaggingConfig`:

```python
tagging_config = TaggingConfig(
    extra={
        "custom_field": "Custom: {tag_artist} - {tag_title}",
        "analysis_note": "Genre: {meta_genre}",
    }
)
```

## LabelExtractor Configuration

The `LabelExtractor.label_names` field accepts two formats:

### Sequence[str] (Classification/Multilabel)

```python
LabelExtractor(
    name="genre400",
    label_names=["Rock", "Pop", "Jazz", ...],  # 400 genres
    task="multilabel"
)
```

### Dict[str, tuple[float, float]] (Regression Threshold Mapping)

```python
LabelExtractor(
    name="danceability",
    label_names={
        "Not Danceable": (0.0, 0.3),
        "Moderately Danceable": (0.3, 0.6),
        "Very Danceable": (0.6, 1.0),
    },
    task="regression"
)
```

When using the dict format, regression scores are mapped to label names based on threshold ranges. On overlap, the lower label wins. Exact boundary values round to the lower label.
