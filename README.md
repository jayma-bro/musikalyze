# musikalize

Bibliothèque Python pour **analyser** des fichiers audio (Essentia + TensorFlow : EffNet, MAEST, têtes de classification), **lire/écrire des tags** (Mutagen) et **exporter** vers plusieurs formats (ffmpeg) avec un **gabarit de chemins**.

Les modèles pré-entraînés Essentia sont soumis à la licence **CC BY-NC-SA 4.0** — voir [Essentia models](https://essentia.upf.edu/documentation/models.html).

**Documentation détaillée des méthodes et des clés `meta_*` / `tag_*`** : [METADATA.md](METADATA.md).

## Prérequis système

- **Python 3.10+** (ex. environnement **mamba** `py311` : `mamba activate py311`)
- **ffmpeg** dans le `PATH`
- Analyse ML : `pip install -e ".[tensorflow]"` (`essentia-tensorflow`)

## Installation

```bash
cd music_organizer
pip install -e .
pip install -e ".[tensorflow]"
```

## Téléchargement des modèles

Exemples : [discogs-effnet](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/), modèles MAEST / genre 519, moods, etc. Les tenseurs `input` / `output` dépendent du fichier `.pb` : les renseigner dans `LabelExtractor` ou `ClassificationHeadSpec`.

## Nouvelle API : plusieurs embeddings + extracteurs (lazy)

Les **embeddings** ne sont calculés que pour les modèles réellement utilisés par un extracteur. Les **prédictions** ne sont lancées que si une métadonnée correspondante est utilisée dans un gabarit ou via `music.label(...)` / `music.meta_`*.

```python
from pathlib import Path

from musikalize import (
    ClassicalConfig,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    MusicProcess,
    TaggingConfig,
)

data = Path("/home/user/Documents/Prog/music process/data")  # exemple

effnet = EmbeddingModel(
    name="effnet",
    embedding_model=data / "discogs-effnet-bs64-1.pb",
    embedding_output="PartitionedCall:1",
    backend="effnet_discogs",
)
maest = EmbeddingModel(
    name="maest",
    embedding_model=data / "discogs-maest-30s-pw-519l-2.pb",
    embedding_output="PartitionedCall/Identity_12",
    backend="generic_tf",
    input_tensor="serving_default_input_1",  # ajuster selon le graphe
)

genre400 = LabelExtractor(
    name="genre400",
    category="genre",
    embedder="effnet",
    graph_path=data / "genre_discogs400-discogs-effnet-1.pb",
    labels_path=data / "genre_discogs400-discogs-effnet-1.json",
    genre_main=True,
    genre_count=5,
    count_thold_policy="intersection",
)
genre519 = LabelExtractor(
    name="genre519",
    category="genre",
    embedder="maest",
    graph_path=data / "genre_discogs519-discogs-maest-30s-pw-519l-1.pb",
    labels_path=data / "genre_discogs519-discogs-maest-30s-pw-519l-1.json",
    count_thold_policy="union",
)

happy = LabelExtractor(
    name="happy",
    category="mood",
    embedder="effnet",
    graph_path=data / "mood_happy-discogs-effnet-1.pb",
    labels_path=data / "mood_happy-discogs-effnet-1.json",
    output_tensor="model/Softmax",
)

music = MusicProcess(
    audio_file=data / "audio.wav",
    embedders=[effnet, maest],
    label_extractors=[genre400, genre519, happy],
    classical_config=ClassicalConfig(bpm=True, key=True),
    tagging_config=TaggingConfig(
        genre="{meta_genre_main}",
        artist="{tag_artist}",
        comment="{meta_mood};{meta_genre_genre400}",
    ),
    export_config=ExportConfig(
        output_root=Path("./output"),
        formats="opus",
        path_template="{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}",
        format_options={"opus": {"audio_bitrate": "160k"}},
    ),
)

music.process_file()
# Accès ponctuel : music.meta_bpm, music.label("meta_mood_happy"), music.label(["meta_key", "meta_bpm"])
```

### Backends d’embedding


| `EmbeddingModel.backend` | Comportement                                                 |
| ------------------------ | ------------------------------------------------------------ |
| `effnet_discogs`         | `TensorflowPredictEffnetDiscogs`                             |
| `maest` / `generic_tf`   | `TensorflowPredict` avec `input_tensor` + `embedding_output` |


Ajuster `input_tensor` / `embedding_output` selon la doc du modèle.

## API historique : un seul `ModelPath`

Toujours supportée : un EffNet + liste de `ClassificationHeadSpec` + `AnalysisConfig` (post-traitement genre global).

```python
from musikalize import (
    AnalysisConfig,
    ClassificationHeadSpec,
    ExportConfig,
    ModelPath,
    MusicProcess,
    TaggingConfig,
)

model_path = ModelPath(
    embedding_model=Path("./data/discogs-effnet-bs64-1.pb"),
    embedding_output="PartitionedCall:1",
    heads=(
        ClassificationHeadSpec(
            name="genre",
            graph_path=Path("./data/genre_discogs400-discogs-effnet-1.pb"),
            labels_path=Path("./data/genre_discogs400-discogs-effnet-1.json"),
        ),
    ),
)

music = MusicProcess(
    audio_file=Path("./data/track.mp3"),
    model_path=model_path,
    analysis_config=AnalysisConfig(genre_main=True, genre_count=5, bpm=True, key=True),
    tagging_config=TaggingConfig(genre="{meta_genre}", artist="{tag_artist}"),
    export_config=ExportConfig(output_root=Path("./output"), formats="opus"),
)
music.process_file()
```

### Étapes séparées (notebook)

```python
music.read_tags()
music.load_audio()
music.analyze_file()
music.tag_file()
music.export_file()
```

Propriétés : `music.analysis` (API historique), `music.tags_original`, `music.tags_resolved`, `music.audio_mono`.

## Gabarits

- `{tag_*}` : tags fichier.
- `{meta_*}` : métadonnées (voir [METADATA.md](METADATA.md)).
- Crochets `[meta_genre]` acceptés.

## Parallélisation

`process_files_parallel` accepte aujourd’hui surtout l’API `**ModelPath**` (sérialisation simple). Pour la nouvelle API (`embedders` + `label_extractors`), enchaîner une boucle sur `list_audio_files` dans le même processus ou étendre le worker (PR / évolution prévue).

```python
from musikalize import ExportConfig, ModelPath, MusicProcess, TaggingConfig, list_audio_files, process_files_parallel

paths = list_audio_files(Path("./library"))
process_files_parallel(
    paths,
    model_path=ModelPath(embedding_model=Path("./data/discogs-effnet-bs64-1.pb")),
    tagging_config=TaggingConfig(),
    export_config=ExportConfig(output_root=Path("./out"), formats="opus"),
    max_workers=4,
)
```

Sous Windows / certains shells, protéger le multiprocessing avec `if __name__ == "__main__":`.

## Tests

```bash
mamba activate py311   # ou votre env
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Limites

- **WMA** : tags en écriture partiels.
- **Tenseurs TF** : à ajuster par modèle.
- **MAEST / graphes exotiques** : vérifier `input_tensor` et `backend` (`generic_tf`).

