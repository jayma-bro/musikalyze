# musikalize

Bibliothèque Python pour **analyser** des fichiers audio (Essentia + modèles TensorFlow EffNet Discogs), **lire/écrire des tags** (Mutagen) et **exporter** vers plusieurs formats (ffmpeg) avec un **gabarit de chemins** (artiste/album/…).

Les modèles pré-entraînés Essentia sont soumis à la licence **CC BY-NC-SA 4.0** — voir [Essentia models](https://essentia.upf.edu/documentation/models.html).

## Prérequis système

- **Python 3.10+**
- **ffmpeg** dans le `PATH` (transcodage et métadonnées d’export)
- Pour l’analyse ML : installer le extra **`tensorflow`** (`essentia-tensorflow`)

## Installation

```bash
cd music_organizer
pip install -e .
pip install -e ".[tensorflow]"   # Essentia + TensorFlow
```

## Téléchargement des modèles

Téléchargez depuis le dépôt officiel (ex. [discogs-effnet](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/)) :

- `discogs-effnet-bs64-1.pb` (embeddings)
- Têtes de classification `.pb` + `.json` (genre, mood, …) compatibles EffNet Discogs

Les noms de tenseurs `input` / `output` peuvent varier : renseignez-les dans `ClassificationHeadSpec` si besoin.

## Utilisation rapide (un fichier)

```python
from pathlib import Path

from musikalize import (
    AnalysisConfig,
    ClassificationHeadSpec,
    ExportConfig,
    ModelPath,
    MusicProcess,
    TaggingConfig,
)

audio_path = Path("./data/track.mp3")
model_path = ModelPath(
    embedding_model=Path("./data/discogs-effnet-bs64-1.pb"),
    embedding_output="PartitionedCall:1",
    heads=(
        ClassificationHeadSpec(
            name="genre",
            graph_path=Path("./data/genre_discogs400-discogs-effnet-1.pb"),
            labels_path=Path("./data/genre_discogs400-discogs-effnet-1.json"),
            input_tensor="serving_default_model_Placeholder",
            output_tensor="PartitionedCall:0",
        ),
    ),
)

analysis_cfg = AnalysisConfig(
    genre_main=True,
    genre_count=5,
    genre_thold=None,
    bpm=True,
    key=True,
)

tagging_cfg = TaggingConfig(
    genre="{meta_genre}",
    artist="{tag_artist}",
    title="{tag_title}",
)

export_cfg = ExportConfig(
    output_root=Path("./output"),
    formats="opus",
    path_template="{tag_artist}/{tag_album}/{tag_track_number:02d} - {tag_title}.{ext}",
    format_options={"opus": {"audio_bitrate": "160k"}},
)

music = MusicProcess(
    audio_file=audio_path,
    model_path=model_path,
    analysis_config=analysis_cfg,
    tagging_config=tagging_cfg,
    export_config=export_cfg,
)

music.process_file()
```

### Étapes séparées (notebook / debug)

```python
music.read_tags()
music.load_audio()
music.analyze_file()
music.tag_file()
music.export_file()
```

Propriétés utiles : `music.analysis`, `music.tags_original`, `music.tags_resolved`, `music.audio_mono`.

### Aperçu de chemin sans écrire

```python
music.preview_path(ext="opus")
```

## Gabarits `{tag_*}` et `{meta_*}`

- **`tag_*`** : tags lus sur le fichier source (`tag_artist`, `tag_title`, …).
- **`meta_*`** : métadonnées produites par l’analyse (`meta_genre`, `meta_bpm`, `meta_key`, `meta_<nom_de_tête>`, …).

Les crochets `[meta_genre]` sont acceptés et convertis en `{meta_genre}`. Les clés manquantes deviennent une chaîne vide.

## Parallélisation (plusieurs fichiers)

Sous **Linux/macOS**, utilisez un script avec garde `if __name__ == "__main__":` pour le multiprocessing :

```python
from pathlib import Path

from musikalize import (
    AnalysisConfig,
    ExportConfig,
    ModelPath,
    MusicProcess,
    TaggingConfig,
    list_audio_files,
    process_files_parallel,
)

def main() -> None:
    model_path = ModelPath(embedding_model=Path("./data/discogs-effnet-bs64-1.pb"))
    paths = list_audio_files(Path("./library"))
    process_files_parallel(
        paths,
        model_path=model_path,
        analysis_config=AnalysisConfig(),
        tagging_config=TaggingConfig(),
        export_config=ExportConfig(output_root=Path("./out"), formats="opus"),
        max_workers=4,
    )

if __name__ == "__main__":
    main()
```

Chaque worker réimporte Essentia/TensorFlow ; pour de très gros débits, limitez `max_workers` selon la RAM et le CPU.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Avec les dépendances de développement : `pip install -e ".[dev]"` puis `pytest`.

## Limites

- **WMA** : lecture souvent possible via ffmpeg ; tags en écriture partiels selon formats.
- **Noms de tenseurs TF** : à ajuster par modèle via `ClassificationHeadSpec`.
