# musikalyze

`musikalyze` analyse des fichiers audio avec Essentia/TensorFlow, expose les résultats sous forme de métadonnées, applique des templates de tags et exporte une bibliothèque vers plusieurs formats.

## Installation

```bash
pip install -e .
```

Pour le développement :

```bash
pip install -e '.[dev]'
```

L’analyse nécessite également les modèles Essentia/TensorFlow et `ffmpeg` pour le transcodage.

## Concepts

- `EmbeddingModel` charge un modèle d’embedding (`effnet` ou `maest`).
- `LabelExtractor` applique une tête de classification, multilabel ou régression.
- `MusicProcess` traite un fichier.
- `MusicBatch` traite récursivement un dossier.
- `TaggingConfig` décrit les tags à remplacer explicitement.
- `ExportConfig` décrit l’export, ou le retag sans réencodage.

Les scores internes des modèles sont des flottants `0..1`. Les seuils configurés par l’utilisateur sont toujours des pourcentages entiers `0..100`.

## Exemple Python

```python
from pathlib import Path
from musikalyze import (
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    MusicProcess,
    TaggingConfig,
)

effnet = EmbeddingModel(
    embedding_model=Path("./models/discogs-effnet-bs64-1.pb"),
    name="effnet",
)

genre = LabelExtractor(
    name="genre400",
    embedder_name="effnet",
    graph_path=Path("./models/genre_discogs400-discogs-effnet-1.pb"),
    labels_path=Path("./models/genre_discogs400-discogs-effnet-1.json"),
    category="genre",
    task="multilabel",
    count=3,
    thold=70,
    count_thold_policy="union",
)

process = MusicProcess(
    audio_file=Path("./library/song.mp3"),
    embedders=[effnet],
    extractors=[genre],
    tagging_config=TaggingConfig(
        tags={
            "genre": "{meta_genres}",
            "key": "{meta_key}",
            "copyright": "{meta_genres_main};{meta_scale}",
        },
        separator=";",
        extra={
            "genre_main": "{meta_genres_main}",
        },
    ),
    export_config=ExportConfig(
        output_root=Path("./output"),
        formats="opus",
        path_template="{tag_artist}/{tag_title}.{ext}",
        format_options={"opus": {"audio_bitrate": "256k"}},
    ),
)

process.process_file()
```

Pour un dossier, `MusicBatch.export()` utilise directement `ExportConfig` :

```python
from musikalyze import MusicBatch

batch = MusicBatch("./library", embedders=[effnet], extractors=[genre], export_config=process.export_config)
batch.export()                    # utilise output_root
batch.export(Path("./temporary-output"))  # surcharge output_root pour cet appel
```

`ExportConfig(delete_after=True)` supprime le fichier source uniquement après
la réussite de son export. Cette option fonctionne de la même manière pour
`MusicProcess` et `MusicBatch`.

## Retag sans réencodage

Pour copier le fichier original et modifier uniquement ses tags :

```python
ExportConfig(
    output_root=Path("./retagged"),
    retag=True,
)
```

En mode `retag=True` :

- le fichier est copié bit à bit avant modification des tags ;
- aucun transcodage n’est effectué ;
- `formats`, `audio_bitrate` et les options ffmpeg sont ignorés ;
- les tags non configurés sont conservés ;
- l’artwork est conservé par la copie du fichier.

## CLI

La CLI utilise des sous-commandes. La commande principale est :

```bash
musikalyze ./library --config ./config.json export ./output_folder
```

Elle accepte également un seul fichier :

```bash
musikalyze ./song.mp3 --config ./config.json export ./output_folder
```

Autres commandes disponibles :

```bash
musikalyze ./library --config ./config.json analyze
musikalyze ./library --config ./config.json analyze --key meta_genres_all
musikalyze ./library --config ./config.json analyze --output analysis.json
musikalyze ./library --config ./config.json preview
```

`export` est une sous-commande plutôt qu’un flag afin de permettre d’ajouter d’autres opérations sans ambiguïté. Le dossier passé après `export` est optionnel dans l’API Python, mais reste le chemin de sortie explicite de la commande CLI et surcharge `export_config.output_root`.

La suppression des originaux ne se configure pas par un flag CLI : elle se fait dans le JSON avec `"delete_after": true`.

Le schéma est disponible dans [`config.schema.json`](config.schema.json), et un exemple complet dans [`config.example.json`](config.example.json).

### Configuration JSON

Les chemins relatifs sont résolus relativement au fichier JSON :

```json
{
  "embedding_models": {
    "effnet": {
      "embedding_model": "models/discogs-effnet-bs64-1.pb",
      "name": "effnet"
    }
  },
  "label_extractors": [
    {
      "name": "genre400",
      "embedder_name": "effnet",
      "graph_path": "models/genre_discogs400-discogs-effnet-1.pb",
      "labels_path": "models/genre_discogs400-discogs-effnet-1.json",
      "category": "genre",
      "task": "multilabel",
      "count": 3,
      "thold": 70,
      "count_thold_policy": "union"
    }
  ],
  "tagging_config": {
    "separator": ";",
    "tags": {
      "genre": "{meta_genres}",
      "key": "{meta_key}",
      "copyright": "{meta_genres_main};{meta_scale}"
    },
    "extra": {
      "energy": "{meta_mood_energy_val_pct}"
    }
  },
  "export_config": {
    "formats": "opus",
    "path_template": "{tag_artist}/{tag_title}.{ext}",
    "format_options": {
      "opus": {"audio_bitrate": "256k"}
    },
    "retag": false
  }
}
```

## Templates

Les templates utilisent les variables `tag_*` et `meta_*` :

```text
{tag_artist}
{tag_track_number:02d}
{meta_genres}
{meta_genres_main}
{meta_mood_happy_val_pct}
{ext}
```

Les listes sont jointes avec `TaggingConfig.separator`. Les valeurs vides sont supprimées avant assemblage.

## Batch et DataFrame

```python
from musikalyze import MusicBatch

batch = MusicBatch("./library", embedders=[effnet], extractors=[genre])
df = batch.analyze("analyze")
```

`MusicBatch.analyze("analyze")` fournit notamment :

- `filename`
- `filepath`
- `artist`
- `album`
- `title`
- `track`
- `metas_all_pct`

Les dictionnaires peuvent être aplatis avec :

```python
df = batch.explode_metas(df, "metas_all_pct")
```

## Visualisation

```python
from musikalyze import MusicEDA

eda = MusicEDA(df, genre_cols="genre400_all")
figure = eda.plot_genre_threshold_analysis()
```

Les visualisations avancées utilisent les dépendances optionnelles :

```bash
pip install -e '.[viz]'
```

## Tests

Tests rapides :

```bash
pytest -q -m 'not integration'
```

Tests d’intégration avec les vrais fichiers audio et modèles :

```bash
pytest -q -m integration
```

Les tests d’intégration peuvent prendre plusieurs minutes car ils exécutent Essentia/TensorFlow sur les fichiers de `tests/fixtures`.
