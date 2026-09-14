# Métadonnées et tagging

Ce document décrit les noms utilisés par `musikalyze` pour lire, analyser et écrire les métadonnées audio.

## Règle fondamentale : `tag_*` et `meta_*`

Les deux préfixes ont des origines différentes.

### `tag_*` : informations déjà présentes dans le fichier

Les tags du fichier original sont lus sans analyse audio et sont disponibles sous la forme :

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
tag_replaygain_track_gain
tag_replaygain_track_peak
tag_replaygain_album_gain
tag_replaygain_album_peak
```

Ces valeurs peuvent servir à construire un chemin d’export ou à conserver les tags originaux.

### `meta_*` : résultats de l’analyse Musikalyze

Ces valeurs sont calculées à partir de l’audio ou des extracteurs configurés :

```text
meta_bpm
meta_key
meta_scale
meta_rgain_gain
meta_rgain_peak
meta_rgain_peak_dbfs
```

Les valeurs `meta_rgain_*` sont différentes des tags `tag_replaygain_*` déjà stockés dans le fichier.

## `TaggingConfig`

```python
TaggingConfig(
    tags={
        "genre": "{meta_genres}",
        "key": "{meta_key}",
        "bpm": "{meta_bpm}",
        "copyright": "{meta_genres_main};{meta_scale}",
    },
    separator=";",
    extra={
        "energy": "{meta_mood_energy_val_pct}",
        "custom_tag": "{tag_artist} - {tag_title}",
    },
)
```

### `tags`

`tags` contient les tags standards écrits dans le fichier de sortie.

Clés standards prises en charge :

```text
artist
title
album
genre
date
tracknumber
discnumber
composer
albumartist
comment
lyrics
copyright
publisher
encodedby
encoder
isrc
language
albumsort
artistsort
titlesort
website
bpm
mood
grouping
key
```

Les espaces éventuels dans cette liste sont uniquement typographiques : les noms de clés ne contiennent pas d’espace.

### `extra`

`extra` permet de créer des tags supplémentaires sans modifier la liste des tags standards :

```python
extra={
    "approachability": "{meta_mood_approachability_val_pct}",
    "energy": "{meta_mood_energy_val_pct}",
    "analysis_version": "musikalyze-0.6",
}
```

Les tags personnalisés sont écrits selon les possibilités du format cible. Un format peut ne pas avoir de correspondance parfaite pour un tag donné.

### Conservation des tags

Par défaut, l’export conserve les tags présents dans le fichier original. Seuls les tags explicitement déclarés dans `tags` ou `extra` sont remplacés ou ajoutés.

Un tag vide ou absent dans la configuration ne supprime pas automatiquement le tag original.

## Genres

Les labels de genre peuvent contenir un genre principal et un sous-genre :

```text
Reggae---Dub
Electronic---Dub
```

Les labels sont triés par score et limités par les paramètres de `LabelExtractor` (`count`, `thold` et `count_thold_policy`).

### Genre principal

Le genre principal est uniquement la partie gauche du label complet ayant le score le plus élevé.

Avec :

```text
Reggae---Dub : 83
Electronic---Dub : 80
```

on obtient :

```text
meta_genres_main = "Reggae"
```

`Electronic` ne devient pas un second genre principal.

### Sous-genres

Les sous-genres sont les parties situées à droite de `---` parmi les labels sélectionnés :

```text
meta_genres_sub = "Dub"
```

Les doublons sont retirés en conservant le premier ordre d’apparition.

Les alias suivants sont disponibles :

```text
meta_genre_main
meta_genre_sub
meta_genres_main
meta_genres_sub
```

Les formes plurielles sont recommandées dans les templates généraux.

## Extracteurs

Pour un extracteur nommé `genre400` dans la catégorie `genre` :

```text
meta_genre_genre400
meta_genre_genre400_val
meta_genre_genre400_val_pct
meta_genre_genre400_dict
meta_genre_genre400_dict_pct
meta_genre_genre400_all
meta_genre_genre400_all_pct
```

Pour un extracteur nommé `happy` dans la catégorie `mood` :

```text
meta_mood_happy
meta_mood_happy_val
meta_mood_happy_val_pct
meta_mood_happy_dict
meta_mood_happy_dict_pct
meta_mood_happy_all
meta_mood_happy_all_pct
```

Les suffixes `_pct` contiennent des scores en pourcentage entier `0..100`.

Les suffixes sans `_pct` contiennent les scores internes, généralement des flottants `0..1`.

## Pourcentages et seuils

Les paramètres configurés par l’utilisateur utilisent toujours des entiers `0..100`.

### Multilabel

```python
LabelExtractor(
    name="genre400",
    embedder_name="effnet",
    graph_path=Path("models/genre.pb"),
    labels_path=Path("models/genre.json"),
    category="genre",
    task="multilabel",
    count=3,
    thold=70,
    count_thold_policy="union",
)
```

### Régression avec labels par intervalles

```python
LabelExtractor(
    name="aggressive",
    embedder_name="effnet",
    graph_path=Path("models/aggressive.pb"),
    labels_path=Path("models/aggressive.json"),
    category="mood",
    task="regression",
    label_names={
        "agrsv_low": (0, 16),
        "agrsv_midlow": (17, 44),
        "agrsv_hi": (45, 100),
    },
)
```

Le modèle produit par exemple `0.443`. Musikalyze le convertit en `44` avant comparaison.

En cas de chevauchement, l’intervalle ayant la borne basse la plus faible est prioritaire. Si aucune plage ne correspond, aucun label n’est attribué.

## Métadonnées classiques

Les métadonnées classiques ne sont pas des prédictions de label. Elles sont calculées directement par Essentia et accessibles comme des valeurs simples :

| Clé | Valeur |
|---|---|
| `meta_bpm` | BPM arrondi |
| `meta_key` | Tonalité, par exemple `C#` |
| `meta_scale` | Mode, par exemple `minor` |
| `meta_rgain_gain` | Gain ReplayGain calculé |
| `meta_rgain_peak` | Peak linéaire calculé |
| `meta_rgain_peak_dbfs` | Peak en dBFS |

Elles peuvent être utilisées directement dans `tags` :

```python
tags={
    "bpm": "{meta_bpm}",
    "key": "{meta_key}",
    "copyright": "{meta_genres_main};{meta_scale}",
}
```

## Numéros de piste et de disque

Les tags originaux peuvent contenir un numéro simple ou une valeur de type `numéro/total` :

```text
tracknumber = "02/12"
discnumber = "1/2"
```

La fonction `format_nbr()` extrait le premier numéro et normalise son affichage pour les templates de chemin :

```python
from musikalyze.tagging import format_nbr

format_nbr("02/12")  # "02"
format_nbr("1/2")    # "01"
format_nbr(3)         # "03"
```

Dans les mappings de templates, les variantes préparées sont disponibles sous :

```text
tag_tracknumber
tag_tracknumber_f
tag_discnumber
tag_discnumber_f
```

La variante `_f` est utile lorsque l’on veut contrôler explicitement la valeur normalisée dans un nom de fichier. Le placeholder recommandé pour un format numérique reste :

```text
{tag_track_number:02d}
```

qui est calculé à partir de `tag_tracknumber`.

## Templates et séparateurs

Le séparateur par défaut est `;` :

```python
TaggingConfig(
    separator=";",
    tags={"genre": "{meta_genres}"},
)
```

Une liste :

```text
["Rock", "Alternative", "Rock"]
```

devient :

```text
Rock;Alternative
```

Les valeurs vides sont supprimées :

```text
["acoustic", "", "sad"] → "acoustic;sad"
```

## Export et retag

### Transcodage

```python
ExportConfig(
    output_root=Path("output"),
    formats="opus",
    format_options={"opus": {"audio_bitrate": "256k"}},
)
```

### Copie sans réencodage

```python
ExportConfig(
    output_root=Path("retagged"),
    retag=True,
    delete_after=False,
)
```

Dans ce mode :

- le fichier original est copié ;
- les tags sont modifiés sur la copie ;
- le flux audio n’est pas réencodé ;
- `formats` et `format_options` sont ignorés ;
- l’artwork est conservé par la copie du fichier ;
- les tags non configurés restent présents.

### Export batch

```python
batch.export()                    # ExportConfig.output_root
batch.export(Path("temporary-output"))  # surcharge temporaire du dossier
```

`delete_after=True` se configure dans `ExportConfig`, jamais comme argument de
`MusicBatch.export()`. Après un export réussi, `MusicProcess` supprime le
fichier source ; en cas d’échec, il est conservé. Les répertoires sources
vides sont ensuite nettoyés par le batch.

### BPM TempoCNN optionnel

`meta_bpm` utilise `RhythmExtractor2013` par défaut. Un modèle Essentia
TempoCNN peut être fourni à `MusicProcess` ou `MusicBatch` :

```python
MusicProcess(
    audio_file=Path("song.mp3"),
    tempo_model_path=Path("models/deeptemp-k16-3.pb"),
)
```

Le même paramètre est disponible dans le JSON CLI sous `tempo_model_path`.

## Formats

Les noms logiques sont traduits vers les noms du conteneur :

| Format | Famille de tags |
|---|---|
| MP3 | ID3v2 / EasyID3 / TXXX |
| FLAC | Vorbis comments |
| OGG | Vorbis comments |
| Opus | Vorbis comments |
| M4A | MP4/iTunes atoms |
| WMA | ASF |

`copyright` est le nom logique recommandé. `TCOP` est le nom d’une frame ID3 correspondant au copyright, pas un champ métier distinct.

Pour les tags issus d’un MP3, `TBPM` est l’alias ID3 brut de `bpm`. Lorsqu’un
export vers Opus/FLAC réécrit le BPM, musikalyze supprime cet alias brut afin
d’éviter un ancien `TBPM` en doublon du tag canonique `bpm`. En revanche,
`TRACKNUMBER` est le nom canonique Vorbis/Opus correspondant à `tracknumber` ;
son affichage en majuscules est normal et nécessaire au conteneur.

## Analyse batch

```python
df = batch.analyze("analyze")
```

La colonne `metas_all_pct` contient les métadonnées regroupées avec des scores en pourcentage. Elle peut être aplatie avec :

```python
df = batch.explode_metas(df, "metas_all_pct")
```

Pour une analyse ciblée :

```python
df = batch.analyze(["meta_genres", "meta_bpm", "tag_artist"])
```

## CLI

```bash
musikalyze ./library --config ./config.json export ./output
```

Pour un seul fichier :

```bash
musikalyze ./song.mp3 --config ./config.json export ./output
```

Voir `README.md` pour le schéma complet de `config.json` et les commandes `analyze` et `preview`.
