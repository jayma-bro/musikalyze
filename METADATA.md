# Référence : méthodes `MusicProcess` et métadonnées

## Classe `MusicProcess`

| Méthode / propriété | Description |
|---------------------|-------------|
| `audio_path` | `Path` du fichier audio. |
| `load_audio()` | Charge le signal **mono 16 kHz** (numpy). Réinitialise le cache d’analyse lazy. |
| `read_tags()` | Lit les tags du fichier (Mutagen) dans des clés logiques (`artist`, `title`, …). |
| `tags_original` | Copie des tags lus. |
| `tags_prefixed` | Tags avec préfixe `tag_*` (usage interne / gabarits). |
| `tags_resolved` | Champs résolus après `tag_file()` (clés logiques pour export / Mutagen). |
| `analyze_file()` | **API historique** (`model_path`) : remplit `analysis` et `analysis.meta`. **Nouvelle API** (`embedders` + `label_extractors`) : retourne le `LazyMetaEngine` sans tout calculer ; les prédictions sont lancées à la demande. |
| `tag_file()` | Applique `TaggingConfig` ; ne calcule que les métadonnées référencées dans les gabarits (+ chemin d’export). |
| `export_file()` | Transcode avec **ffmpeg** et métadonnées résolues. |
| `process_file()` | Enchaîne `read_tags` → `load_audio` → `analyze_file` → `tag_file` → `export_file` (si `export_config`). |
| `preview_path(ext=...)` | Chemin d’export prévu sans écrire. |
| `format_preview(template)` | Résout un gabarit arbitraire avec l’état courant. |
| `label(key)` | Accès programmatique : une clé (`"meta_bpm"`, `"bpm"`) ou une **liste** de clés → liste de valeurs. |
| `meta_*` (attributs) | Ex. `music.meta_bpm` : équivalent à `music.label("meta_bpm")` (nouvelle API lazy ou API historique). |
| `analysis` | Résultat **`AnalysisResult`** uniquement pour l’API **`model_path`** ; sinon `None`. |

## Tags fichier : préfixe `tag_`

| Clé typique | Origine |
|-------------|---------|
| `tag_artist`, `tag_title`, `tag_album`, `tag_genre`, `tag_date`, `tag_tracknumber`, `tag_discnumber`, `tag_composer`, `tag_albumartist`, `tag_comment` | Valeurs lues sur le fichier puis préfixées pour les gabarits. |

Les gabarits utilisent `{tag_artist}`, `{tag_title}`, etc. Les crochets `[tag_artist]` sont convertis automatiquement.

## Métadonnées : préfixe `meta_`

### Descripteurs classiques (Essentia, pas TF)

| Clé | Description |
|-----|-------------|
| `meta_bpm` | BPM entier (**arrondi**), estimateur Percival. |
| `meta_key` | Tonalité (KeyExtractor). |
| `meta_scale` | Mode / gamme (KeyExtractor). |
| `meta_danceability` | Indice 0–1 si `ClassicalConfig.danceability_classical=True`. |

Ils ne sont calculés que si une clé correspondante apparaît dans un gabarit (`tag_file` / `export`) **ou** si vous appelez `label("meta_bpm")` / accès attribut (ce qui force le calcul minimal).

### Agrégats (nouvelle API multi-extracteurs)

| Clé | Description |
|-----|-------------|
| `meta_genre` | Tous les segments de genre (tous extracteurs `category="genre"`), dédupliqués, séparés par `TaggingConfig.separator`. |
| `meta_genre_main` | Segments « principaux » (avant `---` / `//`, etc.). |
| `meta_genre_sub` | Segments « sous-genres » (dernier segment après séparateur). |
| `meta_mood` | Libellés **top-1** de tous les extracteurs `category="mood"`, concaténés. |

### Par extracteur de label

Nommage :

- **Genre** : `meta_genre_<nom_extracteur>` (liste jointe pour cet extracteur), `meta_genre_<nom>_main`, `meta_genre_<nom>_sub`.
- **Mood** : `meta_mood_<nom>` = libellé top-1 pour ce mood.
- **Autre** : `meta_<nom>` (ex. `meta_instrument`).

Suffixes (chaque extracteur concerné) :

| Suffixe | Contenu |
|---------|---------|
| `_val` | Score du libellé top (probabilité ou 1.0 en régression). |
| `_val_str` | Même valeur en chaîne. |
| `_dict` | `dict` libellé → probabilité (0–1). |
| `_dict_str` | JSON de ce dictionnaire. |

## API historique (`ModelPath`)

Les têtes `ClassificationHeadSpec` exposent `meta_<name>` (souvent `meta_genre` si `name="genre"`). Pas d’agrégats `meta_genre_*` par sous-modèle ni suffixes `_dict` automatiques (sauf ce qui est rempli dans `analysis.meta`).

## Erreurs

- **`UnknownEmbedderError`** : nom d’embedding référencé par un `LabelExtractor` absent de `embedders`.
- **`PredictionError`** : échec TensorFlow / Essentia sur une tête.
- Les clés de gabarit inconnues produisent une **chaîne vide** (comportement `str.format` sûr) ; un **warning** peut être loggé pour les accès `label()` incomplets.

## Environnement conseillé

Exemple avec **mamba** / conda :

```bash
mamba activate py311
pip install -e ".[tensorflow]"
```

Modèles locaux : par ex. répertoire `.../music process/data` (déposer les `.pb` / `.json` et pointer les `Path` dans `EmbeddingModel` / `LabelExtractor`).
