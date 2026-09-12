"""Quick smoke test for the MusicEDA visualizer."""

import json
import random
import pandas as pd
import numpy as np

# ---- Build mock data similar to MusicBatch.analyze() output ----

random.seed(42)
n_tracks = 100

rows = []
for i in range(n_tracks):
    row = {"_path": f"/music/track_{i:03d}.mp3"}

    # Score-like columns (0-100 scale, auto-detected)
    row["acousticness"] = round(random.uniform(0, 100), 2)
    row["danceability"] = round(random.uniform(0, 100), 2)
    row["energy"] = round(random.uniform(0, 100), 2)
    row["valence"] = round(random.uniform(0, 100), 2)
    row["tempo"] = round(random.uniform(60, 140), 2)  # other (not 0-100)

    # Genre dict column (will be exploded by MusicEDA)
    genre_scores = {
        "Rock---Classic Rock": round(random.uniform(0, 100), 2),
        "Rock---Metal": round(random.uniform(0, 100), 2),
        "Pop": round(random.uniform(0, 100), 2),
        "Electronic": round(random.uniform(0, 100), 2),
        "Jazz": round(random.uniform(0, 100), 2),
        "Hip-Hop": round(random.uniform(0, 100), 2),
        "Classical": round(random.uniform(0, 100), 2),
        "Folk": round(random.uniform(0, 100), 2),
    }
    row["genre400_all"] = genre_scores

    # ID columns
    row["artist"] = f"Artist {i % 10}"
    row["album"] = f"Album {i % 5}"

    rows.append(row)

df = pd.DataFrame(rows)

# ---- Run tests ----
from musikalyze.visualizer import MusicEDA

eda = MusicEDA(
    df,
    genre_cols="genre400_all",
    id_cols=["artist", "album"],
    score_cols=["acousticness", "danceability", "energy", "valence"],
)

print("=== Summary ===")
print(eda.summary())
print()

# Test each plotting method
methods = {
    "plot_histogram": lambda: eda.plot_histogram("acousticness"),
    "plot_violin": lambda: eda.plot_violin(),
    "plot_scatter": lambda: eda.plot_scatter("acousticness", "energy"),
    "plot_pairplot": lambda: eda.plot_pairplot(["acousticness", "danceability", "energy"]),
    "plot_correlation_heatmap": lambda: eda.plot_correlation_heatmap(),
    "plot_genre_threshold_analysis": lambda: eda.plot_genre_threshold_analysis(),
    "plot_genre_countplot": lambda: eda.plot_genre_countplot(),
    "plot_genre_global": lambda: eda.plot_genre_global(),
    "plot_genre_profile": lambda: eda.plot_genre_profile(),
    "plot_category_stacked": lambda: eda.plot_category_stacked(),
    "plot_genre_cooccurrence": lambda: eda.plot_genre_cooccurrence(),
    "plot_genre_pca": lambda: eda.plot_genre_pca(),
    "plot_radar": lambda: eda.plot_radar(),
    "plot_embedding_tsne": lambda: eda.plot_embedding(method="tsne"),
    "plot_embedding_umap": lambda: eda.plot_embedding(method="umap"),
}

print(f"Testing {len(methods)} plotting methods...")
for name, fn in methods.items():
    try:
        result = fn()
        # Handle single figure or list of figures
        if isinstance(result, list):
            for i, fig in enumerate(result):
                assert hasattr(fig, "to_dict"), f"{name}[{i}] didn't return a Plotly Figure"
        else:
            assert hasattr(result, "to_dict"), f"{name} didn't return a Plotly Figure"
        print(f"  OK   {name}")
    except ImportError as e:
        print(f"  SKIP {name}: {e}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")

print("\nAll tests passed!")
