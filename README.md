# Data Visualization Module

This module provides visualization capabilities for audio analysis data from the musikalyze library. It supports both Plotly and Matplotlib backends with automatic fallback.

## Features

- Distribution plots (histograms, box plots, violin plots)
- Correlation matrices
- Support for multiple visualization backends
- Automatic fallback between Plotly and Matplotlib
- Works with DataFrames from `MusicBatch.analyze()`

## Installation

Install the required visualization libraries:

```bash
pip install plotly matplotlib seaborn
```

## Usage

```python
import pandas as pd
from src.musikalyze.visualizer import DataVisualizer

# Create or load a DataFrame from MusicBatch.analyze()
df = pd.DataFrame(...)  # Your analysis results

# Create visualizer with default plotly backend
visualizer = DataVisualizer(backend='plotly')
visualizer.set_data(df)

# Create distribution plot
fig = visualizer.plot_distribution('loudness', kind='histogram')

# Create correlation matrix
fig = visualizer.plot_correlation_matrix()

# Use matplotlib backend
visualizer = DataVisualizer(backend='matplotlib')
visualizer.set_data(df)
fig = visualizer.plot_distribution('loudness')
```

## Methods

### `plot_distribution(column, title=None, bins=30, kind="histogram")`
Create a distribution plot for a specific column.

### `plot_correlation_matrix(columns=None, title="Correlation Matrix")`
Create a correlation matrix heatmap for specified columns.

## Backend Support

- **Plotly**: Default backend, provides interactive plots
- **Matplotlib**: Fallback backend for static plots
- **Automatic fallback**: If one backend is not available, the other will be used