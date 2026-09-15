"""musikalyze — Essentia-based audio analysis, tagging, and transcoding."""

# Configure noisy native dependencies before any musikalyze module imports
# Essentia/TensorFlow. ``setdefault`` keeps an explicit user override useful
# for debugging.
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Initialise TensorFlow visibility/memory growth before Essentia imports any
# TensorFlow-backed operators. This is especially important for the fresh
# process started by the CLI (notebooks often configure it earlier).
from musikalyze.runtime import configure_tensorflow_memory

configure_tensorflow_memory()

import essentia

essentia.log.infoActive = False

from musikalyze.batch import (
    MusicBatch,
    list_audio_files,
    sample_audio_files,
)
from musikalyze.config import (
    AnalysisResult,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    TaggingConfig,
)
from musikalyze.exceptions import (
    AudioLoadError,
    PredictionError,
    UnknownEmbedderError,
    UnknownMetaKeyError,
    musikalyzeError,
)
from musikalyze.visualizer import MusicEDA

__all__ = [
    "AnalysisResult",
    "AudioLoadError",
    "EmbeddingModel",
    "ExportConfig",
    "LabelExtractor",
    "MusicBatch",
    "MusicEDA",
    "MusicProcess",
    "PredictionError",
    "TaggingConfig",
    "UnknownEmbedderError",
    "UnknownMetaKeyError",
    "list_audio_files",
    "musikalyzeError",
    "sample_audio_files",
]

__version__ = "1.2.1"


def __getattr__(name: str):
    if name == "MusicProcess":
        from musikalyze.process import MusicProcess

        return MusicProcess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__version__", "__doc__"})
