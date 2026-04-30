"""musikalize — Essentia-based audio analysis, tagging, and transcoding."""

from musikalize.config import (
    AnalysisResult,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    TaggingConfig,
)
from musikalize.exceptions import (
    MusikalizeError,
    PredictionError,
    UnknownEmbedderError,
    UnknownMetaKeyError,
)
from musikalize.batch import (
    list_audio_files,
    sample_audio_files,
    process_files_parallel,
)

__all__ = [
    "AnalysisResult",
    "EmbeddingModel",
    "ExportConfig",
    "LabelExtractor",
    "TaggingConfig",
    "MusikalizeError",
    "PredictionError",
    "UnknownEmbedderError",
    "UnknownMetaKeyError",
    "MusicProcess",
    "list_audio_files",
    "sample_audio_files",
    "process_files_parallel",
]

__version__ = "0.3.0"


def __getattr__(name: str):
    if name == "MusicProcess":
        from musikalize.process import MusicProcess

        return MusicProcess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__version__", "__doc__"})
