"""musikalyze — Essentia-based audio analysis, tagging, and transcoding."""

from musikalyze.config import (
    AnalysisResult,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    TaggingConfig,
)
from musikalyze.exceptions import (
    musikalyzeError,
    PredictionError,
    UnknownEmbedderError,
    UnknownMetaKeyError,
)
from musikalyze.batch import (
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
    "musikalyzeError",
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
        from musikalyze.process import MusicProcess

        return MusicProcess
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__version__", "__doc__"})
