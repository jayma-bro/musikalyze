"""musikalize — analyse audio (Essentia), tags et export."""

from musikalize.config import (
    AnalysisConfig,
    AnalysisResult,
    ClassicalConfig,
    ClassificationHeadSpec,
    EmbeddingModel,
    ExportConfig,
    LabelExtractor,
    ModelPath,
    TaggingConfig,
)
from musikalize.exceptions import (
    MusikalizeError,
    PredictionError,
    UnknownEmbedderError,
    UnknownMetaKeyError,
)

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ClassicalConfig",
    "ClassificationHeadSpec",
    "EmbeddingModel",
    "ExportConfig",
    "LabelExtractor",
    "ModelPath",
    "MusicProcess",
    "MusikalizeError",
    "PredictionError",
    "TaggingConfig",
    "UnknownEmbedderError",
    "UnknownMetaKeyError",
    "list_audio_files",
    "process_files_parallel",
]

__version__ = "0.2.0"


def __getattr__(name: str):
    if name == "MusicProcess":
        from musikalize.process import MusicProcess

        return MusicProcess
    if name == "list_audio_files":
        from musikalize.batch import list_audio_files

        return list_audio_files
    if name == "process_files_parallel":
        from musikalize.batch import process_files_parallel

        return process_files_parallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"__version__", "__doc__"})
