"""Explicit errors for musikalyze."""

from __future__ import annotations


class musikalyzeError(Exception):
    """Base error."""


class UnknownEmbedderError(musikalyzeError):
    """Unknown embedding name in the registry."""


class UnknownMetaKeyError(musikalyzeError):
    """Unknown or unavailable metadata key."""


class PredictionError(musikalyzeError):
    """TensorFlow / Essentia inference failure."""
