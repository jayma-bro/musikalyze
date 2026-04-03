"""Explicit errors for musikalize."""

from __future__ import annotations


class MusikalizeError(Exception):
    """Base error."""


class UnknownEmbedderError(MusikalizeError):
    """Unknown embedding name in the registry."""


class UnknownMetaKeyError(MusikalizeError):
    """Unknown or unavailable metadata key."""


class PredictionError(MusikalizeError):
    """TensorFlow / Essentia inference failure."""
