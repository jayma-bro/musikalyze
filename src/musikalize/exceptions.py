"""Erreurs explicites pour musikalize."""

from __future__ import annotations


class MusikalizeError(Exception):
    """Erreur de base."""


class UnknownEmbedderError(MusikalizeError):
    """Référence d'embedding inconnue (nom absent du registre)."""


class UnknownMetaKeyError(MusikalizeError):
    """Clé de métadonnée inconnue ou non calculable."""


class PredictionError(MusikalizeError):
    """Échec d'inférence TensorFlow / Essentia."""
