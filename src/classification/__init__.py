"""Signal classification pipeline for Skaði.

Provides Artemis database access, parametric matching, and
confidence scoring for signal identification.
"""

from src.classification.artemis_db import ArtemisDB, ArtemisSignal
from src.classification.classifier import (
    ClassificationMatch,
    ClassificationResult,
    SignalClassifier,
)
from src.classification.confidence import compute_confidence

__all__ = [
    "ArtemisDB",
    "ArtemisSignal",
    "ClassificationMatch",
    "ClassificationResult",
    "SignalClassifier",
    "compute_confidence",
]
