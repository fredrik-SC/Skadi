"""Signal detection pipeline for Skaði.

Provides noise floor estimation, signal detection from PSD data,
and data models for the detection results.
"""

from src.detection.detector import SignalDetector
from src.detection.exclusions import ExclusionEntry, ExclusionFilter
from src.detection.models import DetectedSignal, ScanResult, ScanStep
from src.detection.noise import NoiseEstimator

__all__ = [
    "DetectedSignal",
    "ExclusionEntry",
    "ExclusionFilter",
    "NoiseEstimator",
    "ScanResult",
    "ScanStep",
    "SignalDetector",
]
