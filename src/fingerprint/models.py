"""Data models for signal fingerprinting.

Defines the modulation type enumeration, intermediate feature vectors,
and the complete fingerprint data structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.detection.models import DetectedSignal


class ModulationType(str, Enum):
    """Detected modulation type categories for v1.0.

    Uses str mixin so values serialise directly to JSON and SQLite TEXT.
    """

    AM = "AM"
    FM = "FM"
    NFM = "NFM"
    FSK = "FSK"
    PSK = "PSK"
    OOK = "OOK"
    UNKNOWN = "UNKNOWN"


@dataclass
class ModulationFeatures:
    """Intermediate feature vector from modulation analysis.

    Retained in the fingerprint for debugging and potential future
    ML training data.

    Attributes:
        envelope_variance: Normalised variance of signal envelope.
            High for AM/OOK, low for FM/PSK.
        inst_freq_variance: Variance of instantaneous frequency.
            High for FM, low for AM.
        inst_freq_kurtosis: Kurtosis of instantaneous frequency.
            High for FSK (discrete jumps).
        phase_discontinuities: Count of abrupt phase jumps.
            High for PSK.
        spectral_flatness: Wiener entropy of power spectrum (0-1).
            Low for tonal/analogue, high for digital.
        num_freq_states: Estimated discrete frequency levels.
            2+ for FSK, ~0 for continuous modulations.
    """

    envelope_variance: float
    inst_freq_variance: float
    inst_freq_kurtosis: float
    phase_discontinuities: int
    spectral_flatness: float
    num_freq_states: int


@dataclass
class SignalFingerprint:
    """Complete fingerprint for a detected signal.

    Extends the DetectedSignal with modulation classification,
    refined bandwidth, and ACF value for Artemis DB matching.

    Attributes:
        signal: The original detected signal.
        modulation: Classified modulation type.
        modulation_confidence: Confidence in the modulation call (0-1).
        bandwidth_hz: Refined bandwidth from isolated signal analysis.
        acf_ms: Primary ACF period in milliseconds, or None.
        acf_strength: Normalised ACF peak strength (0-1), or None.
        features: Raw feature vector for debugging.
    """

    signal: DetectedSignal
    modulation: ModulationType
    modulation_confidence: float
    bandwidth_hz: float
    acf_ms: float | None
    acf_strength: float | None
    features: ModulationFeatures
