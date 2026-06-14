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
        envelope_cv: Coefficient of variation of the envelope (std/mean).
            High for AM/OOK, low for constant-envelope FM/FSK/PSK.
        envelope_bimodality: Sarle's bimodality coefficient of the envelope.
            High for OOK (two-level on/off), low for sinusoidal AM.
        num_freq_states_robust: Discrete frequency-state count from the
            median-filtered instantaneous frequency. 2 for FSK, 1 otherwise.
        freq_state_separation: Between-state / within-state spread ratio of the
            instantaneous frequency. Large for FSK, ~0 for FM/PSK.
        inst_freq_center_norm: |median(inst_freq)| / sample_rate. ~0 for a single
            near-DC carrier (PSK), nonzero for an offset signal.
        phase_jump_rate: Fraction of samples with an abrupt phase jump
            (sample-rate independent). High for PSK, ~0 for FM/CPFSK.
        phase_level_concentration: |mean(exp(2j*angle))| of the derotated signal
            (order-2). ~1 for BPSK parked at {0, pi} AND for real-valued AM.
        phase_single_concentration: |mean(exp(1j*angle))| (order-1). ~1 for a
            single phase level (AM/carrier), ~0 for two opposite levels (BPSK).
            Order-2 high with order-1 low is the BPSK signature.
        symbol_rate_hz: Estimated symbol (baud) rate in Hz, or 0 if undetermined.
            Advisory output (not a decision driver).
    """

    envelope_variance: float
    inst_freq_variance: float
    inst_freq_kurtosis: float
    phase_discontinuities: int
    spectral_flatness: float
    num_freq_states: int
    envelope_cv: float = 0.0
    envelope_bimodality: float = 0.0
    num_freq_states_robust: int = 0
    freq_state_separation: float = 0.0
    inst_freq_center_norm: float = 0.0
    phase_jump_rate: float = 0.0
    phase_level_concentration: float = 0.0
    phase_single_concentration: float = 0.0
    symbol_rate_hz: float = 0.0


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
