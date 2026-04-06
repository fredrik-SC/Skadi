"""Data models for signal detection and spectrum scanning.

Shared dataclasses used by the scanner, detector, and noise estimator
modules throughout the detection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScanStep:
    """PSD result from a single frequency step.

    Attributes:
        centre_freq_hz: Actual tuned centre frequency in Hz.
        freqs_hz: Absolute frequency axis in Hz, shape (fft_size,).
        psd_dbm: Power spectral density in dBm, shape (fft_size,).
        noise_floor_dbm: Estimated noise floor for this step in dBm.
        timestamp: Unix timestamp when capture started.
    """

    centre_freq_hz: float
    freqs_hz: np.ndarray
    psd_dbm: np.ndarray
    noise_floor_dbm: float
    timestamp: float


@dataclass
class DetectedSignal:
    """A single detected signal with its measured parameters.

    Attributes:
        centre_freq_hz: Power-weighted centre frequency in Hz.
        bandwidth_hz: Estimated signal bandwidth in Hz.
        peak_power_dbm: Peak PSD value within the signal in dBm.
        mean_power_dbm: Mean PSD across signal bins in dBm.
        snr_db: Peak power minus noise floor in dB.
        timestamp: Unix timestamp of detection.
        scan_step_freq_hz: Centre frequency of the scan step that found this.
    """

    centre_freq_hz: float
    bandwidth_hz: float
    peak_power_dbm: float
    mean_power_dbm: float
    snr_db: float
    timestamp: float
    scan_step_freq_hz: float


@dataclass
class ScanResult:
    """Complete result from a frequency sweep.

    Attributes:
        freq_start_hz: Start frequency of the sweep in Hz.
        freq_stop_hz: Stop frequency of the sweep in Hz.
        num_steps: Number of frequency steps completed.
        duration_seconds: Total sweep duration in seconds.
        signals: All detected signals across the sweep.
        scan_steps: Per-step PSD data (for debug/plotting).
        timestamp: Unix timestamp when sweep started.
    """

    freq_start_hz: float
    freq_stop_hz: float
    num_steps: int
    duration_seconds: float
    signals: list[DetectedSignal] = field(default_factory=list)
    scan_steps: list[ScanStep] = field(default_factory=list)
    timestamp: float = 0.0
