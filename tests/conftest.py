"""Shared test fixtures and helpers for Skaði tests.

Provides synthetic IQ data generation and mock SDR factories
so tests can run without hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


def generate_tone(
    freq_offset_hz: float,
    sample_rate: float,
    num_samples: int,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Generate a complex tone at a given offset from centre frequency.

    Args:
        freq_offset_hz: Frequency offset from DC in Hz.
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples to generate.
        amplitude: Signal amplitude (linear).

    Returns:
        Complex64 array of the generated tone.
    """
    t = np.arange(num_samples) / sample_rate
    return (amplitude * np.exp(2j * np.pi * freq_offset_hz * t)).astype(np.complex64)


def generate_noise(num_samples: int, power_linear: float = 1e-10) -> np.ndarray:
    """Generate complex Gaussian noise at a given power level.

    Args:
        num_samples: Number of samples to generate.
        power_linear: Noise power in linear scale.

    Returns:
        Complex64 array of Gaussian noise.
    """
    sigma = np.sqrt(power_linear / 2)
    noise = (
        np.random.normal(0, sigma, num_samples)
        + 1j * np.random.normal(0, sigma, num_samples)
    )
    return noise.astype(np.complex64)


def generate_test_iq(
    tones: list[tuple[float, float]] | None = None,
    sample_rate: float = 2_048_000,
    num_samples: int = 40960,
    noise_power: float = 1e-10,
) -> np.ndarray:
    """Generate test IQ data with optional tones in noise.

    Args:
        tones: List of (freq_offset_hz, amplitude) tuples for signals.
        sample_rate: Sample rate in Hz.
        num_samples: Total number of samples.
        noise_power: Background noise power (linear).

    Returns:
        Complex64 IQ array with noise and optional tones.
    """
    iq = generate_noise(num_samples, noise_power)
    if tones:
        for freq_offset, amplitude in tones:
            iq += generate_tone(freq_offset, sample_rate, num_samples, amplitude)
    return iq


def make_mock_sdr(
    iq_data: np.ndarray,
    sample_rate: float = 2_048_000,
) -> MagicMock:
    """Create a mock SDRInterface that returns predefined IQ data.

    Args:
        iq_data: The IQ data to return from capture().
        sample_rate: Sample rate to report.

    Returns:
        MagicMock configured as an SDRInterface.
    """
    mock = MagicMock()
    mock.sample_rate = sample_rate
    mock.connected = True
    mock.tune.return_value = 100e6  # Returns "actual" tuned frequency
    mock.capture.return_value = iq_data
    return mock
