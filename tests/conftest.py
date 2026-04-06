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


def generate_am_signal(
    sample_rate: float = 50_000,
    num_samples: int = 50_000,
    mod_freq: float = 1000.0,
    mod_depth: float = 0.8,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate an AM-modulated signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        mod_freq: Modulating frequency in Hz.
        mod_depth: Modulation depth (0-1).
        amplitude: Carrier amplitude.

    Returns:
        Complex64 AM signal at baseband.
    """
    t = np.arange(num_samples) / sample_rate
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    return (amplitude * envelope).astype(np.complex64)


def generate_fm_signal(
    sample_rate: float = 300_000,
    num_samples: int = 150_000,
    mod_freq: float = 1000.0,
    deviation: float = 75_000.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a wideband FM signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        mod_freq: Modulating audio frequency in Hz.
        deviation: Peak frequency deviation in Hz.
        amplitude: Signal amplitude.

    Returns:
        Complex64 FM signal at baseband.
    """
    t = np.arange(num_samples) / sample_rate
    phase = 2 * np.pi * (deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def generate_nfm_signal(
    sample_rate: float = 50_000,
    num_samples: int = 50_000,
    mod_freq: float = 1000.0,
    deviation: float = 2_500.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a narrowband FM signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        mod_freq: Audio frequency in Hz.
        deviation: Peak deviation in Hz (narrowband: ~2.5 kHz).
        amplitude: Signal amplitude.

    Returns:
        Complex64 NFM signal at baseband.
    """
    return generate_fm_signal(sample_rate, num_samples, mod_freq, deviation, amplitude)


def generate_fsk_signal(
    sample_rate: float = 50_000,
    num_samples: int = 50_000,
    symbol_rate: float = 1200.0,
    freq_shift: float = 1000.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a binary FSK signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        symbol_rate: Symbols per second.
        freq_shift: Frequency shift between states in Hz.
        amplitude: Signal amplitude.

    Returns:
        Complex64 BFSK signal at baseband.
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = num_samples // samples_per_symbol + 1
    np.random.seed(42)
    bits = np.random.randint(0, 2, num_symbols)

    # Build instantaneous frequency
    freq = np.zeros(num_samples)
    for i, bit in enumerate(bits):
        start = i * samples_per_symbol
        end = min(start + samples_per_symbol, num_samples)
        freq[start:end] = freq_shift / 2 if bit else -freq_shift / 2

    # Integrate frequency to get phase
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def generate_psk_signal(
    sample_rate: float = 50_000,
    num_samples: int = 50_000,
    symbol_rate: float = 2400.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a BPSK signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        symbol_rate: Symbols per second.
        amplitude: Signal amplitude.

    Returns:
        Complex64 BPSK signal at baseband.
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = num_samples // samples_per_symbol + 1
    np.random.seed(42)
    bits = np.random.randint(0, 2, num_symbols)

    signal = np.zeros(num_samples, dtype=np.complex64)
    for i, bit in enumerate(bits):
        start = i * samples_per_symbol
        end = min(start + samples_per_symbol, num_samples)
        phase = 0.0 if bit else np.pi
        signal[start:end] = amplitude * np.exp(1j * phase)

    return signal


def generate_ook_signal(
    sample_rate: float = 50_000,
    num_samples: int = 50_000,
    symbol_rate: float = 300.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate an OOK (on-off keying) signal at baseband.

    Args:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        symbol_rate: Symbols per second.
        amplitude: Signal amplitude.

    Returns:
        Complex64 OOK signal at baseband.
    """
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = num_samples // samples_per_symbol + 1
    np.random.seed(42)
    bits = np.random.randint(0, 2, num_symbols)

    signal = np.zeros(num_samples, dtype=np.complex64)
    for i, bit in enumerate(bits):
        start = i * samples_per_symbol
        end = min(start + samples_per_symbol, num_samples)
        if bit:
            signal[start:end] = amplitude

    return signal


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
