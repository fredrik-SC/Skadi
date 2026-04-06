"""Signal isolation from wideband IQ captures.

Extracts individual signals from a 2 MHz IQ capture by frequency-shifting
the target signal to baseband, low-pass filtering, and decimating.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import firwin, lfilter

logger = logging.getLogger(__name__)


class SignalIsolator:
    """Isolate individual signals from a wideband IQ capture.

    Given a 2 MHz IQ capture and a detected signal's centre frequency
    and bandwidth, frequency-shifts and filters to extract only the
    signal of interest.

    Args:
        sample_rate: Sample rate of the wideband capture in Hz.
        guard_factor: Multiply signal bandwidth by this for the filter
            width. Provides margin to capture signal edges.
        filter_numtaps: Number of FIR filter taps.
    """

    def __init__(
        self,
        sample_rate: float = 2_048_000,
        guard_factor: float = 1.5,
        filter_numtaps: int = 101,
    ) -> None:
        self._sample_rate = sample_rate
        self._guard_factor = guard_factor
        self._filter_numtaps = filter_numtaps

    def isolate(
        self,
        iq_data: np.ndarray,
        step_centre_hz: float,
        signal_centre_hz: float,
        signal_bandwidth_hz: float,
    ) -> tuple[np.ndarray, float]:
        """Isolate a signal from a wideband IQ capture.

        Args:
            iq_data: Full wideband IQ capture (complex64).
            step_centre_hz: Centre frequency of the IQ capture.
            signal_centre_hz: Centre frequency of the target signal.
            signal_bandwidth_hz: Detected bandwidth of the target signal.

        Returns:
            Tuple of (isolated_iq, new_sample_rate) where isolated_iq
            is the baseband signal and new_sample_rate is the effective
            sample rate after decimation.
        """
        # Frequency shift to move target signal to DC
        offset_hz = signal_centre_hz - step_centre_hz
        num_samples = len(iq_data)
        t = np.arange(num_samples) / self._sample_rate
        shifted = iq_data * np.exp(-1j * 2 * np.pi * offset_hz * t).astype(np.complex64)

        # Filter bandwidth with guard margin
        filter_bw = signal_bandwidth_hz * self._guard_factor
        # Ensure filter bandwidth doesn't exceed Nyquist
        cutoff_hz = min(filter_bw / 2, self._sample_rate / 2 * 0.95)

        # Normalised cutoff for firwin (relative to Nyquist)
        cutoff_norm = cutoff_hz / (self._sample_rate / 2)
        cutoff_norm = max(cutoff_norm, 0.001)  # Prevent zero cutoff

        # Design low-pass FIR filter
        taps = firwin(self._filter_numtaps, cutoff_norm)

        # Apply filter
        filtered = lfilter(taps, 1.0, shifted)

        # Discard filter transient
        filtered = filtered[self._filter_numtaps:]

        # Compute decimation factor
        decim_factor = max(1, int(self._sample_rate / filter_bw))
        if decim_factor > 1:
            filtered = filtered[::decim_factor]

        new_sample_rate = self._sample_rate / decim_factor

        logger.debug(
            "Isolated signal at %.3f MHz: offset=%.1f kHz, "
            "filter_bw=%.1f kHz, decim=%d, new_sr=%.0f Hz, "
            "%d samples",
            signal_centre_hz / 1e6, offset_hz / 1e3,
            filter_bw / 1e3, decim_factor, new_sample_rate,
            len(filtered),
        )

        return filtered.astype(np.complex64), new_sample_rate
