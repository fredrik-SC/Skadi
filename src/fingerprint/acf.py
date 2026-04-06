"""Autocorrelation function computation for signal fingerprinting.

Finds the primary periodicity in a signal's amplitude envelope,
which corresponds to the ACF value used by the Artemis database
for signal matching.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import correlate, find_peaks

logger = logging.getLogger(__name__)


class ACFComputer:
    """Compute autocorrelation function from IQ samples.

    Finds periodic patterns in the signal's amplitude envelope by
    computing the normalised autocorrelation and searching for peaks.

    Args:
        min_lag_ms: Minimum ACF lag to search in milliseconds.
        max_lag_ms: Maximum ACF lag to search in milliseconds.
        min_peak_strength: Minimum normalised peak height to report (0-1).
    """

    def __init__(
        self,
        min_lag_ms: float = 1.0,
        max_lag_ms: float = 5000.0,
        min_peak_strength: float = 0.3,
    ) -> None:
        self._min_lag_ms = min_lag_ms
        self._max_lag_ms = max_lag_ms
        self._min_peak_strength = min_peak_strength

    def compute(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
    ) -> tuple[float | None, float | None]:
        """Compute the primary ACF period from IQ data.

        Args:
            iq_data: Isolated IQ samples (complex64).
            sample_rate: Sample rate in Hz.

        Returns:
            Tuple of (acf_period_ms, peak_strength) if a significant
            periodicity is found, otherwise (None, None).
        """
        if len(iq_data) < 64:
            return None, None

        # Compute amplitude envelope and remove DC
        envelope = np.abs(iq_data).astype(np.float64)
        envelope = envelope - np.mean(envelope)

        # Decimate for efficiency if sample rate is high
        # ACF periods are typically >1ms, so 10 kHz is sufficient
        target_rate = 10_000
        decim = max(1, int(sample_rate / target_rate))
        if decim > 1:
            envelope = envelope[::decim]
            effective_rate = sample_rate / decim
        else:
            effective_rate = sample_rate

        if len(envelope) < 32:
            return None, None

        # Normalised autocorrelation
        acf = correlate(envelope, envelope, mode="full")
        # Take only the right half (positive lags)
        mid = len(acf) // 2
        acf = acf[mid:]

        # Normalise by zero-lag
        if acf[0] > 0:
            acf = acf / acf[0]
        else:
            return None, None

        # Convert lag bounds to samples
        min_lag = max(1, int(self._min_lag_ms * effective_rate / 1000))
        max_lag = min(len(acf) - 1, int(self._max_lag_ms * effective_rate / 1000))

        if min_lag >= max_lag:
            return None, None

        # Search for peaks in the ACF
        search_region = acf[min_lag:max_lag + 1]
        peaks, properties = find_peaks(
            search_region,
            height=self._min_peak_strength,
            prominence=0.1,
        )

        if len(peaks) == 0:
            return None, None

        # Take the strongest peak
        heights = properties["peak_heights"]
        best_idx = np.argmax(heights)
        best_lag_samples = peaks[best_idx] + min_lag
        best_strength = float(heights[best_idx])

        # Convert lag to milliseconds
        acf_period_ms = best_lag_samples / effective_rate * 1000

        logger.debug(
            "ACF: period=%.2f ms, strength=%.3f (lag=%d samples at %.0f Hz)",
            acf_period_ms, best_strength, best_lag_samples, effective_rate,
        )

        return acf_period_ms, best_strength
