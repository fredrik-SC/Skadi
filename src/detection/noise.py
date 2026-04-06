"""Noise floor estimation for signal detection.

Provides a simple percentile-based noise floor estimator for Session 2.
Session 3 extends this with adaptive/rolling estimation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class NoiseEstimator:
    """Estimate RF noise floor from power spectral density data.

    Uses a percentile-based approach: the noise floor is the Nth percentile
    of PSD bin values. The median (50th percentile) is robust because in
    most scan steps, signals occupy a small fraction of the bandwidth,
    so the median naturally falls on noise.

    Args:
        percentile: Which percentile to use for the estimate (0-100).
            Default 50 (median). Use lower values (e.g. 25) for very
            busy bands where signals occupy a large fraction of bins.
    """

    def __init__(self, percentile: float = 50.0) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"Percentile must be 0-100, got {percentile}")
        self._percentile = percentile

    def estimate(self, psd_dbm: np.ndarray) -> float:
        """Estimate noise floor from a PSD array.

        Args:
            psd_dbm: Power spectral density values in dBm, shape (N,).

        Returns:
            Estimated noise floor in dBm.

        Raises:
            ValueError: If the PSD array is empty.
        """
        if psd_dbm.size == 0:
            raise ValueError("Cannot estimate noise floor from empty PSD array")

        noise_floor = float(np.percentile(psd_dbm, self._percentile))
        logger.debug("Noise floor estimate: %.1f dBm (percentile=%.0f)", noise_floor, self._percentile)
        return noise_floor
