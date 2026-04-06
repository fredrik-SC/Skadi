"""Noise floor estimation for signal detection.

Provides a percentile-based noise floor estimator with optional adaptive
rolling window smoothing across multiple scan steps.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class NoiseEstimator:
    """Estimate RF noise floor from power spectral density data.

    Uses a percentile-based approach per scan step, optionally smoothed
    with an exponential moving average over a rolling window of recent
    estimates. This reduces false detections caused by transient noise
    spikes while still tracking genuine changes in the RF environment.

    Args:
        percentile: Which percentile to use for per-step estimates (0-100).
            Default 50 (median).
        window_size: Number of historical estimates to retain for rolling
            smoothing. Default 10.
        alpha: Exponential moving average weight for the current estimate.
            0.3 means 30% current + 70% historical median. Set to 1.0
            to disable smoothing (pure single-step behaviour).
    """

    def __init__(
        self,
        percentile: float = 50.0,
        window_size: int = 10,
        alpha: float = 0.3,
    ) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"Percentile must be 0-100, got {percentile}")
        if window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {window_size}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be 0-1, got {alpha}")

        self._percentile = percentile
        self._alpha = alpha
        self._history: deque[float] = deque(maxlen=window_size)

    @property
    def history_depth(self) -> int:
        """Number of noise floor estimates currently in the rolling window."""
        return len(self._history)

    def reset(self) -> None:
        """Clear the rolling noise floor history.

        Call this between sweeps or when the scan range changes to
        prevent stale data from biasing the next sweep.
        """
        self._history.clear()

    def estimate(self, psd_dbm: np.ndarray) -> float:
        """Estimate noise floor from a PSD array.

        Computes the per-step percentile, appends it to the rolling
        history, and returns a smoothed estimate when history is
        available.

        Args:
            psd_dbm: Power spectral density values in dBm, shape (N,).

        Returns:
            Estimated noise floor in dBm. On the first call (no history)
            or with alpha=1.0, returns the raw per-step percentile.

        Raises:
            ValueError: If the PSD array is empty.
        """
        if psd_dbm.size == 0:
            raise ValueError("Cannot estimate noise floor from empty PSD array")

        # Per-step percentile estimate
        current = float(np.percentile(psd_dbm, self._percentile))

        # Add to rolling history
        self._history.append(current)

        # If this is the first estimate or smoothing is disabled, return raw
        if len(self._history) < 2 or self._alpha >= 1.0:
            logger.debug(
                "Noise floor estimate: %.1f dBm (raw, history=%d)",
                current, len(self._history),
            )
            return current

        # Smoothed estimate: EMA blend of current with median of history
        history_median = float(np.median(list(self._history)))
        smoothed = self._alpha * current + (1.0 - self._alpha) * history_median

        logger.debug(
            "Noise floor estimate: %.1f dBm (smoothed, current=%.1f, "
            "history_median=%.1f, alpha=%.2f, depth=%d)",
            smoothed, current, history_median, self._alpha, len(self._history),
        )
        return smoothed
