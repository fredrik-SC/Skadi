"""Signal isolation from wideband IQ captures.

Extracts individual signals from a wideband IQ capture by frequency-shifting
the target signal to baseband, low-pass filtering, and decimating.
Supports both wideband (FM broadcast) and narrowband (HF military) signals
with adaptive filter design and multi-stage decimation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import firwin, lfilter

logger = logging.getLogger(__name__)

# Maximum decimation per stage to maintain filter effectiveness
_MAX_DECIM_PER_STAGE = 50


class SignalIsolator:
    """Isolate individual signals from a wideband IQ capture.

    Given a wideband IQ capture and a detected signal's centre frequency
    and bandwidth, frequency-shifts and filters to extract only the
    signal of interest. Uses adaptive filter design and multi-stage
    decimation for narrowband signals.

    Args:
        sample_rate: Sample rate of the wideband capture in Hz.
        guard_factor: Multiply signal bandwidth by this for the filter
            width. Provides margin to capture signal edges.
        filter_numtaps: Minimum number of FIR filter taps. Increased
            adaptively for narrowband signals.
    """

    def __init__(
        self,
        sample_rate: float = 2_048_000,
        guard_factor: float = 3.0,
        filter_numtaps: int = 101,
    ) -> None:
        self._sample_rate = sample_rate
        self._guard_factor = guard_factor
        self._min_filter_numtaps = filter_numtaps

    def isolate(
        self,
        iq_data: np.ndarray,
        step_centre_hz: float,
        signal_centre_hz: float,
        signal_bandwidth_hz: float,
    ) -> tuple[np.ndarray, float]:
        """Isolate a signal from a wideband IQ capture.

        For narrowband signals (high decimation ratios), uses multi-stage
        decimation with adaptive filter orders to maintain signal quality.

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
        # Ensure a reasonable minimum filter bandwidth
        filter_bw = max(filter_bw, 500.0)

        # Compute total decimation needed
        total_decim = max(1, int(self._sample_rate / filter_bw))

        # For wideband signals, use less decimation to preserve quality
        if signal_bandwidth_hz > 100_000:
            total_decim = max(1, int(self._sample_rate / (signal_bandwidth_hz * 4)))

        # Apply multi-stage decimation for large ratios
        filtered = shifted
        current_sr = self._sample_rate

        if total_decim <= 1:
            # No decimation needed — just filter
            filtered = self._apply_filter(filtered, filter_bw, current_sr)
        else:
            # Decompose into stages, each ≤ _MAX_DECIM_PER_STAGE
            stages = self._compute_decimation_stages(total_decim)

            for stage_decim in stages:
                # Filter bandwidth for this stage: target the final bandwidth
                # but at the current sample rate
                stage_cutoff = filter_bw / 2
                filtered = self._apply_filter(filtered, stage_cutoff * 2, current_sr)

                # Decimate
                if stage_decim > 1:
                    filtered = filtered[::stage_decim]
                    current_sr = current_sr / stage_decim

        new_sample_rate = current_sr

        # Ensure we have enough samples for downstream processing
        if len(filtered) < 256:
            logger.warning(
                "Isolated signal has only %d samples (need 256+), skipping decimation",
                len(filtered),
            )
            # Fall back: just frequency-shift and filter without decimation
            filtered = self._apply_filter(shifted, filter_bw, self._sample_rate)
            new_sample_rate = self._sample_rate

        logger.debug(
            "Isolated signal at %.3f MHz: offset=%.1f kHz, "
            "filter_bw=%.1f Hz, total_decim=%d, new_sr=%.0f Hz, "
            "%d samples",
            signal_centre_hz / 1e6, offset_hz / 1e3,
            filter_bw, total_decim, new_sample_rate,
            len(filtered),
        )

        return filtered.astype(np.complex64), new_sample_rate

    def _apply_filter(
        self, data: np.ndarray, bandwidth_hz: float, sample_rate: float
    ) -> np.ndarray:
        """Apply a low-pass FIR filter with adaptive order.

        Args:
            data: Complex IQ data to filter.
            bandwidth_hz: Filter bandwidth in Hz.
            sample_rate: Current sample rate in Hz.

        Returns:
            Filtered data with transient removed.
        """
        cutoff_hz = min(bandwidth_hz / 2, sample_rate / 2 * 0.95)
        cutoff_norm = cutoff_hz / (sample_rate / 2)
        cutoff_norm = max(cutoff_norm, 0.001)

        # Adaptive filter order: more taps for narrower relative bandwidth
        # This ensures adequate stopband rejection at all decimation ratios
        numtaps = max(
            self._min_filter_numtaps,
            int(4.0 / cutoff_norm),  # ~4 / normalised_cutoff for good rejection
        )
        # Keep odd for symmetry, cap at reasonable maximum
        numtaps = min(numtaps, 1001)
        if numtaps % 2 == 0:
            numtaps += 1

        taps = firwin(numtaps, cutoff_norm)
        filtered = lfilter(taps, 1.0, data)

        # Discard filter transient
        if len(filtered) > numtaps * 2:
            filtered = filtered[numtaps:]

        return filtered

    @staticmethod
    def _compute_decimation_stages(total_decim: int) -> list[int]:
        """Decompose a large decimation factor into manageable stages.

        Each stage decimates by at most _MAX_DECIM_PER_STAGE.

        Args:
            total_decim: Total decimation factor needed.

        Returns:
            List of per-stage decimation factors that multiply to
            approximately total_decim.
        """
        if total_decim <= _MAX_DECIM_PER_STAGE:
            return [total_decim]

        stages = []
        remaining = total_decim
        while remaining > _MAX_DECIM_PER_STAGE:
            # Use a moderate factor per stage
            stage = min(_MAX_DECIM_PER_STAGE, int(remaining ** 0.5) + 1)
            stage = max(2, stage)
            stages.append(stage)
            remaining = max(1, remaining // stage)
        if remaining > 1:
            stages.append(remaining)

        return stages
