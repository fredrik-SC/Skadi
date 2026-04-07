"""Signal detection from spectral data.

Identifies signals in power spectral density data by finding contiguous
frequency bins above a configurable threshold above the noise floor.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.detection.models import DetectedSignal, ScanStep

logger = logging.getLogger(__name__)


class SignalDetector:
    """Detect RF signals from power spectral density data.

    Finds contiguous groups of frequency bins whose power exceeds the
    noise floor by a configurable threshold. Each group is reported as
    a detected signal with estimated centre frequency, bandwidth, and
    power measurements.

    Args:
        detection_config: The 'detection' section from default.yaml.
            Expected keys: threshold_db, min_bandwidth_hz, max_signals_per_step.
    """

    def __init__(self, detection_config: dict[str, Any]) -> None:
        self._threshold_db = float(detection_config.get("threshold_db", 10.0))
        self._min_bandwidth_hz = float(detection_config.get("min_bandwidth_hz", 500))
        self._max_signals = int(detection_config.get("max_signals_per_step", 10))

    def detect(self, scan_step: ScanStep) -> list[DetectedSignal]:
        """Detect signals in a single scan step's PSD data.

        Args:
            scan_step: PSD result from a frequency step, including the
                frequency axis, PSD values, and noise floor estimate.

        Returns:
            List of DetectedSignal sorted by peak power (descending),
            limited to max_signals_per_step entries.
        """
        freqs = scan_step.freqs_hz
        psd = scan_step.psd_dbm
        noise_floor = scan_step.noise_floor_dbm
        threshold = noise_floor + self._threshold_db

        # Find bins above threshold
        above = psd > threshold
        if not np.any(above):
            return []

        # Group contiguous bins into signal regions
        regions = self._find_contiguous_regions(above)

        # Compute bin width for bandwidth calculation
        if len(freqs) > 1:
            bin_width_hz = abs(float(freqs[1] - freqs[0]))
        else:
            return []

        signals: list[DetectedSignal] = []
        for start, stop in regions:
            bandwidth_hz = (stop - start) * bin_width_hz

            # Reject signals narrower than minimum bandwidth
            if bandwidth_hz < self._min_bandwidth_hz:
                continue

            region_psd = psd[start:stop]
            region_freqs = freqs[start:stop]

            peak_power_dbm = float(np.max(region_psd))

            # Mean power: average in linear domain, convert back to dBm
            linear_power = 10.0 ** (region_psd / 10.0)
            mean_power_dbm = float(10.0 * np.log10(np.mean(linear_power)))

            # Centre frequency: power-weighted centroid
            centre_freq_hz = float(np.average(region_freqs, weights=linear_power))

            snr_db = peak_power_dbm - noise_floor

            signals.append(DetectedSignal(
                centre_freq_hz=centre_freq_hz,
                bandwidth_hz=bandwidth_hz,
                peak_power_dbm=peak_power_dbm,
                mean_power_dbm=mean_power_dbm,
                snr_db=snr_db,
                timestamp=scan_step.timestamp,
                scan_step_freq_hz=scan_step.centre_freq_hz,
            ))

        # Merge narrowband subcomponents into their parent wideband signals.
        # E.g. RDS subcarriers, stereo pilots within an FM broadcast.
        signals = self._merge_subcomponents(signals)

        # Sort by peak power descending, return top N
        signals.sort(key=lambda s: s.peak_power_dbm, reverse=True)
        if len(signals) > self._max_signals:
            signals = signals[:self._max_signals]

        logger.debug(
            "Detected %d signal(s) at step %.3f MHz (threshold=%.1f dBm)",
            len(signals), scan_step.centre_freq_hz / 1e6, threshold,
        )
        return signals

    @staticmethod
    def _merge_subcomponents(
        signals: list[DetectedSignal],
    ) -> list[DetectedSignal]:
        """Merge narrowband signals that fall within a wider parent signal.

        If a wideband signal (e.g. FM broadcast at 200 kHz) is detected
        alongside narrowband signals (e.g. RDS at 2 kHz) whose centre
        frequency falls within the parent's bandwidth, the narrowband
        signals are absorbed into the parent. This prevents FM subcarriers,
        stereo pilots, and other signal components from being reported as
        independent detections.

        A narrowband signal is only absorbed if:
        - Its centre frequency is within the parent's band (freq +/- bw/2)
        - Its bandwidth is less than 1/4 of the parent's bandwidth
        - The parent is at least 10 kHz wide (don't merge into very narrow signals)

        Args:
            signals: List of detected signals from a single scan step.

        Returns:
            Filtered list with subcomponents removed.
        """
        if len(signals) < 2:
            return signals

        # Sort by bandwidth descending — widest signals are potential parents
        by_bw = sorted(signals, key=lambda s: s.bandwidth_hz, reverse=True)

        absorbed: set[int] = set()  # Indices into the original signals list

        for i, parent in enumerate(by_bw):
            if parent.bandwidth_hz < 10_000:
                break  # No more wide-enough parents

            # Use an absorption zone wider than the measured bandwidth.
            # FM broadcast occupies 200 kHz even if we measure 50 kHz.
            # Use 2x measured bandwidth or 150 kHz, whichever is larger.
            absorption_radius = max(parent.bandwidth_hz, 150_000) / 2
            parent_lower = parent.centre_freq_hz - absorption_radius
            parent_upper = parent.centre_freq_hz + absorption_radius

            for j, child in enumerate(by_bw):
                if i == j or id(child) in absorbed:
                    continue

                # Child must be significantly narrower than parent
                if child.bandwidth_hz >= parent.bandwidth_hz / 4:
                    continue

                # Child's centre must fall within parent's band
                if parent_lower <= child.centre_freq_hz <= parent_upper:
                    absorbed.add(id(child))
                    logger.debug(
                        "Merged subcomponent %.3f MHz (%.1f kHz) into parent %.3f MHz (%.1f kHz)",
                        child.centre_freq_hz / 1e6, child.bandwidth_hz / 1e3,
                        parent.centre_freq_hz / 1e6, parent.bandwidth_hz / 1e3,
                    )

        result = [s for s in signals if id(s) not in absorbed]

        if absorbed:
            logger.debug(
                "Merged %d subcomponent(s) into parent signals (%d -> %d)",
                len(absorbed), len(signals), len(result),
            )

        return result

    @staticmethod
    def _find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
        """Find contiguous True regions in a boolean array.

        Args:
            mask: Boolean array where True indicates bins above threshold.

        Returns:
            List of (start_index, stop_index) tuples. Each region spans
            mask[start:stop] (stop is exclusive).
        """
        diff = np.diff(mask.astype(np.int8))
        starts = list(np.where(diff == 1)[0] + 1)
        stops = list(np.where(diff == -1)[0] + 1)

        # Handle signal at array start
        if mask[0]:
            starts.insert(0, 0)
        # Handle signal at array end
        if mask[-1]:
            stops.append(len(mask))

        return list(zip(starts, stops))
