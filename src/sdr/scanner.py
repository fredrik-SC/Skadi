"""Frequency sweep and spectrum scanning.

Implements the SpectrumScanner which sweeps across a configurable
frequency range, computes FFT-based power spectral density at each
step, and detects signals using the detection pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from src.detection.detector import SignalDetector
from src.detection.exclusions import ExclusionFilter
from src.detection.models import DetectedSignal, ScanResult, ScanStep
from src.detection.noise import NoiseEstimator
from src.detectionlog.database import DetectionLog
from src.sdr.interface import SDRInterface

logger = logging.getLogger(__name__)

# Brief settling time after retuning (seconds) to let AGC stabilise
_RETUNE_SETTLE_TIME = 0.01


class SpectrumScanner:
    """Sweep the RF spectrum and detect signals.

    Orchestrates the SDR interface, FFT computation, noise estimation,
    and signal detection across a configurable frequency range.

    Args:
        sdr: Connected SDRInterface instance.
        scan_config: The 'scan' section from default.yaml.
        sdr_config: The 'sdr' section from default.yaml (for sample_rate).
        detection_config: The 'detection' section from default.yaml.
        noise_estimator: Optional NoiseEstimator (default: adaptive).
        signal_detector: Optional SignalDetector (default: from detection_config).
        exclusion_filter: Optional ExclusionFilter for suppressing known signals.
        detection_log: Optional DetectionLog for persisting detections to SQLite.

    Example:
        with SDRInterface(config["sdr"]) as sdr:
            scanner = SpectrumScanner(sdr, config["scan"], config["sdr"], config["detection"])
            result = scanner.sweep()
            for sig in result.signals:
                print(f"{sig.centre_freq_hz/1e6:.3f} MHz  BW={sig.bandwidth_hz/1e3:.1f} kHz")
    """

    def __init__(
        self,
        sdr: SDRInterface,
        scan_config: dict[str, Any],
        sdr_config: dict[str, Any],
        detection_config: dict[str, Any],
        noise_estimator: NoiseEstimator | None = None,
        signal_detector: SignalDetector | None = None,
        exclusion_filter: ExclusionFilter | None = None,
        detection_log: DetectionLog | None = None,
    ) -> None:
        self._sdr = sdr

        # Scan parameters
        self._freq_start = float(scan_config["freq_start"])
        self._freq_stop = float(scan_config["freq_stop"])
        self._step_size = float(scan_config["step_size"])
        self._dwell_time = float(scan_config.get("dwell_time", 0.5))
        self._fft_size = int(scan_config.get("fft_size", 4096))
        self._fft_averages = int(scan_config.get("fft_averages", 10))

        # SDR parameters
        self._sample_rate = float(sdr_config.get("sample_rate", 2_048_000))

        # Detection components (dependency injection with sensible defaults)
        self._noise_estimator = noise_estimator or NoiseEstimator(
            window_size=int(detection_config.get("noise_window_size", 10)),
            alpha=float(detection_config.get("noise_alpha", 0.3)),
        )
        self._signal_detector = signal_detector or SignalDetector(detection_config)
        self._exclusion_filter = exclusion_filter
        self._detection_log = detection_log

        # Pre-compute the Hann window
        self._window = np.hanning(self._fft_size).astype(np.float32)
        self._window_power = float(np.sum(self._window ** 2))

    @property
    def num_steps(self) -> int:
        """Number of frequency steps in a full sweep."""
        span = self._freq_stop - self._freq_start
        return max(1, int(np.ceil(span / self._step_size)))

    @property
    def step_frequencies(self) -> list[float]:
        """List of centre frequencies for each step.

        Each centre frequency is offset by half the step size from the
        start so that the first step covers [freq_start, freq_start + step_size].
        """
        freqs = []
        # Centre of each step
        freq = self._freq_start + self._step_size / 2
        while freq < self._freq_stop:
            freqs.append(freq)
            freq += self._step_size
        return freqs

    def compute_psd(self, iq_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute averaged power spectral density from IQ samples.

        Splits IQ data into segments, applies a Hann window to each,
        computes the FFT, and averages the power across all segments.

        Args:
            iq_data: Complex IQ samples, dtype complex64.

        Returns:
            Tuple of (freq_offsets_hz, psd_dbm):
                - freq_offsets_hz: Frequency offsets from centre, shape (fft_size,).
                  Add the tuned centre frequency to get absolute frequencies.
                - psd_dbm: Power spectral density in dBm, shape (fft_size,).
        """
        fft_size = self._fft_size
        num_segments = len(iq_data) // fft_size
        if num_segments == 0:
            logger.warning(
                "IQ data (%d samples) shorter than FFT size (%d), zero-padding",
                len(iq_data), fft_size,
            )
            padded = np.zeros(fft_size, dtype=np.complex64)
            padded[:len(iq_data)] = iq_data
            iq_data = padded
            num_segments = 1

        # Limit to configured number of averages
        num_segments = min(num_segments, self._fft_averages)

        # Accumulate power spectra
        power_sum = np.zeros(fft_size, dtype=np.float64)
        for i in range(num_segments):
            segment = iq_data[i * fft_size:(i + 1) * fft_size]
            windowed = segment * self._window
            spectrum = np.fft.fft(windowed)
            power_sum += np.abs(spectrum) ** 2

        # Average and normalise
        averaged_power = power_sum / (num_segments * self._window_power)

        # Convert to dBm (10*log10 of power, with +30 for dBW→dBm nominal offset)
        # Note: without absolute calibration this is relative dBm
        psd_dbm = 10.0 * np.log10(np.maximum(averaged_power, 1e-20)) + 30.0

        # Shift so DC is centred
        psd_dbm = np.fft.fftshift(psd_dbm)
        freq_offsets = np.fft.fftshift(
            np.fft.fftfreq(fft_size, d=1.0 / self._sample_rate)
        )

        return freq_offsets.astype(np.float64), psd_dbm.astype(np.float64)

    def scan_step(self, freq_hz: float) -> ScanStep:
        """Execute a single scan step: tune, capture, compute PSD.

        Args:
            freq_hz: Centre frequency for this step in Hz.

        Returns:
            ScanStep with PSD data and noise floor estimate.
        """
        timestamp = time.time()

        # Tune and let AGC settle
        actual_freq = self._sdr.tune(freq_hz)
        time.sleep(_RETUNE_SETTLE_TIME)

        # Capture IQ samples for the dwell period
        num_samples = int(self._dwell_time * self._sample_rate)
        # Ensure we have at least enough for the FFT averages
        min_samples = self._fft_size * self._fft_averages
        num_samples = max(num_samples, min_samples)

        iq_data = self._sdr.capture(num_samples)

        # Compute PSD
        freq_offsets, psd_dbm = self.compute_psd(iq_data)
        freqs_hz = freq_offsets + actual_freq

        # Estimate noise floor
        noise_floor = self._noise_estimator.estimate(psd_dbm)

        return ScanStep(
            centre_freq_hz=actual_freq,
            freqs_hz=freqs_hz,
            psd_dbm=psd_dbm,
            noise_floor_dbm=noise_floor,
            timestamp=timestamp,
        )

    def sweep(
        self,
        callback: Callable[[ScanStep, int, int], None] | None = None,
        keep_psd: bool = True,
    ) -> ScanResult:
        """Execute a full frequency sweep.

        Args:
            callback: Optional function called after each step with
                (scan_step, step_index, total_steps) for progress reporting.
            keep_psd: Whether to store per-step PSD data in the result.
                Set to False for long sweeps to save memory.

        Returns:
            ScanResult containing all detected signals and optionally
            the per-step PSD data.
        """
        frequencies = self.step_frequencies
        total_steps = len(frequencies)
        all_signals: list[DetectedSignal] = []
        scan_steps: list[ScanStep] = []

        # Reset adaptive noise estimator for fresh sweep
        self._noise_estimator.reset()

        logger.info(
            "Starting sweep: %.3f - %.3f MHz (%d steps, %.1f MHz step size)",
            self._freq_start / 1e6, self._freq_stop / 1e6,
            total_steps, self._step_size / 1e6,
        )

        sweep_start = time.time()

        for i, freq in enumerate(frequencies):
            step = self.scan_step(freq)

            # Detect signals in this step
            signals = self._signal_detector.detect(step)

            # Apply exclusion filter if configured
            if self._exclusion_filter is not None:
                signals = self._exclusion_filter.filter(signals)

            all_signals.extend(signals)

            if keep_psd:
                scan_steps.append(step)

            if callback is not None:
                callback(step, i, total_steps)

            logger.debug(
                "Step %d/%d: %.3f MHz — %d signal(s), noise=%.1f dBm",
                i + 1, total_steps, freq / 1e6,
                len(signals), step.noise_floor_dbm,
            )

        duration = time.time() - sweep_start

        # Sort all signals by peak power
        all_signals.sort(key=lambda s: s.peak_power_dbm, reverse=True)

        # Log detections to database if configured
        if self._detection_log is not None and all_signals:
            self._detection_log.log_signals(all_signals)

        logger.info(
            "Sweep complete: %d signal(s) detected in %.1f seconds",
            len(all_signals), duration,
        )

        return ScanResult(
            freq_start_hz=self._freq_start,
            freq_stop_hz=self._freq_stop,
            num_steps=total_steps,
            duration_seconds=duration,
            signals=all_signals,
            scan_steps=scan_steps,
            timestamp=sweep_start,
        )
