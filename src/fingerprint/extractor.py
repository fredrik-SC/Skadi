"""Fingerprint extraction orchestrator.

Ties signal isolation, modulation classification, bandwidth refinement,
and ACF computation into a single extraction pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.detection.models import DetectedSignal, ScanStep
from src.fingerprint.acf import ACFComputer
from src.fingerprint.isolation import SignalIsolator
from src.fingerprint.models import ModulationType, SignalFingerprint
from src.fingerprint.modulation import ModulationClassifier

logger = logging.getLogger(__name__)


class FingerprintExtractor:
    """Extract signal fingerprints from detected signals.

    Orchestrates signal isolation, modulation classification,
    bandwidth refinement, and ACF computation for each detected
    signal in a scan step.

    Args:
        sample_rate: SDR sample rate in Hz.
        config: Optional fingerprint configuration dict from default.yaml.
    """

    def __init__(
        self,
        sample_rate: float = 2_048_000,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}

        self._sample_rate = sample_rate
        self._isolator = SignalIsolator(
            sample_rate=sample_rate,
            guard_factor=float(cfg.get("guard_factor", 1.5)),
            filter_numtaps=int(cfg.get("filter_numtaps", 101)),
        )
        self._classifier = ModulationClassifier(
            min_snr_db=float(cfg.get("min_snr_db", 8.0)),
            config=cfg.get("modulation"),
        )
        self._acf = ACFComputer(
            min_lag_ms=float(cfg.get("acf_min_lag_ms", 1.0)),
            max_lag_ms=float(cfg.get("acf_max_lag_ms", 5000.0)),
            min_peak_strength=float(cfg.get("acf_min_peak_strength", 0.3)),
        )

    def extract(
        self,
        signal: DetectedSignal,
        scan_step: ScanStep,
    ) -> SignalFingerprint:
        """Extract a complete fingerprint from a detected signal.

        Args:
            signal: The detected signal with frequency and bandwidth.
            scan_step: The scan step containing the wideband IQ data.

        Returns:
            SignalFingerprint with modulation, bandwidth, and ACF.
        """
        if scan_step.iq_data is None:
            logger.warning("No IQ data in scan step, returning UNKNOWN fingerprint")
            from src.fingerprint.models import ModulationFeatures
            return SignalFingerprint(
                signal=signal,
                modulation=ModulationType.UNKNOWN,
                modulation_confidence=0.0,
                bandwidth_hz=signal.bandwidth_hz,
                acf_ms=None,
                acf_strength=None,
                features=ModulationFeatures(0, 0, 0, 0, 0, 0),
            )

        # Isolate the signal from the wideband capture
        isolated_iq, isolated_sr = self._isolator.isolate(
            iq_data=scan_step.iq_data,
            step_centre_hz=scan_step.centre_freq_hz,
            signal_centre_hz=signal.centre_freq_hz,
            signal_bandwidth_hz=signal.bandwidth_hz,
        )

        # Classify modulation
        mod_type, mod_conf, features = self._classifier.classify(
            iq_data=isolated_iq,
            sample_rate=isolated_sr,
            snr_db=signal.snr_db,
            bandwidth_hz=signal.bandwidth_hz,
        )

        # Refine bandwidth from isolated signal's PSD
        refined_bw = self._refine_bandwidth(isolated_iq, isolated_sr, signal.bandwidth_hz)

        # Compute ACF
        acf_ms, acf_strength = self._acf.compute(isolated_iq, isolated_sr)

        return SignalFingerprint(
            signal=signal,
            modulation=mod_type,
            modulation_confidence=mod_conf,
            bandwidth_hz=refined_bw,
            acf_ms=acf_ms,
            acf_strength=acf_strength,
            features=features,
        )

    def extract_batch(
        self,
        signals: list[DetectedSignal],
        scan_step: ScanStep,
    ) -> list[SignalFingerprint]:
        """Extract fingerprints for all signals in a scan step.

        Args:
            signals: List of detected signals from one scan step.
            scan_step: The scan step with IQ data.

        Returns:
            List of fingerprints in the same order as signals.
        """
        fingerprints = []
        for signal in signals:
            try:
                fp = self.extract(signal, scan_step)
                fingerprints.append(fp)
            except Exception as e:
                logger.warning(
                    "Fingerprint extraction failed for signal at %.3f MHz: %s",
                    signal.centre_freq_hz / 1e6, e,
                )
                from src.fingerprint.models import ModulationFeatures
                fingerprints.append(SignalFingerprint(
                    signal=signal,
                    modulation=ModulationType.UNKNOWN,
                    modulation_confidence=0.0,
                    bandwidth_hz=signal.bandwidth_hz,
                    acf_ms=None,
                    acf_strength=None,
                    features=ModulationFeatures(0, 0, 0, 0, 0, 0),
                ))
        return fingerprints

    @staticmethod
    def _refine_bandwidth(
        iq_data: np.ndarray,
        sample_rate: float,
        fallback_bw: float,
    ) -> float:
        """Refine bandwidth estimate from isolated signal's PSD.

        Measures the -10 dB bandwidth from the peak of the isolated
        signal's power spectrum.

        Args:
            iq_data: Isolated IQ samples.
            sample_rate: Sample rate after isolation.
            fallback_bw: Bandwidth to return if estimation fails.

        Returns:
            Refined bandwidth in Hz.
        """
        if len(iq_data) < 64:
            return fallback_bw

        fft_size = min(len(iq_data), 4096)
        psd = np.abs(np.fft.fftshift(np.fft.fft(iq_data[:fft_size]))) ** 2
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))

        peak_db = np.max(psd_db)
        threshold_db = peak_db - 10.0

        # Count bins above -10 dB from peak
        above = psd_db > threshold_db
        num_bins = int(np.sum(above))

        if num_bins == 0:
            return fallback_bw

        bin_width = sample_rate / fft_size
        refined = num_bins * bin_width

        # Sanity check: don't return wildly different from detection estimate
        if refined < fallback_bw * 0.1 or refined > fallback_bw * 10:
            return fallback_bw

        return refined
