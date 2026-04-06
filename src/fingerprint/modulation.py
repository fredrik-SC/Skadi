"""Modulation type classification from IQ samples.

Deterministic feature-based classifier that discriminates between
AM, FM, NFM, FSK, PSK, OOK, and UNKNOWN using energy-based and
spectral feature extraction (no ML for v1.0).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks

from src.fingerprint.models import ModulationFeatures, ModulationType

logger = logging.getLogger(__name__)

# Default classification thresholds
_DEFAULTS = {
    "envelope_variance_threshold": 0.1,
    "envelope_variance_ook": 0.6,
    "inst_freq_variance_high": 0.05,
    "spectral_flatness_digital": 0.4,
    "phase_discontinuity_rate": 0.02,
    "inst_freq_kurtosis_fsk": 3.0,
    "wideband_fm_min_bw_hz": 50_000,
}


class ModulationClassifier:
    """Deterministic modulation type classifier.

    Uses six computed features and a decision tree to classify
    signals into AM, FM, NFM, FSK, PSK, OOK, or UNKNOWN.

    Args:
        min_snr_db: Minimum SNR to attempt classification. Below this,
            returns UNKNOWN. Default 8.0.
        config: Optional dict of threshold overrides. Keys match
            the defaults in _DEFAULTS.
    """

    def __init__(
        self,
        min_snr_db: float = 8.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._min_snr_db = min_snr_db
        self._thresholds = dict(_DEFAULTS)
        if config:
            self._thresholds.update(config)

    def classify(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        snr_db: float,
        bandwidth_hz: float,
    ) -> tuple[ModulationType, float, ModulationFeatures]:
        """Classify modulation type from isolated IQ data.

        Args:
            iq_data: Isolated, baseband IQ samples (complex64).
            sample_rate: Effective sample rate after isolation/decimation.
            snr_db: Signal-to-noise ratio in dB.
            bandwidth_hz: Detected signal bandwidth in Hz.

        Returns:
            Tuple of (modulation_type, confidence, features).
        """
        # Bail on very short or low-SNR signals
        if len(iq_data) < 256 or snr_db < self._min_snr_db:
            features = ModulationFeatures(0, 0, 0, 0, 0, 0)
            return ModulationType.UNKNOWN, 0.0, features

        features = self._compute_features(iq_data, sample_rate)
        mod_type, confidence = self._decide(features, bandwidth_hz)

        logger.debug(
            "Modulation: %s (conf=%.2f) — env_var=%.4f, if_var=%.4f, "
            "if_kurt=%.2f, phase_disc=%d, sf=%.3f, freq_states=%d",
            mod_type.value, confidence,
            features.envelope_variance, features.inst_freq_variance,
            features.inst_freq_kurtosis, features.phase_discontinuities,
            features.spectral_flatness, features.num_freq_states,
        )

        return mod_type, confidence, features

    def _compute_features(
        self, iq_data: np.ndarray, sample_rate: float
    ) -> ModulationFeatures:
        """Compute the six classification features from IQ data."""
        envelope_var = self._compute_envelope_variance(iq_data)
        if_var, if_kurt = self._compute_inst_freq_stats(iq_data, sample_rate)
        phase_disc = self._count_phase_discontinuities(iq_data)
        spectral_flat = self._compute_spectral_flatness(iq_data)
        freq_states = self._estimate_freq_states(iq_data, sample_rate)

        return ModulationFeatures(
            envelope_variance=envelope_var,
            inst_freq_variance=if_var,
            inst_freq_kurtosis=if_kurt,
            phase_discontinuities=phase_disc,
            spectral_flatness=spectral_flat,
            num_freq_states=freq_states,
        )

    def _decide(
        self, features: ModulationFeatures, bandwidth_hz: float
    ) -> tuple[ModulationType, float]:
        """Apply the decision tree to classify modulation.

        The tree uses bandwidth as a strong early discriminator alongside
        spectral features. Wideband signals (>100 kHz) are almost always
        FM broadcast. Narrowband signals (<50 kHz) require feature analysis.
        """
        th = self._thresholds
        env_th = th["envelope_variance_threshold"]
        env_ook = th["envelope_variance_ook"]
        if_high = th["inst_freq_variance_high"]
        sf_digital = th["spectral_flatness_digital"]
        if_kurt_fsk = th["inst_freq_kurtosis_fsk"]
        wfm_bw = th["wideband_fm_min_bw_hz"]

        # Step 0: Bandwidth-aware pre-classification for wideband signals.
        # A signal >100 kHz wide is almost certainly FM broadcast — no digital
        # mode occupies that much bandwidth in VHF/UHF. Only override if
        # envelope variance is very high (clear AM, e.g. AM broadcast).
        if bandwidth_hz >= wfm_bw:
            # For true AM, envelope variance is moderate (0.1-0.5) and
            # inst_freq_variance is very low. Very high envelope variance
            # (>1.0) is more likely filter artefacts on an FM signal.
            if (features.envelope_variance > 0.15
                    and features.envelope_variance < 1.0
                    and features.inst_freq_variance < if_high * 0.5):
                return ModulationType.AM, 0.7
            # Wideband signal: default to FM
            return ModulationType.FM, max(0.5, min(1.0,
                features.inst_freq_variance / (if_high * 3) + 0.4))

        # Step 1: AM-family — high envelope modulation, low freq modulation
        if features.envelope_variance > env_th and features.inst_freq_variance < if_high:
            if features.envelope_variance > env_ook:
                conf = min(1.0, (features.envelope_variance - env_ook) / 0.3 + 0.5)
                return ModulationType.OOK, conf
            conf = min(1.0, (features.envelope_variance - env_th) / 0.2 + 0.4)
            return ModulationType.AM, conf

        # Step 2: FM-family — high instantaneous frequency variation
        if features.inst_freq_variance > if_high:
            if features.envelope_variance < env_th * 2:
                conf = min(1.0, features.inst_freq_variance / (if_high * 5) + 0.3)
                return ModulationType.NFM, conf

        # Step 3: Digital modulations — noise-like spectrum + narrowband
        if features.spectral_flatness > sf_digital and bandwidth_hz < wfm_bw:
            # FSK: discrete frequency states + high kurtosis
            if features.num_freq_states >= 2 and features.inst_freq_kurtosis > if_kurt_fsk:
                conf = min(1.0, 0.4 + 0.1 * features.num_freq_states)
                return ModulationType.FSK, conf

            # PSK: many phase discontinuities + low envelope variance +
            # narrowband (< 50 kHz). This prevents wideband FM from
            # being misclassified as PSK.
            if (features.phase_discontinuities > 200
                    and features.envelope_variance < env_th
                    and bandwidth_hz < 50_000):
                conf = min(1.0, 0.3 + features.spectral_flatness * 0.4)
                return ModulationType.PSK, conf

        # Step 4: Weak FM characteristics
        if features.inst_freq_variance > if_high * 0.3:
            return ModulationType.NFM, 0.3

        return ModulationType.UNKNOWN, 0.0

    @staticmethod
    def _compute_envelope_variance(iq: np.ndarray) -> float:
        """Normalised variance of the signal envelope.

        AM signals have high variance; constant-envelope signals (FM, PSK)
        have low variance.
        """
        envelope = np.abs(iq)
        mean_env = np.mean(envelope)
        if mean_env < 1e-10:
            return 0.0
        normalised = envelope / mean_env
        return float(np.var(normalised))

    @staticmethod
    def _compute_inst_freq_stats(
        iq: np.ndarray, sample_rate: float
    ) -> tuple[float, float]:
        """Variance and kurtosis of instantaneous frequency.

        FM signals have high variance; FSK has high kurtosis (discrete jumps).
        """
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate

        # Normalise by sample rate to make threshold independent of rate
        inst_freq_norm = inst_freq / sample_rate

        variance = float(np.var(inst_freq_norm))
        # Kurtosis (excess) — high for peaked/heavy-tailed distributions
        kurt = float(sp_stats.kurtosis(inst_freq_norm, fisher=True))
        return variance, kurt

    @staticmethod
    def _count_phase_discontinuities(
        iq: np.ndarray, threshold_rad: float = 0.5
    ) -> int:
        """Count abrupt phase jumps exceeding threshold.

        PSK signals exhibit many jumps at symbol boundaries.
        """
        phase_diff = np.abs(np.diff(np.angle(iq)))
        # Handle wrapping
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        return int(np.sum(phase_diff > threshold_rad))

    @staticmethod
    def _compute_spectral_flatness(iq: np.ndarray) -> float:
        """Wiener entropy of the power spectrum.

        Near 0 for tonal/narrowband signals, near 1 for noise-like
        (wideband digital) signals.
        """
        psd = np.abs(np.fft.fft(iq)) ** 2
        psd = psd[psd > 0]  # Remove zeros
        if len(psd) == 0:
            return 0.0
        log_mean = np.mean(np.log(psd))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = np.mean(psd)
        if arithmetic_mean < 1e-20:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    @staticmethod
    def _estimate_freq_states(iq: np.ndarray, sample_rate: float) -> int:
        """Estimate the number of discrete frequency levels.

        FSK signals have 2+ distinct levels in their instantaneous
        frequency histogram.
        """
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate

        # Build histogram
        num_bins = min(100, len(inst_freq) // 10)
        if num_bins < 10:
            return 0
        hist, _ = np.histogram(inst_freq, bins=num_bins)

        # Find peaks in the histogram
        peaks, _ = find_peaks(
            hist.astype(float),
            distance=num_bins // 8,
            prominence=np.max(hist) * 0.1,
        )
        return len(peaks)
