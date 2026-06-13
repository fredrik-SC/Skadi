"""Modulation type classification from IQ samples.

Deterministic, feature-led classifier that discriminates AM, FM, NFM, FSK, PSK,
OOK, and UNKNOWN from the modulation physics (no ML, per the v1.0 spec):

* envelope constancy (CV + bimodality) separates amplitude-modulated AM/OOK from
  constant-envelope FM/FSK/PSK;
* within constant-envelope, a robust discrete-frequency-state measure isolates
  FSK, an order-2 phase concentration isolates PSK, and the remainder is FM/NFM
  by bandwidth.

This replaces an earlier tree that gated all digital modes behind a spectral
flatness test — tonal narrowband digimodes have flatness ~0.01 and never entered,
so FSK/PSK were systematically missed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import medfilt

from src.fingerprint.models import ModulationFeatures, ModulationType

logger = logging.getLogger(__name__)

# Default classification thresholds (overridable via fingerprint.modulation.*).
_DEFAULTS = {
    # Envelope-family gate
    "env_cv_threshold": 0.25,          # above -> amplitude-modulated (AM/OOK)
    "ook_bimodality": 0.80,            # above -> OOK (two-level) vs AM
    # FSK discrete-frequency-state detection
    "freq_state_fit": 0.85,            # R^2 of a 2-level model of inst-freq
    "freq_state_flat": 0.50,           # fraction of plateau (held) samples
    "freq_state_min_share": 0.12,      # min share of samples in each state
    "freq_state_gap_frac": 0.30,       # state gap as fraction of inst-freq spread
    "medfilt_window": 11,              # odd; smooths inter-symbol transitions
    # PSK phase concentration (order-2 high AND order-1 low => two opposite levels)
    "psk_concentration": 0.60,
    "psk_order1_max": 0.50,
    # FM / NFM
    "wideband_fm_min_bw_hz": 50_000,
    "fm_inst_freq_var_min": 5e-4,      # minimum inst-freq variance for FM/NFM
    # Phase-jump feature (advisory; not a gate)
    "phase_jump_thresh_rad": 1.0,
    # Symbol-rate search band (floor 1 Hz so slow modes — RTTY 45, FT8 6.25,
    # WSPR 1.46 baud — are reachable, not just fast ones)
    "symbol_rate_min_hz": 1.0,
}


class ModulationClassifier:
    """Deterministic, feature-led modulation classifier.

    Args:
        min_snr_db: Minimum SNR to attempt classification (else UNKNOWN).
        config: Optional dict of threshold overrides (keys match _DEFAULTS).
    """

    def __init__(
        self,
        min_snr_db: float = 8.0,
        config: dict[str, Any] | None = None,
        model: "Any | None" = None,
    ) -> None:
        self._min_snr_db = min_snr_db
        self._thresholds = dict(_DEFAULTS)
        if config:
            self._thresholds.update(config)
        # Optional trained model (MLModulationModel). When present it makes the
        # decision; otherwise the deterministic _decide() is used (offline fallback).
        self._model = model

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
            bandwidth_hz: Estimated occupied bandwidth in Hz.

        Returns:
            Tuple of (modulation_type, confidence, features).
        """
        if len(iq_data) < 256 or snr_db < self._min_snr_db:
            return ModulationType.UNKNOWN, 0.0, ModulationFeatures(0, 0, 0, 0, 0, 0)

        features = self._compute_features(iq_data, sample_rate)
        if self._model is not None:
            mod_type, confidence = self._model.predict(features, bandwidth_hz)
        else:
            mod_type, confidence = self._decide(features, bandwidth_hz)

        logger.debug(
            "Modulation: %s (conf=%.2f) — env_cv=%.3f bimod=%.2f states=%d "
            "fsep=%.2f conc=%.2f jr=%.3f baud=%.0f",
            mod_type.value, confidence,
            features.envelope_cv, features.envelope_bimodality,
            features.num_freq_states_robust, features.freq_state_separation,
            features.phase_level_concentration, features.phase_jump_rate,
            features.symbol_rate_hz,
        )
        return mod_type, confidence, features

    def _compute_features(
        self, iq_data: np.ndarray, sample_rate: float
    ) -> ModulationFeatures:
        """Compute the full feature vector (legacy + structural features)."""
        env_cv, env_bimod = self._envelope_stats(iq_data)
        if_var, if_kurt = self._compute_inst_freq_stats(iq_data, sample_rate)
        phase_disc = self._count_phase_discontinuities(iq_data)
        spectral_flat = self._compute_spectral_flatness(iq_data)
        legacy_states = self._estimate_freq_states(iq_data, sample_rate)

        num_states, fsep, center = self._freq_state_analysis(iq_data, sample_rate)
        jump_rate, order1, order2 = self._phase_structure(iq_data)
        symbol_rate = self._estimate_symbol_rate(iq_data, sample_rate)

        return ModulationFeatures(
            envelope_variance=float(env_cv ** 2),  # legacy approx (kept for compat)
            inst_freq_variance=if_var,
            inst_freq_kurtosis=if_kurt,
            phase_discontinuities=phase_disc,
            spectral_flatness=spectral_flat,
            num_freq_states=legacy_states,
            envelope_cv=env_cv,
            envelope_bimodality=env_bimod,
            num_freq_states_robust=num_states,
            freq_state_separation=fsep,
            inst_freq_center_norm=center,
            phase_jump_rate=jump_rate,
            phase_level_concentration=order2,
            phase_single_concentration=order1,
            symbol_rate_hz=symbol_rate,
        )

    def _decide(
        self, f: ModulationFeatures, bandwidth_hz: float
    ) -> tuple[ModulationType, float]:
        """Feature-led decision keyed on modulation physics.

        Digital structure (FSK frequency states, PSK phase levels) is checked
        first and is envelope-independent, so real-world amplitude variation
        (fading, AGC, BPSK transition dips) does not steal a digital signal into
        the AM branch.
        """
        th = self._thresholds

        # FSK: sustained discrete frequency states.
        if f.num_freq_states_robust >= 2:
            return ModulationType.FSK, _clip(0.5 + 0.4 * min(f.freq_state_separation, 1.0))

        # PSK: two opposite phase levels (order-2 high, order-1 low). The order-1
        # guard excludes real-valued AM, which also has high order-2.
        if (f.phase_level_concentration > th["psk_concentration"]
                and f.phase_single_concentration < th["psk_order1_max"]):
            return ModulationType.PSK, _clip(0.4 + 0.5 * f.phase_level_concentration)

        # Wideband FM: a wide signal with strong frequency modulation is FM even
        # if isolation filtering introduced envelope ripple (FM-to-AM). Checked
        # before the amplitude branch so that ripple is not mistaken for AM/OOK.
        if (bandwidth_hz >= th["wideband_fm_min_bw_hz"]
                and f.inst_freq_variance > th["fm_inst_freq_var_min"]):
            return ModulationType.FM, _clip(0.5 + 5.0 * f.inst_freq_variance)

        # Amplitude-modulated family: a varying envelope.
        if f.envelope_cv > th["env_cv_threshold"]:
            if f.envelope_bimodality > th["ook_bimodality"]:
                return ModulationType.OOK, _clip(0.5 + (f.envelope_bimodality - th["ook_bimodality"]))
            return ModulationType.AM, _clip(0.5 + min(f.envelope_cv, 0.5))

        # FM / NFM: continuous frequency modulation.
        if f.inst_freq_variance > th["fm_inst_freq_var_min"]:
            if bandwidth_hz >= th["wideband_fm_min_bw_hz"]:
                return ModulationType.FM, _clip(0.5 + 5.0 * f.inst_freq_variance)
            return ModulationType.NFM, _clip(0.4 + 5.0 * f.inst_freq_variance)

        return ModulationType.UNKNOWN, 0.0

    # --- structural features (Phase D) -------------------------------------

    @staticmethod
    def _envelope_stats(iq: np.ndarray) -> tuple[float, float]:
        """Coefficient of variation and Sarle bimodality of the envelope.

        High CV => amplitude-modulated. High bimodality (two-level) => OOK.
        """
        env = np.abs(iq).astype(np.float64)
        mean_env = float(np.mean(env))
        if mean_env < 1e-12:
            return 0.0, 0.0
        cv = float(np.std(env) / mean_env)

        n = len(env)
        # A near-constant envelope is neither AM nor OOK; skip the (numerically
        # unstable) moment calculation.
        if n < 4 or cv < 1e-4:
            return cv, 0.0
        g = float(sp_stats.skew(env))
        k = float(sp_stats.kurtosis(env, fisher=True))  # excess kurtosis
        correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        denom = k + correction
        bimod = (g ** 2 + 1.0) / denom if denom > 1e-9 else 0.0
        return cv, float(max(0.0, min(bimod, 1.0)))

    def _freq_state_analysis(
        self, iq: np.ndarray, sample_rate: float
    ) -> tuple[int, float, float]:
        """Detect sustained discrete frequency states (FSK).

        Median-filters the instantaneous frequency to remove inter-symbol
        transition spikes, then tests whether it is well-described by two
        sustained levels (plateaus). Returns (num_states, fit, center_norm).
        """
        th = self._thresholds
        win = int(th["medfilt_window"])
        if win % 2 == 0:
            win += 1
        phase = np.unwrap(np.angle(iq))
        inst = np.diff(phase) / (2 * np.pi) * sample_rate
        if len(inst) < 2 * win:
            return 1, 0.0, 0.0

        f = medfilt(inst, win)
        center = float(abs(np.median(inst)) / sample_rate) if sample_rate else 0.0
        spread = float(np.percentile(f, 95) - np.percentile(f, 5)) + 1e-9

        # Plateau fraction: held (nearly-flat) samples between transitions.
        flat = float(np.mean(np.abs(np.diff(f)) < 0.02 * spread))

        thr = self._otsu(f)
        lo, hi = f[f <= thr], f[f > thr]
        if len(lo) < 2 or len(hi) < 2:
            return 1, 0.0, center
        level_lo, level_hi = float(np.median(lo)), float(np.median(hi))
        assigned = np.where(np.abs(f - level_lo) < np.abs(f - level_hi), level_lo, level_hi)
        fit = 1.0 - float(np.var(f - assigned)) / (float(np.var(f)) + 1e-12)
        share = min(len(lo), len(hi)) / len(f)
        gap = abs(level_hi - level_lo) / spread

        is_fsk = (
            fit > th["freq_state_fit"]
            and flat > th["freq_state_flat"]
            and share >= th["freq_state_min_share"]
            and gap > th["freq_state_gap_frac"]
        )
        return (2 if is_fsk else 1), float(max(0.0, fit)), center

    def _phase_structure(self, iq: np.ndarray) -> tuple[float, float, float]:
        """Phase-jump rate and order-1 / order-2 phase concentrations.

        Returns (jump_rate, order1, order2). The doubled-carrier line of iq**2 is
        removed so the moments are carrier-independent. order2 ~1 for both BPSK
        (parked at {0, pi}) and real-valued AM; order1 ~1 only for a single phase
        level (AM/carrier), ~0 for BPSK's two opposite levels — so order2 high
        with order1 low is the BPSK signature.
        """
        ang = np.angle(iq)
        dphi = np.abs(np.diff(ang))
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        jump_rate = float(np.mean(dphi > self._thresholds["phase_jump_thresh_rad"]))

        sq = iq.astype(np.complex128) ** 2
        n = len(sq)
        fc2 = float(np.fft.fftfreq(n)[int(np.argmax(np.abs(np.fft.fft(sq))))])
        derot_ang = ang - np.pi * fc2 * np.arange(n)  # half the order-2 carrier
        order1 = float(abs(np.mean(np.exp(1j * derot_ang))))
        order2 = float(abs(np.mean(np.exp(2j * derot_ang))))
        return jump_rate, min(order1, 1.0), min(order2, 1.0)

    def _estimate_symbol_rate(self, iq: np.ndarray, sample_rate: float) -> float:
        """Estimate symbol (baud) rate from a transition train (advisory).

        Returns the dominant spectral line of the transition train, or 0.0.
        """
        if len(iq) < 64:
            return 0.0
        phase = np.unwrap(np.angle(iq))
        win = int(self._thresholds["medfilt_window"])
        if win % 2 == 0:
            win += 1
        inst = medfilt(np.diff(phase) / (2 * np.pi) * sample_rate, win)
        ap = np.abs(np.diff(np.angle(iq)))
        tr_psk = np.minimum(ap, 2 * np.pi - ap)
        tr_fsk = np.abs(np.diff(inst))

        fmin = self._thresholds["symbol_rate_min_hz"]
        best_f, best_prom = 0.0, 0.0
        for tr in (tr_psk, tr_fsk):
            tr = tr - np.mean(tr)
            if len(tr) < 32:
                continue
            # Zero-pad to ~0.5 Hz bin spacing so slow bauds (down to ~1 Hz) are
            # resolvable — a few-thousand-sample window at tens of kHz otherwise
            # has bins far wider than the target rate.
            nfft = min(1 << 16, max(len(tr), int(sample_rate / 0.5)))
            spec = np.abs(np.fft.rfft(tr, n=nfft))
            freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
            band = (freqs >= fmin) & (freqs <= sample_rate / 4)
            if not band.any():
                continue
            i = int(np.argmax(spec[band]))
            prom = float(spec[band][i] / (np.median(spec[band]) + 1e-12))
            if prom > best_prom:
                best_prom, best_f = prom, float(freqs[band][i])
        return best_f

    @staticmethod
    def _otsu(x: np.ndarray, nbins: int = 64) -> float:
        """Otsu threshold that best splits x into two classes."""
        hist, edges = np.histogram(x, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2
        total = hist.sum()
        if total == 0:
            return float(np.median(x))
        w = np.cumsum(hist)
        mu = np.cumsum(hist * centers)
        mu_t = mu[-1]
        w_f = total - w
        with np.errstate(invalid="ignore", divide="ignore"):
            m_b = mu / np.where(w == 0, 1, w)
            m_f = (mu_t - mu) / np.where(w_f == 0, 1, w_f)
            between = w * w_f * (m_b - m_f) ** 2
        return float(centers[int(np.nanargmax(between))])

    # --- legacy features (kept for the feature vector / debugging) ----------

    @staticmethod
    def _compute_inst_freq_stats(
        iq: np.ndarray, sample_rate: float
    ) -> tuple[float, float]:
        """Variance and excess kurtosis of normalised instantaneous frequency."""
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate
        inst_freq_norm = inst_freq / sample_rate
        return float(np.var(inst_freq_norm)), float(
            sp_stats.kurtosis(inst_freq_norm, fisher=True)
        )

    @staticmethod
    def _count_phase_discontinuities(
        iq: np.ndarray, threshold_rad: float = 0.5
    ) -> int:
        """Count abrupt phase jumps exceeding a threshold (legacy feature)."""
        phase_diff = np.abs(np.diff(np.angle(iq)))
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        return int(np.sum(phase_diff > threshold_rad))

    @staticmethod
    def _compute_spectral_flatness(iq: np.ndarray) -> float:
        """Wiener entropy of the power spectrum (legacy feature; no longer a gate)."""
        psd = np.abs(np.fft.fft(iq)) ** 2
        psd = psd[psd > 0]
        if len(psd) == 0:
            return 0.0
        geometric_mean = np.exp(np.mean(np.log(psd)))
        arithmetic_mean = np.mean(psd)
        if arithmetic_mean < 1e-20:
            return 0.0
        return float(geometric_mean / arithmetic_mean)

    @staticmethod
    def _estimate_freq_states(iq: np.ndarray, sample_rate: float) -> int:
        """Legacy histogram-peak frequency-state estimate (kept for the vector)."""
        from scipy.signal import find_peaks
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate
        num_bins = min(100, len(inst_freq) // 10)
        if num_bins < 10:
            return 0
        hist, _ = np.histogram(inst_freq, bins=num_bins)
        peaks, _ = find_peaks(
            hist.astype(float), distance=num_bins // 8, prominence=np.max(hist) * 0.1
        )
        return len(peaks)


def _clip(x: float) -> float:
    """Clip a confidence to [0, 1]."""
    return float(max(0.0, min(1.0, x)))
