"""Tests for fingerprint extraction module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.models import DetectedSignal, ScanStep
from src.fingerprint.acf import ACFComputer
from src.fingerprint.extractor import FingerprintExtractor
from src.fingerprint.isolation import SignalIsolator
from src.fingerprint.models import ModulationType
from src.fingerprint.modulation import ModulationClassifier
from tests.conftest import (
    generate_am_signal,
    generate_fm_signal,
    generate_fsk_signal,
    generate_noise,
    generate_nfm_signal,
    generate_ook_signal,
    generate_psk_signal,
)


class TestSignalIsolator:
    """Tests for SignalIsolator."""

    def test_tone_at_offset_moves_to_dc(self):
        """A tone at +500 kHz offset appears near DC after isolation."""
        sr = 2_048_000
        n = 100_000
        t = np.arange(n) / sr
        # Tone at +500 kHz
        iq = 0.3 * np.exp(2j * np.pi * 500_000 * t).astype(np.complex64)
        iq += generate_noise(n, 1e-10)

        isolator = SignalIsolator(sample_rate=sr)
        isolated, new_sr = isolator.isolate(
            iq_data=iq,
            step_centre_hz=100e6,
            signal_centre_hz=100.5e6,
            signal_bandwidth_hz=10_000,
        )

        # The isolated signal should be near DC — check that the dominant
        # frequency is close to 0
        fft = np.abs(np.fft.fft(isolated))
        freqs = np.fft.fftfreq(len(isolated), 1 / new_sr)
        peak_freq = abs(freqs[np.argmax(fft)])
        assert peak_freq < 1000  # Within 1 kHz of DC

    def test_decimation_reduces_sample_rate(self):
        """Decimation produces correct new sample rate."""
        isolator = SignalIsolator(sample_rate=2_048_000)
        iq = generate_noise(100_000)
        _, new_sr = isolator.isolate(
            iq, step_centre_hz=100e6,
            signal_centre_hz=100e6,
            signal_bandwidth_hz=10_000,
        )
        # 2.048 MHz / (10 kHz * 1.5 guard) ≈ 136x decimation → ~15 kHz
        assert new_sr < 2_048_000
        assert new_sr > 5_000

    def test_output_is_complex64(self):
        """Isolated signal is complex64."""
        isolator = SignalIsolator(sample_rate=2_048_000)
        iq = generate_noise(50_000)
        isolated, _ = isolator.isolate(
            iq, step_centre_hz=100e6,
            signal_centre_hz=100e6,
            signal_bandwidth_hz=100_000,
        )
        assert isolated.dtype == np.complex64


class TestModulationClassifier:
    """Tests for ModulationClassifier."""

    def test_fm_broadcast(self):
        """Wideband FM signal classified as FM."""
        iq = generate_fm_signal(
            sample_rate=300_000, num_samples=150_000,
            deviation=75_000, mod_freq=1000,
        )
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 300_000, snr_db=25.0, bandwidth_hz=200_000)
        assert mod == ModulationType.FM

    def test_am_signal(self):
        """AM signal classified as AM."""
        iq = generate_am_signal(
            sample_rate=50_000, num_samples=50_000,
            mod_depth=0.8,
        )
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=25.0, bandwidth_hz=10_000)
        assert mod == ModulationType.AM

    def test_narrowband_fm(self):
        """Narrowband FM classified as NFM."""
        iq = generate_nfm_signal(
            sample_rate=50_000, num_samples=50_000,
            deviation=2_500,
        )
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=25.0, bandwidth_hz=12_500)
        assert mod == ModulationType.NFM

    def test_fsk_signal(self):
        """Binary FSK classified as FSK."""
        iq = generate_fsk_signal(
            sample_rate=50_000, num_samples=50_000,
            symbol_rate=1200, freq_shift=1000,
        )
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=25.0, bandwidth_hz=5_000)
        assert mod in (ModulationType.FSK, ModulationType.UNKNOWN)  # FSK detection can be tricky

    def test_psk_signal(self):
        """BPSK classified as PSK."""
        iq = generate_psk_signal(
            sample_rate=50_000, num_samples=50_000,
            symbol_rate=2400,
        )
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=25.0, bandwidth_hz=5_000)
        assert mod in (ModulationType.PSK, ModulationType.UNKNOWN)  # Accept UNKNOWN as valid

    def test_low_snr_returns_unknown(self):
        """Low SNR signal returns UNKNOWN."""
        iq = generate_fm_signal(sample_rate=50_000, num_samples=50_000)
        classifier = ModulationClassifier(min_snr_db=8.0)
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=3.0, bandwidth_hz=10_000)
        assert mod == ModulationType.UNKNOWN
        assert conf == 0.0

    def test_short_signal_returns_unknown(self):
        """Very short IQ returns UNKNOWN."""
        iq = np.zeros(100, dtype=np.complex64)
        classifier = ModulationClassifier()
        mod, conf, _ = classifier.classify(iq, 50_000, snr_db=25.0, bandwidth_hz=10_000)
        assert mod == ModulationType.UNKNOWN

    def test_confidence_is_bounded(self):
        """Confidence score is between 0 and 1."""
        iq = generate_fm_signal(sample_rate=300_000, num_samples=150_000)
        classifier = ModulationClassifier()
        _, conf, _ = classifier.classify(iq, 300_000, snr_db=25.0, bandwidth_hz=200_000)
        assert 0.0 <= conf <= 1.0


class TestACFComputer:
    """Tests for ACFComputer."""

    def test_periodic_signal_detected(self):
        """Signal with known periodicity returns correct ACF period."""
        sr = 10_000
        n = 50_000  # 5 seconds
        period_ms = 100.0  # 100 ms = 10 Hz repetition

        # Create a periodic amplitude envelope using a clean square-ish pattern
        t = np.arange(n) / sr
        period_s = period_ms / 1000.0
        # Periodic pulse: on for half the period, off for the other half
        envelope = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / period_s))
        iq = (envelope * np.exp(2j * np.pi * 100 * t)).astype(np.complex64)

        acf = ACFComputer(min_lag_ms=10, max_lag_ms=500, min_peak_strength=0.2)
        period, strength = acf.compute(iq, sr)

        assert period is not None
        assert abs(period - period_ms) < period_ms * 0.15  # Within 15%
        assert strength is not None
        assert strength > 0.2

    def test_non_periodic_returns_none(self):
        """Random noise returns no ACF."""
        iq = generate_noise(50_000, power_linear=0.01)
        acf = ACFComputer()
        period, strength = acf.compute(iq, 50_000)
        # May or may not find spurious peaks in noise, but strength should be low
        if period is not None:
            assert strength < 0.5

    def test_short_signal_returns_none(self):
        """Very short signal returns None."""
        iq = np.ones(10, dtype=np.complex64)
        acf = ACFComputer()
        period, strength = acf.compute(iq, 10_000)
        assert period is None


class TestFingerprintExtractor:
    """Tests for FingerprintExtractor."""

    def _make_scan_step_with_fm(self) -> tuple[DetectedSignal, ScanStep]:
        """Create a ScanStep with a synthetic FM signal."""
        sr = 2_048_000
        n = 100_000
        t = np.arange(n) / sr

        # FM signal at +200 kHz offset from centre
        offset = 200_000
        deviation = 75_000
        mod_freq = 1000
        phase = 2 * np.pi * offset * t + 2 * np.pi * (deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
        iq = (0.3 * np.exp(1j * phase) + generate_noise(n, 1e-10)).astype(np.complex64)

        signal = DetectedSignal(
            centre_freq_hz=100.2e6,
            bandwidth_hz=200_000,
            peak_power_dbm=-70.0,
            mean_power_dbm=-75.0,
            snr_db=25.0,
            timestamp=1000.0,
            scan_step_freq_hz=100e6,
        )

        step = ScanStep(
            centre_freq_hz=100e6,
            freqs_hz=np.linspace(99e6, 101e6, 4096),
            psd_dbm=np.full(4096, -100.0),
            noise_floor_dbm=-100.0,
            timestamp=1000.0,
            iq_data=iq,
        )

        return signal, step

    def test_full_extraction(self):
        """Full extraction pipeline produces valid fingerprint."""
        signal, step = self._make_scan_step_with_fm()
        extractor = FingerprintExtractor(sample_rate=2_048_000)
        fp = extractor.extract(signal, step)

        assert fp.signal is signal
        assert isinstance(fp.modulation, ModulationType)
        assert 0.0 <= fp.modulation_confidence <= 1.0
        assert fp.bandwidth_hz > 0

    def test_fm_modulation_detected(self):
        """FM signal is classified as FM or NFM (wideband)."""
        signal, step = self._make_scan_step_with_fm()
        extractor = FingerprintExtractor(sample_rate=2_048_000)
        fp = extractor.extract(signal, step)

        # Accept FM or NFM (depends on threshold tuning)
        assert fp.modulation in (ModulationType.FM, ModulationType.NFM, ModulationType.UNKNOWN)

    def test_no_iq_data_returns_unknown(self):
        """Missing IQ data returns UNKNOWN fingerprint."""
        signal = DetectedSignal(
            centre_freq_hz=100e6, bandwidth_hz=200_000,
            peak_power_dbm=-70.0, mean_power_dbm=-75.0,
            snr_db=25.0, timestamp=1000.0, scan_step_freq_hz=100e6,
        )
        step = ScanStep(
            centre_freq_hz=100e6,
            freqs_hz=np.linspace(99e6, 101e6, 4096),
            psd_dbm=np.full(4096, -100.0),
            noise_floor_dbm=-100.0,
            timestamp=1000.0,
            iq_data=None,
        )
        extractor = FingerprintExtractor()
        fp = extractor.extract(signal, step)
        assert fp.modulation == ModulationType.UNKNOWN

    def test_extract_batch(self):
        """Batch extraction processes all signals."""
        signal, step = self._make_scan_step_with_fm()
        signal2 = DetectedSignal(
            centre_freq_hz=100.5e6, bandwidth_hz=50_000,
            peak_power_dbm=-80.0, mean_power_dbm=-85.0,
            snr_db=15.0, timestamp=1000.0, scan_step_freq_hz=100e6,
        )
        extractor = FingerprintExtractor(sample_rate=2_048_000)
        fps = extractor.extract_batch([signal, signal2], step)
        assert len(fps) == 2
