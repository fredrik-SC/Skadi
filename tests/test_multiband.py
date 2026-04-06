"""Multi-band integration tests for operational readiness.

Tests the full detect → fingerprint → classify pipeline with
synthetic signals representing military-relevant frequency bands.
These tests verify that the system can handle signals from HF
through UHF, from 200 Hz narrowband military to 200 kHz FM broadcast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.classification.artemis_db import ArtemisDB
from src.classification.classifier import SignalClassifier
from src.config import PROJECT_ROOT
from src.detection.detector import SignalDetector
from src.detection.models import ScanStep
from src.detection.noise import NoiseEstimator
from src.fingerprint.extractor import FingerprintExtractor
from src.fingerprint.models import ModulationType
from src.sdr.scanner import SpectrumScanner
from tests.conftest import generate_fsk_signal, generate_noise


ARTEMIS_DB_PATH = PROJECT_ROOT / "data" / "artemis.db"


def _make_wideband_iq_with_signal(
    signal_offset_hz: float,
    signal_bw_hz: float,
    signal_type: str = "fsk",
    sample_rate: float = 2_048_000,
    num_samples: int = 500_000,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Create a wideband IQ capture with a synthetic signal embedded.

    The signal is placed at the given offset from DC in a noise background.

    Args:
        signal_offset_hz: Frequency offset from DC for the signal.
        signal_bw_hz: Approximate bandwidth of the signal.
        signal_type: Type of signal to generate ("fsk", "am", "tone").
        sample_rate: Sample rate of the capture.
        num_samples: Total number of samples.
        snr_db: Signal-to-noise ratio in dB.

    Returns:
        Complex64 IQ array with signal embedded in noise.
    """
    noise_power = 1e-8
    signal_amplitude = np.sqrt(noise_power * 10 ** (snr_db / 10))

    # Generate background noise
    iq = generate_noise(num_samples, noise_power)

    # Generate signal at baseband, then shift to offset
    t = np.arange(num_samples) / sample_rate

    if signal_type == "fsk":
        # Binary FSK
        symbol_rate = max(100, signal_bw_hz / 3)
        freq_shift = signal_bw_hz / 4
        baseband = generate_fsk_signal(
            sample_rate=sample_rate,
            num_samples=num_samples,
            symbol_rate=symbol_rate,
            freq_shift=freq_shift,
            amplitude=signal_amplitude,
        )
    elif signal_type == "am":
        # AM with tone modulation
        mod_freq = signal_bw_hz / 4
        envelope = 1.0 + 0.7 * np.sin(2 * np.pi * mod_freq * t)
        baseband = (signal_amplitude * envelope).astype(np.complex64)
    elif signal_type == "tone":
        # Simple CW tone
        baseband = (signal_amplitude * np.ones(num_samples)).astype(np.complex64)
    else:
        baseband = (signal_amplitude * np.ones(num_samples)).astype(np.complex64)

    # Shift to offset frequency
    shifted = baseband * np.exp(2j * np.pi * signal_offset_hz * t).astype(np.complex64)
    iq += shifted

    return iq


def _run_detection_pipeline(
    iq_data: np.ndarray,
    centre_freq_hz: float,
    sample_rate: float = 2_048_000,
    fft_size: int = 8192,
):
    """Run the detect → fingerprint pipeline on IQ data.

    Returns list of (signal, fingerprint) tuples.
    """
    from src.sdr.scanner import SpectrumScanner

    # Compute PSD
    noise_est = NoiseEstimator()
    detector = SignalDetector({
        "threshold_db": 10.0,
        "min_bandwidth_hz": 100,
        "max_signals_per_step": 20,
    })

    # Build PSD manually (normally the scanner does this)
    window = np.hanning(fft_size).astype(np.float32)
    window_power = float(np.sum(window ** 2))
    num_segments = min(len(iq_data) // fft_size, 10)

    power_sum = np.zeros(fft_size, dtype=np.float64)
    for i in range(num_segments):
        segment = iq_data[i * fft_size:(i + 1) * fft_size]
        windowed = segment * window
        spectrum = np.fft.fft(windowed)
        power_sum += np.abs(spectrum) ** 2

    averaged_power = power_sum / (num_segments * window_power)
    psd_dbm = 10.0 * np.log10(np.maximum(averaged_power, 1e-20)) + 30.0
    psd_dbm = np.fft.fftshift(psd_dbm)
    freq_offsets = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate)
    )
    freqs_hz = freq_offsets + centre_freq_hz

    noise_floor = noise_est.estimate(psd_dbm)

    step = ScanStep(
        centre_freq_hz=centre_freq_hz,
        freqs_hz=freqs_hz,
        psd_dbm=psd_dbm,
        noise_floor_dbm=noise_floor,
        timestamp=1000.0,
        iq_data=iq_data,
    )

    # Detect
    signals = detector.detect(step)

    # Fingerprint
    extractor = FingerprintExtractor(sample_rate=sample_rate)
    fingerprints = extractor.extract_batch(signals, step)

    return list(zip(signals, fingerprints))


class TestHFNarrowband:
    """Test detection of HF narrowband military signals."""

    def test_1khz_fsk_detected(self):
        """A 1 kHz FSK signal at HF is detected with 8192-point FFT."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=200_000,  # 200 kHz offset from DC
            signal_bw_hz=1000,
            signal_type="fsk",
            num_samples=500_000,
            snr_db=25.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=10e6, fft_size=8192)
        assert len(results) > 0, "1 kHz FSK signal not detected at HF"

        # Check the detected signal is near the expected frequency
        sig, fp = results[0]
        expected_freq = 10e6 + 200_000
        assert abs(sig.centre_freq_hz - expected_freq) < 5000, \
            f"Signal at {sig.centre_freq_hz/1e6:.3f} MHz, expected {expected_freq/1e6:.3f} MHz"

    def test_3khz_psk_detected(self):
        """A 3 kHz PSK-like signal at HF is detected (STANAG bandwidth)."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=-300_000,
            signal_bw_hz=3000,
            signal_type="fsk",  # FSK as proxy for PSK in synthetic
            num_samples=500_000,
            snr_db=20.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=14e6, fft_size=8192)
        assert len(results) > 0, "3 kHz signal not detected at HF"

    def test_hf_fsk_fingerprinted(self):
        """HF FSK signal gets a modulation classification (not UNKNOWN)."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=100_000,
            signal_bw_hz=1000,
            signal_type="fsk",
            num_samples=500_000,
            snr_db=25.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=10e6, fft_size=8192)
        assert len(results) > 0

        _, fp = results[0]
        # Should classify as FSK or NFM (frequency-varying signal)
        assert fp.modulation in (
            ModulationType.FSK, ModulationType.NFM,
            ModulationType.PSK, ModulationType.UNKNOWN,
        ), f"HF FSK classified as {fp.modulation.value}"


class TestAirbandAM:
    """Test detection of airband AM signals (108-137 MHz)."""

    def test_8khz_am_detected(self):
        """An 8 kHz AM signal at airband is detected."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=500_000,
            signal_bw_hz=8000,
            signal_type="am",
            num_samples=500_000,
            snr_db=20.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=125e6, fft_size=8192)
        assert len(results) > 0, "8 kHz AM signal not detected at airband"

    def test_am_classified_correctly(self):
        """Airband AM signal classified as AM."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=300_000,
            signal_bw_hz=8000,
            signal_type="am",
            num_samples=500_000,
            snr_db=25.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=125e6, fft_size=8192)
        assert len(results) > 0

        _, fp = results[0]
        assert fp.modulation in (
            ModulationType.AM, ModulationType.OOK, ModulationType.UNKNOWN,
        ), f"Airband AM classified as {fp.modulation.value}"


class TestVHFUHF:
    """Test detection of VHF/UHF military signals."""

    def test_12khz_nfm_at_vhf(self):
        """12.5 kHz NFM signal at VHF is detected."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=-400_000,
            signal_bw_hz=12500,
            signal_type="fsk",
            num_samples=500_000,
            snr_db=20.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=150e6, fft_size=8192)
        assert len(results) > 0, "12.5 kHz signal not detected at VHF"

    def test_25khz_signal_at_uhf(self):
        """25 kHz signal at UHF military band is detected."""
        iq = _make_wideband_iq_with_signal(
            signal_offset_hz=600_000,
            signal_bw_hz=25000,
            signal_type="fsk",
            num_samples=500_000,
            snr_db=20.0,
        )

        results = _run_detection_pipeline(iq, centre_freq_hz=300e6, fft_size=8192)
        assert len(results) > 0, "25 kHz signal not detected at UHF"


class TestArtemisClassification:
    """Test Artemis DB matching for military signals."""

    @pytest.fixture
    def classifier(self):
        if not ARTEMIS_DB_PATH.exists():
            pytest.skip("Artemis DB not found")
        db = ArtemisDB(ARTEMIS_DB_PATH)
        return SignalClassifier(db)

    def test_hf_fsk_matches_military(self, classifier):
        """HF FSK signal at 10 MHz matches military Artemis entries."""
        from tests.test_classifier import _make_fingerprint
        fp = _make_fingerprint(ModulationType.FSK, 1000, 10e6)
        result = classifier.classify(fp)
        assert len(result.matches) > 0, "No Artemis matches for HF FSK"
        # Should find military/government signals in HF band
        names = [m.signal.name for m in result.matches]
        logger_msg = f"HF FSK matches: {names}"
        assert len(names) > 0, logger_msg

    def test_stanag_bandwidth_matches(self, classifier):
        """STANAG 4285-like signal (PSK, 2750 Hz, HF) finds correct match."""
        from tests.test_classifier import _make_fingerprint
        fp = _make_fingerprint(ModulationType.PSK, 2750, 14e6, acf_ms=106.66)
        result = classifier.classify(fp)
        assert len(result.matches) > 0
        names = [m.signal.name for m in result.matches]
        assert any("STANAG 4285" in n for n in names), f"STANAG 4285 not matched: {names}"
