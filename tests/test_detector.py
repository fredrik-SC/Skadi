"""Tests for signal detection module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.detector import SignalDetector
from src.detection.models import ScanStep


def _make_scan_step(
    psd_dbm: np.ndarray,
    noise_floor_dbm: float = -100.0,
    centre_freq_hz: float = 100e6,
    sample_rate: float = 2_048_000,
) -> ScanStep:
    """Helper to create a ScanStep from PSD data."""
    fft_size = len(psd_dbm)
    freq_offsets = np.fft.fftshift(
        np.fft.fftfreq(fft_size, d=1.0 / sample_rate)
    )
    freqs_hz = freq_offsets + centre_freq_hz
    return ScanStep(
        centre_freq_hz=centre_freq_hz,
        freqs_hz=freqs_hz,
        psd_dbm=psd_dbm,
        noise_floor_dbm=noise_floor_dbm,
        timestamp=0.0,
    )


class TestSignalDetector:
    """Tests for SignalDetector."""

    @pytest.fixture
    def config(self) -> dict:
        return {
            "threshold_db": 10.0,
            "min_bandwidth_hz": 500,
            "max_signals_per_step": 10,
        }

    def test_no_signals_below_threshold(self, config):
        """Flat PSD below threshold returns empty list."""
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert signals == []

    def test_single_signal_detected(self, config):
        """A single peak above threshold is detected."""
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        # Add a signal ~400 bins wide (~200 kHz at 500 Hz/bin)
        psd[1800:2200] = -80.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 1
        assert signals[0].bandwidth_hz == pytest.approx(200_000, rel=0.01)
        assert signals[0].peak_power_dbm == pytest.approx(-80.0)
        assert signals[0].snr_db == pytest.approx(20.0)

    def test_multiple_signals(self, config):
        """Two separated peaks produce two detections sorted by power."""
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        # Stronger signal
        psd[1000:1100] = -70.0
        # Weaker signal
        psd[3000:3100] = -85.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 2
        # Sorted by peak power descending
        assert signals[0].peak_power_dbm > signals[1].peak_power_dbm

    def test_narrow_signal_rejected(self, config):
        """Signal narrower than min_bandwidth_hz is filtered out."""
        config["min_bandwidth_hz"] = 5000  # 5 kHz minimum
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        # Only 2 bins wide = 1 kHz at 500 Hz/bin — should be rejected
        psd[2000:2002] = -70.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert signals == []

    def test_max_signals_limit(self, config):
        """Only top N signals are returned."""
        config["max_signals_per_step"] = 2
        config["min_bandwidth_hz"] = 0  # Accept everything
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        # Three separate signals
        psd[500:510] = -75.0
        psd[2000:2010] = -70.0
        psd[3500:3510] = -80.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 2
        assert signals[0].peak_power_dbm == pytest.approx(-70.0)
        assert signals[1].peak_power_dbm == pytest.approx(-75.0)

    def test_signal_at_start(self, config):
        """Signal at the start of the PSD array is detected."""
        config["min_bandwidth_hz"] = 0
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        psd[0:50] = -75.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 1

    def test_signal_at_end(self, config):
        """Signal at the end of the PSD array is detected."""
        config["min_bandwidth_hz"] = 0
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        psd[4046:4096] = -75.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 1

    def test_bandwidth_matches_bin_count(self, config):
        """Bandwidth equals contiguous bins × bin width."""
        config["min_bandwidth_hz"] = 0
        detector = SignalDetector(config)
        psd = np.full(4096, -100.0)
        num_bins = 100  # 100 bins × 500 Hz/bin = 50 kHz
        psd[2000:2100] = -75.0
        step = _make_scan_step(psd, noise_floor_dbm=-100.0)
        signals = detector.detect(step)
        assert len(signals) == 1
        expected_bw = num_bins * (2_048_000 / 4096)
        assert signals[0].bandwidth_hz == pytest.approx(expected_bw, rel=0.01)
