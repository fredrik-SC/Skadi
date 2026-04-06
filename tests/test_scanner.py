"""Tests for spectrum scanner module."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from tests.conftest import generate_test_iq, make_mock_sdr
from src.detection.models import ScanStep
from src.detection.noise import NoiseEstimator
from src.detection.detector import SignalDetector
from src.sdr.scanner import SpectrumScanner


@pytest.fixture
def scan_config() -> dict:
    """FM band scan config for testing."""
    return {
        "freq_start": 88e6,
        "freq_stop": 108e6,
        "step_size": 2e6,
        "dwell_time": 0.5,
        "fft_size": 4096,
        "fft_averages": 10,
    }


@pytest.fixture
def sdr_config() -> dict:
    return {"sample_rate": 2_048_000}


@pytest.fixture
def detection_config() -> dict:
    return {
        "threshold_db": 10.0,
        "min_bandwidth_hz": 500,
        "max_signals_per_step": 10,
    }


class TestSpectrumScanner:
    """Tests for SpectrumScanner."""

    def test_num_steps_fm_band(self, scan_config, sdr_config, detection_config):
        """FM band (88-108 MHz) at 2 MHz steps = 10 steps."""
        mock_sdr = MagicMock()
        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        assert scanner.num_steps == 10

    def test_step_frequencies(self, scan_config, sdr_config, detection_config):
        """Step frequencies are correctly spaced across the range."""
        mock_sdr = MagicMock()
        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        freqs = scanner.step_frequencies
        assert len(freqs) == 10
        # First step centred at 89 MHz (88 + 1 MHz offset)
        assert freqs[0] == pytest.approx(89e6)
        # Last step centred at 107 MHz
        assert freqs[-1] == pytest.approx(107e6)
        # Steps are evenly spaced
        diffs = np.diff(freqs)
        assert all(d == pytest.approx(2e6) for d in diffs)

    def test_compute_psd_shape(self, scan_config, sdr_config, detection_config):
        """PSD output has shape (fft_size,)."""
        mock_sdr = MagicMock()
        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        iq = generate_test_iq(num_samples=40960)
        freq_offsets, psd_dbm = scanner.compute_psd(iq)
        assert freq_offsets.shape == (4096,)
        assert psd_dbm.shape == (4096,)

    def test_compute_psd_tone_peak(self, scan_config, sdr_config, detection_config):
        """A pure tone produces a peak at the correct frequency offset."""
        mock_sdr = MagicMock()
        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)

        # Generate a tone at +500 kHz offset
        tone_offset = 500_000
        iq = generate_test_iq(
            tones=[(tone_offset, 0.5)],
            num_samples=40960,
            noise_power=1e-12,
        )
        freq_offsets, psd_dbm = scanner.compute_psd(iq)

        # Find peak bin
        peak_idx = np.argmax(psd_dbm)
        peak_freq_offset = freq_offsets[peak_idx]

        # Peak should be near +500 kHz (within one bin width = 500 Hz)
        bin_width = 2_048_000 / 4096
        assert abs(peak_freq_offset - tone_offset) < bin_width

    def test_scan_step_calls_sdr(self, scan_config, sdr_config, detection_config):
        """scan_step calls tune then capture on the SDR."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)

        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        step = scanner.scan_step(100e6)

        mock_sdr.tune.assert_called_once_with(100e6)
        mock_sdr.capture.assert_called_once()
        assert isinstance(step, ScanStep)
        assert step.centre_freq_hz == 100e6

    def test_sweep_iterates_all_steps(self, scan_config, sdr_config, detection_config):
        """sweep calls scan_step for each frequency in the range."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)

        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        result = scanner.sweep()

        assert mock_sdr.tune.call_count == 10
        assert result.num_steps == 10
        assert result.duration_seconds > 0

    def test_sweep_callback(self, scan_config, sdr_config, detection_config):
        """Callback is invoked with correct step index and total."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)

        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        callback = MagicMock()
        scanner.sweep(callback=callback)

        assert callback.call_count == 10
        # Check first and last call args
        first_call = callback.call_args_list[0]
        assert first_call[0][1] == 0   # step_index
        assert first_call[0][2] == 10  # total_steps
        last_call = callback.call_args_list[-1]
        assert last_call[0][1] == 9

    def test_sweep_keep_psd_false(self, scan_config, sdr_config, detection_config):
        """With keep_psd=False, scan_steps list is empty."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)

        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        result = scanner.sweep(keep_psd=False)

        assert result.scan_steps == []
        assert result.num_steps == 10

    def test_psd_frequency_axis_centred(self, scan_config, sdr_config, detection_config):
        """Frequency offsets are centred around DC."""
        mock_sdr = MagicMock()
        scanner = SpectrumScanner(mock_sdr, scan_config, sdr_config, detection_config)
        iq = generate_test_iq(num_samples=40960)
        freq_offsets, _ = scanner.compute_psd(iq)

        # Should span from -sample_rate/2 to +sample_rate/2
        assert freq_offsets[0] < 0
        assert freq_offsets[-1] > 0
        assert freq_offsets[0] == pytest.approx(-2_048_000 / 2, abs=500)

    def test_sweep_resets_noise_estimator(self, scan_config, sdr_config, detection_config):
        """Sweep resets the noise estimator at the start."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)
        noise_est = MagicMock()
        noise_est.estimate.return_value = -100.0

        scanner = SpectrumScanner(
            mock_sdr, scan_config, sdr_config, detection_config,
            noise_estimator=noise_est,
        )
        scanner.sweep()
        noise_est.reset.assert_called_once()

    def test_sweep_with_exclusion_filter(self, scan_config, sdr_config, detection_config):
        """Exclusion filter is applied and removes matching signals."""
        iq = generate_test_iq(num_samples=1_024_000)
        mock_sdr = make_mock_sdr(iq)
        mock_filter = MagicMock()
        # Filter removes all signals
        mock_filter.filter.return_value = []

        scanner = SpectrumScanner(
            mock_sdr, scan_config, sdr_config, detection_config,
            exclusion_filter=mock_filter,
        )
        result = scanner.sweep()

        # Filter should have been called for each step
        assert mock_filter.filter.call_count == 10
        assert result.signals == []

    def test_sweep_with_detection_log(self, scan_config, sdr_config, detection_config):
        """Detection log receives signals after the sweep."""
        iq = generate_test_iq(
            tones=[(500_000, 0.5)],  # Add a tone so signals are detected
            num_samples=1_024_000,
            noise_power=1e-12,
        )
        mock_sdr = make_mock_sdr(iq)
        mock_log = MagicMock()
        mock_log.log_signal.return_value = 1

        scanner = SpectrumScanner(
            mock_sdr, scan_config, sdr_config, detection_config,
            detection_log=mock_log,
        )
        result = scanner.sweep()

        if result.signals:
            # log_signal called once per signal
            assert mock_log.log_signal.call_count == len(result.signals)
