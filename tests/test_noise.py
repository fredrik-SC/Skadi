"""Tests for noise floor estimation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection.noise import NoiseEstimator


class TestNoiseEstimator:
    """Tests for NoiseEstimator."""

    def test_flat_spectrum(self):
        """Flat PSD returns the flat value as noise floor."""
        estimator = NoiseEstimator(percentile=50.0)
        psd = np.full(4096, -100.0)
        assert estimator.estimate(psd) == pytest.approx(-100.0)

    def test_spectrum_with_signal(self):
        """Median ignores a narrow signal spike in the PSD."""
        estimator = NoiseEstimator(percentile=50.0)
        psd = np.full(4096, -100.0)
        # Add a narrow signal (5% of bins)
        psd[2000:2200] = -60.0
        noise = estimator.estimate(psd)
        assert noise == pytest.approx(-100.0)

    def test_custom_percentile(self):
        """Lower percentile returns a lower value on non-uniform data."""
        psd = np.linspace(-120.0, -80.0, 1000)
        estimator_25 = NoiseEstimator(percentile=25.0)
        estimator_75 = NoiseEstimator(percentile=75.0)
        assert estimator_25.estimate(psd) < estimator_75.estimate(psd)

    def test_empty_array_raises(self):
        """Empty PSD array raises ValueError."""
        estimator = NoiseEstimator()
        with pytest.raises(ValueError, match="empty"):
            estimator.estimate(np.array([]))

    def test_invalid_percentile_raises(self):
        """Percentile outside 0-100 raises ValueError."""
        with pytest.raises(ValueError, match="Percentile"):
            NoiseEstimator(percentile=101.0)
        with pytest.raises(ValueError, match="Percentile"):
            NoiseEstimator(percentile=-1.0)

    def test_single_bin(self):
        """Single-bin PSD returns that value."""
        estimator = NoiseEstimator()
        assert estimator.estimate(np.array([-95.0])) == pytest.approx(-95.0)


class TestAdaptiveNoiseEstimator:
    """Tests for adaptive rolling window noise estimation."""

    def test_first_estimate_is_unsmoothed(self):
        """First call returns raw percentile (no history yet)."""
        estimator = NoiseEstimator(alpha=0.3, window_size=10)
        psd = np.full(4096, -100.0)
        result = estimator.estimate(psd)
        assert result == pytest.approx(-100.0)
        assert estimator.history_depth == 1

    def test_rolling_window_smooths_estimate(self):
        """Smoothing dampens the effect of a single outlier step."""
        estimator = NoiseEstimator(alpha=0.3, window_size=10)
        normal_psd = np.full(4096, -100.0)

        # Build up stable history
        for _ in range(5):
            estimator.estimate(normal_psd)

        # Now inject a noisy step
        noisy_psd = np.full(4096, -80.0)  # 20 dB higher
        smoothed = estimator.estimate(noisy_psd)

        # Smoothed should be between -80 and -100 (dampened)
        assert -100.0 < smoothed < -80.0
        # And much closer to -100 than -80 (history dominates)
        assert smoothed < -90.0

    def test_alpha_one_disables_smoothing(self):
        """With alpha=1.0, returns raw per-step percentile."""
        estimator = NoiseEstimator(alpha=1.0, window_size=10)
        psd1 = np.full(4096, -100.0)
        psd2 = np.full(4096, -80.0)

        estimator.estimate(psd1)
        result = estimator.estimate(psd2)
        assert result == pytest.approx(-80.0)

    def test_reset_clears_history(self):
        """After reset, next estimate is unsmoothed."""
        estimator = NoiseEstimator(alpha=0.3, window_size=10)
        psd = np.full(4096, -100.0)

        for _ in range(5):
            estimator.estimate(psd)
        assert estimator.history_depth == 5

        estimator.reset()
        assert estimator.history_depth == 0

        # Next estimate should be unsmoothed (first call)
        result = estimator.estimate(np.full(4096, -80.0))
        assert result == pytest.approx(-80.0)

    def test_history_depth_tracks_correctly(self):
        """history_depth increments with each estimate call."""
        estimator = NoiseEstimator(window_size=5)
        psd = np.full(100, -95.0)

        for i in range(7):
            estimator.estimate(psd)

        # Window size is 5, so depth is capped at 5
        assert estimator.history_depth == 5

    def test_backward_compatible_defaults(self):
        """Constructing with no args works and produces valid estimates."""
        estimator = NoiseEstimator()
        psd = np.full(4096, -100.0)
        result = estimator.estimate(psd)
        assert result == pytest.approx(-100.0)

    def test_invalid_window_size_raises(self):
        """Window size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Window size"):
            NoiseEstimator(window_size=0)

    def test_invalid_alpha_raises(self):
        """Alpha outside 0-1 raises ValueError."""
        with pytest.raises(ValueError, match="Alpha"):
            NoiseEstimator(alpha=1.5)
