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
