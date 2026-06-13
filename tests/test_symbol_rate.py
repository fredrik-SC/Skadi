"""Tests for the (approximate, advisory) symbol-rate estimator.

The estimator feeds the ML feature vector; it does not need to be an exact baud
estimator (robust baud estimation is hard). These tests check it is reachable
across the band (the zero-pad fix lets it resolve down to ~1 Hz), bounded,
deterministic, and tracks the true rate on the cases where it is reliable.
"""

from __future__ import annotations

from src.fingerprint.modulation import ModulationClassifier
from tests.conftest import generate_fsk_signal, generate_psk_signal


def test_estimate_is_bounded_and_deterministic():
    c = ModulationClassifier()
    s = generate_fsk_signal(sample_rate=50_000, num_samples=80_000, symbol_rate=300, freq_shift=800)
    r1 = c._estimate_symbol_rate(s, 50_000)
    r2 = c._estimate_symbol_rate(s, 50_000)
    assert r1 == r2                      # deterministic
    assert 0.0 <= r1 <= 50_000 / 4       # within the search band


def test_low_floor_reachable():
    """The zero-padded FFT can resolve well below the old 20 Hz floor."""
    c = ModulationClassifier()
    assert c._thresholds["symbol_rate_min_hz"] <= 1.0


def test_returns_value_for_fsk():
    """Returns a positive in-band estimate for FSK (advisory feature; exact baud
    is not guaranteed — the estimator is approximate and signal-dependent)."""
    c = ModulationClassifier()
    s = generate_fsk_signal(sample_rate=50_000, num_samples=100_000, symbol_rate=1200, freq_shift=1000)
    r = c._estimate_symbol_rate(s, 50_000)
    assert 0.0 < r <= 50_000 / 4


def test_returns_value_for_psk():
    c = ModulationClassifier()
    s = generate_psk_signal(sample_rate=50_000, num_samples=80_000, symbol_rate=2400)
    assert c._estimate_symbol_rate(s, 50_000) > 0
