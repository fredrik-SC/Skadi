"""Tests for the ITU-R SM.443 occupied-bandwidth estimator."""

from __future__ import annotations

import numpy as np

from src.dsp.bandwidth import occupied_bandwidth

SR = 2_048_000
N = 8192
FREQS = np.fft.fftshift(np.fft.fftfreq(N, 1.0 / SR))


def _spectrum(tones, floor_lin=1e-9):
    """Build a linear PSD with delta tones (freq_hz, amplitude) on a noise floor."""
    lin = np.full(N, floor_lin)
    for f, amp in tones:
        lin[int(np.argmin(np.abs(FREQS - f)))] += amp
    return lin


def test_captures_am_sidebands():
    """AM carrier + sidebands at +/-4 kHz -> ~8 kHz, not just the carrier."""
    lin = _spectrum([(0, 1.0), (4000, 0.3), (-4000, 0.3)])
    psd_db = 10 * np.log10(lin)
    res = occupied_bandwidth(FREQS, psd_db, peak_freq_hz=0, beta=0.01,
                             power_is_db=True, noise_floor_db=-90, window_hz=50_000)
    assert res is not None
    assert 7000 < res[2] < 9500


def test_narrowband_digital():
    """A ~2.4 kHz spread returns a few kHz, not collapsed."""
    lin = _spectrum([(-1200, 0.5), (0, 0.6), (1200, 0.5)])
    res = occupied_bandwidth(FREQS, 10 * np.log10(lin), peak_freq_hz=0,
                             power_is_db=True, noise_floor_db=-90, window_hz=50_000)
    assert res is not None
    assert 1500 < res[2] < 4000


def test_wideband_block():
    """A flat 180 kHz block returns ~180 kHz."""
    lin = np.full(N, 1e-9)
    lin[np.abs(FREQS) <= 90_000] += 0.01
    res = occupied_bandwidth(FREQS, 10 * np.log10(lin), beta=0.01,
                             power_is_db=True, noise_floor_db=-90)
    assert res is not None
    assert 170_000 < res[2] < 185_000


def test_signal_at_edge_clamps():
    """A peak in the first bins clamps without error."""
    lin = np.full(N, 1e-9)
    lin[0] += 1.0
    lin[1] += 0.5
    res = occupied_bandwidth(FREQS, 10 * np.log10(lin), noise_floor_db=-90)
    assert res is not None
    assert res[2] > 0


def test_all_noise_returns_none():
    """A flat noise spectrum (everything subtracted) returns None."""
    res = occupied_bandwidth(FREQS, np.full(N, -90.0), noise_floor_db=-90)
    assert res is None


def test_malformed_returns_none():
    assert occupied_bandwidth(np.array([1.0]), np.array([1.0])) is None
    assert occupied_bandwidth(FREQS, FREQS[:10]) is None  # mismatched lengths


def test_single_bin_never_zero():
    """One bin above floor returns >= one bin width, never <= 0."""
    lin = np.full(N, 1e-12)
    lin[N // 2] = 1.0
    res = occupied_bandwidth(FREQS, 10 * np.log10(lin), noise_floor_db=-110)
    assert res is not None
    assert res[2] >= (FREQS[1] - FREQS[0]) - 1e-6
    assert res[2] > 0


def test_beta_monotonicity():
    """99% occupied bandwidth >= 90% for the same spectrum."""
    lin = _spectrum([(0, 1.0), (4000, 0.3), (-4000, 0.3)])
    db = 10 * np.log10(lin)
    a = occupied_bandwidth(FREQS, db, peak_freq_hz=0, beta=0.01,
                           noise_floor_db=-90, window_hz=50_000)[2]
    b = occupied_bandwidth(FREQS, db, peak_freq_hz=0, beta=0.10,
                           noise_floor_db=-90, window_hz=50_000)[2]
    assert a >= b


def test_db_and_linear_agree():
    """Passing dB vs linear power yields the same band."""
    lin = _spectrum([(0, 1.0), (4000, 0.3), (-4000, 0.3)])
    r_db = occupied_bandwidth(FREQS, 10 * np.log10(lin), peak_freq_hz=0,
                              power_is_db=True, noise_floor_db=-90, window_hz=50_000)
    r_lin = occupied_bandwidth(FREQS, lin, peak_freq_hz=0,
                               power_is_db=False, noise_floor_db=-90, window_hz=50_000)
    assert abs(r_db[2] - r_lin[2]) < 1.0  # within a sub-bin


def test_noise_margin_suppresses_pedestal():
    """A higher noise margin yields a narrower band by suppressing the pedestal.

    A strong carrier sits on a broadband pedestal ~5 dB above the floor across
    the window. With no margin the pedestal is integrated (wide band); with a
    6 dB margin it is suppressed, leaving the carrier (narrow band).
    """
    lin = np.full(N, 1e-6)                       # base floor
    lin[np.abs(FREQS) <= 100_000] += 2e-6        # broadband pedestal (~5 dB)
    lin[N // 2] += 5e-4                           # modest carrier at DC
    db = 10 * np.log10(lin)
    floor_db = 10 * np.log10(1e-6)
    wide = occupied_bandwidth(FREQS, db, peak_freq_hz=0, noise_floor_db=floor_db,
                              noise_margin_db=0.0, window_hz=200_000)[2]
    narrow = occupied_bandwidth(FREQS, db, peak_freq_hz=0, noise_floor_db=floor_db,
                                noise_margin_db=6.0, window_hz=200_000)[2]
    assert narrow < wide
