"""Occupied-bandwidth estimation (ITU-R SM.443 beta-power method).

Measures signal bandwidth by integrating spectral power rather than counting
threshold-crossing bins. The occupied bandwidth is the band that contains
(1 - beta) of the total power around the signal peak, with beta/2 excluded from
each side. Unless specified otherwise SM.443 takes beta/2 = 0.5%, i.e. the
99%-power occupied bandwidth (``beta = 0.01``).

This is far more robust than an x-dB or bare-threshold width for signals whose
energy sits in sidebands a few dB above the noise floor (e.g. AM voice), which
a threshold method clips down to the carrier alone.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def occupied_bandwidth(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    *,
    peak_freq_hz: float | None = None,
    beta: float = 0.01,
    power_is_db: bool = True,
    window_hz: float | None = None,
    noise_floor_db: float | None = None,
    noise_margin_db: float = 0.0,
) -> tuple[float, float, float] | None:
    """Estimate occupied bandwidth via the ITU-R SM.443 beta-power method.

    Args:
        freqs_hz: Ascending frequency axis (absolute or offset), length N.
        power: Spectral power at each frequency, length N. Interpreted as dB
            when ``power_is_db`` else as linear power.
        peak_freq_hz: Frequency to anchor the measurement window/region on.
            Defaults to the bin of maximum power.
        beta: Total fraction of power excluded (beta/2 each side). 0.01 = 99%
            occupied bandwidth.
        power_is_db: Whether ``power`` is in dB (converted to linear internally).
        window_hz: If given, restrict integration to +/- this many Hz around
            the peak. Guards against a second signal in the same span biasing
            the integral. None integrates the whole array.
        noise_floor_db: If given, subtract this floor (in dB) from the linear
            power before integrating, so a noise pedestal does not dominate and
            genuine sideband energy is counted. Clamped at zero.
        noise_margin_db: Extra dB added to ``noise_floor_db`` before subtracting,
            to suppress noise fluctuations above the floor. Without this, a wide
            integration window of mostly-noise inflates the bandwidth to the
            window width. Only meaningful when ``noise_floor_db`` is given.

    Returns:
        ``(f_lo_hz, f_hi_hz, bandwidth_hz)`` for the occupied band, or ``None``
        if it cannot be determined (too few bins, no energy, flat spectrum).
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    pwr = np.asarray(power, dtype=np.float64)
    if freqs.ndim != 1 or freqs.shape != pwr.shape or len(freqs) < 2:
        return None

    # Convert to linear power.
    lin = 10.0 ** (pwr / 10.0) if power_is_db else pwr.copy()

    # Subtract the noise floor (plus a margin to suppress noise fluctuations)
    # so sideband energy counts but the broadband noise pedestal does not
    # inflate the result to the window width.
    if noise_floor_db is not None:
        lin = lin - 10.0 ** ((noise_floor_db + noise_margin_db) / 10.0)
    lin = np.maximum(lin, 0.0)

    # Anchor on the peak.
    if peak_freq_hz is None:
        peak_idx = int(np.argmax(lin))
    else:
        peak_idx = int(np.argmin(np.abs(freqs - peak_freq_hz)))

    # Restrict to a window around the peak if requested.
    if window_hz is not None:
        peak_f = freqs[peak_idx]
        in_window = np.abs(freqs - peak_f) <= window_hz
        lo_i = int(np.argmax(in_window))                 # first True
        hi_i = len(freqs) - int(np.argmax(in_window[::-1]))  # one past last True
    else:
        lo_i, hi_i = 0, len(freqs)

    w_freqs = freqs[lo_i:hi_i]
    w_lin = lin[lo_i:hi_i]
    if len(w_freqs) < 2:
        return None

    total = float(np.sum(w_lin))
    if total <= 0.0:
        return None

    cum = np.cumsum(w_lin)
    cum_frac = cum / total

    lower_target = beta / 2.0
    upper_target = 1.0 - beta / 2.0

    f_lo = _crossing_freq(w_freqs, cum_frac, lower_target, lower=True)
    f_hi = _crossing_freq(w_freqs, cum_frac, upper_target, lower=False)

    if f_lo is None or f_hi is None:
        return None

    bin_width = float(w_freqs[1] - w_freqs[0])
    bw = max(f_hi - f_lo, bin_width)  # never collapse to zero/negative
    return float(f_lo), float(f_hi), float(bw)


def _crossing_freq(
    freqs: np.ndarray,
    cum_frac: np.ndarray,
    target: float,
    *,
    lower: bool,
) -> float | None:
    """Frequency where the cumulative power fraction reaches ``target``.

    Linearly interpolates between adjacent bins for sub-bin precision (avoids
    quantisation jitter on narrowband signals). Clamps to the window edge when
    the crossing falls in the first/last bin (signal runs off the edge).
    """
    idx = int(np.searchsorted(cum_frac, target))
    if idx <= 0:
        return float(freqs[0])
    if idx >= len(freqs):
        return float(freqs[-1])

    # Interpolate between idx-1 and idx.
    c0, c1 = cum_frac[idx - 1], cum_frac[idx]
    f0, f1 = freqs[idx - 1], freqs[idx]
    if c1 == c0:
        return float(f1)
    frac = (target - c0) / (c1 - c0)
    return float(f0 + frac * (f1 - f0))
