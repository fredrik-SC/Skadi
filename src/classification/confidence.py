"""Confidence scoring for signal classification.

Computes a weighted confidence score comparing a detected signal's
fingerprint against an Artemis database candidate across four
dimensions: modulation, bandwidth, frequency, and ACF.
"""

from __future__ import annotations

from src.classification.artemis_db import ArtemisSignal
from src.fingerprint.models import SignalFingerprint

# Scoring weights per dimension
WEIGHT_MODULATION = 0.4
WEIGHT_BANDWIDTH = 0.3
WEIGHT_FREQUENCY = 0.2
WEIGHT_ACF = 0.1


def compute_confidence(
    fingerprint: SignalFingerprint,
    candidate: ArtemisSignal,
    modulation_terms: list[str],
    bandwidth_tolerance: float = 0.15,
) -> tuple[float, float, float, float, float]:
    """Compute weighted confidence score for a candidate match.

    Args:
        fingerprint: The detected signal's fingerprint.
        candidate: An Artemis database signal record.
        modulation_terms: The Artemis modulation terms that correspond
            to the fingerprint's modulation type (from MODULATION_MAP).
        bandwidth_tolerance: Fractional tolerance for bandwidth matching
            (0.15 = ±15%).

    Returns:
        Tuple of (total_confidence, mod_score, bw_score, freq_score, acf_score).
        All values are in the range [0.0, 1.0].
    """
    mod_score = _score_modulation(candidate, modulation_terms)
    bw_score = _score_bandwidth(fingerprint, candidate, bandwidth_tolerance)
    freq_score = _score_frequency(fingerprint, candidate)
    acf_score = _score_acf(fingerprint, candidate)

    total = (
        WEIGHT_MODULATION * mod_score
        + WEIGHT_BANDWIDTH * bw_score
        + WEIGHT_FREQUENCY * freq_score
        + WEIGHT_ACF * acf_score
    )

    return total, mod_score, bw_score, freq_score, acf_score


def _score_modulation(candidate: ArtemisSignal, modulation_terms: list[str]) -> float:
    """Score modulation match: 1.0 if any term matches, 0.0 otherwise."""
    if not modulation_terms:
        return 0.0
    terms_upper = {t.upper().strip() for t in modulation_terms}
    cand_mods = {m.upper().strip() for m in candidate.modulation_types}
    return 1.0 if terms_upper.intersection(cand_mods) else 0.0


def _score_bandwidth(
    fingerprint: SignalFingerprint,
    candidate: ArtemisSignal,
    tolerance: float,
) -> float:
    """Score bandwidth match with tolerance.

    Returns 1.0 if fingerprint bandwidth falls within the candidate's
    bandwidth range (expanded by tolerance). Linear falloff outside.
    Returns 0.5 if candidate has no bandwidth data.
    """
    fp_bw = fingerprint.bandwidth_hz
    bw_min = candidate.bandwidth_min_hz
    bw_max = candidate.bandwidth_max_hz

    if bw_min is None and bw_max is None:
        return 0.5  # Neutral — don't penalise missing data

    # Use available bounds
    if bw_min is None:
        bw_min = bw_max
    if bw_max is None:
        bw_max = bw_min

    # Expand range by tolerance
    lower = bw_min * (1.0 - tolerance)
    upper = bw_max * (1.0 + tolerance)

    if lower <= fp_bw <= upper:
        return 1.0

    # Linear falloff outside the expanded range
    range_width = max(upper - lower, 1.0)
    if fp_bw < lower:
        distance = lower - fp_bw
    else:
        distance = fp_bw - upper

    score = max(0.0, 1.0 - distance / range_width)
    return score


def _score_frequency(fingerprint: SignalFingerprint, candidate: ArtemisSignal) -> float:
    """Score frequency range match.

    Returns 1.0 if detected frequency is inside the candidate's range.
    Linear falloff over 10% of range width at the edges.
    Returns 0.5 if candidate has no frequency data.
    """
    freq = fingerprint.signal.centre_freq_hz
    f_min = candidate.freq_min_hz
    f_max = candidate.freq_max_hz

    if f_min is None or f_max is None:
        return 0.5  # Neutral

    if f_min <= freq <= f_max:
        return 1.0

    # Linear falloff
    range_width = max(f_max - f_min, 1.0)
    margin = range_width * 0.1

    if freq < f_min:
        distance = f_min - freq
    else:
        distance = freq - f_max

    score = max(0.0, 1.0 - distance / margin)
    return score


def _score_acf(fingerprint: SignalFingerprint, candidate: ArtemisSignal) -> float:
    """Score ACF match.

    Returns the best match between fingerprint ACF and any candidate
    ACF value. Returns 0.5 (neutral) if either side has no ACF data.
    """
    if fingerprint.acf_ms is None or not candidate.acf_values_ms:
        return 0.5  # Neutral

    best_score = 0.0
    for cand_acf in candidate.acf_values_ms:
        if cand_acf <= 0:
            continue
        ratio = abs(fingerprint.acf_ms - cand_acf) / cand_acf
        score = max(0.0, 1.0 - ratio)
        best_score = max(best_score, score)

    return best_score
