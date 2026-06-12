"""Propose a starter ``truth.json`` by raw-PSD peak-picking a recording.

This deliberately bypasses the detection pipeline under test — it computes a
plain averaged PSD per step and picks spectral peaks directly — so the seeded
ground truth is independent of the detector we are scoring (avoiding circular
evaluation). Every seeded entry is marked ``verified: false``; an operator must
confirm the frequencies (and fill in modulation/signal_type) against an external
source before the numbers can be trusted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from src.benchmark.truth import GroundTruth, GroundTruthSignal
from src.sdr.recorder import MANIFEST_NAME
from src.sdr.sigmf import read_iq

logger = logging.getLogger(__name__)

_FFT_SIZE = 8192
_MAX_SEGMENTS = 16


def seed_truth(
    session_dir: Path,
    *,
    snr_db: float = 12.0,
    min_separation_hz: float = 150_000.0,
) -> GroundTruth:
    """Propose ground-truth signals from raw spectral peaks in a recording.

    Args:
        session_dir: Recorded session directory (with a ``session.json``).
        snr_db: Minimum peak height above the per-step median PSD, in dB.
        min_separation_hz: Minimum spacing between proposed signals.

    Returns:
        A :class:`GroundTruth` whose signals are all ``verified=False``.
    """
    session_dir = Path(session_dir)
    manifest = json.loads((session_dir / MANIFEST_NAME).read_text())
    sample_rate = float(manifest["sample_rate"])
    bin_hz = sample_rate / _FFT_SIZE
    min_sep_bins = max(1, int(min_separation_hz / bin_hz))

    # Collect (abs_freq_hz, peak_height_db) across all steps.
    candidates: list[tuple[float, float]] = []
    for step in manifest["steps"]:
        iq = read_iq(session_dir / step["data_file"])
        psd_db = _averaged_psd_db(iq)
        offsets = np.fft.fftshift(np.fft.fftfreq(_FFT_SIZE, d=1.0 / sample_rate))
        rel = psd_db - np.median(psd_db)
        peaks, props = find_peaks(rel, height=snr_db, distance=min_sep_bins)
        for pk in peaks:
            abs_freq = step["center_freq_hz"] + offsets[pk]
            candidates.append((float(abs_freq), float(rel[pk])))

    merged = _merge_candidates(candidates, min_separation_hz)

    signals = [
        GroundTruthSignal(center_freq_hz=freq, verified=False)
        for freq, _ in merged
    ]
    logger.info("Seeded %d candidate signal(s) from %s", len(signals), session_dir)
    return GroundTruth(
        signals=signals,
        match_tolerance_hz=min_separation_hz / 2,
        session=manifest.get("label"),
    )


def _averaged_psd_db(iq: np.ndarray) -> np.ndarray:
    """Averaged Hann-windowed PSD in dB (independent of the scanner)."""
    num_segments = min(_MAX_SEGMENTS, max(1, len(iq) // _FFT_SIZE))
    window = np.hanning(_FFT_SIZE)
    power = np.zeros(_FFT_SIZE, dtype=np.float64)
    for i in range(num_segments):
        seg = iq[i * _FFT_SIZE:(i + 1) * _FFT_SIZE]
        if len(seg) < _FFT_SIZE:
            break
        power += np.abs(np.fft.fft(seg * window)) ** 2
    power /= num_segments
    return np.fft.fftshift(10.0 * np.log10(np.maximum(power, 1e-20)))


def _merge_candidates(
    candidates: list[tuple[float, float]], min_separation_hz: float
) -> list[tuple[float, float]]:
    """Merge peaks within min_separation_hz, keeping the strongest."""
    merged: list[tuple[float, float]] = []
    for freq, height in sorted(candidates, key=lambda c: c[0]):
        if merged and abs(freq - merged[-1][0]) < min_separation_hz:
            # Keep whichever is stronger.
            if height > merged[-1][1]:
                merged[-1] = (freq, height)
        else:
            merged.append((freq, height))
    return merged


def write_truth_json(ground_truth: GroundTruth, path: Path) -> None:
    """Write a GroundTruth to a truth.json file (seeded entries flagged)."""
    data = {
        "_note": "Seeded by raw-PSD peak picking — VERIFY frequencies and fill "
                 "in modulation/signal_type against a known source before use.",
        "schema_version": 1,
        "session": ground_truth.session,
        "match_tolerance_hz": ground_truth.match_tolerance_hz,
        "signals": [
            {
                "center_freq_hz": s.center_freq_hz,
                "label": s.label,
                "modulation": s.modulation,
                "signal_type": s.signal_type,
                "bandwidth_hz": s.bandwidth_hz,
                "threat_level": s.threat_level,
                "verified": s.verified,
            }
            for s in ground_truth.signals
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))
