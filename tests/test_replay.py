"""Tests for ReplaySource and deterministic offline replay."""

from __future__ import annotations

import numpy as np
import pytest

from src.sdr import SDRError
from src.sdr.base import SdrSource
from src.sdr.recorder import SigmfRecorder
from src.sdr.replay import ReplaySource
from src.sdr.scanner import SpectrumScanner
from tests.conftest import generate_test_iq

SAMPLE_RATE = 2_048_000
FFT_SIZE = 4096
FFT_AVERAGES = 4
STEP_SAMPLES = FFT_SIZE * FFT_AVERAGES

# Scan config whose num_samples == STEP_SAMPLES (tiny dwell), so replay matches
# the recorded length exactly.
SCAN_CONFIG = {
    "freq_start": 88e6,
    "freq_stop": 108e6,
    "step_size": 2e6,
    "dwell_time": 0.001,
    "fft_size": FFT_SIZE,
    "fft_averages": FFT_AVERAGES,
}
SDR_CONFIG = {"driver": "sdrplay", "sample_rate": SAMPLE_RATE}
DETECTION_CONFIG = {"threshold_db": 10.0, "min_bandwidth_hz": 500, "max_signals_per_step": 10}


def _record_session(tmp_path, *, num_samples=STEP_SAMPLES, tone=True):
    """Record a session whose step centres match scanner.step_frequencies."""
    recorder = SigmfRecorder(tmp_path, SCAN_CONFIG, SDR_CONFIG, label="t")
    centers = []
    freq = SCAN_CONFIG["freq_start"] + SCAN_CONFIG["step_size"] / 2
    while freq < SCAN_CONFIG["freq_stop"]:
        tones = [(500_000.0, 0.6)] if tone else None
        iq = generate_test_iq(tones=tones, sample_rate=SAMPLE_RATE, num_samples=num_samples)
        recorder.record_step(freq, SAMPLE_RATE, iq)
        centers.append(freq)
        freq += SCAN_CONFIG["step_size"]
    recorder.finalize()
    return centers


def test_is_sdr_source(tmp_path):
    """ReplaySource structurally satisfies the SdrSource protocol."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    assert isinstance(replay, SdrSource)
    assert replay.connected is True
    assert replay.sample_rate == SAMPLE_RATE


def test_restores_config_from_manifest(tmp_path):
    """The manifest's scan/SDR config is exposed for the caller to restore."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    assert replay.scan_config["step_size"] == 2e6
    assert replay.sdr_config["driver"] == "sdrplay"


def test_nearest_match_within_tolerance(tmp_path):
    """tune() snaps a near request to the recorded centre."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    # 89.0 MHz recorded; request 88.9 MHz (within step_size/2 = 1 MHz tolerance).
    assert replay.tune(88.9e6) == pytest.approx(89e6)


def test_out_of_tolerance_raises(tmp_path):
    """tune() with no nearby recorded step is a loud error."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    with pytest.raises(SDRError, match="No recorded step"):
        replay.tune(150e6)


def test_capture_before_tune_raises(tmp_path):
    """capture() without a prior successful tune() raises."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    with pytest.raises(SDRError, match="before tune"):
        replay.capture(STEP_SAMPLES)


def test_capture_exact_length(tmp_path):
    """Exact-length request returns the recorded IQ unchanged."""
    _record_session(tmp_path)
    replay = ReplaySource(tmp_path)
    replay.tune(89e6)
    iq = replay.capture(STEP_SAMPLES)
    assert len(iq) == STEP_SAMPLES
    assert iq.dtype == np.complex64


def test_capture_truncates_when_longer(tmp_path):
    """A longer recording is truncated to the requested length."""
    _record_session(tmp_path, num_samples=2000)
    replay = ReplaySource(tmp_path)
    replay.tune(89e6)
    assert len(replay.capture(500)) == 500


def test_capture_tiles_when_shorter(tmp_path):
    """A shorter recording is tiled up to the requested length."""
    _record_session(tmp_path, num_samples=500)
    replay = ReplaySource(tmp_path)
    replay.tune(89e6)
    out = replay.capture(1200)
    assert len(out) == 1200


def test_missing_manifest_raises(tmp_path):
    """Constructing on a directory without a manifest raises."""
    with pytest.raises(SDRError, match="manifest"):
        ReplaySource(tmp_path)


def test_empty_session_raises(tmp_path):
    """A session with no recorded steps raises."""
    recorder = SigmfRecorder(tmp_path, SCAN_CONFIG, SDR_CONFIG)
    recorder.finalize()  # no steps recorded
    with pytest.raises(SDRError, match="no steps"):
        ReplaySource(tmp_path)


def test_full_pipeline_replay_is_deterministic(tmp_path):
    """Replaying the same session twice yields identical detections."""
    _record_session(tmp_path)

    def run():
        replay = ReplaySource(tmp_path)
        scanner = SpectrumScanner(replay, SCAN_CONFIG, SDR_CONFIG, DETECTION_CONFIG)
        result = scanner.sweep()
        return [(round(s.centre_freq_hz, 3), round(s.bandwidth_hz, 3)) for s in result.signals]

    first = run()
    second = run()
    assert first == second
    # The 500 kHz tone in every step should produce detections.
    assert len(first) > 0
