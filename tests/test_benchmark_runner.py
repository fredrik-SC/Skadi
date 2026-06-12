"""Integration test for the benchmark runner over a synthetic session.

Records a tiny session to tmp_path (reusing SigmfRecorder + generate_test_iq the
same way tests/test_replay.py does), injects a stub classifier so the test does
not depend on the Artemis database, and checks the report is well-formed and
JSON-serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.benchmark.report import to_json
from src.benchmark.runner import run_benchmark
from src.benchmark.truth import GroundTruth, GroundTruthSignal
from src.config import load_config
from src.sdr.recorder import SigmfRecorder
from tests.conftest import generate_test_iq

SAMPLE_RATE = 2_048_000
TONE_OFFSET = 500_000.0
STEP_SAMPLES = 4096 * 4

SCAN_CONFIG = {
    "freq_start": 88e6, "freq_stop": 92e6, "step_size": 2e6,
    "dwell_time": 0.001, "fft_size": 4096, "fft_averages": 4,
}
SDR_CONFIG = {"driver": "sdrplay", "sample_rate": SAMPLE_RATE}


# --- Stub classifier (mimics SignalClassifier.classify's return shape) -------
@dataclass
class _StubArtemis:
    name: str


@dataclass
class _StubMatch:
    signal: _StubArtemis
    confidence: float


@dataclass
class _StubResult:
    matches: list


class _StubClassifier:
    def classify(self, fingerprint):
        return _StubResult(matches=[
            _StubMatch(_StubArtemis("TONE"), 0.9),
            _StubMatch(_StubArtemis("Other"), 0.4),
        ])


def _record_session(tmp_path) -> list[float]:
    """Record a session with one tone per step; return the tone frequencies."""
    recorder = SigmfRecorder(tmp_path, SCAN_CONFIG, SDR_CONFIG, label="synthetic")
    tone_freqs = []
    freq = SCAN_CONFIG["freq_start"] + SCAN_CONFIG["step_size"] / 2
    while freq < SCAN_CONFIG["freq_stop"]:
        iq = generate_test_iq(
            tones=[(TONE_OFFSET, 0.6)], sample_rate=SAMPLE_RATE, num_samples=STEP_SAMPLES,
        )
        recorder.record_step(freq, SAMPLE_RATE, iq)
        tone_freqs.append(freq + TONE_OFFSET)
        freq += SCAN_CONFIG["step_size"]
    recorder.finalize()
    return tone_freqs


def test_runner_scores_synthetic_session(tmp_path):
    tone_freqs = _record_session(tmp_path)
    truth = GroundTruth(
        signals=[GroundTruthSignal(f, modulation=None, signal_type="TONE") for f in tone_freqs],
        match_tolerance_hz=100_000.0,
    )

    config = load_config()
    config["detection"]["threshold_db"] = 10.0
    config["detection"]["min_bandwidth_hz"] = 500

    report = run_benchmark(
        tmp_path, truth, config,
        artemis_path=Path("/nonexistent/artemis.db"),  # unused: classifier injected
        classifier=_StubClassifier(),
    )

    # Both tones should be found.
    assert report.recall > 0.0
    assert report.num_truth == len(tone_freqs)
    # Stub classifies everything as "TONE", which is our truth signal_type.
    assert report.classification_top1_accuracy == 1.0
    # Report serialises cleanly to JSON-compatible structures.
    payload = to_json(report)
    import json
    json.loads(json.dumps(payload))  # round-trips without error


def test_runner_without_classifier_skips_classification(tmp_path):
    tone_freqs = _record_session(tmp_path)
    truth = GroundTruth(
        signals=[GroundTruthSignal(f, signal_type="TONE") for f in tone_freqs],
        match_tolerance_hz=100_000.0,
    )
    config = load_config()
    config["detection"]["threshold_db"] = 10.0
    config["detection"]["min_bandwidth_hz"] = 500

    # No artemis DB and no injected classifier -> classification dimension skipped.
    report = run_benchmark(
        tmp_path, truth, config, artemis_path=Path("/nonexistent/artemis.db"),
    )
    assert report.classification_top1_accuracy in (None, 0.0)
