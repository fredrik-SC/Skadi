"""Test for the raw-PSD truth seeder."""

from __future__ import annotations

from src.benchmark.seed import seed_truth, write_truth_json
from src.benchmark.truth import load_truth
from src.sdr.recorder import SigmfRecorder
from tests.conftest import generate_test_iq

SAMPLE_RATE = 2_048_000
TONE_OFFSET = 400_000.0


def test_seed_finds_tone_and_flags_unverified(tmp_path):
    scan = {"freq_start": 88e6, "freq_stop": 90e6, "step_size": 2e6,
            "dwell_time": 0.01, "fft_size": 8192, "fft_averages": 8}
    sdr = {"driver": "sdrplay", "sample_rate": SAMPLE_RATE}
    recorder = SigmfRecorder(tmp_path, scan, sdr, label="seedtest")
    # One step at 89 MHz with a strong tone at +400 kHz -> 89.4 MHz.
    iq = generate_test_iq(tones=[(TONE_OFFSET, 0.8)], sample_rate=SAMPLE_RATE, num_samples=8192 * 8)
    recorder.record_step(89e6, SAMPLE_RATE, iq)
    recorder.finalize()

    gt = seed_truth(tmp_path)

    assert gt.signals, "expected at least one seeded candidate"
    assert all(not s.verified for s in gt.signals)
    # A candidate should land near the tone frequency (89.4 MHz).
    assert any(abs(s.center_freq_hz - 89.4e6) < 50_000 for s in gt.signals)


def test_write_and_reload_seeded_truth(tmp_path):
    scan = {"freq_start": 88e6, "freq_stop": 90e6, "step_size": 2e6,
            "dwell_time": 0.01, "fft_size": 8192, "fft_averages": 8}
    sdr = {"driver": "sdrplay", "sample_rate": SAMPLE_RATE}
    recorder = SigmfRecorder(tmp_path, scan, sdr)
    iq = generate_test_iq(tones=[(TONE_OFFSET, 0.8)], sample_rate=SAMPLE_RATE, num_samples=8192 * 8)
    recorder.record_step(89e6, SAMPLE_RATE, iq)
    recorder.finalize()

    gt = seed_truth(tmp_path)
    out = tmp_path / "truth.json"
    write_truth_json(gt, out)

    # The written file is valid and reloads through the standard loader.
    reloaded = load_truth(out)
    assert len(reloaded.signals) == len(gt.signals)
