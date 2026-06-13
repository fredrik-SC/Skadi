"""Tests for the audio -> IQ ingest tool."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import find_peaks

from src.ingest.audio_iq import audio_to_iq, build_session, infer_modulation, load_audio
from src.sdr.replay import ReplaySource
from src.benchmark.truth import load_truth

SR_AUDIO = 8000
DUR = 3.0


def _inst_freq_states(iq, sr):
    """Count distinct instantaneous-frequency levels (full-rate, reliable)."""
    phase = np.unwrap(np.angle(iq))
    inst = np.diff(phase) / (2 * np.pi) * sr
    hist, _ = np.histogram(inst, bins=60)
    peaks, _ = find_peaks(hist.astype(float), prominence=hist.max() * 0.15)
    return len(peaks)


def _fsk_audio(shift_hz=2000.0, baud=50.0):
    n = int(SR_AUDIO * DUR)
    spb = int(SR_AUDIO / baud)
    rng = np.random.default_rng(3)
    bits = np.repeat(rng.integers(0, 2, n // spb + 1), spb)[:n]
    freq = np.where(bits > 0, 1500 + shift_hz / 2, 1500 - shift_hz / 2)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR_AUDIO).astype(np.float32)


def _bpsk_audio(carrier=1200.0, baud=50.0):
    n = int(SR_AUDIO * DUR)
    t = np.arange(n) / SR_AUDIO
    spb = int(SR_AUDIO / baud)
    rng = np.random.default_rng(4)
    ph = np.repeat(rng.integers(0, 2, n // spb + 1) * np.pi, spb)[:n]
    return np.sin(2 * np.pi * carrier * t + ph).astype(np.float32)


def test_infer_modulation():
    assert infer_modulation("RTTY45") == "FSK"
    assert infer_modulation("PSK31") == "PSK"
    assert infer_modulation("FT8") == "FSK"
    assert infer_modulation("MFSK16") == "FSK"
    assert infer_modulation("CW_morse") == "OOK"
    assert infer_modulation("mystery_mode") is None


def test_load_wav_mono(tmp_path):
    p = tmp_path / "tone.wav"
    wavfile.write(p, SR_AUDIO, (0.5 * np.sin(np.linspace(0, 100, 8000))).astype(np.float32))
    data, rate = load_audio(p)
    assert rate == SR_AUDIO
    assert data.ndim == 1
    assert np.abs(data).max() <= 1.0


def test_audio_to_iq_single_tone_constant_envelope():
    t = np.arange(SR_AUDIO * 2) / SR_AUDIO
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    iq, rate = audio_to_iq(tone, SR_AUDIO, 48_000)
    assert rate == 48_000
    assert iq.dtype == np.complex64
    env = np.abs(iq)
    # A single tone is constant-envelope after analytic conversion.
    assert np.std(env) / np.mean(env) < 0.1


def test_audio_to_iq_preserves_fsk_states():
    """A two-tone FSK audio yields >=2 instantaneous-frequency states in IQ."""
    iq, rate = audio_to_iq(_fsk_audio(), SR_AUDIO, 48_000)
    assert _inst_freq_states(iq, rate) >= 2


def test_audio_to_iq_preserves_psk_phase():
    """A BPSK audio yields many phase discontinuities in IQ."""
    iq, rate = audio_to_iq(_bpsk_audio(), SR_AUDIO, 48_000)
    phase_diff = np.abs(np.diff(np.angle(iq)))
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
    jumps = int(np.sum(phase_diff > 1.5))
    assert jumps > 50  # phase reversals preserved


def test_build_session_replayable_and_labelled(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    wavfile.write(samples / "RTTY45.wav", SR_AUDIO, (_fsk_audio() * 32767).astype(np.int16))
    wavfile.write(samples / "PSK31.wav", SR_AUDIO, (_bpsk_audio() * 32767).astype(np.int16))

    out = tmp_path / "session"
    summary = build_session(
        [samples / "RTTY45.wav", samples / "PSK31.wav"], out,
        target_rate=96_000, fft_size=2048, fft_averages=4, seed=0,
    )
    assert summary["ingested"] == 2

    # Session replays through the standard pipeline.
    rp = ReplaySource(out)
    assert len(rp.entries if hasattr(rp, "entries") else rp._centers) == 2

    # Truth loads and carries inferred modulations.
    truth = load_truth(out / "truth.json")
    mods = {s.label: s.modulation for s in truth.signals}
    assert mods["RTTY45"] == "FSK"
    assert mods["PSK31"] == "PSK"


def test_label_map_overrides(tmp_path):
    samples = tmp_path / "s"
    samples.mkdir()
    wavfile.write(samples / "mystery.wav", SR_AUDIO, (_bpsk_audio() * 32767).astype(np.int16))
    out = tmp_path / "sess"
    build_session(
        [samples / "mystery.wav"], out, target_rate=96_000,
        fft_size=2048, fft_averages=4,
        label_map={"mystery": {"modulation": "PSK", "signal_type": "STANAG 4285"}},
    )
    truth = load_truth(out / "truth.json")
    assert truth.signals[0].modulation == "PSK"
    assert truth.signals[0].signal_type == "STANAG 4285"
