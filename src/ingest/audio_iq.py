"""Core audio → IQ conversion and session building for the ingest tool."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

from src.sdr.recorder import SigmfRecorder

logger = logging.getLogger(__name__)

# Map a mode name (substring, case-insensitive) to our ModulationType value.
# First match wins, so list more-specific keys first.
_MODULATION_KEYS: list[tuple[str, str]] = [
    ("bpsk", "PSK"), ("qpsk", "PSK"), ("8psk", "PSK"), ("psk", "PSK"),
    ("rtty", "FSK"), ("sitor", "FSK"), ("amtor", "FSK"), ("navtex", "FSK"),
    ("mfsk", "FSK"), ("olivia", "FSK"), ("contestia", "FSK"), ("domino", "FSK"),
    ("thor", "FSK"), ("mt63", "FSK"), ("packet", "FSK"), ("ax25", "FSK"),
    ("ft8", "FSK"), ("ft4", "FSK"), ("fst4", "FSK"), ("jt", "FSK"),
    ("wspr", "FSK"), ("fsk", "FSK"), ("gmsk", "FSK"), ("msk", "FSK"),
    ("hell", "OOK"), ("feld", "OOK"), ("cw", "OOK"), ("morse", "OOK"),
]

_WAV_EXTS = {".wav", ".wave"}


def infer_modulation(name: str) -> str | None:
    """Infer the ModulationType value from a mode name, or None if unknown."""
    low = name.lower()
    for key, mod in _MODULATION_KEYS:
        if key in low:
            return mod
    return None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 in [-1, 1], returning (samples, rate).

    WAV is read natively (no dependencies). Other formats (OGG/MP3/FLAC) are
    decoded via the optional ``soundfile`` package or an ``ffmpeg`` binary if
    either is available.

    Raises:
        RuntimeError: For a non-WAV file when no decoder is available.
    """
    path = Path(path)
    if path.suffix.lower() in _WAV_EXTS:
        rate, data = wavfile.read(path)
        return _to_mono_float(data), int(rate)

    # Try soundfile (libsndfile: OGG/FLAC/WAV).
    try:
        import soundfile  # type: ignore
        data, rate = soundfile.read(str(path), always_2d=False)
        return _to_mono_float(np.asarray(data)), int(rate)
    except ImportError:
        pass

    # Fall back to ffmpeg -> temp WAV.
    if _have_ffmpeg():
        return _decode_with_ffmpeg(path)

    raise RuntimeError(
        f"Cannot decode {path.suffix} file {path.name}: no decoder available. "
        f"Install 'soundfile' (pip install soundfile) or ffmpeg, or convert to WAV."
    )


def audio_to_iq(
    audio: np.ndarray,
    sr_audio: int,
    target_rate: float,
    *,
    center: bool = True,
) -> tuple[np.ndarray, float]:
    """Convert real audio to complex baseband IQ at ``target_rate``.

    Recovers the analytic signal (Hilbert), optionally shifts the spectral
    centroid to DC, and resamples to the target rate. AFSK/SSB audio is
    spectrally equivalent to the RF signal at baseband, so the modulation
    structure (FSK tone states, PSK phase reversals) is preserved.

    Returns:
        (iq_complex64, target_rate).
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size < 2:
        raise ValueError("Audio is too short to convert")

    analytic = hilbert(audio)  # complex analytic signal (positive freqs only)

    if center:
        centroid = _spectral_centroid(analytic, sr_audio)
        t = np.arange(len(analytic)) / sr_audio
        analytic = analytic * np.exp(-2j * np.pi * centroid * t)

    # Resample to the target IQ rate via polyphase filtering.
    up, down = _ratio(target_rate, sr_audio)
    iq = resample_poly(analytic, up, down)

    # Normalise to unit RMS so SNR is controllable downstream.
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    if rms > 0:
        iq = iq / rms

    return iq.astype(np.complex64), float(target_rate)


def embed_in_step(
    iq: np.ndarray,
    target_rate: float,
    offset_hz: float,
    snr_db: float,
    rng: np.random.Generator,
    min_samples: int,
) -> np.ndarray:
    """Place a baseband signal at ``offset_hz`` in a noisy wideband step.

    The signal is frequency-shifted to the offset, tiled to at least
    ``min_samples``, and broadband complex Gaussian noise is added at the
    requested SNR (signal power is unit RMS, so noise power = 10**(-snr/10)).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < min_samples:
        reps = int(np.ceil(min_samples / len(iq)))
        iq = np.tile(iq, reps)[:min_samples]

    t = np.arange(len(iq)) / target_rate
    shifted = iq * np.exp(2j * np.pi * offset_hz * t).astype(np.complex64)

    noise_power = 10.0 ** (-snr_db / 10.0)
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) * sigma
    return (shifted + noise).astype(np.complex64)


def build_session(
    audio_files: list[Path],
    out_dir: Path,
    *,
    target_rate: float = 96_000.0,
    base_freq_hz: float = 14_000_000.0,
    offset_hz: float = 24_000.0,
    snr_db: float = 25.0,
    fft_size: int = 8192,
    fft_averages: int = 10,
    label_map: dict[str, dict] | None = None,
    seed: int = 0,
) -> dict:
    """Convert audio files into a benchmark-ready SigMF session + truth.json.

    Each file becomes one scan step on a regular frequency grid, with the mode
    signal placed at ``offset_hz`` within the step span. A ``truth.json`` labels
    each signal with its inferred modulation and the mode name.

    Args:
        audio_files: Audio files to ingest (label derived from the file stem).
        out_dir: Session directory to create.
        target_rate: IQ sample rate for the synthetic steps.
        base_freq_hz: Centre frequency of the first step.
        offset_hz: Offset of the signal from each step centre (< target_rate/2).
        snr_db: Signal-to-noise ratio of the embedded signal.
        fft_size, fft_averages: Stored scan config (drive the minimum step length).
        label_map: Optional {stem: {modulation, signal_type, freq_hz}} overrides.
        seed: RNG seed for reproducible noise.

    Returns:
        A summary dict (counts, output paths).
    """
    out_dir = Path(out_dir)
    label_map = label_map or {}
    rng = np.random.default_rng(seed)
    min_samples = fft_size * fft_averages

    step_size = target_rate
    n = len(audio_files)
    scan_config = {
        "freq_start": base_freq_hz - step_size / 2.0,
        "freq_stop": base_freq_hz + (n - 0.5) * step_size + 1.0,
        "step_size": step_size,
        "dwell_time": min_samples / target_rate,
        "fft_size": fft_size,
        "fft_averages": fft_averages,
    }
    sdr_config = {"driver": "ingest", "sample_rate": target_rate}

    recorder = SigmfRecorder(out_dir, scan_config, sdr_config, label="audio-ingest")
    truth_signals: list[dict] = []
    skipped: list[str] = []

    for i, path in enumerate(audio_files):
        path = Path(path)
        stem = path.stem
        override = label_map.get(stem, {})
        try:
            audio, sr_audio = load_audio(path)
            iq, _ = audio_to_iq(audio, sr_audio, target_rate)
            step_iq = embed_in_step(iq, target_rate, offset_hz, snr_db, rng, min_samples)
        except Exception as e:  # noqa: BLE001 — report and continue the batch
            logger.warning("Skipping %s: %s", path.name, e)
            skipped.append(path.name)
            continue

        center = override.get("freq_hz", base_freq_hz + i * step_size)
        recorder.record_step(center, target_rate, step_iq)
        truth_signals.append({
            "center_freq_hz": float(center + offset_hz),
            "label": stem,
            "modulation": override.get("modulation", infer_modulation(stem)),
            "signal_type": override.get("signal_type", stem),
            "verified": True,
            "source": path.name,
        })

    recorder.finalize()

    truth = {
        "_note": "Synthetic IQ built from demodulated audio samples (audio->IQ "
                 "ingest). Modulation is inferred from the mode name; correct it "
                 "if the inference is wrong. Signals are clean injections, not "
                 "live captures.",
        "schema_version": 1,
        "session": out_dir.name,
        "match_tolerance_hz": max(offset_hz, 5000.0),
        "signals": truth_signals,
    }
    truth_path = out_dir / "truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))

    logger.info(
        "Ingested %d/%d file(s) into %s (%d skipped)",
        len(truth_signals), len(audio_files), out_dir, len(skipped),
    )
    return {
        "session_dir": str(out_dir),
        "ingested": len(truth_signals),
        "skipped": skipped,
        "truth_path": str(truth_path),
    }


# --- helpers ---------------------------------------------------------------

def _to_mono_float(data: np.ndarray) -> np.ndarray:
    """Convert (possibly int/stereo) audio samples to mono float32 in [-1, 1]."""
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)
    return data.astype(np.float32)


def _spectral_centroid(analytic: np.ndarray, sr: float) -> float:
    """Power-weighted mean frequency of an analytic signal, in Hz."""
    spec = np.abs(np.fft.fft(analytic)) ** 2
    freqs = np.fft.fftfreq(len(analytic), d=1.0 / sr)
    total = np.sum(spec)
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spec) / total)


def _ratio(target_rate: float, sr_audio: int) -> tuple[int, int]:
    """Reduced integer up/down ratio for resampling to target_rate."""
    from math import gcd
    up = int(round(target_rate))
    down = int(sr_audio)
    g = gcd(up, down)
    return up // g, down // g


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    """Decode any audio file to mono float via an ffmpeg subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ac", "1", str(tmp_path)],
            capture_output=True, check=True,
        )
        rate, data = wavfile.read(tmp_path)
        return _to_mono_float(data), int(rate)
    finally:
        tmp_path.unlink(missing_ok=True)
