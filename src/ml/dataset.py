"""Build a labelled (feature, modulation) training dataset.

Turns the few available signals into a usable dataset through augmentation:
synthetic generators swept over parameters and SNR, live captures sliced into
many windows with noise/frequency jitter, OGG-ingested sessions, and operator
corrections. Every source ends in the same ``_featurize`` path the production
pipeline uses (isolate -> compute features -> features_to_vector), so training
rows and live predictions share identical column semantics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from src.dsp.bandwidth import occupied_bandwidth
from src.fingerprint.extractor import FingerprintExtractor
from src.fingerprint.isolation import SignalIsolator
from src.fingerprint.modulation import ModulationClassifier
from src.ml.features import FEATURE_NAMES, features_to_vector
from src.sdr.replay import ReplaySource
from src.sdr.sigmf import read_iq

logger = logging.getLogger(__name__)

Row = tuple[np.ndarray, str]  # (feature_vector, modulation_label)


@dataclass
class DatasetSpec:
    """Augmentation parameters for dataset construction."""

    snr_db_levels: tuple[float, ...] = (10.0, 20.0, 30.0, 1e6)  # 1e6 ~ clean
    capture_snr_levels: tuple[float, ...] = (1e6, 20.0)
    capture_freq_shifts_hz: tuple[float, ...] = (0.0, 200.0)
    capture_seconds: float = 3.0  # truncate huge captures before isolating
    window_samples: int = 8192
    window_stride: int = 4096
    max_windows_per_capture: int = 200
    synthetic_reps: int = 2
    seed: int = 0
    guard_factor: float = 3.0
    min_filter_bw_hz: float = 500.0


# --- featurization (shared by every source) --------------------------------

_classifier = ModulationClassifier()  # only for _compute_features (stateless)


def _add_noise(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex Gaussian noise at the given SNR (signal assumed ~unit RMS)."""
    if snr_db >= 1e5:
        return iq
    power = float(np.mean(np.abs(iq) ** 2)) or 1.0
    noise_power = power * 10.0 ** (-snr_db / 10.0)
    sigma = np.sqrt(noise_power / 2.0)
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))) * sigma
    return (iq + noise).astype(np.complex64)


def _occupied_bw(iq: np.ndarray, sample_rate: float) -> float:
    """Measure occupied bandwidth of a baseband signal (fallback to a sane value)."""
    n = min(len(iq), 16384)
    psd = np.abs(np.fft.fftshift(np.fft.fft(iq[:n]))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    obw = occupied_bandwidth(freqs, psd, beta=0.01, power_is_db=False)
    return obw[2] if obw and obw[2] > 0 else sample_rate / 20.0


def _featurize_windows(
    isolated: np.ndarray, isolated_sr: float, spec: DatasetSpec
) -> Iterator[np.ndarray]:
    """Slide a window over an isolated signal, yielding a feature vector each."""
    w, s = spec.window_samples, spec.window_stride
    count = 0
    for start in range(0, max(1, len(isolated) - w + 1), s):
        if count >= spec.max_windows_per_capture:
            break
        window = isolated[start:start + w]
        if len(window) < 256:
            break
        feats = _classifier._compute_features(window, isolated_sr)
        bw = FingerprintExtractor._refine_bandwidth(window, isolated_sr, _occupied_bw(window, isolated_sr))
        yield features_to_vector(feats, bw)
        count += 1


def _isolate(
    iq: np.ndarray, sample_rate: float, step_center: float, signal_center: float,
    occ_bw: float, spec: DatasetSpec,
) -> tuple[np.ndarray, float]:
    isolator = SignalIsolator(
        sample_rate=sample_rate, guard_factor=spec.guard_factor,
        min_filter_bw_hz=spec.min_filter_bw_hz,
    )
    return isolator.isolate(
        iq_data=iq, step_centre_hz=step_center,
        signal_centre_hz=signal_center, signal_bandwidth_hz=occ_bw,
    )


# --- sources ----------------------------------------------------------------

def _rows_from_synthetic(spec: DatasetSpec) -> Iterator[Row]:
    """Synthetic generators swept over parameters and SNR."""
    from tests import conftest as gen

    configs: list[tuple[str, callable, dict]] = []
    for sr_baud in (45, 50, 300, 1200, 4800, 9600):
        for shift in (170, 450, 1000, 3000, 5000):
            configs.append(("FSK", gen.generate_fsk_signal,
                            dict(sample_rate=50_000, num_samples=80_000, symbol_rate=sr_baud, freq_shift=shift)))
    for sr_baud in (31, 300, 1200, 2400, 4800):
        configs.append(("PSK", gen.generate_psk_signal,
                        dict(sample_rate=50_000, num_samples=80_000, symbol_rate=sr_baud)))
    for sr_baud in (10, 20, 50, 100, 300):
        configs.append(("OOK", gen.generate_ook_signal,
                        dict(sample_rate=50_000, num_samples=80_000, symbol_rate=sr_baud)))
    for mf in (300, 1000, 3000):
        for depth in (0.3, 0.6, 0.9):
            configs.append(("AM", gen.generate_am_signal,
                            dict(sample_rate=50_000, num_samples=80_000, mod_freq=mf, mod_depth=depth)))
    for mf in (300, 1000, 3000):
        for dev in (25_000, 50_000, 75_000):
            configs.append(("FM", gen.generate_fm_signal,
                            dict(sample_rate=300_000, num_samples=200_000, mod_freq=mf, deviation=dev)))
    for mf in (300, 1000):
        for dev in (2_500, 5_000):
            configs.append(("NFM", gen.generate_nfm_signal,
                            dict(sample_rate=50_000, num_samples=80_000, mod_freq=mf, deviation=dev)))

    rng = np.random.default_rng(spec.seed)
    for label, fn, kwargs in configs:
        sr = kwargs["sample_rate"]
        base = fn(**kwargs)
        occ = _occupied_bw(base, sr)
        for snr in spec.snr_db_levels:
            for _ in range(spec.synthetic_reps):
                noisy = _add_noise(base, snr, rng)
                try:
                    iso, iso_sr = _isolate(noisy, sr, 0.0, 0.0, occ, spec)
                    if len(iso) < 256:
                        continue
                    # one feature vector per synthetic example (already ~stationary)
                    feats = _classifier._compute_features(iso, iso_sr)
                    bw = FingerprintExtractor._refine_bandwidth(iso, iso_sr, occ)
                    yield features_to_vector(feats, bw), label
                except Exception as e:  # noqa: BLE001
                    logger.debug("synthetic %s skipped: %s", label, e)


def _rows_from_session(session_dir: Path, spec: DatasetSpec, sliced: bool) -> Iterator[Row]:
    """Rows from a recorded SigMF session with a truth.json (live captures / OGG)."""
    truth_path = session_dir / "truth.json"
    if not truth_path.exists():
        return
    truth = json.loads(truth_path.read_text())
    rp = ReplaySource(session_dir)
    sr = rp.sample_rate
    # Map each recorded step centre -> its IQ.
    for sig in truth.get("signals", []):
        label = sig.get("modulation")
        if not label:
            continue
        center = float(sig["center_freq_hz"])
        idx = int(np.argmin(np.abs(rp._centers - center)))
        step_center = float(rp._centers[idx])
        iq = read_iq(session_dir / rp._data_files[idx])
        if sliced and spec.capture_seconds:  # huge live captures -> truncate
            iq = iq[:int(spec.capture_seconds * sr)]
        occ = float(sig.get("bandwidth_hz") or 0) or _occupied_bw(iq, sr)
        snr_levels = spec.capture_snr_levels if sliced else (1e6, 15.0)
        rng = np.random.default_rng(spec.seed + idx)
        for snr in snr_levels:
            for shift in (spec.capture_freq_shifts_hz if sliced else (0.0,)):
                try:
                    iq_aug = _add_noise(iq, snr, rng)
                    iso, iso_sr = _isolate(iq_aug, sr, step_center, center + shift, occ, spec)
                    if len(iso) < 256:
                        continue
                    # Always emit the whole-signal row — that is the granularity
                    # production classifies at. Slices then augment around it.
                    feats = _classifier._compute_features(iso, iso_sr)
                    bw = FingerprintExtractor._refine_bandwidth(iso, iso_sr, occ)
                    yield features_to_vector(feats, bw), label
                    if sliced:
                        for vec in _featurize_windows(iso, iso_sr, spec):
                            yield vec, label
                except Exception as e:  # noqa: BLE001
                    logger.debug("session %s signal skipped: %s", session_dir.name, e)


def _rows_from_corrections(db_path: Path) -> Iterator[Row]:
    """Rows from operator-corrected detections (loop closure)."""
    if not db_path.exists():
        return
    from src.detectionlog.database import DetectionLog
    log = DetectionLog(db_path)
    try:
        for row in log.query_corrections():
            fv = row.get("feature_vector")
            mod = row.get("corrected_modulation")
            if not fv or not mod:
                continue
            vec = np.array(json.loads(fv), dtype=np.float64)
            if len(vec) == len(FEATURE_NAMES):
                yield vec, mod
    finally:
        log.close()


# --- public API -------------------------------------------------------------

# Session names treated as live captures (sliced) vs OGG/ingest (whole-signal).
_LIVE_CAPTURE_PREFIXES = ("live_",)


def build_dataset(
    spec: DatasetSpec | None = None,
    *,
    sessions_dir: Path | None = None,
    db_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, feature_names) from all available sources."""
    spec = spec or DatasetSpec()
    rows: list[Row] = []

    n0 = len(rows)
    rows.extend(_rows_from_synthetic(spec))
    logger.info("synthetic rows: %d", len(rows) - n0)

    if sessions_dir and sessions_dir.exists():
        for sub in sorted(sessions_dir.iterdir()):
            if not (sub / "truth.json").exists():
                continue
            sliced = sub.name.startswith(_LIVE_CAPTURE_PREFIXES)
            n = len(rows)
            rows.extend(_rows_from_session(sub, spec, sliced))
            logger.info("session %s rows: %d (%s)", sub.name, len(rows) - n,
                        "sliced" if sliced else "whole")

    if db_path:
        n = len(rows)
        rows.extend(_rows_from_corrections(db_path))
        logger.info("correction rows: %d", len(rows) - n)

    if not rows:
        raise RuntimeError("No training rows produced")
    X = np.array([r[0] for r in rows], dtype=np.float64)
    y = np.array([r[1] for r in rows], dtype=object)
    return X, y, list(FEATURE_NAMES)


def save_dataset(path: Path, X: np.ndarray, y: np.ndarray, feature_names: list[str], meta: dict) -> None:
    """Save a dataset to an .npz archive."""
    np.savez(path, X=X, y=y, feature_names=np.array(feature_names, dtype=object),
             meta=np.array(json.dumps(meta), dtype=object))


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a dataset from an .npz archive."""
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"], list(d["feature_names"])
