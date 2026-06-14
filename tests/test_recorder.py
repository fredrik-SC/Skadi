"""Tests for SigMF recording (SigmfRecorder + RecordingSDR decorator)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.sdr.base import SdrSource
from src.sdr.recorder import MANIFEST_NAME, RecordingSDR, SigmfRecorder
from src.sdr.sigmf import read_iq, read_meta


class _FakeSDR:
    """Minimal SdrSource returning a constant, frequency-tagged capture."""

    def __init__(self, sample_rate: float = 2_048_000, actual_offset: float = 0.0):
        self._sample_rate = sample_rate
        self._actual_offset = actual_offset
        self._last = 0.0
        self.entered = False
        self.exited = False

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def connected(self) -> bool:
        return True

    def tune(self, frequency_hz: float) -> float:
        # Report an "actual" frequency slightly offset from requested.
        self._last = frequency_hz + self._actual_offset
        return self._last

    def capture(self, num_samples: int) -> np.ndarray:
        # DC offset present so we can prove raw (unprocessed) IQ is recorded.
        return np.full(num_samples, 1 + 1j, dtype=np.complex64)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *a):
        self.exited = True
        return None


def _record_three_steps(tmp_path, actual_offset=0.0):
    scan_config = {"freq_start": 88e6, "freq_stop": 108e6, "step_size": 2e6}
    sdr_config = {"driver": "sdrplay", "sample_rate": 2048000}
    recorder = SigmfRecorder(tmp_path, scan_config, sdr_config, label="fm")
    inner = _FakeSDR(actual_offset=actual_offset)
    with RecordingSDR(inner, recorder) as sdr:
        for f in (89e6, 91e6, 93e6):
            sdr.tune(f)
            sdr.capture(256)
    return inner


def test_writes_step_files_and_manifest(tmp_path):
    """One recording per step plus a valid session.json manifest."""
    _record_three_steps(tmp_path)

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert len(manifest["steps"]) == 3
    assert manifest["label"] == "fm"
    assert manifest["scan_config"]["step_size"] == 2e6
    assert manifest["sdr_config"]["driver"] == "sdrplay"
    # Each step has its data file on disk.
    for step in manifest["steps"]:
        assert (tmp_path / f"{step['data_file']}.sigmf-data").exists()


def test_records_actual_tuned_frequency(tmp_path):
    """The recorded centre is the actual (offset) frequency, not requested."""
    _record_three_steps(tmp_path, actual_offset=1_000.0)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    centers = [s["center_freq_hz"] for s in manifest["steps"]]
    assert centers == [89e6 + 1000, 91e6 + 1000, 93e6 + 1000]


def test_records_raw_iq_including_dc(tmp_path):
    """Recorded IQ is the raw capture (DC offset preserved, not cleaned)."""
    _record_three_steps(tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text())
    first = manifest["steps"][0]["data_file"]
    iq = read_iq(tmp_path / first)
    assert np.allclose(iq, 1 + 1j)  # DC component intact
    meta = read_meta(tmp_path / first)
    assert meta["global"]["skadi:label"] == "fm"


def test_decorator_passes_through_and_manages_context(tmp_path):
    """Decorator forwards sample_rate/connected and enters/exits inner."""
    scan_config = {"step_size": 2e6}
    sdr_config = {"driver": "sdrplay", "sample_rate": 2048000}
    recorder = SigmfRecorder(tmp_path, scan_config, sdr_config)
    inner = _FakeSDR()
    rec = RecordingSDR(inner, recorder)
    assert isinstance(rec, SdrSource)
    assert rec.sample_rate == 2_048_000
    assert rec.connected is True
    with rec:
        rec.tune(89e6)
        rec.capture(64)
    assert inner.entered and inner.exited
