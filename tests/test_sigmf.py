"""Tests for the minimal SigMF reader/writer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.sdr import sigmf
from tests.conftest import generate_test_iq


def test_roundtrip_bit_exact(tmp_path):
    """IQ written and read back is bit-for-bit identical complex64."""
    iq = generate_test_iq(tones=[(50_000, 0.5)], num_samples=2048).astype(np.complex64)
    base = tmp_path / "rec"

    sigmf.write_recording(
        base, iq,
        sample_rate=2_048_000,
        center_freq_hz=100e6,
        iso_timestamp="2026-06-12T10:00:00+00:00",
    )
    back = sigmf.read_iq(base)

    assert back.dtype == np.complex64
    assert np.array_equal(iq, back)


def test_meta_fields(tmp_path):
    """Meta carries the SigMF core fields plus skadi label/driver."""
    iq = np.ones(16, dtype=np.complex64)
    base = tmp_path / "rec"

    sigmf.write_recording(
        base, iq,
        sample_rate=2_048_000,
        center_freq_hz=146.5e6,
        iso_timestamp="2026-06-12T10:00:00+00:00",
        label="VHF tactical",
        driver="sdrplay",
    )
    meta = sigmf.read_meta(base)

    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["core:sample_rate"] == 2_048_000
    assert meta["global"]["skadi:label"] == "VHF tactical"
    assert meta["global"]["skadi:driver"] == "sdrplay"
    assert meta["captures"][0]["core:frequency"] == 146.5e6
    assert meta["captures"][0]["core:datetime"] == "2026-06-12T10:00:00+00:00"


def test_files_written_with_sigmf_extensions(tmp_path):
    """Both .sigmf-data and .sigmf-meta are produced; meta is valid JSON."""
    base = tmp_path / "rec"
    sigmf.write_recording(
        base, np.zeros(8, dtype=np.complex64),
        sample_rate=1e6, center_freq_hz=1e6,
        iso_timestamp="2026-06-12T10:00:00+00:00",
    )
    assert (tmp_path / "rec.sigmf-data").exists()
    assert (tmp_path / "rec.sigmf-meta").exists()
    json.loads((tmp_path / "rec.sigmf-meta").read_text())  # parses


def test_path_accessors_accept_either_member(tmp_path):
    """Base path helpers accept the base or either file path."""
    base = tmp_path / "rec"
    assert sigmf.data_path_for(base).name == "rec.sigmf-data"
    assert sigmf.meta_path_for(base).name == "rec.sigmf-meta"
    # Passing the data file should resolve back to the same pair.
    assert sigmf.meta_path_for(base.with_name("rec.sigmf-data")).name == "rec.sigmf-meta"
