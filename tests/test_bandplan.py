"""Tests for the BandPlan classification-prior lookup."""

from __future__ import annotations

import json

import pytest

from src.classification.bandplan import BandPlan
from src.config import PROJECT_ROOT


@pytest.fixture
def band_plan():
    return BandPlan(PROJECT_ROOT / "config" / "band_plan.yaml")


def test_loads_entries(band_plan):
    assert len(band_plan.entries) > 0


def test_lookup_fm_broadcast(band_plan):
    entry = band_plan.lookup(98_000_000)
    assert entry is not None
    assert "FM Broadcast" in entry.service
    assert "FM" in entry.expected_modulations


def test_lookup_airband_am(band_plan):
    entry = band_plan.lookup(124_000_000)
    assert entry is not None
    assert "AM" in entry.expected_modulations
    assert "Airband" in entry.service or "Aeronautical" in entry.service


def test_lookup_2m_amateur(band_plan):
    entry = band_plan.lookup(145_000_000)
    assert entry is not None
    assert "Amateur" in entry.service


def test_lookup_gap_returns_none(band_plan):
    # 80 MHz is between FM broadcast and nothing defined below it.
    assert band_plan.lookup(80_000_000) is None


def test_no_path_is_empty():
    bp = BandPlan()
    assert bp.entries == []
    assert bp.lookup(98_000_000) is None


def test_first_match_wins(tmp_path):
    p = tmp_path / "bp.yaml"
    p.write_text(json.dumps({"bands": [
        {"freq_start_hz": 118e6, "freq_stop_hz": 137e6, "service": "Airband",
         "expected_modulations": ["AM"]},
        {"freq_start_hz": 100e6, "freq_stop_hz": 200e6, "service": "Broad",
         "expected_modulations": ["FM"]},
    ]}))
    bp = BandPlan(p)
    # 124 MHz is in both; the narrower (first-listed) Airband wins.
    assert bp.lookup(124e6).service == "Airband"


def test_invalid_entry_skipped(tmp_path, caplog):
    p = tmp_path / "bp.yaml"
    p.write_text(json.dumps({"bands": [
        {"service": "missing freqs"},
        {"freq_start_hz": 118e6, "freq_stop_hz": 137e6, "service": "OK",
         "expected_modulations": ["AM"]},
    ]}))
    with caplog.at_level("WARNING"):
        bp = BandPlan(p)
    assert len(bp.entries) == 1
