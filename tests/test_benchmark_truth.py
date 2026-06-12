"""Tests for the ground-truth loader/validator."""

from __future__ import annotations

import json

import pytest

from src.benchmark.truth import (
    DEFAULT_TOLERANCE_HZ,
    BenchmarkError,
    load_truth,
)


def _write(tmp_path, obj):
    p = tmp_path / "truth.json"
    p.write_text(json.dumps(obj))
    return p


def test_valid_load(tmp_path):
    p = _write(tmp_path, {
        "match_tolerance_hz": 80000,
        "signals": [
            {"center_freq_hz": 95.1e6, "label": "A", "modulation": "fm",
             "signal_type": "FM Broadcast Radio", "threat_level": "informational"},
            {"center_freq_hz": 99.0e6},
        ],
    })
    gt = load_truth(p)
    assert len(gt.signals) == 2
    assert gt.match_tolerance_hz == 80000
    assert gt.signals[0].modulation == "FM"            # upper-cased
    assert gt.signals[0].threat_level == "INFORMATIONAL"
    assert gt.signals[1].modulation is None            # optional field absent


def test_default_tolerance_applied(tmp_path):
    p = _write(tmp_path, {"signals": [{"center_freq_hz": 95e6}]})
    assert load_truth(p).match_tolerance_hz == DEFAULT_TOLERANCE_HZ


def test_missing_file_raises(tmp_path):
    with pytest.raises(BenchmarkError, match="not found"):
        load_truth(tmp_path / "nope.json")


def test_missing_center_freq_raises(tmp_path):
    p = _write(tmp_path, {"signals": [{"label": "no freq"}]})
    with pytest.raises(BenchmarkError, match="center_freq_hz"):
        load_truth(p)


def test_empty_signals_raises(tmp_path):
    p = _write(tmp_path, {"signals": []})
    with pytest.raises(BenchmarkError, match="no 'signals'"):
        load_truth(p)


def test_bad_json_raises(tmp_path):
    p = tmp_path / "truth.json"
    p.write_text("{not json")
    with pytest.raises(BenchmarkError, match="not valid JSON"):
        load_truth(p)


def test_unknown_modulation_warns_not_raises(tmp_path, caplog):
    p = _write(tmp_path, {"signals": [{"center_freq_hz": 95e6, "modulation": "QAM"}]})
    with caplog.at_level("WARNING"):
        gt = load_truth(p)
    assert gt.signals[0].modulation == "QAM"
    assert any("unknown modulation" in r.message for r in caplog.records)
