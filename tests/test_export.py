"""Tests for JSON export module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.models import DetectedSignal
from src.detectionlog.database import DetectionLog
from src.detectionlog.export import export_json, export_json_string


def _make_signal(freq_hz: float = 100e6, timestamp: float = 1712400000.0) -> DetectedSignal:
    return DetectedSignal(
        centre_freq_hz=freq_hz,
        bandwidth_hz=200_000,
        peak_power_dbm=-70.0,
        mean_power_dbm=-75.0,
        snr_db=20.0,
        timestamp=timestamp,
        scan_step_freq_hz=freq_hz,
    )


class TestJSONExport:
    """Tests for JSON export functions."""

    def test_export_produces_valid_json(self, tmp_path):
        """Exported file is valid JSON."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.log_signal(_make_signal())
        log.close()

        output = tmp_path / "export.json"
        count = export_json(db_path, output)
        assert count == 1
        assert output.exists()

        with output.open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_contains_all_fields(self, tmp_path):
        """Exported records contain all v1.0 schema fields."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.log_signal(
            _make_signal(),
            modulation="FM",
            signal_type="FM Broadcast Radio",
            confidence_score=0.95,
            threat_level="INFORMATIONAL",
        )
        log.close()

        output = tmp_path / "export.json"
        export_json(db_path, output)

        with output.open() as f:
            data = json.load(f)

        record = data[0]
        expected_fields = [
            "id", "timestamp_utc", "frequency_hz", "bandwidth_hz",
            "modulation", "signal_strength_dbm", "signal_type",
            "confidence_score", "threat_level",
        ]
        for field in expected_fields:
            assert field in record, f"Missing field: {field}"

    def test_export_empty_db(self, tmp_path):
        """Empty database exports empty array."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.close()

        output = tmp_path / "export.json"
        count = export_json(db_path, output)
        assert count == 0

        with output.open() as f:
            data = json.load(f)
        assert data == []

    def test_export_frequency_filter(self, tmp_path):
        """Frequency filter works in export."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.log_signal(_make_signal(freq_hz=88e6))
        log.log_signal(_make_signal(freq_hz=100e6))
        log.log_signal(_make_signal(freq_hz=200e6))
        log.close()

        output = tmp_path / "export.json"
        count = export_json(db_path, output, freq_min_hz=90e6, freq_max_hz=150e6)
        assert count == 1

    def test_export_json_string(self, tmp_path):
        """String export returns valid JSON."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.log_signal(_make_signal())
        log.close()

        json_str = export_json_string(db_path)
        data = json.loads(json_str)
        assert len(data) == 1
