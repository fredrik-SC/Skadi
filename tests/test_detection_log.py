"""Tests for detection log database."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.detection.models import DetectedSignal
from src.detectionlog.database import DetectionLog


def _make_signal(
    freq_hz: float = 100e6,
    bandwidth_hz: float = 200_000,
    peak_power: float = -70.0,
    timestamp: float = 1712400000.0,  # 2024-04-06T12:00:00Z
) -> DetectedSignal:
    """Helper to create a DetectedSignal for testing."""
    return DetectedSignal(
        centre_freq_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        peak_power_dbm=peak_power,
        mean_power_dbm=peak_power + 5.0,
        snr_db=20.0,
        timestamp=timestamp,
        scan_step_freq_hz=freq_hz,
    )


class TestDetectionLog:
    """Tests for DetectionLog."""

    def test_create_table_on_init(self, tmp_path):
        """Database and table are created automatically."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        assert db_path.exists()

        # Verify table exists by querying it
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections'")
        assert cursor.fetchone() is not None
        conn.close()
        log.close()

    def test_log_signal_returns_row_id(self, tmp_path):
        """Inserting one signal returns an integer row ID."""
        log = DetectionLog(tmp_path / "test.db")
        row_id = log.log_signal(_make_signal())
        assert isinstance(row_id, int)
        assert row_id >= 1
        log.close()

    def test_log_signals_batch(self, tmp_path):
        """Batch insert returns correct number of row IDs."""
        log = DetectionLog(tmp_path / "test.db")
        signals = [_make_signal(freq_hz=f) for f in [100e6, 101e6, 102e6]]
        row_ids = log.log_signals(signals)
        assert len(row_ids) == 3
        assert log.count() == 3
        log.close()

    def test_timestamp_is_utc_iso8601(self, tmp_path):
        """Stored timestamp is in UTC ISO 8601 format."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signal(_make_signal(timestamp=1712400000.0))
        rows = log.query()
        assert len(rows) == 1
        ts = rows[0]["timestamp_utc"]
        # Should parse as a valid ISO 8601 datetime
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None  # Has timezone info
        log.close()

    def test_query_returns_all(self, tmp_path):
        """Query with no filters returns all entries."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signals([_make_signal(freq_hz=f) for f in [100e6, 101e6, 102e6]])
        rows = log.query()
        assert len(rows) == 3
        log.close()

    def test_query_since_filter(self, tmp_path):
        """Since filter only returns events after the given timestamp."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signal(_make_signal(timestamp=1712400000.0))  # Earlier
        log.log_signal(_make_signal(timestamp=1712500000.0))  # Later

        # Query for events after the midpoint
        midpoint = datetime.fromtimestamp(1712450000.0, tz=timezone.utc).isoformat()
        rows = log.query(since=midpoint)
        assert len(rows) == 1
        log.close()

    def test_query_frequency_filter(self, tmp_path):
        """Frequency range filter works."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signals([
            _make_signal(freq_hz=88e6),
            _make_signal(freq_hz=100e6),
            _make_signal(freq_hz=200e6),
        ])
        rows = log.query(freq_min_hz=90e6, freq_max_hz=150e6)
        assert len(rows) == 1
        assert rows[0]["frequency_hz"] == 100e6
        log.close()

    def test_classification_fields_are_null(self, tmp_path):
        """Classification fields are NULL for basic signal logging."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signal(_make_signal())
        rows = log.query()
        row = rows[0]
        assert row["modulation"] is None
        assert row["signal_type"] is None
        assert row["confidence_score"] is None
        assert row["threat_level"] is None
        assert row["acf_value"] is None
        log.close()

    def test_log_signal_with_classification(self, tmp_path):
        """Classification fields can be populated via log_signal kwargs."""
        log = DetectionLog(tmp_path / "test.db")
        log.log_signal(
            _make_signal(),
            modulation="FSK",
            signal_type="Test Signal",
            confidence_score=0.85,
            threat_level="LOW",
        )
        rows = log.query()
        row = rows[0]
        assert row["modulation"] == "FSK"
        assert row["signal_type"] == "Test Signal"
        assert row["confidence_score"] == pytest.approx(0.85)
        assert row["threat_level"] == "LOW"
        log.close()

    def test_count(self, tmp_path):
        """Count returns correct total."""
        log = DetectionLog(tmp_path / "test.db")
        assert log.count() == 0
        log.log_signals([_make_signal(), _make_signal()])
        assert log.count() == 2
        log.close()

    def test_data_persists_across_close_reopen(self, tmp_path):
        """Data persists after closing and reopening."""
        db_path = tmp_path / "test.db"
        log = DetectionLog(db_path)
        log.log_signal(_make_signal())
        log.close()

        log2 = DetectionLog(db_path)
        assert log2.count() == 1
        log2.close()
