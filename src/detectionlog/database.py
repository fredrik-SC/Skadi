"""SQLite detection log for recording signal detection events.

Implements the v1.0 detection log schema from PRD Section 3.6.1.
Classification fields (modulation, signal_type, threat_level, etc.)
are nullable and will be populated in later sessions.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.detection.models import DetectedSignal

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    frequency_hz REAL NOT NULL,
    bandwidth_hz REAL NOT NULL,
    modulation TEXT,
    signal_strength_dbm REAL NOT NULL,
    signal_type TEXT,
    confidence_score REAL,
    alt_match_1 TEXT,
    alt_match_1_confidence REAL,
    alt_match_2 TEXT,
    alt_match_2_confidence REAL,
    known_users TEXT,
    threat_level TEXT,
    acf_value REAL,
    notes TEXT
)
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_detections_frequency ON detections(frequency_hz)",
]

_INSERT_SQL = """
INSERT INTO detections (
    timestamp_utc, frequency_hz, bandwidth_hz, modulation,
    signal_strength_dbm, signal_type, confidence_score,
    alt_match_1, alt_match_1_confidence,
    alt_match_2, alt_match_2_confidence,
    known_users, threat_level, acf_value, notes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class DetectionLog:
    """SQLite-backed detection event log.

    Records all signal detection events with full metadata as defined
    in PRD Section 3.6.1. Serves as the integration point for SEIARA.

    Args:
        db_path: Path to the SQLite database file. Created automatically
            if it does not exist.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info("Detection log opened at %s", db_path)

    def _ensure_schema(self) -> None:
        """Create the detections table and indexes if they don't exist."""
        cursor = self._conn.cursor()
        cursor.execute(_CREATE_TABLE_SQL)
        for index_sql in _CREATE_INDEXES_SQL:
            cursor.execute(index_sql)
        self._conn.commit()

    def log_signal(
        self,
        signal: DetectedSignal,
        modulation: str | None = None,
        signal_type: str | None = None,
        confidence_score: float | None = None,
        alt_match_1: str | None = None,
        alt_match_1_confidence: float | None = None,
        alt_match_2: str | None = None,
        alt_match_2_confidence: float | None = None,
        known_users: str | None = None,
        threat_level: str | None = None,
        acf_value: float | None = None,
        notes: str | None = None,
    ) -> int:
        """Insert a single detection event.

        Args:
            signal: The detected signal with measured parameters.
            modulation: Detected modulation type (Session 4+).
            signal_type: Best Artemis DB match (Session 5+).
            confidence_score: Classification confidence (Session 5+).
            alt_match_1: Second-best match (Session 5+).
            alt_match_1_confidence: Second match confidence.
            alt_match_2: Third-best match (Session 5+).
            alt_match_2_confidence: Third match confidence.
            known_users: Known operators from Artemis (Session 5+).
            threat_level: Assigned threat level (Session 6+).
            acf_value: Measured ACF value (Session 4+).
            notes: Optional operator notes.

        Returns:
            The row ID of the inserted detection.
        """
        timestamp_utc = datetime.fromtimestamp(
            signal.timestamp, tz=timezone.utc
        ).isoformat()

        cursor = self._conn.cursor()
        cursor.execute(_INSERT_SQL, (
            timestamp_utc,
            signal.centre_freq_hz,
            signal.bandwidth_hz,
            modulation,
            signal.peak_power_dbm,
            signal_type,
            confidence_score,
            alt_match_1,
            alt_match_1_confidence,
            alt_match_2,
            alt_match_2_confidence,
            known_users,
            threat_level,
            acf_value,
            notes,
        ))
        self._conn.commit()
        return cursor.lastrowid

    def log_signals(self, signals: list[DetectedSignal]) -> list[int]:
        """Insert multiple detection events in a single transaction.

        Args:
            signals: List of detected signals to log.

        Returns:
            List of row IDs for the inserted detections.
        """
        row_ids = []
        cursor = self._conn.cursor()

        for signal in signals:
            timestamp_utc = datetime.fromtimestamp(
                signal.timestamp, tz=timezone.utc
            ).isoformat()

            cursor.execute(_INSERT_SQL, (
                timestamp_utc,
                signal.centre_freq_hz,
                signal.bandwidth_hz,
                None,  # modulation
                signal.peak_power_dbm,
                None,  # signal_type
                None,  # confidence_score
                None,  # alt_match_1
                None,  # alt_match_1_confidence
                None,  # alt_match_2
                None,  # alt_match_2_confidence
                None,  # known_users
                None,  # threat_level
                None,  # acf_value
                None,  # notes
            ))
            row_ids.append(cursor.lastrowid)

        self._conn.commit()
        logger.info("Logged %d detection(s) to database", len(row_ids))
        return row_ids

    def query(
        self,
        since: str | None = None,
        freq_min_hz: float | None = None,
        freq_max_hz: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query detection events with optional filters.

        Args:
            since: ISO 8601 timestamp — only return events after this time.
            freq_min_hz: Minimum frequency filter in Hz.
            freq_max_hz: Maximum frequency filter in Hz.
            limit: Maximum number of results to return.

        Returns:
            List of detection event dictionaries, ordered by timestamp
            descending.
        """
        conditions = []
        params: list[Any] = []

        if since is not None:
            conditions.append("timestamp_utc >= ?")
            params.append(since)
        if freq_min_hz is not None:
            conditions.append("frequency_hz >= ?")
            params.append(freq_min_hz)
        if freq_max_hz is not None:
            conditions.append("frequency_hz <= ?")
            params.append(freq_max_hz)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM detections
            {where_clause}
            ORDER BY timestamp_utc DESC
            LIMIT ?
        """
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Return total number of detection events."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM detections")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("Detection log closed")
