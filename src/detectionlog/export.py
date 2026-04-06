"""JSON export for detection log.

Exports the detection log database to JSON format for external
consumption (SEIARA integration, operator review, archival).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.detectionlog.database import DetectionLog

logger = logging.getLogger(__name__)


def export_json(
    db_path: Path,
    output_path: Path,
    since: str | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    limit: int = 10_000,
) -> int:
    """Export detection log to a JSON file.

    Args:
        db_path: Path to the detections SQLite database.
        output_path: Path to write the JSON output file.
        since: Optional ISO 8601 timestamp to filter events after.
        freq_min_hz: Optional minimum frequency filter.
        freq_max_hz: Optional maximum frequency filter.
        limit: Maximum number of records to export.

    Returns:
        Number of records exported.
    """
    log = DetectionLog(db_path)
    try:
        rows = log.query(
            since=since,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            limit=limit,
        )
    finally:
        log.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    logger.info("Exported %d detection(s) to %s", len(rows), output_path)
    return len(rows)


def export_json_string(
    db_path: Path,
    since: str | None = None,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    limit: int = 10_000,
) -> str:
    """Export detection log to a JSON string.

    Args:
        db_path: Path to the detections SQLite database.
        since: Optional ISO 8601 timestamp to filter events after.
        freq_min_hz: Optional minimum frequency filter.
        freq_max_hz: Optional maximum frequency filter.
        limit: Maximum number of records to export.

    Returns:
        JSON string containing the detection records.
    """
    log = DetectionLog(db_path)
    try:
        rows = log.query(
            since=since,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            limit=limit,
        )
    finally:
        log.close()

    return json.dumps(rows, indent=2, default=str)
