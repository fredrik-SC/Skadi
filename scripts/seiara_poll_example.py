#!/usr/bin/env python3
"""SEIARA integration polling example.

Demonstrates how SEIARA (or any external system) can poll the Skadi
detection log database for new signal detection events. This is the
primary integration interface between Skadi and SEIARA.

The script queries the SQLite detection log at regular intervals
for entries newer than the last poll timestamp, prints them as JSON,
and tracks the polling state.

Usage:
    python scripts/seiara_poll_example.py [--db data/detections.db] [--interval 60]

Integration pattern for SEIARA:
    1. Open the Skadi detections.db file (read-only)
    2. Query: SELECT * FROM detections WHERE timestamp_utc > ? ORDER BY timestamp_utc
    3. Process new detections (correlate, assess risk, display)
    4. Update the last-seen timestamp
    5. Repeat at polling interval (~60 seconds)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectionlog.database import DetectionLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def poll_loop(db_path: Path, interval: float = 60.0) -> None:
    """Poll the detection log for new entries.

    Args:
        db_path: Path to the Skadi detections.db file.
        interval: Polling interval in seconds.
    """
    if not db_path.exists():
        logger.error("Detection database not found: %s", db_path)
        logger.info("Start Skadi first: python -m src.main --preset vhf")
        return

    # Start polling from now (or from the beginning if --all is used)
    last_poll = datetime.now(timezone.utc).isoformat()
    total_received = 0

    logger.info("SEIARA polling started")
    logger.info("  Database: %s", db_path)
    logger.info("  Interval: %.0f seconds", interval)
    logger.info("  Polling from: %s", last_poll)
    logger.info("")

    try:
        while True:
            log = DetectionLog(db_path)
            try:
                new_detections = log.query(since=last_poll, limit=1000)
            finally:
                log.close()

            if new_detections:
                total_received += len(new_detections)
                logger.info(
                    "Received %d new detection(s) (total: %d)",
                    len(new_detections), total_received,
                )

                # Output as JSON (this is what SEIARA would consume)
                for det in new_detections:
                    print(json.dumps(det, indent=None, default=str))

                # Update poll timestamp to the newest detection
                last_poll = new_detections[0]["timestamp_utc"]
            else:
                logger.debug("No new detections since %s", last_poll)

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Polling stopped. Total detections received: %d", total_received)


def main() -> None:
    """Entry point for the SEIARA polling example."""
    project_root = Path(__file__).resolve().parent.parent
    default_db = project_root / "data" / "detections.db"

    parser = argparse.ArgumentParser(
        description="SEIARA integration — poll Skadi detection log for new events",
    )
    parser.add_argument(
        "--db", type=Path, default=default_db,
        help=f"Path to detections.db (default: {default_db})",
    )
    parser.add_argument(
        "--interval", type=float, default=60.0,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Start from the beginning (show all existing detections)",
    )
    args = parser.parse_args()

    if args.all:
        # Show existing detections first
        if args.db.exists():
            log = DetectionLog(args.db)
            existing = log.query(limit=10000)
            log.close()
            logger.info("Existing detections: %d", len(existing))
            for det in existing:
                print(json.dumps(det, indent=None, default=str))

    poll_loop(args.db, args.interval)


if __name__ == "__main__":
    main()
