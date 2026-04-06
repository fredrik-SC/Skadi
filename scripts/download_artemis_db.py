#!/usr/bin/env python3
"""Download and convert Artemis signal database to SQLite.

Fetches the Artemis DB CSV from the AresValley website, parses the
*-delimited format, and stores it as a SQLite database for offline
signal classification.

Usage:
    python scripts/download_artemis_db.py [--output data/artemis.db] [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARTEMIS_DATA_URL = "https://aresvalley.com/Storage/Artemis/Database/data.zip"
CSV_PATH_IN_ZIP = "Data/db.csv"

# Column mapping: CSV field index -> (db_column_name, db_type, parser)
COLUMNS = [
    ("name", "TEXT NOT NULL", str),
    ("freq_min_hz", "INTEGER", lambda v: int(v) if v.strip() else None),
    ("freq_max_hz", "INTEGER", lambda v: int(v) if v.strip() else None),
    ("mode", "TEXT", lambda v: v.strip() or None),
    ("bandwidth_min_hz", "INTEGER", lambda v: int(v) if v.strip() else None),
    ("bandwidth_max_hz", "INTEGER", lambda v: int(v) if v.strip() else None),
    ("location", "TEXT", lambda v: v.strip() or None),
    ("wiki_url", "TEXT", lambda v: v.strip() or None),
    ("description", "TEXT", lambda v: v.strip() or None),
    ("modulation", "TEXT", lambda v: v.strip() or None),
    ("category_bitmap", "TEXT", lambda v: v.strip() or None),
    ("acf_value", "TEXT", lambda v: v.strip() or None),
]


def download_zip(url: str) -> Path:
    """Download the Artemis data.zip to a temporary file.

    Args:
        url: URL of the data.zip file.

    Returns:
        Path to the downloaded temporary file.
    """
    logger.info("Downloading Artemis database from %s", url)
    request = Request(url, headers={"User-Agent": "Skadi/1.0"})
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(tmp_path_str)
    try:
        with urlopen(request) as response, open(tmp_fd, "wb") as out_file:
            out_file.write(response.read())
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Downloaded to %s (%.1f MB)", tmp_path, tmp_path.stat().st_size / 1e6)
    return tmp_path


def extract_csv(zip_path: Path) -> str:
    """Extract db.csv content from the zip file.

    Args:
        zip_path: Path to the downloaded zip file.

    Returns:
        Content of db.csv as a string.

    Raises:
        KeyError: If db.csv is not found in the archive.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(CSV_PATH_IN_ZIP) as f:
            content = f.read().decode("utf-8")
    logger.info("Extracted %s (%d bytes)", CSV_PATH_IN_ZIP, len(content))
    return content


def parse_csv(csv_content: str) -> list[dict]:
    """Parse the *-delimited Artemis CSV into a list of signal dicts.

    Args:
        csv_content: Raw CSV content string.

    Returns:
        List of dictionaries, one per signal, with parsed values.
    """
    signals = []
    reader = csv.reader(io.StringIO(csv_content), delimiter="*")

    for row_num, row in enumerate(reader, 1):
        if len(row) < len(COLUMNS):
            logger.warning("Row %d has only %d fields, skipping", row_num, len(row))
            continue

        signal = {}
        for i, (col_name, _, parser) in enumerate(COLUMNS):
            try:
                signal[col_name] = parser(row[i])
            except (ValueError, IndexError) as e:
                logger.warning(
                    "Row %d, column '%s': parse error (%s), setting to None",
                    row_num, col_name, e,
                )
                signal[col_name] = None

        signals.append(signal)

    logger.info("Parsed %d signal records", len(signals))
    return signals


def create_database(signals: list[dict], db_path: Path) -> None:
    """Create SQLite database with the signals table.

    Args:
        signals: List of parsed signal dictionaries.
        db_path: Path where the SQLite database will be created.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB if present
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        # Create table
        col_defs = ",\n            ".join(
            f"{name} {dtype}" for name, dtype, _ in COLUMNS
        )
        cursor.execute(f"""
            CREATE TABLE signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {col_defs}
            )
        """)

        # Create indexes for classification queries
        cursor.execute(
            "CREATE INDEX idx_freq_range ON signals(freq_min_hz, freq_max_hz)"
        )
        cursor.execute(
            "CREATE INDEX idx_modulation ON signals(modulation)"
        )
        cursor.execute(
            "CREATE INDEX idx_mode ON signals(mode)"
        )

        # Insert signals
        col_names = [name for name, _, _ in COLUMNS]
        placeholders = ", ".join(["?"] * len(col_names))
        insert_sql = (
            f"INSERT INTO signals ({', '.join(col_names)}) "
            f"VALUES ({placeholders})"
        )

        for signal in signals:
            values = [signal[col] for col in col_names]
            cursor.execute(insert_sql, values)

        conn.commit()
        logger.info("Created database at %s with %d signals", db_path, len(signals))

    finally:
        conn.close()


def print_summary(db_path: Path) -> None:
    """Print a summary of the database contents."""
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        total = cursor.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        with_mod = cursor.execute(
            "SELECT COUNT(*) FROM signals WHERE modulation IS NOT NULL"
        ).fetchone()[0]
        with_acf = cursor.execute(
            "SELECT COUNT(*) FROM signals WHERE acf_value IS NOT NULL"
        ).fetchone()[0]

        # Top modulation types
        mod_counts = cursor.execute("""
            SELECT modulation, COUNT(*) as cnt
            FROM signals
            WHERE modulation IS NOT NULL
            GROUP BY modulation
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()

        print("\n" + "=" * 50)
        print("ARTEMIS DATABASE SUMMARY")
        print("=" * 50)
        print(f"  Total signals:           {total}")
        print(f"  With modulation data:    {with_mod}")
        print(f"  With ACF values:         {with_acf}")
        print(f"  Database size:           {db_path.stat().st_size / 1024:.1f} KB")
        print(f"\n  Top modulation types:")
        for mod, count in mod_counts:
            print(f"    {mod:<20s}  {count:>4d}")
        print("=" * 50)

    finally:
        conn.close()


def main() -> None:
    """Download, parse, and store the Artemis database."""
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "artemis.db"

    parser = argparse.ArgumentParser(
        description="Download and convert Artemis signal database to SQLite"
    )
    parser.add_argument(
        "--output", type=Path, default=default_output,
        help=f"Output SQLite database path (default: {default_output})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing database without prompting",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"Database already exists at {args.output}")
        print("Use --force to overwrite.")
        return

    # Download
    zip_path = download_zip(ARTEMIS_DATA_URL)

    try:
        # Extract and parse
        csv_content = extract_csv(zip_path)
        signals = parse_csv(csv_content)

        # Store
        create_database(signals, args.output)

        # Summary
        print_summary(args.output)

    finally:
        # Clean up temp file
        zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
