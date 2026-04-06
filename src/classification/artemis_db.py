"""Artemis signal database access layer.

Provides read-only access to the Artemis SQLite database containing
500+ known signal types. Pre-loads all records into memory at init
for fast in-memory filtering during classification.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ArtemisSignal:
    """A single signal record from the Artemis database.

    Attributes:
        id: Database primary key.
        name: Signal name (e.g. "FM Broadcast Radio").
        freq_min_hz: Minimum frequency in Hz, or None.
        freq_max_hz: Maximum frequency in Hz, or None.
        mode: Demodulation mode (USB, LSB, AM, FM, etc.).
        bandwidth_min_hz: Minimum bandwidth in Hz, or None.
        bandwidth_max_hz: Maximum bandwidth in Hz, or None.
        modulation: Raw modulation string from Artemis.
        modulation_types: Parsed list of individual modulation types.
        acf_values_ms: Parsed ACF periods in milliseconds.
        description: Human-readable signal description.
        location: Known countries/regions of origin.
    """

    id: int
    name: str
    freq_min_hz: int | None = None
    freq_max_hz: int | None = None
    mode: str | None = None
    bandwidth_min_hz: int | None = None
    bandwidth_max_hz: int | None = None
    modulation: str | None = None
    modulation_types: list[str] = field(default_factory=list)
    acf_values_ms: list[float] = field(default_factory=list)
    description: str | None = None
    location: str | None = None


class ArtemisDB:
    """Read-only access to the Artemis signal database.

    Loads all signal records into memory at initialization and provides
    filtered query methods for the classification engine.

    Args:
        db_path: Path to the Artemis SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        if not db_path.exists():
            raise FileNotFoundError(f"Artemis database not found: {db_path}")

        self._signals: list[ArtemisSignal] = []
        self._load(db_path)
        logger.info("Loaded %d signals from Artemis DB at %s", len(self._signals), db_path)

    def _load(self, db_path: Path) -> None:
        """Load all signals from the database, parsing modulation and ACF."""
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, freq_min_hz, freq_max_hz, mode,
                       bandwidth_min_hz, bandwidth_max_hz, modulation,
                       description, location, acf_value
                FROM signals
            """)
            for row in cursor.fetchall():
                mod_raw = row["modulation"]
                acf_raw = row["acf_value"]

                self._signals.append(ArtemisSignal(
                    id=row["id"],
                    name=row["name"],
                    freq_min_hz=row["freq_min_hz"],
                    freq_max_hz=row["freq_max_hz"],
                    mode=row["mode"],
                    bandwidth_min_hz=row["bandwidth_min_hz"],
                    bandwidth_max_hz=row["bandwidth_max_hz"],
                    modulation=mod_raw,
                    modulation_types=self._parse_modulation(mod_raw),
                    acf_values_ms=self._parse_acf(acf_raw),
                    description=row["description"],
                    location=row["location"],
                ))
        finally:
            conn.close()

    @property
    def signals(self) -> list[ArtemisSignal]:
        """All loaded Artemis signals."""
        return self._signals

    def query_candidates(
        self,
        modulation_terms: list[str],
        freq_hz: float | None = None,
    ) -> list[ArtemisSignal]:
        """Filter signals by modulation terms and/or frequency.

        Args:
            modulation_terms: List of Artemis modulation strings to match.
                A signal matches if any of its modulation_types intersect.
                Pass an empty list to skip modulation filtering.
            freq_hz: Optional centre frequency to filter by. Keeps signals
                whose frequency range includes this value.

        Returns:
            List of matching ArtemisSignal records.
        """
        terms_set = {t.upper().strip() for t in modulation_terms}

        results = []
        for sig in self._signals:
            # Modulation filter
            if terms_set:
                sig_mods = {m.upper().strip() for m in sig.modulation_types}
                if not terms_set.intersection(sig_mods):
                    continue

            # Frequency filter
            if freq_hz is not None:
                if sig.freq_min_hz is not None and sig.freq_max_hz is not None:
                    if not (sig.freq_min_hz <= freq_hz <= sig.freq_max_hz):
                        continue

            results.append(sig)

        return results

    @staticmethod
    def _parse_modulation(raw: str | None) -> list[str]:
        """Parse semicolon-separated modulation string into a list.

        Args:
            raw: Raw modulation string like "PSK; QAM" or None.

        Returns:
            List of individual modulation types, e.g. ["PSK", "QAM"].
        """
        if not raw or not raw.strip():
            return []
        return [part.strip() for part in raw.split(";") if part.strip()]

    @staticmethod
    def _parse_acf(raw: str | None) -> list[float]:
        """Parse ACF text field into numeric millisecond values.

        Handles formats like:
            "Main - 20" → [20.0]
            "Header - 200 ; Body - 66.66" → [200.0, 66.66]
            "Main - variable" → [] (skips non-numeric)

        Args:
            raw: Raw ACF text string or None.

        Returns:
            List of parsed ACF values in milliseconds.
        """
        if not raw or not raw.strip():
            return []

        values = []
        for segment in raw.split(";"):
            segment = segment.strip()
            # Extract number after " - "
            match = re.search(r"-\s*([\d.]+)", segment)
            if match:
                try:
                    values.append(float(match.group(1)))
                except ValueError:
                    pass
        return values
