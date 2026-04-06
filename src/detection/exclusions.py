"""Configurable exclusion list for known/benign signals.

Filters detected signals against a list of known frequencies that
should be ignored during scanning (e.g., local FM stations, known
infrastructure signals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.detection.models import DetectedSignal

logger = logging.getLogger(__name__)


@dataclass
class ExclusionEntry:
    """A single exclusion list entry.

    Attributes:
        freq_hz: Centre frequency to exclude in Hz.
        bandwidth_hz: Bandwidth of the exclusion zone in Hz.
            If 0, treated as a point exclusion — any signal whose band
            contains this frequency will be excluded.
        label: Human-readable label for logging.
    """

    freq_hz: float
    bandwidth_hz: float = 0.0
    label: str = ""


class ExclusionFilter:
    """Filters detected signals against a configurable exclusion list.

    Signals whose frequency band overlaps any exclusion entry's frequency
    band are removed from detection results.

    Args:
        exclusions_path: Path to exclusions.yaml. If None, starts with
            an empty exclusion list.
    """

    def __init__(self, exclusions_path: Path | None = None) -> None:
        self._entries: list[ExclusionEntry] = []
        if exclusions_path is not None:
            self.load(exclusions_path)

    @property
    def entries(self) -> list[ExclusionEntry]:
        """The currently loaded exclusion entries."""
        return list(self._entries)

    def load(self, exclusions_path: Path) -> None:
        """Load exclusion entries from a YAML file.

        Args:
            exclusions_path: Path to the exclusions YAML file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not exclusions_path.exists():
            raise FileNotFoundError(f"Exclusions file not found: {exclusions_path}")

        with exclusions_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_entries = (data or {}).get("exclusions") or []
        self._entries = []
        for entry in raw_entries:
            if not isinstance(entry, dict) or "freq_hz" not in entry:
                logger.warning("Skipping invalid exclusion entry: %s", entry)
                continue
            self._entries.append(ExclusionEntry(
                freq_hz=float(entry["freq_hz"]),
                bandwidth_hz=float(entry.get("bandwidth_hz", 0)),
                label=str(entry.get("label", "")),
            ))

        logger.info("Loaded %d exclusion entries from %s", len(self._entries), exclusions_path)

    def is_excluded(self, freq_hz: float, bandwidth_hz: float = 0.0) -> bool:
        """Check whether a frequency/bandwidth overlaps any exclusion.

        Args:
            freq_hz: Centre frequency to check in Hz.
            bandwidth_hz: Signal bandwidth in Hz. If 0, treated as a point.

        Returns:
            True if the frequency band overlaps any exclusion entry.
        """
        sig_lower = freq_hz - bandwidth_hz / 2
        sig_upper = freq_hz + bandwidth_hz / 2

        for entry in self._entries:
            if entry.bandwidth_hz > 0:
                excl_lower = entry.freq_hz - entry.bandwidth_hz / 2
                excl_upper = entry.freq_hz + entry.bandwidth_hz / 2
            else:
                # Point exclusion — check if signal band contains the point
                excl_lower = entry.freq_hz
                excl_upper = entry.freq_hz

            # Bands overlap if they are not disjoint
            if sig_upper >= excl_lower and sig_lower <= excl_upper:
                return True

        return False

    def filter(self, signals: list[DetectedSignal]) -> list[DetectedSignal]:
        """Return only signals that are NOT excluded.

        Args:
            signals: List of detected signals to filter.

        Returns:
            Filtered list with excluded signals removed.
        """
        if not self._entries:
            return signals

        kept = []
        for signal in signals:
            if self.is_excluded(signal.centre_freq_hz, signal.bandwidth_hz):
                logger.debug(
                    "Excluded signal at %.3f MHz (BW=%.1f kHz)",
                    signal.centre_freq_hz / 1e6, signal.bandwidth_hz / 1e3,
                )
            else:
                kept.append(signal)

        excluded_count = len(signals) - len(kept)
        if excluded_count > 0:
            logger.info("Excluded %d of %d signals", excluded_count, len(signals))

        return kept
