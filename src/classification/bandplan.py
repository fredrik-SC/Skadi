"""Frequency band plan used as a classification prior.

Maps frequency ranges to the service expected there and the modulations typical
of it (ITU Region 1 / Sweden-Denmark focus). The classifier consults this to
mildly boost band-consistent candidates and to flag signals whose modulation is
unexpected for the band. It informs and flags — it never vetoes — because an
unexpected mode in a band (a data mode where only voice is expected) is exactly
the kind of anomaly the tool exists to surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BandEntry:
    """A single band-plan entry.

    Attributes:
        freq_start_hz: Lower bound of the band in Hz (inclusive).
        freq_stop_hz: Upper bound of the band in Hz (inclusive).
        service: Human-readable service name (e.g. "Aeronautical Mobile").
        expected_modulations: Modulations typical of this band (upper-case).
        typical_bandwidth_min_hz: Lower bound of typical bandwidth, or None.
        typical_bandwidth_max_hz: Upper bound of typical bandwidth, or None.
        notes: Free-text notes.
    """

    freq_start_hz: float
    freq_stop_hz: float
    service: str
    expected_modulations: list[str] = field(default_factory=list)
    typical_bandwidth_min_hz: float | None = None
    typical_bandwidth_max_hz: float | None = None
    notes: str = ""

    def contains(self, freq_hz: float) -> bool:
        """Whether ``freq_hz`` falls within this band."""
        return self.freq_start_hz <= freq_hz <= self.freq_stop_hz


class BandPlan:
    """Frequency allocation lookup loaded from a YAML band plan.

    Args:
        band_plan_path: Path to band_plan.yaml. If None, starts empty (every
            lookup returns None, so the prior is a no-op).
    """

    def __init__(self, band_plan_path: Path | None = None) -> None:
        self._entries: list[BandEntry] = []
        if band_plan_path is not None:
            self.load(band_plan_path)

    @property
    def entries(self) -> list[BandEntry]:
        """The loaded band entries."""
        return list(self._entries)

    def load(self, band_plan_path: Path) -> None:
        """Load band entries from a YAML file.

        Args:
            band_plan_path: Path to the band plan YAML.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not band_plan_path.exists():
            raise FileNotFoundError(f"Band plan not found: {band_plan_path}")

        with band_plan_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw = (data or {}).get("bands") or []
        self._entries = []
        for entry in raw:
            if (not isinstance(entry, dict)
                    or "freq_start_hz" not in entry
                    or "freq_stop_hz" not in entry):
                logger.warning("Skipping invalid band-plan entry: %s", entry)
                continue
            mods = [str(m).strip().upper() for m in entry.get("expected_modulations", [])]
            self._entries.append(BandEntry(
                freq_start_hz=float(entry["freq_start_hz"]),
                freq_stop_hz=float(entry["freq_stop_hz"]),
                service=str(entry.get("service", "")),
                expected_modulations=mods,
                typical_bandwidth_min_hz=_opt_float(entry.get("typical_bandwidth_min_hz")),
                typical_bandwidth_max_hz=_opt_float(entry.get("typical_bandwidth_max_hz")),
                notes=str(entry.get("notes", "")),
            ))

        logger.info("Loaded %d band-plan entries from %s", len(self._entries), band_plan_path)

    def lookup(self, freq_hz: float) -> BandEntry | None:
        """Return the first band containing ``freq_hz``, or None.

        First match wins, so list narrower/more-specific bands earlier.
        """
        for entry in self._entries:
            if entry.contains(freq_hz):
                return entry
        return None


def _opt_float(value: object) -> float | None:
    """Coerce an optional value to float, or None."""
    return None if value is None else float(value)
