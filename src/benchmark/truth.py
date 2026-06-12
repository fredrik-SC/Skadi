"""Ground-truth schema and loader for the benchmark harness.

Ground truth for a recorded session lives in a hand-editable ``truth.json``
inside the session directory, which keeps the SigMF recording immutable. Each
entry describes a signal known to be present, by centre frequency plus optional
modulation, Artemis signal type, bandwidth, and threat level. Entries missing an
optional field simply skip that scoring dimension while still counting toward
detection recall.

(A future-portable alternative is to embed the same fields as SigMF
``annotations`` on each recording; ``truth.json`` is used here because it is one
editable file per session and needs no SigMF tooling.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.classification.threat import THREAT_LEVELS
from src.fingerprint.models import ModulationType

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_HZ = 100_000.0
SCHEMA_VERSION = 1

# Valid modulation labels (the ModulationType value set).
_MODULATIONS = {m.value for m in ModulationType}


class BenchmarkError(ValueError):
    """Raised when a truth file is missing or fails schema validation."""


@dataclass(frozen=True)
class GroundTruthSignal:
    """A single known signal in a recorded session.

    Attributes:
        center_freq_hz: Centre frequency in Hz (the only required field).
        label: Optional human-readable name (e.g. a station call sign).
        modulation: Optional expected modulation, normalised upper-case
            (e.g. "FM"). Scored against the fingerprint when present.
        signal_type: Optional Artemis signal name (e.g. "FM Broadcast Radio").
            Scored against the classifier when present.
        bandwidth_hz: Optional expected bandwidth in Hz.
        threat_level: Optional expected threat level.
        verified: Whether this entry has been confirmed against an external
            source. The seeder writes ``False``; hand-verified entries are ``True``.
    """

    center_freq_hz: float
    label: str | None = None
    modulation: str | None = None
    signal_type: str | None = None
    bandwidth_hz: float | None = None
    threat_level: str | None = None
    verified: bool = True


@dataclass(frozen=True)
class GroundTruth:
    """The full ground-truth label set for one recorded session.

    Attributes:
        signals: The known signals.
        match_tolerance_hz: Maximum centre-frequency distance for a detection
            to be considered a match to a truth signal.
        session: Optional session name (informational).
    """

    signals: list[GroundTruthSignal]
    match_tolerance_hz: float = DEFAULT_TOLERANCE_HZ
    session: str | None = None


def load_truth(path: Path) -> GroundTruth:
    """Load and validate a ``truth.json`` file.

    Args:
        path: Path to the truth JSON file.

    Returns:
        The parsed :class:`GroundTruth`.

    Raises:
        BenchmarkError: If the file is missing, not valid JSON, or fails schema
            validation (no ``signals``, or a signal lacking ``center_freq_hz``).
    """
    path = Path(path)
    if not path.exists():
        raise BenchmarkError(f"Truth file not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise BenchmarkError(f"Truth file {path} is not valid JSON: {e}") from e

    if not isinstance(raw, dict):
        raise BenchmarkError(f"Truth file {path} must contain a JSON object")

    raw_signals = raw.get("signals")
    if not raw_signals:
        raise BenchmarkError(f"Truth file {path} has no 'signals'")
    if not isinstance(raw_signals, list):
        raise BenchmarkError(f"Truth file {path} 'signals' must be a list")

    signals: list[GroundTruthSignal] = []
    for i, entry in enumerate(raw_signals):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"{path}: signal {i} must be an object")
        if "center_freq_hz" not in entry:
            raise BenchmarkError(f"{path}: signal {i} missing 'center_freq_hz'")
        try:
            center = float(entry["center_freq_hz"])
        except (TypeError, ValueError) as e:
            raise BenchmarkError(
                f"{path}: signal {i} has non-numeric 'center_freq_hz'"
            ) from e

        signals.append(GroundTruthSignal(
            center_freq_hz=center,
            label=entry.get("label"),
            modulation=_norm_modulation(entry.get("modulation"), path, i),
            signal_type=entry.get("signal_type"),
            bandwidth_hz=_opt_float(entry.get("bandwidth_hz")),
            threat_level=_norm_threat(entry.get("threat_level"), path, i),
            verified=bool(entry.get("verified", True)),
        ))

    tolerance = _opt_float(raw.get("match_tolerance_hz")) or DEFAULT_TOLERANCE_HZ

    _warn_on_close_duplicates(signals, tolerance, path)

    return GroundTruth(
        signals=signals,
        match_tolerance_hz=tolerance,
        session=raw.get("session"),
    )


def _opt_float(value: object) -> float | None:
    """Coerce an optional value to float, or None if absent."""
    if value is None:
        return None
    return float(value)


def _norm_modulation(value: object, path: Path, idx: int) -> str | None:
    """Normalise a modulation label to upper-case; warn on unknown values."""
    if value is None:
        return None
    norm = str(value).strip().upper()
    if norm not in _MODULATIONS:
        logger.warning(
            "%s: signal %d has unknown modulation '%s' (not one of %s)",
            path, idx, norm, sorted(_MODULATIONS),
        )
    return norm


def _norm_threat(value: object, path: Path, idx: int) -> str | None:
    """Normalise a threat level to upper-case; warn on unknown values."""
    if value is None:
        return None
    norm = str(value).strip().upper()
    if norm not in THREAT_LEVELS:
        logger.warning(
            "%s: signal %d has unknown threat_level '%s'", path, idx, norm,
        )
    return norm


def _warn_on_close_duplicates(
    signals: list[GroundTruthSignal], tolerance: float, path: Path
) -> None:
    """Warn if two truth signals fall within the match tolerance of each other."""
    ordered = sorted(signals, key=lambda s: s.center_freq_hz)
    for a, b in zip(ordered, ordered[1:]):
        if abs(b.center_freq_hz - a.center_freq_hz) <= tolerance:
            logger.warning(
                "%s: truth signals at %.3f and %.3f MHz are within the match "
                "tolerance (%.0f kHz); matching may be ambiguous",
                path, a.center_freq_hz / 1e6, b.center_freq_hz / 1e6,
                tolerance / 1e3,
            )
