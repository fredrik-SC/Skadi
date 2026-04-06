"""Signal classification engine using the Artemis database.

Matches detected signal fingerprints against known signal types in
the Artemis DB using parametric comparison: modulation type, bandwidth,
frequency range, and ACF value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.classification.artemis_db import ArtemisDB, ArtemisSignal
from src.classification.confidence import compute_confidence
from src.fingerprint.models import ModulationType, SignalFingerprint

logger = logging.getLogger(__name__)

# Maps our broad modulation types to Artemis-specific terms.
# A candidate matches if ANY of its modulation_types intersects these terms.
MODULATION_MAP: dict[ModulationType, list[str]] = {
    ModulationType.AM: ["AM", "ASK", "DSB", "SSB", "USB", "LSB", "VSB"],
    ModulationType.FM: ["FM", "FMCW", "FMOP"],
    ModulationType.NFM: ["FM", "NFM", "FFSK", "C4FM", "GMSK"],
    ModulationType.FSK: [
        "FSK", "2FSK", "4FSK", "AFSK", "FFSK", "GFSK",
        "GMSK", "MSK", "MFSK", "IFK", "IFK+", "C4FM",
    ],
    ModulationType.PSK: [
        "PSK", "BPSK", "QPSK", "OQPSK", "8PSK", "D8PSK",
        "DQPSK", "DPSK",
    ],
    ModulationType.OOK: ["OOK", "CW", "ASK", "PWM", "PPM", "Pulse", "Pulsed"],
    ModulationType.UNKNOWN: [],
}


@dataclass
class ClassificationMatch:
    """A single match from the Artemis database.

    Attributes:
        signal: The matched Artemis signal record.
        confidence: Overall confidence score (0.0-1.0).
        modulation_score: Modulation dimension score.
        bandwidth_score: Bandwidth dimension score.
        frequency_score: Frequency dimension score.
        acf_score: ACF dimension score.
    """

    signal: ArtemisSignal
    confidence: float
    modulation_score: float
    bandwidth_score: float
    frequency_score: float
    acf_score: float


@dataclass
class ClassificationResult:
    """Complete classification result for a fingerprint.

    Attributes:
        matches: Up to max_matches ranked matches, sorted by confidence.
        fingerprint: The input fingerprint that was classified.
    """

    matches: list[ClassificationMatch]
    fingerprint: SignalFingerprint


class SignalClassifier:
    """Classify detected signals against the Artemis database.

    Uses parametric matching with weighted confidence scoring across
    modulation type, bandwidth, frequency range, and ACF value.

    Args:
        artemis_db: Loaded ArtemisDB instance.
        config: Optional classification config dict from default.yaml.
            Keys: bandwidth_tolerance, max_matches, min_confidence.
    """

    def __init__(
        self,
        artemis_db: ArtemisDB,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        self._db = artemis_db
        self._bandwidth_tolerance = float(cfg.get("bandwidth_tolerance", 0.15))
        self._max_matches = int(cfg.get("max_matches", 3))
        self._min_confidence = float(cfg.get("min_confidence", 0.1))

    def classify(self, fingerprint: SignalFingerprint) -> ClassificationResult:
        """Classify a signal fingerprint against the Artemis database.

        Args:
            fingerprint: The signal's extracted fingerprint.

        Returns:
            ClassificationResult with up to max_matches ranked matches.
        """
        # Get modulation search terms
        mod_terms = MODULATION_MAP.get(fingerprint.modulation, [])

        # Get candidates from Artemis DB
        if mod_terms:
            candidates = self._db.query_candidates(
                modulation_terms=mod_terms,
                freq_hz=fingerprint.signal.centre_freq_hz,
            )
        else:
            # UNKNOWN modulation — search all signals, filter by frequency only
            candidates = self._db.query_candidates(
                modulation_terms=[],
                freq_hz=fingerprint.signal.centre_freq_hz,
            )

        # Score each candidate
        matches: list[ClassificationMatch] = []
        for candidate in candidates:
            total, mod_s, bw_s, freq_s, acf_s = compute_confidence(
                fingerprint, candidate, mod_terms, self._bandwidth_tolerance,
            )

            if total >= self._min_confidence:
                matches.append(ClassificationMatch(
                    signal=candidate,
                    confidence=total,
                    modulation_score=mod_s,
                    bandwidth_score=bw_s,
                    frequency_score=freq_s,
                    acf_score=acf_s,
                ))

        # Sort by confidence descending, limit to max_matches
        matches.sort(key=lambda m: m.confidence, reverse=True)
        matches = matches[:self._max_matches]

        if matches:
            logger.debug(
                "Classified %.3f MHz: top match '%s' (conf=%.2f)",
                fingerprint.signal.centre_freq_hz / 1e6,
                matches[0].signal.name, matches[0].confidence,
            )
        else:
            logger.debug(
                "No matches for %.3f MHz (mod=%s, bw=%.0f Hz)",
                fingerprint.signal.centre_freq_hz / 1e6,
                fingerprint.modulation.value, fingerprint.bandwidth_hz,
            )

        return ClassificationResult(matches=matches, fingerprint=fingerprint)
