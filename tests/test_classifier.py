"""Tests for signal classification module.

Includes Artemis DB access tests, confidence scoring tests,
and 6 known-signal acceptance tests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.classification.artemis_db import ArtemisDB, ArtemisSignal
from src.classification.classifier import MODULATION_MAP, SignalClassifier
from src.classification.confidence import compute_confidence
from src.config import PROJECT_ROOT
from src.detection.models import DetectedSignal
from src.fingerprint.models import ModulationFeatures, ModulationType, SignalFingerprint

ARTEMIS_DB_PATH = PROJECT_ROOT / "data" / "artemis.db"


def _make_fingerprint(
    modulation: ModulationType,
    bandwidth_hz: float,
    centre_freq_hz: float,
    acf_ms: float | None = None,
) -> SignalFingerprint:
    """Build a minimal fingerprint for classification testing."""
    signal = DetectedSignal(
        centre_freq_hz=centre_freq_hz,
        bandwidth_hz=bandwidth_hz,
        peak_power_dbm=-70.0,
        mean_power_dbm=-75.0,
        snr_db=25.0,
        timestamp=time.time(),
        scan_step_freq_hz=centre_freq_hz,
    )
    return SignalFingerprint(
        signal=signal,
        modulation=modulation,
        modulation_confidence=0.8,
        bandwidth_hz=bandwidth_hz,
        acf_ms=acf_ms,
        acf_strength=0.7 if acf_ms else None,
        features=ModulationFeatures(0, 0, 0, 0, 0, 0),
    )


@pytest.fixture
def artemis_db() -> ArtemisDB:
    """Load the real Artemis DB."""
    if not ARTEMIS_DB_PATH.exists():
        pytest.skip("Artemis DB not found — run scripts/download_artemis_db.py first")
    return ArtemisDB(ARTEMIS_DB_PATH)


@pytest.fixture
def classifier(artemis_db) -> SignalClassifier:
    """Create a classifier with default config."""
    return SignalClassifier(artemis_db)


class TestArtemisDB:
    """Tests for Artemis DB access layer."""

    def test_load_all_signals(self, artemis_db):
        """All 432 signals are loaded."""
        assert len(artemis_db.signals) == 432

    def test_modulation_parsing_simple(self, artemis_db):
        """Simple modulation parsed correctly."""
        # Find a signal with known modulation
        fsk_signals = [s for s in artemis_db.signals if "FSK" in (s.modulation_types or [])]
        assert len(fsk_signals) > 0

    def test_modulation_parsing_compound(self, artemis_db):
        """Compound modulation (e.g. 'PSK; OFDM') parsed into list."""
        compound = [s for s in artemis_db.signals if len(s.modulation_types) > 1]
        assert len(compound) > 0
        # Each should have 2+ types
        for sig in compound[:5]:
            assert len(sig.modulation_types) >= 2

    def test_acf_parsing_single(self):
        """Single ACF value parsed correctly."""
        values = ArtemisDB._parse_acf("Main - 20")
        assert len(values) == 1
        assert values[0] == pytest.approx(20.0)

    def test_acf_parsing_multiple(self):
        """Multiple ACF values parsed correctly."""
        values = ArtemisDB._parse_acf("Header - 200 ; Body - 66.66")
        assert len(values) == 2
        assert values[0] == pytest.approx(200.0)
        assert values[1] == pytest.approx(66.66)

    def test_acf_parsing_empty(self):
        """Empty ACF returns empty list."""
        assert ArtemisDB._parse_acf("") == []
        assert ArtemisDB._parse_acf(None) == []

    def test_query_candidates_by_modulation(self, artemis_db):
        """Query with FSK terms returns FSK signals."""
        candidates = artemis_db.query_candidates(modulation_terms=["FSK"])
        assert len(candidates) > 0
        for c in candidates:
            assert "FSK" in [m.upper() for m in c.modulation_types]

    def test_query_candidates_by_frequency(self, artemis_db):
        """Query with frequency filter narrows results."""
        all_fsk = artemis_db.query_candidates(modulation_terms=["FM"])
        fm_at_100mhz = artemis_db.query_candidates(modulation_terms=["FM"], freq_hz=100e6)
        assert len(fm_at_100mhz) <= len(all_fsk)


class TestConfidenceScoring:
    """Tests for confidence scoring functions."""

    def _make_candidate(self, **kwargs) -> ArtemisSignal:
        """Build a test ArtemisSignal."""
        defaults = dict(
            id=1, name="Test Signal",
            freq_min_hz=88_000_000, freq_max_hz=108_000_000,
            mode="FM", bandwidth_min_hz=150_000, bandwidth_max_hz=200_000,
            modulation="FM", modulation_types=["FM"],
            acf_values_ms=[], description="Test", location="UK",
        )
        defaults.update(kwargs)
        return ArtemisSignal(**defaults)

    def test_perfect_match(self):
        """All dimensions matching gives high confidence."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        cand = self._make_candidate()
        total, mod, bw, freq, acf = compute_confidence(fp, cand, ["FM"])
        assert total > 0.8
        assert mod == 1.0
        assert bw == 1.0
        assert freq == 1.0

    def test_modulation_mismatch(self):
        """Wrong modulation gives zero modulation score."""
        fp = _make_fingerprint(ModulationType.PSK, 200_000, 98e6)
        cand = self._make_candidate()
        total, mod, bw, freq, acf = compute_confidence(fp, cand, ["PSK"])
        assert mod == 0.0

    def test_bandwidth_within_tolerance(self):
        """Bandwidth within ±15% scores well."""
        fp = _make_fingerprint(ModulationType.FM, 180_000, 98e6)  # 10% off
        cand = self._make_candidate(bandwidth_min_hz=200_000, bandwidth_max_hz=200_000)
        _, _, bw, _, _ = compute_confidence(fp, cand, ["FM"], bandwidth_tolerance=0.15)
        assert bw > 0.5

    def test_bandwidth_out_of_range(self):
        """Bandwidth way outside range scores low."""
        fp = _make_fingerprint(ModulationType.FM, 10_000, 98e6)  # 10kHz vs 200kHz
        cand = self._make_candidate(bandwidth_min_hz=200_000, bandwidth_max_hz=200_000)
        _, _, bw, _, _ = compute_confidence(fp, cand, ["FM"])
        assert bw < 0.3

    def test_frequency_in_range(self):
        """Frequency within range scores 1.0."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        cand = self._make_candidate(freq_min_hz=88e6, freq_max_hz=108e6)
        _, _, _, freq, _ = compute_confidence(fp, cand, ["FM"])
        assert freq == 1.0

    def test_frequency_out_of_range(self):
        """Frequency outside range scores low."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 200e6)
        cand = self._make_candidate(freq_min_hz=88e6, freq_max_hz=108e6)
        _, _, _, freq, _ = compute_confidence(fp, cand, ["FM"])
        assert freq < 0.5

    def test_acf_exact_match(self):
        """ACF within tolerance scores high."""
        fp = _make_fingerprint(ModulationType.FSK, 5000, 100e6, acf_ms=20.0)
        cand = self._make_candidate(acf_values_ms=[20.0])
        _, _, _, _, acf = compute_confidence(fp, cand, ["FSK"])
        assert acf > 0.9

    def test_missing_acf_neutral(self):
        """No ACF on either side scores 0.5."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        cand = self._make_candidate(acf_values_ms=[])
        _, _, _, _, acf = compute_confidence(fp, cand, ["FM"])
        assert acf == 0.5


class TestSignalClassifier:
    """Acceptance tests with known signal types."""

    def test_fm_broadcast(self, classifier):
        """FM Broadcast Radio matched correctly."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        result = classifier.classify(fp)
        assert len(result.matches) > 0
        assert "FM Broadcast" in result.matches[0].signal.name

    def test_nfm_voice(self, classifier):
        """NFM Voice is among the top matches for narrowband FM at VHF."""
        fp = _make_fingerprint(ModulationType.NFM, 12_500, 145e6)
        result = classifier.classify(fp)
        assert len(result.matches) > 0
        names = [m.signal.name for m in result.matches]
        assert any("NFM" in n for n in names), f"NFM Voice not in matches: {names}"

    def test_morse_code(self, classifier):
        """Morse Code / CW matched correctly."""
        fp = _make_fingerprint(ModulationType.OOK, 100, 7e6)
        result = classifier.classify(fp)
        assert len(result.matches) > 0
        # Morse code or CW should be in the results
        names = [m.signal.name.lower() for m in result.matches]
        assert any("morse" in n or "cw" in n for n in names)

    def test_stanag_4285(self, classifier):
        """STANAG 4285 matched with PSK modulation and ACF."""
        fp = _make_fingerprint(ModulationType.PSK, 2_750, 10e6, acf_ms=106.66)
        result = classifier.classify(fp)
        assert len(result.matches) > 0
        names = [m.signal.name for m in result.matches]
        assert any("STANAG 4285" in n for n in names)

    def test_unknown_modulation_still_matches(self, classifier):
        """UNKNOWN modulation still returns matches based on freq+BW."""
        fp = _make_fingerprint(ModulationType.UNKNOWN, 200_000, 98e6)
        result = classifier.classify(fp)
        # Should still find FM Broadcast based on bandwidth and frequency
        assert len(result.matches) > 0

    def test_max_matches_limit(self, classifier):
        """Returns at most max_matches results."""
        fp = _make_fingerprint(ModulationType.FSK, 5000, 10e6)
        result = classifier.classify(fp)
        assert len(result.matches) <= 3

    def test_results_sorted_by_confidence(self, classifier):
        """Matches are sorted by confidence descending."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        result = classifier.classify(fp)
        if len(result.matches) >= 2:
            for i in range(len(result.matches) - 1):
                assert result.matches[i].confidence >= result.matches[i + 1].confidence

    def test_confidence_scores_bounded(self, classifier):
        """All confidence scores are between 0 and 1."""
        fp = _make_fingerprint(ModulationType.FM, 200_000, 98e6)
        result = classifier.classify(fp)
        for match in result.matches:
            assert 0.0 <= match.confidence <= 1.0
