"""Tests for threat level assignment module."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.classification.threat import ThreatMapper
from src.config import PROJECT_ROOT


class TestThreatMapper:
    """Tests for ThreatMapper."""

    @pytest.fixture
    def mapper(self) -> ThreatMapper:
        """Load the real threat_levels.yaml."""
        return ThreatMapper(PROJECT_ROOT / "config" / "threat_levels.yaml")

    def test_military_signal_critical_or_high(self, mapper):
        """Military signals get CRITICAL or HIGH."""
        level = mapper.assess("Russian military 20bd FSK", "Military encrypted signal")
        assert level in ("CRITICAL", "HIGH")

    def test_stanag_is_high(self, mapper):
        """STANAG signals get HIGH (NATO/military keyword)."""
        level = mapper.assess("STANAG 4285", "NATO military communication standard")
        assert level == "HIGH"

    def test_fm_broadcast_informational(self, mapper):
        """FM Broadcast gets INFORMATIONAL."""
        level = mapper.assess("FM Broadcast Radio", "Standard FM broadcast")
        assert level == "INFORMATIONAL"

    def test_gps_informational(self, mapper):
        """GPS/navigation signals get INFORMATIONAL."""
        level = mapper.assess("GPS L1", "Navigation satellite signal")
        assert level == "INFORMATIONAL"

    def test_unknown_defaults_to_medium(self, mapper):
        """Completely unknown signal gets MEDIUM (default)."""
        level = mapper.assess("XYZ-12345", "No recognisable keywords here")
        assert level == "MEDIUM"

    def test_none_inputs_return_default(self, mapper):
        """None name and description returns default."""
        level = mapper.assess(None, None)
        assert level == "MEDIUM"

    def test_amateur_is_low(self, mapper):
        """Amateur radio signals get LOW."""
        level = mapper.assess("Amateur Packet", "Amateur ham radio digital mode")
        assert level == "LOW"

    def test_case_insensitive(self, mapper):
        """Keyword matching is case-insensitive."""
        level = mapper.assess("MILITARY STANAG", "GOVERNMENT ENCRYPTED")
        assert level in ("CRITICAL", "HIGH")

    def test_first_rule_wins(self, mapper):
        """Rules are evaluated in order; first match wins."""
        # "military encrypted" matches CRITICAL before "military" matches HIGH
        level = mapper.assess("military encrypted comm", "")
        assert level == "CRITICAL"

    def test_load_custom_config(self, tmp_path):
        """Custom config loads correctly."""
        config = tmp_path / "threats.yaml"
        config.write_text(dedent("""\
            default_threat_level: LOW
            rules:
              - threat_level: CRITICAL
                keywords:
                  - "test keyword"
        """))
        mapper = ThreatMapper(config)
        assert mapper.default_level == "LOW"
        assert mapper.assess("test keyword signal", "") == "CRITICAL"
        assert mapper.assess("other signal", "") == "LOW"
