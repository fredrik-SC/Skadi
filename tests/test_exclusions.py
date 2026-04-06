"""Tests for exclusion list filtering."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.detection.exclusions import ExclusionEntry, ExclusionFilter
from src.detection.models import DetectedSignal


def _make_signal(freq_hz: float, bandwidth_hz: float = 200_000) -> DetectedSignal:
    """Helper to create a DetectedSignal for testing."""
    return DetectedSignal(
        centre_freq_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        peak_power_dbm=-70.0,
        mean_power_dbm=-75.0,
        snr_db=20.0,
        timestamp=1000.0,
        scan_step_freq_hz=freq_hz,
    )


class TestExclusionFilter:
    """Tests for ExclusionFilter."""

    def test_empty_list_passes_all(self):
        """No exclusions, all signals returned."""
        filt = ExclusionFilter()
        signals = [_make_signal(100e6), _make_signal(101e6)]
        assert filt.filter(signals) == signals

    def test_signal_within_exclusion_band(self):
        """Signal centre inside exclusion band is filtered out."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=1e6)]
        signals = [_make_signal(100.2e6, 200_000)]  # Within 100 ± 0.5 MHz
        result = filt.filter(signals)
        assert result == []

    def test_signal_outside_exclusion_band(self):
        """Signal well outside exclusion band passes through."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=200_000)]
        signals = [_make_signal(105e6)]  # 5 MHz away
        result = filt.filter(signals)
        assert len(result) == 1

    def test_overlapping_bandwidth(self):
        """Partial bandwidth overlap causes exclusion."""
        filt = ExclusionFilter()
        # Exclusion: 100 MHz ± 100 kHz = 99.9 - 100.1 MHz
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=200_000)]
        # Signal: 100.05 MHz ± 100 kHz = 99.95 - 100.15 MHz (overlaps)
        signals = [_make_signal(100.05e6, 200_000)]
        assert filt.filter(signals) == []

    def test_point_exclusion_filters_overlapping_signal(self):
        """Exclusion with bandwidth=0 filters signal whose band contains the point."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=0)]
        # Signal: 100 MHz ± 100 kHz — contains the point
        signals = [_make_signal(100e6, 200_000)]
        assert filt.filter(signals) == []

    def test_point_exclusion_passes_distant_signal(self):
        """Exclusion with bandwidth=0 doesn't affect distant signals."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=0)]
        signals = [_make_signal(101e6, 200_000)]
        assert len(filt.filter(signals)) == 1

    def test_multiple_exclusions(self):
        """Multiple entries, only matching signals removed."""
        filt = ExclusionFilter()
        filt._entries = [
            ExclusionEntry(freq_hz=100e6, bandwidth_hz=200_000),
            ExclusionEntry(freq_hz=105e6, bandwidth_hz=200_000),
        ]
        signals = [
            _make_signal(100e6),   # Excluded
            _make_signal(102e6),   # Passes
            _make_signal(105e6),   # Excluded
        ]
        result = filt.filter(signals)
        assert len(result) == 1
        assert result[0].centre_freq_hz == 102e6

    def test_is_excluded_method(self):
        """Direct is_excluded check works correctly."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6, bandwidth_hz=200_000)]
        assert filt.is_excluded(100e6) is True
        assert filt.is_excluded(100.05e6, 200_000) is True
        assert filt.is_excluded(105e6) is False

    def test_load_from_yaml(self, tmp_path):
        """Loads exclusion entries from a YAML file."""
        yaml_file = tmp_path / "exclusions.yaml"
        yaml_file.write_text(dedent("""\
            exclusions:
              - freq_hz: 100000000
                bandwidth_hz: 200000
                label: "FM 100 MHz"
              - freq_hz: 105000000
                bandwidth_hz: 150000
        """))
        filt = ExclusionFilter(yaml_file)
        assert len(filt.entries) == 2
        assert filt.entries[0].label == "FM 100 MHz"
        assert filt.entries[1].freq_hz == 105e6

    def test_load_empty_yaml(self, tmp_path):
        """Empty exclusions list results in no filtering."""
        yaml_file = tmp_path / "exclusions.yaml"
        yaml_file.write_text("exclusions:\n")
        filt = ExclusionFilter(yaml_file)
        assert filt.entries == []

    def test_entries_property_is_copy(self):
        """entries property returns a copy, not the internal list."""
        filt = ExclusionFilter()
        filt._entries = [ExclusionEntry(freq_hz=100e6)]
        entries = filt.entries
        entries.clear()
        assert len(filt._entries) == 1
