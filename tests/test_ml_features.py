"""Tests for the ML feature-vector contract."""

from __future__ import annotations

import numpy as np

from src.fingerprint.models import ModulationFeatures
from src.ml.features import FEATURE_NAMES, features_to_vector


def test_vector_length_and_order():
    f = ModulationFeatures(0, 0, 0, 0, 0, 0)
    v = features_to_vector(f, 5000.0)
    assert v.shape == (len(FEATURE_NAMES),)
    assert v.dtype == np.float64


def test_feature_names_stable():
    """FEATURE_NAMES is a frozen contract — a reorder/removal must break loudly."""
    assert FEATURE_NAMES[:6] == [
        "envelope_variance", "inst_freq_variance", "inst_freq_kurtosis",
        "phase_discontinuities", "spectral_flatness", "num_freq_states",
    ]
    assert FEATURE_NAMES[-2:] == ["bandwidth_hz", "symbol_rate_norm"]
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))  # no duplicates


def test_values_map_correctly():
    f = ModulationFeatures(
        envelope_variance=0.5, inst_freq_variance=0.01, inst_freq_kurtosis=3.0,
        phase_discontinuities=42, spectral_flatness=0.2, num_freq_states=2,
        envelope_cv=0.7, envelope_bimodality=0.9, num_freq_states_robust=2,
        freq_state_separation=1.5, inst_freq_center_norm=0.001, phase_jump_rate=0.05,
        phase_level_concentration=0.8, phase_single_concentration=0.1, symbol_rate_hz=1200.0,
    )
    v = features_to_vector(f, 6000.0)
    d = dict(zip(FEATURE_NAMES, v))
    assert d["envelope_cv"] == 0.7
    assert d["phase_discontinuities"] == 42.0
    assert d["bandwidth_hz"] == 6000.0
    assert abs(d["symbol_rate_norm"] - 1200.0 / 6000.0) < 1e-9


def test_zero_bandwidth_safe():
    f = ModulationFeatures(0, 0, 0, 0, 0, 0, symbol_rate_hz=100.0)
    v = features_to_vector(f, 0.0)
    d = dict(zip(FEATURE_NAMES, v))
    assert d["symbol_rate_norm"] == 0.0  # no divide-by-zero
    assert np.isfinite(v).all()


def test_nan_inf_sanitised():
    f = ModulationFeatures(float("nan"), float("inf"), 0, 0, 0, 0)
    v = features_to_vector(f, 5000.0)
    assert np.isfinite(v).all()
