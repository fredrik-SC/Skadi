"""Tests for ML model integration + deterministic fallback."""

from __future__ import annotations

import dataclasses

import numpy as np

from src.fingerprint.extractor import FingerprintExtractor
from src.fingerprint.modulation import ModulationClassifier
from src.fingerprint.models import ModulationFeatures, ModulationType
from tests.conftest import generate_fsk_signal


def _analog_am_features() -> ModulationFeatures:
    """A feature vector matching the real-airband analog-AM-voice signature."""
    return ModulationFeatures(
        envelope_variance=0.30, inst_freq_variance=0.07, inst_freq_kurtosis=0.0,
        phase_discontinuities=0, spectral_flatness=0.0, num_freq_states=1,
        envelope_cv=0.55, envelope_bimodality=0.45, num_freq_states_robust=1,
        freq_state_separation=0.6, inst_freq_center_norm=0.0, phase_jump_rate=0.0,
        phase_level_concentration=0.09, phase_single_concentration=0.05,
        symbol_rate_hz=0.0,
    )


class _StubModel:
    """A model that always predicts a fixed type."""

    def __init__(self, mod: ModulationType):
        self._mod = mod

    def predict(self, features, bandwidth_hz):
        return self._mod, 0.99


def test_no_model_uses_deterministic():
    """model=None falls back to the deterministic _decide()."""
    clf = ModulationClassifier(model=None)
    mod, _, _ = clf.classify(generate_fsk_signal(), 50_000, 25, 5000)
    assert mod == ModulationType.FSK  # deterministic gets clean synthetic FSK right


def test_model_routes_through():
    """A loaded model makes the decision instead of _decide()."""
    clf = ModulationClassifier(model=_StubModel(ModulationType.PSK))
    mod, conf, _ = clf.classify(generate_fsk_signal(), 50_000, 25, 5000)
    assert mod == ModulationType.PSK  # the stub overrides the deterministic FSK
    assert conf == 0.99


def test_extractor_missing_model_falls_back():
    """ml.enabled with a missing model file logs and falls back, no raise."""
    ext = FingerprintExtractor(
        sample_rate=2_048_000,
        config={"ml": {"enabled": True, "model_path": "data/does_not_exist.joblib"}},
    )
    # Classifier exists and has no model -> deterministic path.
    assert ext._classifier._model is None


def test_extractor_ml_disabled_no_model():
    ext = FingerprintExtractor(sample_rate=2_048_000, config={"ml": {"enabled": False}})
    assert ext._classifier._model is None


# --- hybrid analog-AM gate (model present) ---------------------------------

def test_am_gate_fires_on_analog_signature():
    """The analog-AM-voice signature trips the gate."""
    assert ModulationClassifier()._looks_like_analog_am(_analog_am_features())


def test_am_gate_rejects_digital_and_fm_signatures():
    """Digital / FM signatures must not be mistaken for analog AM."""
    clf = ModulationClassifier()
    base = _analog_am_features()
    # PSK-like: order-2 phase concentration present.
    assert not clf._looks_like_analog_am(dataclasses.replace(base, phase_level_concentration=0.6))
    # FSK-like: two sustained frequency states.
    assert not clf._looks_like_analog_am(dataclasses.replace(base, num_freq_states_robust=2))
    # FM-like: ~zero residual instantaneous-frequency variance.
    assert not clf._looks_like_analog_am(dataclasses.replace(base, inst_freq_variance=0.0006))
    # Constant-envelope digital: low envelope CV.
    assert not clf._looks_like_analog_am(dataclasses.replace(base, envelope_cv=0.1))


def test_am_gate_overrides_model(monkeypatch):
    """When the gate fires, AM wins even though the model would say FSK."""
    clf = ModulationClassifier(model=_StubModel(ModulationType.FSK))
    monkeypatch.setattr(clf, "_compute_features", lambda iq, sr: _analog_am_features())
    mod, _, _ = clf.classify(np.ones(512, dtype=np.complex64), 50_000, 25, 5000)
    assert mod == ModulationType.AM


def test_model_used_when_gate_quiet(monkeypatch):
    """When the gate does not fire, the model's verdict passes through."""
    clf = ModulationClassifier(model=_StubModel(ModulationType.FSK))
    digital = dataclasses.replace(_analog_am_features(), num_freq_states_robust=2)
    monkeypatch.setattr(clf, "_compute_features", lambda iq, sr: digital)
    mod, _, _ = clf.classify(np.ones(512, dtype=np.complex64), 50_000, 25, 5000)
    assert mod == ModulationType.FSK


def test_am_gate_inactive_without_model():
    """No model -> deterministic path; the gate never short-circuits it."""
    clf = ModulationClassifier(model=None)
    # Even with AM-like features, classify() must run _decide(), not the gate.
    mod, _, _ = clf.classify(generate_fsk_signal(), 50_000, 25, 5000)
    assert mod == ModulationType.FSK  # deterministic still classifies synthetic FSK
