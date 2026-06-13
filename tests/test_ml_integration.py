"""Tests for ML model integration + deterministic fallback."""

from __future__ import annotations

from src.fingerprint.extractor import FingerprintExtractor
from src.fingerprint.modulation import ModulationClassifier
from src.fingerprint.models import ModulationType
from tests.conftest import generate_fsk_signal


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
