"""Tests for training + the model load/predict round-trip."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.fingerprint.modulation import ModulationClassifier
from src.fingerprint.models import ModulationType
from src.ml.dataset import DatasetSpec, _rows_from_synthetic
from src.ml.features import FEATURE_NAMES
from src.ml.model import MLModelError, MLModulationModel
from src.ml.train import train
from tests.conftest import generate_fsk_signal, generate_am_signal, generate_fm_signal


def _tiny_dataset():
    spec = DatasetSpec(snr_db_levels=(20.0, 1e6), synthetic_reps=2)
    rows = list(_rows_from_synthetic(spec))
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=object)
    return X, y


def test_train_and_predict_roundtrip(tmp_path):
    X, y = _tiny_dataset()
    model, report = train(X, y, n_estimators=100, seed=0)
    assert report["held_out_accuracy"] > 0.7  # synthetic should be learnable

    import sklearn
    bundle = {
        "model": model, "feature_names": list(FEATURE_NAMES),
        "classes": sorted(set(y.tolist())), "sklearn_version": sklearn.__version__,
        "meta": {},
    }
    path = tmp_path / "m.joblib"
    joblib.dump(bundle, path)

    loaded = MLModulationModel.load(path)
    assert loaded.feature_names == FEATURE_NAMES

    # Classify fresh clean synthetic through the loaded model (high tolerance —
    # this proves the train->predict path, not production accuracy).
    clf = ModulationClassifier(model=loaded)
    fsk, _, _ = clf.classify(generate_fsk_signal(), 50_000, 25, 5000)
    am, _, _ = clf.classify(generate_am_signal(), 50_000, 25, 10_000)
    assert fsk == ModulationType.FSK
    assert am == ModulationType.AM


def test_load_rejects_mismatched_feature_names(tmp_path):
    bundle = {"model": object(), "feature_names": ["wrong"], "classes": ["FSK"]}
    path = tmp_path / "bad.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(MLModelError, match="feature_names"):
        MLModulationModel.load(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(MLModelError, match="not found"):
        MLModulationModel.load(tmp_path / "nope.joblib")
