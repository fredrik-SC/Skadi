"""Tests for the ML dataset builder."""

from __future__ import annotations

import numpy as np

from src.ml.dataset import DatasetSpec, _rows_from_synthetic, load_dataset, save_dataset
from src.ml.features import FEATURE_NAMES


def test_synthetic_rows_cover_classes():
    spec = DatasetSpec(snr_db_levels=(1e6,), synthetic_reps=1)
    rows = list(_rows_from_synthetic(spec))
    assert rows
    labels = {label for _, label in rows}
    # Every modulation family appears.
    assert {"AM", "FM", "NFM", "FSK", "PSK", "OOK"} <= labels
    for vec, _ in rows:
        assert vec.shape == (len(FEATURE_NAMES),)
        assert np.isfinite(vec).all()


def test_save_load_roundtrip(tmp_path):
    X = np.random.rand(10, len(FEATURE_NAMES))
    y = np.array(["FSK"] * 5 + ["PSK"] * 5, dtype=object)
    p = tmp_path / "ds.npz"
    save_dataset(p, X, y, list(FEATURE_NAMES), {"note": "test"})
    X2, y2, names = load_dataset(p)
    assert np.allclose(X, X2)
    assert list(y2) == list(y)
    assert names == FEATURE_NAMES
