"""Load and run a trained modulation model (inference wrapper)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from src.fingerprint.models import ModulationFeatures, ModulationType
from src.ml.features import FEATURE_NAMES, features_to_vector

logger = logging.getLogger(__name__)


class MLModelError(Exception):
    """Raised when a model bundle is missing, corrupt, or misaligned."""


class MLModulationModel:
    """A loaded modulation classifier that predicts from the feature vector.

    Wraps a scikit-learn estimator persisted as a bundle. Validates that the
    bundle's feature ordering matches the current :data:`FEATURE_NAMES` so a
    stale model can never be fed misaligned inputs.
    """

    def __init__(self, model, classes: list[str], meta: dict) -> None:
        self._model = model
        self._classes = classes
        self._meta = meta

    @classmethod
    def load(cls, path: Path) -> "MLModulationModel":
        """Load a model bundle from disk.

        Raises:
            MLModelError: If the file is missing or its feature ordering does
                not match the current FEATURE_NAMES.
        """
        path = Path(path)
        if not path.exists():
            raise MLModelError(f"Model not found: {path}")
        try:
            bundle = joblib.load(path)
        except Exception as e:  # noqa: BLE001
            raise MLModelError(f"Failed to load model {path}: {e}") from e

        names = list(bundle.get("feature_names", []))
        if names != FEATURE_NAMES:
            raise MLModelError(
                "Model feature_names do not match current FEATURE_NAMES — retrain."
            )
        sk_ver = bundle.get("sklearn_version")
        import sklearn
        if sk_ver and sk_ver != sklearn.__version__:
            logger.warning(
                "Model trained with sklearn %s, running %s — predictions may differ",
                sk_ver, sklearn.__version__,
            )
        return cls(bundle["model"], list(bundle["classes"]), bundle.get("meta", {}))

    def predict(
        self, features: ModulationFeatures, bandwidth_hz: float
    ) -> tuple[ModulationType, float]:
        """Predict modulation type and confidence from a feature vector."""
        vec = features_to_vector(features, bandwidth_hz).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0]
        i = int(np.argmax(proba))
        return ModulationType(self._classes[i]), float(proba[i])

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)
