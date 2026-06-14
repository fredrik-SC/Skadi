"""The feature vector contract shared by training and prediction.

``FEATURE_NAMES`` is the single source of truth for the model's input columns and
their order. Both dataset construction and live prediction call
``features_to_vector`` so the column semantics can never drift. NEVER reorder or
remove a name without retraining — the model is keyed to index position, and
:class:`~src.ml.model.MLModulationModel` refuses to load a bundle whose stored
``feature_names`` no longer matches this list.
"""

from __future__ import annotations

import numpy as np

from src.fingerprint.models import ModulationFeatures

# Ordered feature columns. The first 15 mirror ModulationFeatures; the last two
# are derived. Append-only: add new features at the end, never reorder.
FEATURE_NAMES: list[str] = [
    "envelope_variance",
    "inst_freq_variance",
    "inst_freq_kurtosis",
    "phase_discontinuities",
    "spectral_flatness",
    "num_freq_states",
    "envelope_cv",
    "envelope_bimodality",
    "num_freq_states_robust",
    "freq_state_separation",
    "inst_freq_center_norm",
    "phase_jump_rate",
    "phase_level_concentration",
    "phase_single_concentration",
    "symbol_rate_hz",
    "bandwidth_hz",
    "symbol_rate_norm",  # symbol_rate_hz / bandwidth_hz (scale-robust)
]


def features_to_vector(f: ModulationFeatures, bandwidth_hz: float) -> np.ndarray:
    """Flatten a ModulationFeatures + bandwidth into the model input vector.

    Args:
        f: The computed modulation features.
        bandwidth_hz: The signal's occupied bandwidth in Hz.

    Returns:
        A float64 array of length ``len(FEATURE_NAMES)`` in FEATURE_NAMES order.
    """
    bw = float(bandwidth_hz) if bandwidth_hz else 0.0
    symbol_rate_norm = (f.symbol_rate_hz / bw) if bw > 0 else 0.0
    values = {
        "envelope_variance": f.envelope_variance,
        "inst_freq_variance": f.inst_freq_variance,
        "inst_freq_kurtosis": f.inst_freq_kurtosis,
        "phase_discontinuities": float(f.phase_discontinuities),
        "spectral_flatness": f.spectral_flatness,
        "num_freq_states": float(f.num_freq_states),
        "envelope_cv": f.envelope_cv,
        "envelope_bimodality": f.envelope_bimodality,
        "num_freq_states_robust": float(f.num_freq_states_robust),
        "freq_state_separation": f.freq_state_separation,
        "inst_freq_center_norm": f.inst_freq_center_norm,
        "phase_jump_rate": f.phase_jump_rate,
        "phase_level_concentration": f.phase_level_concentration,
        "phase_single_concentration": f.phase_single_concentration,
        "symbol_rate_hz": f.symbol_rate_hz,
        "bandwidth_hz": bw,
        "symbol_rate_norm": symbol_rate_norm,
    }
    vec = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
