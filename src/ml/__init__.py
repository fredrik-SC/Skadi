"""Trained (v2.0) modulation classification.

A feature-based machine-learning layer that replaces the hand-tuned modulation
decision when enabled. It trains on the same engineered features the
deterministic classifier already computes (see :mod:`src.ml.features`), so the
training and prediction paths are identical. The deterministic classifier
remains the offline fallback when no model is loaded.
"""
