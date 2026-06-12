"""Offline benchmark harness for the Skaði signal pipeline.

Replays a recorded IQ session through detect → fingerprint → classify and
scores the output against hand-labelled ground truth, turning subjective
accuracy impressions into measurable numbers (detection recall, fragmentation
factor, spurious rate, modulation/classification accuracy and confusion
matrices). The same recording can be scored under different configurations,
making detection/classification tuning a measured activity.
"""

from src.benchmark.scorer import (
    BenchmarkReport,
    DetectionRecord,
    PerTruthOutcome,
    score_benchmark,
)
from src.benchmark.truth import (
    BenchmarkError,
    GroundTruth,
    GroundTruthSignal,
    load_truth,
)

__all__ = [
    "BenchmarkError",
    "BenchmarkReport",
    "DetectionRecord",
    "GroundTruth",
    "GroundTruthSignal",
    "PerTruthOutcome",
    "load_truth",
    "score_benchmark",
]
