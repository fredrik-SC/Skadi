"""Tests for the pure scoring core."""

from __future__ import annotations

from src.benchmark.scorer import DetectionRecord, score_benchmark
from src.benchmark.truth import GroundTruth, GroundTruthSignal


def _det(freq, power=-50.0, mod="FM", names=()):
    return DetectionRecord(freq, power, 180e3, mod, tuple(names))


def _truth(signals, tol=100_000.0):
    return GroundTruth(signals=signals, match_tolerance_hz=tol)


def test_clean_one_to_one():
    truth = _truth([
        GroundTruthSignal(92.4e6, "A", "FM", "FM Broadcast Radio"),
        GroundTruthSignal(99.0e6, "B", "FM", "FM Broadcast Radio"),
    ])
    dets = [
        _det(92.41e6, names=("FM Broadcast Radio",)),
        _det(99.0e6, names=("FM Broadcast Radio",)),
    ]
    r = score_benchmark(dets, truth)
    assert r.recall == 1.0
    assert r.fragmentation_factor == 1.0
    assert r.spurious_count == 0
    assert r.modulation_accuracy == 1.0
    assert r.classification_top1_accuracy == 1.0


def test_fragmentation_picks_strongest_primary():
    truth = _truth([GroundTruthSignal(95.0e6, "X", "FM")])
    dets = [
        _det(95.00e6, power=-30.0, mod="NFM"),   # strongest -> primary
        _det(95.02e6, power=-60.0, mod="NFM"),
        _det(94.98e6, power=-55.0, mod="NFM"),
        _det(95.04e6, power=-52.0, mod="NFM"),
    ]
    r = score_benchmark(dets, truth)
    assert r.recall == 1.0
    assert r.fragmentation_factor == 4.0
    assert r.spurious_count == 0
    assert r.per_truth[0].primary.peak_power_dbm == -30.0
    # Primary is NFM but truth is FM -> modulation accuracy 0.
    assert r.modulation_accuracy == 0.0


def test_spurious_detections_ordered_by_power():
    truth = _truth([GroundTruthSignal(95.0e6, "X", "FM")])
    dets = [
        _det(95.0e6, power=-40.0),
        _det(80.0e6, power=-20.0),   # spurious, strongest
        _det(110.0e6, power=-35.0),  # spurious
    ]
    r = score_benchmark(dets, truth)
    assert r.recall == 1.0
    assert r.spurious_count == 2
    assert [round(d.center_freq_hz / 1e6, 1) for d in r.worst_spurious] == [80.0, 110.0]
    assert abs(r.spurious_rate - 2 / 3) < 1e-9


def test_detection_between_two_truths_goes_to_nearest():
    truth = _truth([
        GroundTruthSignal(95.0e6, "low"),
        GroundTruthSignal(95.15e6, "high"),
    ], tol=100_000.0)
    # 95.06 is nearer 95.0 than 95.15
    r = score_benchmark([_det(95.06e6)], truth)
    assert r.per_truth[0].found is True
    assert r.per_truth[1].found is False


def test_equidistant_tie_breaks_to_lower_freq():
    truth = _truth([
        GroundTruthSignal(95.0e6, "low"),
        GroundTruthSignal(95.10e6, "high"),
    ], tol=100_000.0)
    # 95.05 is exactly between the two -> assigned to lower freq truth.
    r = score_benchmark([_det(95.05e6)], truth)
    assert r.per_truth[0].found is True
    assert r.per_truth[1].found is False


def test_truth_with_zero_detections_is_false_negative():
    truth = _truth([
        GroundTruthSignal(95.0e6, "found", "FM"),
        GroundTruthSignal(120.0e6, "missed", "FM"),
    ])
    r = score_benchmark([_det(95.0e6, mod="FM")], truth)
    assert r.recall == 0.5
    assert r.per_truth[1].found is False
    assert r.per_truth[1].modulation_correct is None  # not scored when not found


def test_modulation_confusion_matrix_counts():
    truth = _truth([
        GroundTruthSignal(91.0e6, "a", "FM"),
        GroundTruthSignal(95.0e6, "b", "FM"),
        GroundTruthSignal(99.0e6, "c", "FM"),
    ])
    dets = [
        _det(91.0e6, mod="FM"),
        _det(95.0e6, mod="NFM"),
        _det(99.0e6, mod="UNKNOWN"),
    ]
    r = score_benchmark(dets, truth)
    rows, cols, matrix = r.modulation_confusion
    assert rows == ["FM"]
    assert matrix[0][cols.index("FM")] == 1
    assert matrix[0][cols.index("NFM")] == 1
    assert matrix[0][cols.index("UNKNOWN")] == 1
    assert abs(r.modulation_accuracy - 1 / 3) < 1e-9


def test_modulation_accuracy_skips_truth_without_modulation():
    truth = _truth([
        GroundTruthSignal(95.0e6, "labelled", "FM"),
        GroundTruthSignal(99.0e6, "unlabelled"),   # no modulation -> excluded
    ])
    r = score_benchmark([_det(95.0e6, mod="FM"), _det(99.0e6, mod="NFM")], truth)
    assert r.modulation_accuracy == 1.0  # only the FM truth counted


def test_classification_top3_hit_case_insensitive():
    truth = _truth([GroundTruthSignal(95.0e6, "a", "FM", "STANAG 4285")])
    det = _det(95.0e6, names=("Other A", "Other B", "  stanag 4285  "))
    r = score_benchmark([det], truth)
    assert r.classification_top1_accuracy == 0.0
    assert r.classification_top3_accuracy == 1.0


def test_classification_top3_miss():
    truth = _truth([GroundTruthSignal(95.0e6, "a", "FM", "STANAG 4285")])
    det = _det(95.0e6, names=("A", "B", "C", "STANAG 4285"))  # only at rank 4
    r = score_benchmark([det], truth)
    assert r.classification_top3_accuracy == 0.0


def test_near_miss_listed_but_not_counted():
    truth = _truth([GroundTruthSignal(95.0e6, "a", "FM", "FM Broadcast")])
    det = _det(95.0e6, names=("FM Broadcast Radio",))  # substring overlap, not exact
    r = score_benchmark([det], truth)
    assert r.classification_top1_accuracy == 0.0
    assert r.classification_top3_accuracy == 0.0
    assert len(r.near_misses) == 1
    assert r.near_misses[0].matched_name == "FM Broadcast Radio"


def test_multiclass_classification_confusion_built():
    truth = _truth([
        GroundTruthSignal(91.0e6, "a", "FM", "FM Broadcast Radio"),
        GroundTruthSignal(95.0e6, "b", "PSK", "STANAG 4285"),
    ])
    dets = [
        _det(91.0e6, names=("FM Broadcast Radio",)),
        _det(95.0e6, names=("CIS-12",)),
    ]
    r = score_benchmark(dets, truth)
    assert r.classification_confusion is not None
    rows, cols, _ = r.classification_confusion
    assert "FM Broadcast Radio" in rows and "STANAG 4285" in rows
