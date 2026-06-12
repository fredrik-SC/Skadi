"""Pure scoring core for the benchmark harness.

Given a flat list of detections (each with its fingerprint modulation and the
classifier's top-N match names) and a :class:`~src.benchmark.truth.GroundTruth`,
compute detection/modulation/classification metrics. This module is deliberately
free of SDR, pipeline, and I/O dependencies so it is fast and trivial to unit
test on hand-built inputs.

Scoring is built around the failure mode we need to measure: one real signal
fragmenting into many detections. Each detection is matched to the nearest truth
signal within a tolerance; one **primary** detection (highest power) represents
each matched truth for modulation/classification scoring, while the extras are
counted as **fragments**. Detections matching no truth are **spurious**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.benchmark.truth import GroundTruth, GroundTruthSignal
from src.fingerprint.models import ModulationType

# Stable column order for the modulation confusion matrix.
_MODULATION_COLS = [m.value for m in ModulationType] + ["(none)"]
_NONE_LABEL = "(none)"

# How many spurious detections to retain for the report.
_WORST_SPURIOUS_N = 5


class DetectionLabel(str, Enum):
    """How a detection relates to ground truth."""

    PRIMARY = "PRIMARY"      # the representative detection for a matched truth
    FRAGMENT = "FRAGMENT"    # an extra detection of an already-matched truth
    SPURIOUS = "SPURIOUS"    # matched no truth signal


@dataclass(frozen=True)
class DetectionRecord:
    """A single detection flattened for scoring.

    Attributes:
        center_freq_hz: Detection centre frequency in Hz.
        peak_power_dbm: Peak power (used to choose the primary detection).
        bandwidth_hz: Estimated bandwidth in Hz.
        modulation: Fingerprint modulation value (e.g. "FM"), or None.
        top_match_names: Classifier match names, best first (may be empty).
    """

    center_freq_hz: float
    peak_power_dbm: float
    bandwidth_hz: float
    modulation: str | None
    top_match_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class NearMiss:
    """A classification near-miss (substring overlap, not counted as a hit)."""

    truth_label: str | None
    truth_signal_type: str
    matched_name: str


@dataclass(frozen=True)
class PerTruthOutcome:
    """Per-truth-signal scoring outcome."""

    truth: GroundTruthSignal
    found: bool
    num_detections: int
    primary: DetectionRecord | None
    modulation_correct: bool | None      # None = not scored (no truth modulation / not found)
    classification_top1: bool | None
    classification_top3: bool | None


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete scoring result for one benchmark run."""

    num_truth: int
    num_detections: int
    recall: float
    primary_precision: float
    fragmentation_factor: float
    spurious_count: int
    spurious_rate: float
    modulation_accuracy: float | None
    modulation_confusion: tuple[list[str], list[str], np.ndarray] | None
    classification_top1_accuracy: float | None
    classification_top3_accuracy: float | None
    classification_confusion: tuple[list[str], list[str], np.ndarray] | None
    per_truth: list[PerTruthOutcome]
    near_misses: list[NearMiss]
    worst_spurious: list[DetectionRecord]
    config_summary: dict = field(default_factory=dict)


def _norm_name(name: str) -> str:
    """Normalise a signal-type name for exact matching."""
    return name.strip().casefold()


def _assign(
    detections: list[DetectionRecord], truth: GroundTruth
) -> tuple[dict[int, list[int]], list[int]]:
    """Assign each detection to the nearest truth within tolerance.

    Returns:
        (assignments, spurious) where assignments maps a truth index to the
        list of detection indices nearest it, and spurious is the list of
        detection indices that matched no truth.
    """
    # Truth centres sorted ascending so argmin breaks ties toward lower freq.
    order = sorted(range(len(truth.signals)), key=lambda i: truth.signals[i].center_freq_hz)
    centers = np.array([truth.signals[i].center_freq_hz for i in order], dtype=np.float64)

    assignments: dict[int, list[int]] = {}
    spurious: list[int] = []

    for di, det in enumerate(detections):
        if len(centers) == 0:
            spurious.append(di)
            continue
        nearest_pos = int(np.argmin(np.abs(centers - det.center_freq_hz)))
        if abs(centers[nearest_pos] - det.center_freq_hz) <= truth.match_tolerance_hz:
            truth_idx = order[nearest_pos]
            assignments.setdefault(truth_idx, []).append(di)
        else:
            spurious.append(di)

    return assignments, spurious


def _pick_primary(detections: list[DetectionRecord], indices: list[int], truth_freq: float) -> int:
    """Pick the primary detection index: highest power, ties to closest freq."""
    return max(
        indices,
        key=lambda di: (
            detections[di].peak_power_dbm,
            -abs(detections[di].center_freq_hz - truth_freq),
        ),
    )


def _modulation_confusion(
    scored: list[tuple[GroundTruthSignal, DetectionRecord]],
) -> tuple[list[str], list[str], np.ndarray] | None:
    """Build the modulation confusion matrix (truth rows x predicted cols)."""
    if not scored:
        return None
    rows = sorted({gt.modulation for gt, _ in scored if gt.modulation})
    if not rows:
        return None
    cols = list(_MODULATION_COLS)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}
    matrix = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for gt, det in scored:
        if not gt.modulation:
            continue
        pred = det.modulation if det.modulation else _NONE_LABEL
        if pred not in col_idx:
            pred = _NONE_LABEL
        matrix[row_idx[gt.modulation], col_idx[pred]] += 1
    return rows, cols, matrix


def _classification_confusion(
    scored: list[tuple[GroundTruthSignal, DetectionRecord]],
) -> tuple[list[str], list[str], np.ndarray] | None:
    """Build the top-1 classification confusion matrix when multi-class.

    Skipped (returns None) for a single distinct truth signal_type, where a
    1-row matrix is uninformative.
    """
    typed = [(gt, det) for gt, det in scored if gt.signal_type]
    truth_types = sorted({gt.signal_type for gt, _ in typed})
    if len(truth_types) <= 1:
        return None
    predicted = sorted({
        det.top_match_names[0] for _, det in typed if det.top_match_names
    } | {"(no match)"})
    row_idx = {t: i for i, t in enumerate(truth_types)}
    col_idx = {p: i for i, p in enumerate(predicted)}
    matrix = np.zeros((len(truth_types), len(predicted)), dtype=np.int64)
    for gt, det in typed:
        pred = det.top_match_names[0] if det.top_match_names else "(no match)"
        if pred not in col_idx:
            continue
        matrix[row_idx[gt.signal_type], col_idx[pred]] += 1
    return truth_types, predicted, matrix


def score_benchmark(
    detections: list[DetectionRecord],
    truth: GroundTruth,
    config_summary: dict | None = None,
) -> BenchmarkReport:
    """Score detections against ground truth.

    Args:
        detections: All detections from one pipeline run.
        truth: The ground-truth labels for the session.
        config_summary: Optional echo of the config/profile used, stored on the
            report for provenance.

    Returns:
        A fully populated :class:`BenchmarkReport`.
    """
    assignments, spurious = _assign(detections, truth)

    num_truth = len(truth.signals)
    num_det = len(detections)
    matched = len(assignments)
    matched_dets = sum(len(v) for v in assignments.values())
    spurious_count = len(spurious)

    recall = matched / num_truth if num_truth else 0.0
    fragmentation = matched_dets / matched if matched else 0.0
    spurious_rate = spurious_count / num_det if num_det else 0.0
    primary_precision = matched / (matched + spurious_count) if (matched + spurious_count) else 0.0

    # Per-truth outcomes + collect scored (truth, primary) pairs.
    per_truth: list[PerTruthOutcome] = []
    scored: list[tuple[GroundTruthSignal, DetectionRecord]] = []
    near_misses: list[NearMiss] = []

    mod_total = mod_correct = 0
    top1_total = top1_correct = top3_correct = 0

    for ti, gt in enumerate(truth.signals):
        det_indices = assignments.get(ti, [])
        found = bool(det_indices)
        primary_det: DetectionRecord | None = None
        mod_correct_flag: bool | None = None
        top1_flag: bool | None = None
        top3_flag: bool | None = None

        if found:
            primary_idx = _pick_primary(detections, det_indices, gt.center_freq_hz)
            primary_det = detections[primary_idx]
            scored.append((gt, primary_det))

            if gt.modulation:
                mod_total += 1
                mod_correct_flag = (
                    primary_det.modulation is not None
                    and primary_det.modulation.upper() == gt.modulation
                )
                if mod_correct_flag:
                    mod_correct += 1

            if gt.signal_type:
                top1_total += 1
                names = [_norm_name(n) for n in primary_det.top_match_names]
                target = _norm_name(gt.signal_type)
                top1_flag = bool(names) and names[0] == target
                top3_flag = target in set(names[:3])
                if top1_flag:
                    top1_correct += 1
                if top3_flag:
                    top3_correct += 1
                if not top3_flag:
                    near_misses.extend(_find_near_misses(gt, primary_det))

        per_truth.append(PerTruthOutcome(
            truth=gt,
            found=found,
            num_detections=len(det_indices),
            primary=primary_det,
            modulation_correct=mod_correct_flag,
            classification_top1=top1_flag,
            classification_top3=top3_flag,
        ))

    modulation_accuracy = (mod_correct / mod_total) if mod_total else None
    top1_accuracy = (top1_correct / top1_total) if top1_total else None
    top3_accuracy = (top3_correct / top1_total) if top1_total else None

    worst_spurious = sorted(
        (detections[i] for i in spurious),
        key=lambda d: d.peak_power_dbm,
        reverse=True,
    )[:_WORST_SPURIOUS_N]

    return BenchmarkReport(
        num_truth=num_truth,
        num_detections=num_det,
        recall=recall,
        primary_precision=primary_precision,
        fragmentation_factor=fragmentation,
        spurious_count=spurious_count,
        spurious_rate=spurious_rate,
        modulation_accuracy=modulation_accuracy,
        modulation_confusion=_modulation_confusion(scored),
        classification_top1_accuracy=top1_accuracy,
        classification_top3_accuracy=top3_accuracy,
        classification_confusion=_classification_confusion(scored),
        per_truth=per_truth,
        near_misses=near_misses,
        worst_spurious=worst_spurious,
        config_summary=config_summary or {},
    )


def _find_near_misses(gt: GroundTruthSignal, primary: DetectionRecord) -> list[NearMiss]:
    """Find substring-overlap near-misses among the top-3 names (advisory only)."""
    if not gt.signal_type:
        return []
    target = _norm_name(gt.signal_type)
    out: list[NearMiss] = []
    for name in primary.top_match_names[:3]:
        norm = _norm_name(name)
        if norm == target:
            continue
        if target in norm or norm in target:
            out.append(NearMiss(
                truth_label=gt.label,
                truth_signal_type=gt.signal_type,
                matched_name=name,
            ))
    return out
