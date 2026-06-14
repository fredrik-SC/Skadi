"""Render a :class:`~src.benchmark.scorer.BenchmarkReport` as text or JSON.

The text form is the human-facing summary printed to stdout; the JSON form is a
stable, git-diffable record for tracking metrics across config changes over time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.benchmark.scorer import BenchmarkReport, DetectionRecord, PerTruthOutcome


def _pct(value: float | None) -> str:
    """Format a 0-1 fraction as a percentage, or n/a."""
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _mhz(hz: float) -> str:
    return f"{hz / 1e6:.3f}"


def _flag(value: bool | None) -> str:
    if value is None:
        return "-"
    return "OK" if value else "X"


def _render_confusion(confusion: tuple[list[str], list[str], np.ndarray], title: str) -> str:
    """Render a confusion matrix as a fixed-width ASCII table."""
    rows, cols, matrix = confusion
    # Only show columns that have at least one count, plus keep order stable.
    used_cols = [j for j in range(len(cols)) if matrix[:, j].any()]
    col_labels = [cols[j] for j in used_cols]
    row_w = max((len(r) for r in rows), default=8)
    row_w = max(row_w, len("truth\\pred"))
    col_w = max((len(c) for c in col_labels), default=4)
    col_w = max(col_w, 4)

    lines = [title]
    header = "truth\\pred".ljust(row_w) + " | " + " ".join(c.rjust(col_w) for c in col_labels)
    lines.append("  " + header)
    for i, r in enumerate(rows):
        cells = " ".join(str(int(matrix[i, j])).rjust(col_w) for j in used_cols)
        lines.append("  " + r.ljust(row_w) + " | " + cells)
    return "\n".join(lines)


def render_text(report: BenchmarkReport) -> str:
    """Render a full human-readable benchmark report."""
    out: list[str] = []
    out.append("=" * 66)
    out.append("  SKADI BENCHMARK REPORT")
    out.append("=" * 66)

    if report.config_summary:
        knobs = ", ".join(
            f"{k}={v}" for k, v in report.config_summary.items() if v is not None
        )
        out.append(f"  Profile: {knobs}")
        out.append("-" * 66)

    # Detection
    out.append("DETECTION")
    out.append(f"  Truth signals:        {report.num_truth}")
    out.append(f"  Detections:           {report.num_detections}")
    out.append(f"  Station recall:       {_pct(report.recall)}  "
               f"({sum(1 for p in report.per_truth if p.found)}/{report.num_truth} found)")
    out.append(f"  Primary precision:    {_pct(report.primary_precision)}  "
               f"(matched truths / (matched + spurious))")
    out.append(f"  Fragmentation factor: {report.fragmentation_factor:.2f}x  "
               f"(detections per matched station; 1.0 = clean)")
    out.append(f"  Spurious detections:  {report.spurious_count}  "
               f"({_pct(report.spurious_rate)} of detections)")

    # Modulation
    out.append("")
    out.append("MODULATION")
    out.append(f"  Accuracy:             {_pct(report.modulation_accuracy)}  "
               f"(over found stations with a known modulation)")
    if report.modulation_confusion is not None:
        out.append(_render_confusion(report.modulation_confusion, "  Confusion (truth -> predicted):"))

    # Classification
    out.append("")
    out.append("CLASSIFICATION")
    out.append(f"  Top-1 accuracy:       {_pct(report.classification_top1_accuracy)}")
    out.append(f"  Top-3 accuracy:       {_pct(report.classification_top3_accuracy)}")
    if report.classification_confusion is not None:
        out.append(_render_confusion(report.classification_confusion, "  Confusion (truth -> top-1):"))
    if report.near_misses:
        out.append("  Near-misses (not counted as hits):")
        for nm in report.near_misses[:10]:
            label = f" [{nm.truth_label}]" if nm.truth_label else ""
            out.append(f"    '{nm.truth_signal_type}'{label}  ~  '{nm.matched_name}'")

    # Per-truth table
    out.append("")
    out.append("PER-TRUTH OUTCOMES")
    out.append(f"  {'Freq (MHz)':>11}  {'Label':<16} {'Found':>5} {'#Det':>4} "
               f"{'Mod':>3} {'T1':>2} {'T3':>2}")
    for p in report.per_truth:
        label = (p.truth.label or "")[:16]
        out.append(
            f"  {_mhz(p.truth.center_freq_hz):>11}  {label:<16} "
            f"{('yes' if p.found else 'NO'):>5} {p.num_detections:>4} "
            f"{_flag(p.modulation_correct):>3} {_flag(p.classification_top1):>2} "
            f"{_flag(p.classification_top3):>2}"
        )

    # Worst spurious
    if report.worst_spurious:
        out.append("")
        out.append("WORST SPURIOUS DETECTIONS (by power)")
        for d in report.worst_spurious:
            out.append(
                f"  {_mhz(d.center_freq_hz):>11} MHz  {d.peak_power_dbm:6.1f} dBm  "
                f"BW={d.bandwidth_hz / 1e3:.1f} kHz  mod={d.modulation}"
            )

    out.append("=" * 66)
    return "\n".join(out)


def _detection_to_dict(d: DetectionRecord) -> dict[str, Any]:
    return {
        "center_freq_hz": d.center_freq_hz,
        "peak_power_dbm": d.peak_power_dbm,
        "bandwidth_hz": d.bandwidth_hz,
        "modulation": d.modulation,
        "top_match_names": list(d.top_match_names),
    }


def _per_truth_to_dict(p: PerTruthOutcome) -> dict[str, Any]:
    return {
        "center_freq_hz": p.truth.center_freq_hz,
        "label": p.truth.label,
        "found": p.found,
        "num_detections": p.num_detections,
        "modulation_correct": p.modulation_correct,
        "classification_top1": p.classification_top1,
        "classification_top3": p.classification_top3,
        "primary": _detection_to_dict(p.primary) if p.primary else None,
    }


def _confusion_to_dict(confusion: tuple[list[str], list[str], np.ndarray] | None) -> dict | None:
    if confusion is None:
        return None
    rows, cols, matrix = confusion
    return {"rows": rows, "cols": cols, "matrix": matrix.tolist()}


def to_json(report: BenchmarkReport) -> dict[str, Any]:
    """Convert a report to a JSON-serialisable dict (stable key order)."""
    return {
        "config_summary": report.config_summary,
        "num_truth": report.num_truth,
        "num_detections": report.num_detections,
        "recall": report.recall,
        "primary_precision": report.primary_precision,
        "fragmentation_factor": report.fragmentation_factor,
        "spurious_count": report.spurious_count,
        "spurious_rate": report.spurious_rate,
        "modulation_accuracy": report.modulation_accuracy,
        "modulation_confusion": _confusion_to_dict(report.modulation_confusion),
        "classification_top1_accuracy": report.classification_top1_accuracy,
        "classification_top3_accuracy": report.classification_top3_accuracy,
        "classification_confusion": _confusion_to_dict(report.classification_confusion),
        "near_misses": [
            {
                "truth_label": nm.truth_label,
                "truth_signal_type": nm.truth_signal_type,
                "matched_name": nm.matched_name,
            }
            for nm in report.near_misses
        ],
        "per_truth": [_per_truth_to_dict(p) for p in report.per_truth],
        "worst_spurious": [_detection_to_dict(d) for d in report.worst_spurious],
    }
