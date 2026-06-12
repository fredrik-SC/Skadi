"""Benchmark runner: replay a session through the pipeline and score it.

Wires a recorded session through the real detect → fingerprint → classify path
(reusing :class:`~src.sdr.replay.ReplaySource` and
:class:`~src.sdr.scanner.SpectrumScanner`) and scores the flattened detections
against ground truth. The scanner runs with a fingerprint extractor but no
classifier and no detection log, so nothing is written to disk and the
classification is done explicitly here for structured top-N access.

Because the manifest stores the recording's scan/SDR config, those win over the
loaded config, while detection/fingerprint/classification/capture overrides flow
through — which is what lets the same recording be scored under different
profiles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from src.benchmark.scorer import BenchmarkReport, DetectionRecord, score_benchmark
from src.benchmark.truth import GroundTruth
from src.fingerprint.extractor import FingerprintExtractor
from src.sdr.replay import ReplaySource
from src.sdr.scanner import SpectrumScanner

logger = logging.getLogger(__name__)


class _Classifier(Protocol):
    """Minimal classifier interface the runner needs (real or stubbed)."""

    def classify(self, fingerprint: Any) -> Any: ...


def _build_classifier(artemis_path: Path, classification_config: dict) -> _Classifier | None:
    """Build the Artemis classifier, or None if the database is unavailable."""
    if not artemis_path.exists():
        logger.warning(
            "Artemis DB not found at %s; classification dimension will be skipped",
            artemis_path,
        )
        return None
    from src.classification.artemis_db import ArtemisDB
    from src.classification.classifier import SignalClassifier

    db = ArtemisDB(artemis_path)
    return SignalClassifier(db, classification_config)


def run_benchmark(
    session_dir: Path,
    truth: GroundTruth,
    config: dict[str, Any],
    *,
    artemis_path: Path,
    classifier: _Classifier | None = None,
    classification_top_n: int = 3,
) -> BenchmarkReport:
    """Replay a recorded session and score it against ground truth.

    Args:
        session_dir: The recorded session directory.
        truth: Ground-truth labels for the session.
        config: Loaded config (already preset/override-mutated by the caller).
        artemis_path: Path to the Artemis SQLite database.
        classifier: Optional pre-built classifier (used by tests to inject a
            stub). When None, one is built from ``artemis_path`` if present.
        classification_top_n: Number of match names to retain per detection.

    Returns:
        The :class:`BenchmarkReport`.
    """
    replay = ReplaySource(session_dir)

    # Manifest scan/SDR config wins so step frequencies regenerate exactly;
    # detection/fingerprint/capture/classification overrides flow through.
    scan_config = {**config.get("scan", {}), **replay.scan_config}
    sdr_config = {**config.get("sdr", {}), **replay.sdr_config}

    extractor = FingerprintExtractor(
        sample_rate=replay.sample_rate,
        config=config.get("fingerprint", {}),
    )
    scanner = SpectrumScanner(
        replay,
        scan_config,
        sdr_config,
        config.get("detection", {}),
        fingerprint_extractor=extractor,
        signal_classifier=None,
        detection_log=None,
        capture_config=config.get("capture", {}),
    )
    result = scanner.sweep()

    if classifier is None:
        classifier = _build_classifier(artemis_path, config.get("classification", {}))

    detections: list[DetectionRecord] = []
    for fp in result.fingerprints:
        names: tuple[str, ...] = ()
        if classifier is not None:
            cr = classifier.classify(fp)
            names = tuple(m.signal.name for m in cr.matches[:classification_top_n])
        detections.append(DetectionRecord(
            center_freq_hz=fp.signal.centre_freq_hz,
            peak_power_dbm=fp.signal.peak_power_dbm,
            bandwidth_hz=fp.bandwidth_hz,
            modulation=fp.modulation.value,
            top_match_names=names,
        ))

    logger.info(
        "Benchmark: %d fingerprintable detection(s) over %d step(s) in %.1fs",
        len(detections), result.num_steps, result.duration_seconds,
    )

    return score_benchmark(detections, truth, config_summary=_config_summary(config))


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Echo the knobs that affect the run, for report provenance."""
    det = config.get("detection", {})
    cap = config.get("capture", {})
    cls = config.get("classification", {})
    return {
        "threshold_db": det.get("threshold_db"),
        "min_bandwidth_hz": det.get("min_bandwidth_hz"),
        "edge_guard_fraction": det.get("edge_guard_fraction"),
        "dc_removal": cap.get("dc_removal"),
        "bandwidth_tolerance": cls.get("bandwidth_tolerance"),
        "min_confidence": cls.get("min_confidence"),
    }
