"""Skaði application entry point.

Orchestrates the complete RF signal identification pipeline:
scan → detect → fingerprint → classify → threat assess → log.

Usage:
    python -m src.main [--single] [--start 88e6] [--stop 108e6] [--export]
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from src.classification.artemis_db import ArtemisDB
from src.classification.classifier import SignalClassifier
from src.classification.threat import ThreatMapper
from src.config import PROJECT_ROOT, load_config
from src.detection.exclusions import ExclusionFilter
from src.detectionlog.database import DetectionLog
from src.detectionlog.export import export_json
from src.fingerprint.extractor import FingerprintExtractor
from src.sdr.interface import SDRInterface
from src.sdr.scanner import SpectrumScanner

logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested... finishing current sweep.")


def _print_sweep_summary(result, detection_log):
    """Print a summary table by reading from the detection log."""
    # Read the most recent detections from the log
    rows = detection_log.query(limit=len(result.signals) + 10)
    # Sort by signal_strength descending (already stored)
    rows.sort(key=lambda r: r.get("signal_strength_dbm") or -999, reverse=True)

    print(f"\n{'=' * 110}")
    print(f"{'SKAÐI DETECTION REPORT':^110}")
    print(f"{'=' * 110}")
    print(
        f"{'#':<4} {'Freq (MHz)':<12} {'Mod':<8} {'Signal Type':<32} "
        f"{'Conf':<6} {'Threat':<14} {'BW (kHz)':<10} {'Power (dBm)':<12}"
    )
    print("-" * 110)

    if not rows:
        print("  No signals detected.")
    else:
        for i, row in enumerate(rows[:30], 1):  # Top 30
            freq = row.get("frequency_hz", 0) / 1e6
            mod = row.get("modulation") or "?"
            sig_type = (row.get("signal_type") or "UNKNOWN")[:30]
            conf = row.get("confidence_score")
            conf_str = f"{conf:.2f}" if conf is not None else "-"
            threat = row.get("threat_level") or "?"
            bw = (row.get("bandwidth_hz") or 0) / 1e3
            power = row.get("signal_strength_dbm") or 0

            print(
                f"{i:<4} {freq:<12.3f} "
                f"{mod:<8} {sig_type:<32} "
                f"{conf_str:<6} {threat:<14} "
                f"{bw:<10.1f} {power:<12.1f}"
            )

    print("-" * 110)
    print(
        f"Total: {len(result.signals)} signal(s) in {result.duration_seconds:.1f}s  |  "
        f"Range: {result.freq_start_hz / 1e6:.0f}-{result.freq_stop_hz / 1e6:.0f} MHz"
    )
    print("=" * 110)


def main() -> None:
    """Run the Skaði RF signal identification pipeline."""
    parser = argparse.ArgumentParser(
        description="Skaði — RF Signal Identification Tool",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run a single sweep and exit (default: continuous)",
    )
    parser.add_argument(
        "--start", type=float, default=None,
        help="Override start frequency in Hz (default: from config)",
    )
    parser.add_argument(
        "--stop", type=float, default=None,
        help="Override stop frequency in Hz (default: from config)",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export detection log to JSON file after sweep(s)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    config = load_config()
    sdr_config = config["sdr"]
    scan_config = config["scan"].copy()
    detection_config = config["detection"]

    if args.start:
        scan_config["freq_start"] = args.start
    if args.stop:
        scan_config["freq_stop"] = args.stop

    print("\n" + "=" * 60)
    print("  SKAÐI — RF Signal Identification Tool v1.0")
    print("=" * 60)
    print(f"  Scan range: {scan_config['freq_start'] / 1e6:.0f} - {scan_config['freq_stop'] / 1e6:.0f} MHz")
    print(f"  Step size:  {scan_config['step_size'] / 1e6:.1f} MHz")
    print(f"  Mode:       {'Single sweep' if args.single else 'Continuous'}")

    # Initialise components
    exclusion_filter = ExclusionFilter(PROJECT_ROOT / "config" / "exclusions.yaml")
    if exclusion_filter.entries:
        print(f"  Exclusions: {len(exclusion_filter.entries)} entries")

    log_path = Path(PROJECT_ROOT / config["logging"]["database_path"])
    detection_log = DetectionLog(log_path)

    fingerprint_extractor = FingerprintExtractor(
        sample_rate=float(sdr_config.get("sample_rate", 2_048_000)),
        config=config.get("fingerprint", {}),
    )

    # Artemis classifier
    artemis_path = PROJECT_ROOT / "data" / "artemis.db"
    signal_classifier = None
    if artemis_path.exists():
        artemis_db = ArtemisDB(artemis_path)
        signal_classifier = SignalClassifier(artemis_db, config.get("classification", {}))
        print(f"  Artemis DB: {len(artemis_db.signals)} signals")

    # Threat mapper
    threat_mapper = ThreatMapper(PROJECT_ROOT / "config" / "threat_levels.yaml")
    print(f"  Threats:    {len(threat_mapper._rules)} rules (default: {threat_mapper.default_level})")
    print("=" * 60 + "\n")

    # Set up graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)

    sweep_count = 0

    try:
        with SDRInterface(sdr_config) as sdr:
            scanner = SpectrumScanner(
                sdr, scan_config, sdr_config, detection_config,
                exclusion_filter=exclusion_filter,
                detection_log=detection_log,
                fingerprint_extractor=fingerprint_extractor,
                signal_classifier=signal_classifier,
                threat_mapper=threat_mapper,
            )

            while not _shutdown_requested:
                sweep_count += 1
                print(f"\n--- Sweep #{sweep_count} ---")

                def progress(step, idx, total):
                    print(
                        f"  [{idx + 1}/{total}] {step.centre_freq_hz / 1e6:.1f} MHz "
                        f"(noise: {step.noise_floor_dbm:.1f} dBm)",
                        flush=True,
                    )

                result = scanner.sweep(callback=progress)
                _print_sweep_summary(result, detection_log)

                if args.single:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        detection_log.close()
        print(f"\nSkaði shutdown. {sweep_count} sweep(s) completed.")

        # Export if requested
        if args.export:
            export_path = Path(args.export)
            count = export_json(log_path, export_path)
            print(f"Exported {count} detection(s) to {export_path}")


if __name__ == "__main__":
    main()
