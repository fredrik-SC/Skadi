"""Skadi application entry point.

Orchestrates the complete RF signal identification pipeline:
scan -> detect -> fingerprint -> classify -> threat assess -> log.

Runs a web dashboard alongside the scan loop for real-time monitoring.

Usage:
    python -m src.main [--single] [--preset hf] [--start 88e6] [--stop 108e6] [--no-web]
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime, timezone
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

_shutdown_requested = False


def _run_scan_loop(
    sdr_config: dict,
    scan_config: dict,
    detection_config: dict,
    config: dict,
    exclusion_filter,
    detection_log: DetectionLog,
    fingerprint_extractor,
    signal_classifier,
    threat_mapper,
    app=None,
    broadcaster=None,
    single: bool = False,
) -> None:
    """Run the scan loop with SDR crash resilience and UI control."""
    global _shutdown_requested
    sweep_count = 0
    active_scan_config = scan_config.copy()

    def _set_status(scanning: bool, error: str | None = None, **extra):
        if app:
            app.config["SCANNER_ACTIVE"] = scanning
            app.config["LAST_ERROR"] = error
            for k, v in extra.items():
                app.config[k] = v
        if broadcaster:
            broadcaster.broadcast_status({
                "scanning": scanning,
                "sweep_count": sweep_count,
                "total_detections": detection_log.count() if detection_log else 0,
                "error": error,
                **extra,
            })

    while not _shutdown_requested:
        # Check if UI requested a scan start (when in web mode)
        if app and not single:
            # Check for frequency range updates from UI
            ui_start = app.config.get("SCAN_FREQ_START")
            ui_stop = app.config.get("SCAN_FREQ_STOP")
            if ui_start:
                active_scan_config["freq_start"] = ui_start
                app.config["SCAN_FREQ_START"] = None
            if ui_stop:
                active_scan_config["freq_stop"] = ui_stop
                app.config["SCAN_FREQ_STOP"] = None

            # Check for stop request
            if app.config.get("SCAN_REQUESTED") is False and sweep_count > 0:
                _set_status(False, last_sweep_time="Stopped by operator")
                logger.info("Scan stopped by operator, waiting for start request...")
                while not _shutdown_requested and not app.config.get("SCAN_REQUESTED", True):
                    time.sleep(0.5)
                    # Check for new freq range while paused
                    ui_start = app.config.get("SCAN_FREQ_START")
                    ui_stop = app.config.get("SCAN_FREQ_STOP")
                    if ui_start:
                        active_scan_config["freq_start"] = ui_start
                        app.config["SCAN_FREQ_START"] = None
                    if ui_stop:
                        active_scan_config["freq_stop"] = ui_stop
                        app.config["SCAN_FREQ_STOP"] = None
                if _shutdown_requested:
                    break
                logger.info("Scan resumed by operator")

        # Try to connect to SDR and run a sweep
        try:
            with SDRInterface(sdr_config) as sdr:
                scanner = SpectrumScanner(
                    sdr, active_scan_config, sdr_config, detection_config,
                    exclusion_filter=exclusion_filter,
                    detection_log=detection_log,
                    fingerprint_extractor=fingerprint_extractor,
                    signal_classifier=signal_classifier,
                    threat_mapper=threat_mapper,
                )

                # Run sweeps until stopped or SDR disconnects
                while not _shutdown_requested:
                    # Check stop request mid-loop
                    if app and app.config.get("SCAN_REQUESTED") is False:
                        break

                    sweep_count += 1
                    _set_status(True, last_sweep_time=f"Sweep #{sweep_count} starting...")

                    def step_callback(step, idx, total_steps):
                        if app:
                            app.config["TOTAL_DETECTIONS"] = detection_log.count()
                        if broadcaster:
                            recent = detection_log.query(limit=20)
                            broadcaster.broadcast_detections(recent)
                            broadcaster.broadcast_status({
                                "scanning": True,
                                "sweep_count": sweep_count,
                                "total_detections": detection_log.count(),
                                "last_sweep_time": f"Step {idx+1}/{total_steps}",
                            })

                    result = scanner.sweep(callback=step_callback)

                    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                    _set_status(
                        False,
                        LAST_SWEEP_TIME=now_str,
                        TOTAL_DETECTIONS=detection_log.count(),
                        SWEEP_COUNT=sweep_count,
                    )

                    logger.info(
                        "Sweep #%d complete: %d signal(s) in %.1fs",
                        sweep_count, len(result.signals), result.duration_seconds,
                    )

                    if single:
                        return

        except Exception as e:
            error_msg = str(e)
            logger.error("SDR/scan error: %s", error_msg)
            _set_status(False, error=f"SDR Error: {error_msg}")

            if _shutdown_requested or single:
                break

            # Wait before retry — don't spam reconnection attempts
            logger.info("Retrying SDR connection in 5 seconds...")
            for _ in range(10):
                if _shutdown_requested:
                    return
                time.sleep(0.5)

    logger.info("Scan loop stopped after %d sweep(s)", sweep_count)


def main() -> None:
    """Run the Skadi RF signal identification pipeline."""
    parser = argparse.ArgumentParser(
        description="Skadi -- RF Signal Identification Tool",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run a single sweep and exit (default: continuous)",
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        choices=["hf", "vhf", "uhf", "airband", "military_hf"],
        help="Use a scan preset (overrides freq range, FFT size, dwell time)",
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
        "--no-web", action="store_true",
        help="Disable the web server (CLI-only mode)",
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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    config = load_config()
    sdr_config = config["sdr"]
    scan_config = config["scan"].copy()
    detection_config = config["detection"]
    web_config = config.get("web", {})

    # Apply scan preset
    if args.preset:
        presets = config.get("scan_presets", {})
        if args.preset in presets:
            scan_config.update(presets[args.preset])
            logger.info("Applied scan preset: %s", args.preset)

    if args.start:
        scan_config["freq_start"] = args.start
    if args.stop:
        scan_config["freq_stop"] = args.stop

    print("\n" + "=" * 60)
    print("  SKADI -- RF Signal Identification Tool v1.0")
    print("=" * 60)
    print(f"  Scan range: {scan_config['freq_start'] / 1e6:.0f} - {scan_config['freq_stop'] / 1e6:.0f} MHz")
    print(f"  Step size:  {scan_config['step_size'] / 1e6:.1f} MHz")
    print(f"  FFT size:   {scan_config.get('fft_size', 8192)} ({scan_config.get('sample_rate', 2048000) / scan_config.get('fft_size', 8192):.0f} Hz/bin)")
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

    artemis_path = PROJECT_ROOT / "data" / "artemis.db"
    signal_classifier = None
    if artemis_path.exists():
        artemis_db = ArtemisDB(artemis_path)
        signal_classifier = SignalClassifier(artemis_db, config.get("classification", {}))
        print(f"  Artemis DB: {len(artemis_db.signals)} signals")

    threat_mapper = ThreatMapper(PROJECT_ROOT / "config" / "threat_levels.yaml")
    print(f"  Threats:    {len(threat_mapper._rules)} rules (default: {threat_mapper.default_level})")

    if not args.no_web:
        host = web_config.get("host", "127.0.0.1")
        port = int(web_config.get("port", 8050))
        print(f"  Web GUI:    http://{host}:{port}")

    print("=" * 60 + "\n")

    scan_kwargs = dict(
        sdr_config=sdr_config,
        scan_config=scan_config,
        detection_config=detection_config,
        config=config,
        exclusion_filter=exclusion_filter,
        detection_log=detection_log,
        fingerprint_extractor=fingerprint_extractor,
        signal_classifier=signal_classifier,
        threat_mapper=threat_mapper,
        single=args.single,
    )

    if args.no_web:
        import signal as sig_mod
        sig_mod.signal(sig_mod.SIGINT, lambda s, f: globals().update(_shutdown_requested=True))
        _run_scan_loop(**scan_kwargs)
        detection_log.close()
    else:
        from src.web.server import create_app, socketio
        from src.web.websocket import AlertBroadcaster

        host = web_config.get("host", "127.0.0.1")
        port = int(web_config.get("port", 8050))

        app = create_app(log_path, web_config)
        broadcaster = AlertBroadcaster(socketio)

        # Set initial scan state — scanning starts automatically
        app.config["SCAN_REQUESTED"] = True

        scan_kwargs["app"] = app
        scan_kwargs["broadcaster"] = broadcaster

        scan_thread = threading.Thread(
            target=_run_scan_loop,
            kwargs=scan_kwargs,
            daemon=True,
            name="scan-loop",
        )
        scan_thread.start()

        try:
            socketio.run(
                app, host=host, port=port,
                debug=False, use_reloader=False,
                log_output=False, allow_unsafe_werkzeug=True,
            )
        except KeyboardInterrupt:
            pass
        finally:
            global _shutdown_requested
            _shutdown_requested = True
            scan_thread.join(timeout=5)
            detection_log.close()

    print(f"\nSkadi shutdown complete.")

    if args.export:
        export_path = Path(args.export)
        count = export_json(log_path, export_path)
        print(f"Exported {count} detection(s) to {export_path}")


if __name__ == "__main__":
    main()
