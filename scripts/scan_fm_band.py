#!/usr/bin/env python3
"""Scan the FM broadcast band (88-108 MHz) and report detected signals.

Session 2 acceptance test: verify that the scanning engine detects
known FM broadcast stations with reasonable frequency and bandwidth
estimates.

Usage:
    python scripts/scan_fm_band.py [--start 88e6] [--stop 108e6] [--plot]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, PROJECT_ROOT
from src.detection.detector import SignalDetector
from src.detection.exclusions import ExclusionFilter
from src.detection.noise import NoiseEstimator
from src.detectionlog.database import DetectionLog
from src.sdr.interface import SDRInterface
from src.sdr.scanner import SpectrumScanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def progress_callback(step, step_index: int, total_steps: int) -> None:
    """Print scan progress."""
    pct = (step_index + 1) / total_steps * 100
    print(
        f"  Step {step_index + 1}/{total_steps} "
        f"({pct:.0f}%) — {step.centre_freq_hz / 1e6:.3f} MHz, "
        f"noise floor: {step.noise_floor_dbm:.1f} dBm",
        flush=True,
    )


def save_plot(result, output_path: Path) -> None:
    """Save a stitched PSD plot with detected signals marked."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot stitched PSD from all scan steps
    for step in result.scan_steps:
        ax.plot(
            step.freqs_hz / 1e6, step.psd_dbm,
            linewidth=0.5, color="steelblue", alpha=0.8,
        )

    # Mark detected signals
    for sig in result.signals:
        ax.axvline(
            x=sig.centre_freq_hz / 1e6, color="red",
            linewidth=0.8, alpha=0.6, linestyle="--",
        )
        ax.annotate(
            f"{sig.centre_freq_hz / 1e6:.2f} MHz\n{sig.bandwidth_hz / 1e3:.0f} kHz",
            xy=(sig.centre_freq_hz / 1e6, sig.peak_power_dbm),
            fontsize=7, color="red", ha="center",
            xytext=(0, 10), textcoords="offset points",
        )

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power Spectral Density (dBm)")
    ax.set_title(
        f"Skaði FM Band Scan — "
        f"{result.freq_start_hz / 1e6:.0f}-{result.freq_stop_hz / 1e6:.0f} MHz  "
        f"({len(result.signals)} signals detected)"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def main() -> None:
    """Run the FM band scan."""
    parser = argparse.ArgumentParser(
        description="Scan the FM broadcast band and detect signals"
    )
    parser.add_argument(
        "--start", type=float, default=88e6,
        help="Start frequency in Hz (default: 88 MHz)",
    )
    parser.add_argument(
        "--stop", type=float, default=108e6,
        help="Stop frequency in Hz (default: 108 MHz)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a spectrum plot to data/fm_band_scan.png",
    )
    args = parser.parse_args()

    # Load config and override scan range
    config = load_config()
    scan_config = config["scan"].copy()
    scan_config["freq_start"] = args.start
    scan_config["freq_stop"] = args.stop

    sdr_config = config["sdr"]
    detection_config = config["detection"]

    print(f"\nScanning {args.start / 1e6:.0f} - {args.stop / 1e6:.0f} MHz...")
    print(f"Step size: {scan_config['step_size'] / 1e6:.1f} MHz, "
          f"Dwell time: {scan_config['dwell_time']}s\n")

    # Load exclusion list
    exclusions_path = PROJECT_ROOT / "config" / "exclusions.yaml"
    exclusion_filter = ExclusionFilter(exclusions_path)
    if exclusion_filter.entries:
        print(f"Exclusion list: {len(exclusion_filter.entries)} entries loaded")

    # Set up detection log
    log_path = Path(PROJECT_ROOT / config["logging"]["database_path"])
    detection_log = DetectionLog(log_path)

    with SDRInterface(sdr_config) as sdr:
        scanner = SpectrumScanner(
            sdr, scan_config, sdr_config, detection_config,
            exclusion_filter=exclusion_filter,
            detection_log=detection_log,
        )
        print(f"Total steps: {scanner.num_steps}\n")

        result = scanner.sweep(callback=progress_callback)

    detection_log.close()

    # Print results table
    print("\n" + "=" * 75)
    print(f"{'DETECTED SIGNALS':^75}")
    print("=" * 75)
    print(f"{'#':<4} {'Freq (MHz)':<14} {'BW (kHz)':<12} {'Peak (dBm)':<13} "
          f"{'Mean (dBm)':<13} {'SNR (dB)':<10}")
    print("-" * 75)

    if not result.signals:
        print("  No signals detected.")
    else:
        for i, sig in enumerate(result.signals, 1):
            print(
                f"{i:<4} {sig.centre_freq_hz / 1e6:<14.3f} "
                f"{sig.bandwidth_hz / 1e3:<12.1f} "
                f"{sig.peak_power_dbm:<13.1f} "
                f"{sig.mean_power_dbm:<13.1f} "
                f"{sig.snr_db:<10.1f}"
            )

    print("-" * 75)
    print(f"Total: {len(result.signals)} signal(s) in {result.duration_seconds:.1f}s")
    print("=" * 75)

    if args.plot:
        output_path = Path(__file__).resolve().parent.parent / "data" / "fm_band_scan.png"
        save_plot(result, output_path)


if __name__ == "__main__":
    main()
