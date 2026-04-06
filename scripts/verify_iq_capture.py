#!/usr/bin/env python3
"""Verify SDR IQ capture by plotting a spectrum from a known frequency.

Captures IQ samples from the RSPduo at a given frequency (default 100 MHz,
targeting FM broadcast) and produces a power spectral density plot and
spectrogram to confirm valid signal acquisition.

Usage:
    python scripts/verify_iq_capture.py [--freq 100e6] [--duration 0.5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Add project root to path so we can import src package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.sdr.interface import SDRInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_psd(
    iq_data: np.ndarray, sample_rate: float, fft_size: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method.

    Args:
        iq_data: Complex IQ samples.
        sample_rate: Sample rate in Hz.
        fft_size: FFT segment length.

    Returns:
        Tuple of (frequencies in Hz, PSD in dB).
    """
    freqs, psd = signal.welch(
        iq_data,
        fs=sample_rate,
        nperseg=fft_size,
        return_onesided=False,
    )
    # Shift so DC is in the centre
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    # Convert to dB (avoid log of zero)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd_db


def main() -> None:
    """Run the IQ capture verification."""
    parser = argparse.ArgumentParser(
        description="Verify SDR IQ capture with spectrum plot"
    )
    parser.add_argument(
        "--freq", type=float, default=100e6,
        help="Centre frequency in Hz (default: 100 MHz)",
    )
    parser.add_argument(
        "--duration", type=float, default=0.5,
        help="Capture duration in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    sdr_config = config.get("sdr", {})
    sample_rate = float(sdr_config.get("sample_rate", 2_048_000))
    num_samples = int(sample_rate * args.duration)

    logger.info(
        "Capturing %.2f seconds (%d samples) at %.3f MHz",
        args.duration, num_samples, args.freq / 1e6,
    )

    # Capture IQ data
    with SDRInterface(sdr_config) as sdr:
        sdr.tune(args.freq)
        iq_data = sdr.capture(num_samples)

    logger.info("Capture complete. Processing...")

    # Compute PSD
    freqs, psd_db = compute_psd(iq_data, sample_rate)
    freqs_mhz = (freqs + args.freq) / 1e6  # Convert to absolute MHz

    # Find peak
    peak_idx = np.argmax(psd_db)
    peak_freq_mhz = freqs_mhz[peak_idx]
    peak_power_db = psd_db[peak_idx]
    noise_floor_db = float(np.median(psd_db))

    logger.info("Peak: %.3f MHz at %.1f dB", peak_freq_mhz, peak_power_db)
    logger.info("Noise floor: %.1f dB", noise_floor_db)
    logger.info("SNR: %.1f dB", peak_power_db - noise_floor_db)

    # Create figure with two subplots
    fig, (ax_psd, ax_spec) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        f"Skaði IQ Capture Verification — {args.freq / 1e6:.3f} MHz",
        fontsize=14,
    )

    # Top: Power spectral density
    ax_psd.plot(freqs_mhz, psd_db, linewidth=0.5)
    ax_psd.axhline(
        y=noise_floor_db, color="r", linestyle="--", alpha=0.5,
        label=f"Noise floor: {noise_floor_db:.1f} dB",
    )
    ax_psd.set_xlabel("Frequency (MHz)")
    ax_psd.set_ylabel("Power Spectral Density (dB)")
    ax_psd.set_title("Power Spectrum")
    ax_psd.legend()
    ax_psd.grid(True, alpha=0.3)

    # Bottom: Spectrogram
    ax_spec.specgram(
        iq_data, NFFT=1024, Fs=sample_rate, Fc=args.freq,
        noverlap=512, cmap="viridis",
    )
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Spectrogram")

    plt.tight_layout()

    # Save to data/
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "verify_spectrum.png"
    fig.savefig(output_path, dpi=150)
    logger.info("Spectrum plot saved to %s", output_path)

    # Print summary
    print("\n" + "=" * 50)
    print("CAPTURE SUMMARY")
    print("=" * 50)
    print(f"  Centre frequency:  {args.freq / 1e6:.3f} MHz")
    print(f"  Sample rate:       {sample_rate / 1e6:.3f} MHz")
    print(f"  Duration:          {args.duration:.2f} s")
    print(f"  Samples captured:  {len(iq_data):,}")
    print(f"  Peak frequency:    {peak_freq_mhz:.3f} MHz")
    print(f"  Peak power:        {peak_power_db:.1f} dB")
    print(f"  Noise floor:       {noise_floor_db:.1f} dB")
    print(f"  SNR:               {peak_power_db - noise_floor_db:.1f} dB")
    print(f"  Plot saved:        {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
