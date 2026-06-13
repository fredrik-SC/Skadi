"""Audio → IQ ingest CLI: build a labelled digimode session from audio samples.

Usage:
    # Ingest a folder of WAV samples into a benchmark-ready session
    python -m src.ingest --input samples/bartg --output sessions/digimodes

    # Then score the modulation classifier against it
    python -m src.ingest --input samples/ --output sessions/digimodes
    python -m src.benchmark --session sessions/digimodes --band-plan
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.ingest.audio_iq import build_session

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".wave", ".ogg", ".mp3", ".flac"}


def _collect_inputs(inputs: list[Path]) -> list[Path]:
    """Expand files and directories into a sorted list of audio files."""
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files.extend(
                p for p in sorted(item.iterdir())
                if p.suffix.lower() in _AUDIO_EXTS
            )
        elif item.is_file():
            files.append(item)
        else:
            logger.warning("Input not found: %s", item)
    return files


def main() -> int:
    """Run the ingest CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Convert demodulated digimode audio into a labelled IQ session",
    )
    parser.add_argument("--input", required=True, nargs="+", type=Path,
                        help="Audio files and/or directories to ingest")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output session directory")
    parser.add_argument("--rate", type=float, default=96_000.0,
                        help="Target IQ sample rate in Hz (default: 96000)")
    parser.add_argument("--offset", type=float, default=24_000.0,
                        help="Signal offset from step centre in Hz (default: 24000)")
    parser.add_argument("--snr", type=float, default=25.0,
                        help="Embedded-signal SNR in dB (default: 25)")
    parser.add_argument("--base-freq", type=float, default=14_000_000.0,
                        help="Centre frequency of the first step (default: 14e6)")
    parser.add_argument("--label-map", type=Path, default=None,
                        help="YAML of {stem: {modulation, signal_type, freq_hz}} overrides")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Truncate each clip to this many seconds (default: full)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for noise")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    files = _collect_inputs(args.input)
    if not files:
        print("No audio files found in the given input(s).", file=sys.stderr)
        return 1

    label_map = {}
    if args.label_map is not None:
        label_map = (yaml.safe_load(args.label_map.read_text()) or {})

    summary = build_session(
        files,
        args.output,
        target_rate=args.rate,
        base_freq_hz=args.base_freq,
        offset_hz=args.offset,
        snr_db=args.snr,
        max_seconds=args.max_seconds,
        label_map=label_map,
        seed=args.seed,
    )

    print(f"\nIngested {summary['ingested']} file(s) into {summary['session_dir']}")
    if summary["skipped"]:
        print(f"Skipped {len(summary['skipped'])}: {', '.join(summary['skipped'])}")
    print(f"Truth: {summary['truth_path']}")
    print(f"\nNext: python -m src.benchmark --session {summary['session_dir']} --band-plan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
