"""Benchmark CLI: score a recorded session against ground truth.

Usage:
    # Propose a starter truth.json (then hand-edit and verify it)
    python -m src.benchmark --session sessions/fm_groundtruth --seed-truth

    # Score under a clean FM profile, write JSON results
    python -m src.benchmark --session sessions/fm_groundtruth \\
        --threshold-db 12 --min-bandwidth-hz 50000 --json out_fm.json

    # Score under the fragmenting default profile for comparison
    python -m src.benchmark --session sessions/fm_groundtruth \\
        --threshold-db 10 --min-bandwidth-hz 100 --json out_default.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.benchmark.report import render_text, to_json
from src.benchmark.runner import run_benchmark
from src.benchmark.seed import seed_truth, write_truth_json
from src.benchmark.truth import BenchmarkError, load_truth
from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the benchmark CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Score a recorded session against ground truth",
    )
    parser.add_argument("--session", required=True, type=Path,
                        help="Recorded session directory")
    parser.add_argument("--truth", type=Path, default=None,
                        help="Truth file (default: <session>/truth.json)")
    parser.add_argument("--preset", type=str, default=None,
                        help="Apply a scan preset's settings before overrides")
    parser.add_argument("--threshold-db", type=float, default=None,
                        help="Override detection.threshold_db")
    parser.add_argument("--min-bandwidth-hz", type=float, default=None,
                        help="Override detection.min_bandwidth_hz")
    parser.add_argument("--dc-removal", action="store_true",
                        help="Enable capture.dc_removal")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Classification match depth (default: 3)")
    parser.add_argument("--json", type=Path, default=None, dest="json_out",
                        help="Write JSON results to this path")
    parser.add_argument("--seed-truth", action="store_true",
                        help="Propose a starter truth.json from raw PSD peaks, then exit")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    truth_path = args.truth or (args.session / "truth.json")

    # Seed mode: write a starter truth file and exit.
    if args.seed_truth:
        if truth_path.exists():
            print(f"Refusing to overwrite existing {truth_path}", file=sys.stderr)
            return 1
        gt = seed_truth(args.session)
        write_truth_json(gt, truth_path)
        print(f"Seeded {len(gt.signals)} candidate signal(s) to {truth_path}")
        print("VERIFY frequencies and add modulation/signal_type before benchmarking.")
        return 0

    # Load and validate ground truth.
    try:
        truth = load_truth(truth_path)
    except BenchmarkError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Tip: run with --seed-truth to propose a starter truth.json.",
              file=sys.stderr)
        return 1

    # Build the config with preset + overrides.
    config = load_config()
    if args.preset:
        presets = config.get("scan_presets", {})
        if args.preset in presets:
            config["scan"].update(presets[args.preset])
        else:
            print(f"Unknown preset '{args.preset}'", file=sys.stderr)
            return 1
    if args.threshold_db is not None:
        config["detection"]["threshold_db"] = args.threshold_db
    if args.min_bandwidth_hz is not None:
        config["detection"]["min_bandwidth_hz"] = args.min_bandwidth_hz
    if args.dc_removal:
        config.setdefault("capture", {})["dc_removal"] = True

    artemis_path = PROJECT_ROOT / "data" / "artemis.db"

    report = run_benchmark(
        args.session,
        truth,
        config,
        artemis_path=artemis_path,
        classification_top_n=args.top_n,
    )

    print(render_text(report))

    if args.json_out:
        args.json_out.write_text(json.dumps(to_json(report), indent=2))
        print(f"\nResults written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
