"""Train the modulation classifier and save a model bundle.

Usage:
    # Build the dataset from sessions/ + synthetic, train, and save
    python -m src.ml.train --build --sessions-dir sessions --out data/modulation_model.joblib

    # Retrain from a saved dataset, folding in operator corrections
    python -m src.ml.train --dataset data/ml_dataset.npz --db data/detections.db
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT
from src.ml.dataset import DatasetSpec, build_dataset, load_dataset, save_dataset
from src.ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def train(
    X: np.ndarray, y: np.ndarray, *,
    n_estimators: int = 300, seed: int = 0, test_size: float = 0.25,
) -> tuple[RandomForestClassifier, dict]:
    """Train a RandomForest and return (model, report)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed,
        class_weight="balanced", n_jobs=1,  # n_jobs=-1 can deadlock under joblib on macOS
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, y_pred))
    labels = sorted(set(y))
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1]
    )
    return model, {
        "held_out_accuracy": acc,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_features": importances[:10],
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


def _print_report(report: dict) -> None:
    print(f"\nHeld-out accuracy: {report['held_out_accuracy']*100:.1f}%  "
          f"(train={report['n_train']}, test={report['n_test']})")
    labels = report["labels"]
    print("Confusion (truth rows x predicted cols):")
    print("  " + " ".join(f"{l:>5}" for l in [""] + labels))
    for r, row in zip(labels, report["confusion_matrix"]):
        print("  " + f"{r:>5} " + " ".join(f"{c:>5}" for c in row))
    print("Top features:")
    for name, imp in report["top_features"]:
        print(f"  {name:<26} {imp:.3f}")


def main(argv: list[str] | None = None) -> int:
    """Train CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train the modulation classifier")
    parser.add_argument("--build", action="store_true", help="Build the dataset first")
    parser.add_argument("--sessions-dir", type=Path, default=PROJECT_ROOT / "sessions")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "ml_dataset.npz")
    parser.add_argument("--db", type=Path, default=None, help="Fold in operator corrections from this DB")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "modulation_model.joblib")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.build:
        print("Building dataset...")
        X, y, names = build_dataset(DatasetSpec(seed=args.seed),
                                    sessions_dir=args.sessions_dir, db_path=args.db)
        save_dataset(args.dataset, X, y, names, {"built_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Dataset: {len(X)} rows -> {args.dataset}")
    else:
        X, y, names = load_dataset(args.dataset)
        if args.db:  # fold corrections into a previously-built dataset
            from src.ml.dataset import _rows_from_corrections
            extra = list(_rows_from_corrections(args.db))
            if extra:
                X = np.vstack([X, np.array([r[0] for r in extra])])
                y = np.concatenate([y, np.array([r[1] for r in extra], dtype=object)])
                print(f"Folded in {len(extra)} correction rows")

    from collections import Counter
    print(f"Class balance: {dict(Counter(y.tolist()))}")

    model, report = train(X, y, n_estimators=args.n_estimators, seed=args.seed)
    _print_report(report)

    if report["held_out_accuracy"] < args.min_accuracy:
        print(f"\nFAIL: held-out accuracy {report['held_out_accuracy']*100:.1f}% "
              f"< gate {args.min_accuracy*100:.0f}%. Not saving.", file=sys.stderr)
        return 1

    import sklearn
    bundle = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "classes": sorted(set(y.tolist())),
        "sklearn_version": sklearn.__version__,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "meta": {"held_out_accuracy": report["held_out_accuracy"]},
    }
    joblib.dump(bundle, args.out)
    print(f"\nSaved model -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
