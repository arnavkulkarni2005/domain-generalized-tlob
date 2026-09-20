"""Score saved prediction/label arrays and maintain an auditable metric record."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def score(predictions: Path, labels: Path) -> dict:
    y_pred = np.load(predictions).reshape(-1)
    y_true = np.load(labels).reshape(-1)
    if len(y_pred) != len(y_true):
        raise ValueError(f"Prediction/label length mismatch: {len(y_pred)} != {len(y_true)}")
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(predictions),
        "labels": str(labels),
        "samples": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = score(args.predictions, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
