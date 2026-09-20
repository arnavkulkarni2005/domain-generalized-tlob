"""Create visual benchmark summaries from completed run metric files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "experiments" / "benchmark_grid" / "runs"


def collect(runs: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in runs.glob("*/**/metrics.json"):
        metadata_path = metrics_path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"experiment": metadata.get("experiment", metrics_path.parent.parent.name), "label": metadata.get("label", ""), "macro_f1": metrics["macro_f1"]})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "experiments" / "benchmark_grid" / "reports")
    args = parser.parse_args()
    frame = collect(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print("No metrics.json files found; run training and score predictions first.")
        return 0
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame.to_csv(args.output / "metrics_long.csv", index=False)
    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(13, 7))
    ordered = frame.sort_values("macro_f1")
    sns.barplot(data=ordered, x="macro_f1", y="label", hue="label", legend=False, palette="viridis", ax=axis)
    axis.set(xlabel="Macro F1", ylabel="", title="TLOB Benchmark Experimental Grid")
    figure.tight_layout()
    figure.savefig(args.output / "macro_f1_summary.png", dpi=180)
    figure.savefig(args.output / "macro_f1_summary.svg")
    plt.close(figure)
    print(f"Wrote reports to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
