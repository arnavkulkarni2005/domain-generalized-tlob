# TLOB Benchmark Experimental Grid

This folder is the reproducibility harness for the domain-shift grid. The canonical architecture remains `models/tlob.py`; this suite only controls experiments, provenance, scoring, and reports.

## Grid

| Experiment | Training data | FI-2010 | ETH | BTC zero-shot | Status |
| --- | --- | --- | --- | --- | --- |
| TLOB (ERM - Baseline 1) | FI-2010 | High | Low | Low | runnable |
| TLOB (ERM - Baseline 2) | ETH | Low | High | Moderate | backend required |
| TLOB (Naive Pooled ERM) | FI-2010 + ETH | Moderate | Moderate | Moderate | backend required |
| TLOB + DeepCORAL | FI-2010 + ETH | High | High | target result | runnable with existing FI source path |
| TLOB + GroupDRO | FI-2010 + ETH | High | High | target result robust | group-aware backend required |
| TLOB Oracle | BTC | N/A | N/A | theoretical maximum | BTC TLOB path required |

The `expected_results` fields are hypotheses/targets, not measured values.

## Run

From the repository root:

```powershell
python experiments/benchmark_grid/runner.py --dry-run
python experiments/benchmark_grid/runner.py --experiment baseline_fi2010
python experiments/benchmark_grid/runner.py --experiment deepcoral
```

The runner requires CUDA for runnable entries and refuses to label a CPU fallback as a GPU run. Every attempt creates a unique folder under `data/experiments/benchmark_grid/runs/<experiment>/<UTC timestamp>/` containing the config snapshot, SHA-256 provenance, `metadata.json`, `events.jsonl`, and, for executed jobs, `stdout.log` and `stderr.log`. Existing runs are never overwritten.

The current repository's `run.py` supports FI-2010 training and FI-2010-to-ETH DeepCORAL, but it does not yet expose ETH-only, pooled multi-domain, GroupDRO, or BTC TLOB training. Those rows are kept as explicit blocked entries so the grid is complete without fabricating results. Their `status` must be changed only when the corresponding training backend is implemented.

## Scoring and visualization

After a run produces prediction and label NumPy arrays:

```powershell
python experiments/benchmark_grid/metrics.py predictions.npy labels.npy --output data/experiments/benchmark_grid/runs/<experiment>/<timestamp>/metrics.json
python experiments/benchmark_grid/visualize.py
```

Reports are written to `data/experiments/benchmark_grid/reports/` as CSV, PNG, and SVG. Do not store credentials or W&B API keys in these folders; use environment variables or the existing W&B configuration.
