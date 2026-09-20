"""Run and audit the benchmark grid without modifying the canonical TLOB model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent / "configs"
RUNS_DIR = ROOT / "data" / "experiments" / "benchmark_grid" / "runs"
RUNNABLE = {"baseline_fi2010", "deepcoral"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_configs(selected: str | None) -> list[tuple[Path, dict]]:
    paths = sorted(CONFIG_DIR.glob("*/config.yaml"))
    if selected:
        paths = [CONFIG_DIR / selected / "config.yaml"]
    loaded = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Unknown experiment: {path.parent.name}")
        with path.open(encoding="utf-8") as handle:
            loaded.append((path, yaml.safe_load(handle)))
    return loaded


def require_gpu() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU-enforced experiments") from exc
    hardware = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
    if not hardware["cuda_available"]:
        raise RuntimeError("CUDA is unavailable; refusing to record this as a GPU experiment")
    hardware["device_name"] = torch.cuda.get_device_name(0)
    return hardware


def command_for(config: dict) -> list[str]:
    experiment = config["name"]
    overrides = [
        "+model=tlob",
        "+dataset=fi_2010",
        "experiment.is_wandb=false",
        "experiment.is_data_preprocessed=false",
        "experiment.type=[TRAINING]",
        f"experiment.seed={config['seed']}",
        f"experiment.horizon={config['horizon']}",
        f"experiment.max_epochs={config['max_epochs']}",
        f"experiment.deepcoral={'true' if config.get('deepcoral', False) else 'false'}",
        f"experiment.coral_weight={config.get('coral_weight', 1.0)}",
        f"experiment.target_data_path={config.get('target_data_path', '')}",
        "hydra.job.chdir=False",
    ]
    return [sys.executable, "main.py", *overrides, f"experiment.dir_ckpt=benchmark_{experiment}"]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_one(config_path: Path, config: dict, dry_run: bool) -> dict:
    run_dir = RUNS_DIR / config["name"] / utc_stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    config_bytes = config_path.read_bytes()
    snapshot = run_dir / "config.yaml"
    snapshot.write_bytes(config_bytes)
    metadata = {
        "experiment": config["name"],
        "label": config["label"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "config_source": str(config_path.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "model_architecture": config["architecture_contract"],
        "requested_status": config["status"],
        "gpu_required": config.get("gpu_required", True),
        "command": command_for(config),
    }
    events = run_dir / "events.jsonl"
    with events.open("w", encoding="utf-8") as event_log:
        event_log.write(json.dumps({"event": "created", "metadata": metadata}) + "\n")
        if config["name"] not in RUNNABLE:
            metadata["status"] = "not_run"
            metadata["reason"] = config["status"]
            event_log.write(json.dumps({"event": "skipped", "reason": config["status"]}) + "\n")
            write_json(run_dir / "metadata.json", metadata)
            return metadata
        if config.get("gpu_required", True) and not dry_run:
            metadata["hardware"] = require_gpu()
        if dry_run:
            metadata["status"] = "dry_run"
            event_log.write(json.dumps({"event": "dry_run"}) + "\n")
            write_json(run_dir / "metadata.json", metadata)
            return metadata
        command = metadata["command"]
        metadata["status"] = "running"
        write_json(run_dir / "metadata.json", metadata)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        metadata["return_code"] = process.returncode
        metadata["status"] = "completed" if process.returncode == 0 else "failed"
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        event_log.write(json.dumps({"event": "finished", "return_code": process.returncode}) + "\n")
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Run one config directory; omit to process the whole grid")
    parser.add_argument("--dry-run", action="store_true", help="Create audited run folders without training")
    args = parser.parse_args()
    results = []
    for path, config in load_configs(args.experiment):
        try:
            result = run_one(path, config, args.dry_run)
        except Exception as exc:
            result = {"experiment": config.get("name"), "status": "error", "error": repr(exc)}
        results.append(result)
        print(f"{result['experiment']}: {result['status']}")
    RUNS_DIR.parent.mkdir(parents=True, exist_ok=True)
    write_json(RUNS_DIR.parent / "latest_manifest.json", {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "runs": results})
    return 0 if all(item["status"] in {"completed", "dry_run", "not_run"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
