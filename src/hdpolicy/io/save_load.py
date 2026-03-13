from __future__ import annotations
from pathlib import Path
import pickle
import json
from datetime import datetime
from dataclasses import asdict


def make_run_dir(results_root: Path, tag: str | None = None) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = ts if tag is None else f"{ts}_{tag}"
    run_dir = results_root / name
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir



def save_results(run_dir: Path, results: dict) -> None:
    path = run_dir / "results.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)



def save_config(run_dir: Path, cfg) -> None:
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent = 2)



def load_results(run_dir: Path) -> dict:
    path = run_dir / "results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)



def load_config(run_dir: Path):
    path = run_dir / "config.json"
    with open(path, "r") as f:
        return json.load(f)


