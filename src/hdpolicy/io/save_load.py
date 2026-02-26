from __future__ import annotations
from pathlib import Path
import pickle
from datetime import datetime


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



def load_results(run_dir: Path) -> dict:
    path = run_dir / "results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)




