"""
Shared validation-window loading and baseline metric JSON I/O.
Metric computation uses evaluate_metrics helpers (imported from scripts at call sites).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from paths import (
    BASELINE_DATASETS,
    DATASET_DIRS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    REPORTS_DATASET_DIRS,
    baseline_eegdenoisenet_metrics_path,
    baseline_ica_metrics_path,
)

DEFAULT_SAMPLE_RATE: dict[str, float] = {
    "seed": 200.0,
    "deap": 128.0,
    "clinical": 200.0,
    "artifact_benchmark": 200.0,
}

# FastICA is per-window CPU-heavy; cap DEAP val windows unless --max-windows is set.
DEFAULT_ICA_MAX_VAL_WINDOWS: dict[str, int | None] = {
    "clinical": None,
    "deap": 512,
    "seed": None,
}

METRIC_KEYS = (
    "loss_mse",
    "snr_improvement_db",
    "psd_alpha_beta_correlation",
    "artifact_magnitude_rmse",
    "physics_residual_rmse",
)


def resolve_sample_rate(dataset: str) -> float:
    if dataset not in DEFAULT_SAMPLE_RATE:
        raise ValueError(f"Unknown dataset for sample rate: {dataset}")
    return DEFAULT_SAMPLE_RATE[dataset]


def dataset_length(data_dir: Path) -> int:
    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        raise FileNotFoundError(f"Missing {mix_path}")
    return int(np.load(mix_path, mmap_mode="r").shape[0])


def cap_val_indices(
    val_idx: list[int],
    max_windows: int | None,
    subsample_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[int], bool]:
    """Random subsample of val windows when len(val_idx) > max_windows."""
    if max_windows is None or len(val_idx) <= max_windows:
        return val_idx, False
    rng = np.random.default_rng(subsample_seed)
    picked = rng.choice(np.asarray(val_idx, dtype=np.int64), size=max_windows, replace=False)
    return sorted(int(i) for i in picked), True


def resolve_ica_max_windows(dataset: str, cli_max: int | None) -> int | None:
    if cli_max is not None:
        return cli_max
    return DEFAULT_ICA_MAX_VAL_WINDOWS.get(dataset)


def val_indices(
    n_total: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> list[int]:
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    placeholder = TensorDataset(torch.zeros(n_total))
    _, val_ds = random_split(
        placeholder,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(split_seed),
    )
    return list(val_ds.indices)


def train_indices(
    n_total: int,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> list[int]:
    val_set = set(val_indices(n_total, val_fraction, split_seed))
    return [i for i in range(n_total) if i not in val_set]


def resolve_data_dir(dataset: str, data_dir: str | None = None) -> Path:
    if data_dir:
        p = Path(data_dir)
        return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p
    if dataset not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASET_DIRS[dataset]


def load_mix_window(data_dir: Path, index: int) -> np.ndarray:
    mix_mmap = np.load(data_dir / "mix.npy", mmap_mode="r")
    return np.array(mix_mmap[index], dtype=np.float32)


def load_mix_batch(data_dir: Path, indices: list[int]) -> np.ndarray:
    mix_mmap = np.load(data_dir / "mix.npy", mmap_mode="r")
    return np.array([mix_mmap[i] for i in indices], dtype=np.float32)


def load_refs_batch(data_dir: Path, indices: list[int]) -> dict[str, np.ndarray] | None:
    eog_path = data_dir / "ref_eog.npy"
    if not eog_path.is_file():
        return None
    try:
        ref_eog = np.load(eog_path, mmap_mode="r")
        ref_emg = np.load(data_dir / "ref_emg.npy", mmap_mode="r")
        ref_ecg = np.load(data_dir / "ref_ecg.npy", mmap_mode="r")
    except OSError as exc:
        raise OSError(f"Could not load reference arrays from {data_dir}: {exc}") from exc
    return {
        "ref_eog": np.array([ref_eog[i] for i in indices], dtype=np.float32),
        "ref_emg": np.array([ref_emg[i] for i in indices], dtype=np.float32),
        "ref_ecg": np.array([ref_ecg[i] for i in indices], dtype=np.float32),
    }


def metrics_dict_to_report(
    aggregated: dict[str, dict[str, float]],
    *,
    dataset: str,
    method: str,
    n_val: int,
    val_fraction: float,
    split_seed: int,
    raw_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "n_val_windows": n_val,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "metrics": aggregated,
    }
    if raw_metrics is not None:
        payload["raw_metrics"] = raw_metrics
    return payload


def save_baseline_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write baseline JSON to {path}: {exc}") from exc


def load_baseline_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def extract_metric_means(payload: dict[str, Any] | None) -> dict[str, float]:
    if payload is None:
        return {}
    block = payload.get("metrics", {})
    if not isinstance(block, dict):
        return {}
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        entry = block.get(key, {})
        if isinstance(entry, dict) and "mean" in entry:
            try:
                out[key] = float(entry["mean"])
            except (TypeError, ValueError):
                continue
    return out


def extract_raw_metric_means(payload: dict[str, Any] | None) -> dict[str, float]:
    if payload is None:
        return {}
    block = payload.get("raw_metrics", {})
    if not isinstance(block, dict):
        return {}
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        entry = block.get(key, {})
        if isinstance(entry, dict) and "mean" in entry:
            try:
                out[key] = float(entry["mean"])
            except (TypeError, ValueError):
                continue
    return out


def ensure_reports_dir(dataset: str) -> Path:
    out_dir = REPORTS_DATASET_DIRS.get(dataset)
    if out_dir is None:
        raise ValueError(f"No report directory for dataset: {dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def list_baseline_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(BASELINE_DATASETS)
    if dataset_arg not in BASELINE_DATASETS:
        raise ValueError(f"dataset must be one of {BASELINE_DATASETS} or 'all', got {dataset_arg}")
    return [dataset_arg]
