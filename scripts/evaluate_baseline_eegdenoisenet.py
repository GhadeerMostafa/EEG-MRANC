"""
Train and evaluate EEGdenoiseNet-style FCN autoencoder baseline (clinical, deap, seed).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from runtime import setup_src_path
from torch.utils.data import DataLoader, Dataset

setup_src_path()

from baselines.eegdenoisenet_fcn import EEGDenoiseNetFCN
from baseline_eval import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    baseline_eegdenoisenet_metrics_path,
    dataset_length,
    ensure_reports_dir,
    list_baseline_datasets,
    load_mix_batch,
    metrics_dict_to_report,
    resolve_data_dir,
    resolve_sample_rate,
    save_baseline_json,
    train_indices,
    val_indices,
)
from checkpoint_paths import (
    new_run_id,
    record_stacking_checkpoint,
    resolve_stacking_latest,
    versioned_weight_path,
)
from evaluate_metrics import aggregate_metrics, compute_batch_metrics_torch
from paths import BASELINE_EEGDENOISENET_CHECKPOINT, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_DATASETS = ("deap", "seed")
# DEAP windows are 256 samples; SEED/clinical are 1000 — harmonize for mixed training batches.
DEFAULT_TRAIN_WINDOW_SAMPLES = 256


def harmonize_window_length(x: torch.Tensor, target_t: int) -> torch.Tensor:
    """(C, T) -> (C, target_t) via center crop or trailing zero-pad."""
    t = x.shape[-1]
    if t == target_t:
        return x
    if t > target_t:
        start = (t - target_t) // 2
        return x[..., start : start + target_t]
    return F.pad(x, (0, target_t - t))


class MixWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        indices: list[int],
        target_samples: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.indices = indices
        self.target_samples = target_samples
        self.mix_mmap = np.load(data_dir / "mix.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        i = self.indices[idx]
        x = torch.from_numpy(np.array(self.mix_mmap[i], dtype=np.float32))
        if self.target_samples is not None:
            x = harmonize_window_length(x, self.target_samples)
        return x


class CombinedMixDataset(Dataset):
    def __init__(
        self,
        parts: list[tuple[Path, list[int]]],
        target_samples: int,
    ) -> None:
        self.target_samples = target_samples
        self.parts = parts
        self.offsets: list[tuple[int, int, Path, list[int]]] = []
        cursor = 0
        for data_dir, indices in parts:
            self.offsets.append((cursor, cursor + len(indices), data_dir, indices))
            cursor += len(indices)

    def __len__(self) -> int:
        return self.offsets[-1][1] if self.offsets else 0

    def __getitem__(self, idx: int) -> torch.Tensor:
        for start, end, data_dir, indices in self.offsets:
            if start <= idx < end:
                local = indices[idx - start]
                mix_mmap = np.load(data_dir / "mix.npy", mmap_mode="r")
                x = torch.from_numpy(np.array(mix_mmap[local], dtype=np.float32))
                return harmonize_window_length(x, self.target_samples)
        raise IndexError(idx)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate EEGdenoiseNet FCN baseline")
    p.add_argument("--dataset", type=str, default="all", help="clinical, deap, seed, or all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise-sigma", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-if-exists", action="store_true", help="Skip dataset if metrics JSON exists")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id for versioned baseline checkpoint filename",
    )
    p.add_argument(
        "--train-window-samples",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_SAMPLES,
        help="Crop/pad DEAP+SEED windows to this length for mixed training (DEAP=256)",
    )
    return p.parse_args()


def train_model(
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> EEGDenoiseNetFCN:
    train_parts: list[tuple[Path, list[int]]] = []
    val_loaders: list[DataLoader] = []

    for ds in TRAIN_DATASETS:
        data_dir = resolve_data_dir(ds)
        n_total = dataset_length(data_dir)
        tr_idx = train_indices(
            n_total,
            args.val_fraction,
            args.split_seed,
            data_dir=data_dir,
            dataset=ds,
        )
        va_idx = val_indices(
            n_total,
            args.val_fraction,
            args.split_seed,
            data_dir=data_dir,
            dataset=ds,
        )
        train_parts.append((data_dir, tr_idx))
        val_loaders.append(
            DataLoader(
                MixWindowDataset(data_dir, va_idx, target_samples=args.train_window_samples),
                batch_size=args.batch_size,
                shuffle=False,
            )
        )

    logger.info(
        "Mixed training: harmonizing DEAP+SEED windows to %d samples (center crop / pad)",
        args.train_window_samples,
    )
    train_loader = DataLoader(
        CombinedMixDataset(train_parts, target_samples=args.train_window_samples),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_parts) > 0,
    )

    model = EEGDenoiseNetFCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for mix in train_loader:
            mix = mix.to(device)
            noisy = mix + args.noise_sigma * torch.randn_like(mix)
            pred = model(noisy)
            loss = F.mse_loss(pred, mix)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_train += 1

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for val_loader in val_loaders:
                for mix in val_loader:
                    mix = mix.to(device)
                    noisy = mix + args.noise_sigma * torch.randn_like(mix)
                    pred = model(noisy)
                    val_loss += float(F.mse_loss(pred, mix).item())
                    n_val += 1

        mean_train = train_loss / max(n_train, 1)
        mean_val = val_loss / max(n_val, 1)
        logger.info("Epoch %d/%d train_mse=%.6f val_mse=%.6f", epoch, args.epochs, mean_train, mean_val)

        if mean_val < best_val:
            best_val = mean_val
            try:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {"train_window_samples": args.train_window_samples},
                    },
                    checkpoint_path,
                )
            except OSError as exc:
                logger.error("Could not save checkpoint: %s", exc)

    if checkpoint_path.is_file():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        except OSError as exc:
            logger.warning("Could not reload best checkpoint: %s", exc)

    return model


def evaluate_dataset(
    model: EEGDenoiseNetFCN,
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    out_path = baseline_eegdenoisenet_metrics_path(dataset)
    if args.skip_if_exists and out_path.is_file():
        logger.info("Skipping %s (exists): %s", dataset, out_path)
        return

    data_dir = resolve_data_dir(dataset)
    n_total = dataset_length(data_dir)
    val_idx = val_indices(
        n_total,
        args.val_fraction,
        args.split_seed,
        data_dir=data_dir,
        dataset=dataset,
    )
    if args.max_windows is not None:
        val_idx = val_idx[: args.max_windows]

    fs = resolve_sample_rate(dataset)
    per_window: list[dict] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_idx), args.batch_size):
            batch_ids = val_idx[start : start + args.batch_size]
            mix_batch = load_mix_batch(data_dir, batch_ids)
            mix_t = torch.from_numpy(mix_batch).to(device)
            clean_t = model(mix_t)
            artifact_t = mix_t - clean_t
            batch_metrics, _ = compute_batch_metrics_torch(mix_t, clean_t, artifact_t, fs)
            for j in range(len(batch_ids)):
                per_window.append({k: float(batch_metrics[k][j].cpu()) for k in batch_metrics})

    aggregated = aggregate_metrics(per_window)
    out_path = baseline_eegdenoisenet_metrics_path(dataset)
    ensure_reports_dir(dataset)
    payload = metrics_dict_to_report(
        aggregated,
        dataset=dataset,
        method="eegdenoisenet",
        n_val=len(val_idx),
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    payload["train_window_samples"] = args.train_window_samples
    save_baseline_json(out_path, payload)
    logger.info("Wrote EEGdenoiseNet baseline metrics to %s", out_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.skip_train:
        latest = resolve_stacking_latest("baseline_eegdenoisenet", BASELINE_EEGDENOISENET_CHECKPOINT)
        if latest is None or not latest.is_file():
            logger.error("No baseline checkpoint found; run without --skip-train")
            sys.exit(1)
        checkpoint_path = latest
        logger.info("Loading checkpoint %s", checkpoint_path)
        model = EEGDenoiseNetFCN().to(device)
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        except OSError as exc:
            logger.error("Could not load checkpoint: %s", exc)
            sys.exit(1)
    else:
        run_id = args.run_id or new_run_id()
        checkpoint_path = versioned_weight_path(BASELINE_EEGDENOISENET_CHECKPOINT, run_id)
        logger.info("Training EEGdenoiseNet FCN on DEAP + SEED; checkpoint -> %s", checkpoint_path)
        model = train_model(args, device, checkpoint_path)
        record_stacking_checkpoint("baseline_eegdenoisenet", checkpoint_path, run_id)

    try:
        datasets = list_baseline_datasets(args.dataset)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    for ds in datasets:
        try:
            evaluate_dataset(model, ds, args, device)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error("EEGdenoiseNet evaluation failed for %s: %s", ds, exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
