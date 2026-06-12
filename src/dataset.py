"""
MRANC dataset utilities for processed DEAP arrays.

Storage modes:
  - storage='cpu'  : full arrays in system RAM (default for low VRAM)
  - storage='cuda' : full arrays in GPU VRAM (frees CPU RAM; needs ~3GB+ VRAM for DEAP)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from group_splits import audit_split_leakage, train_val_indices_for_dir

logger = logging.getLogger(__name__)


def _npy_to_device(
    path: Path,
    device: torch.device,
    chunk_windows: int = 2048,
) -> torch.Tensor:
    """
    Stream .npy -> torch on `device` in chunks to limit peak CPU RAM.
    """
    mmap = np.load(path, mmap_mode="r")
    if mmap.dtype != np.float32:
        mmap = mmap.astype(np.float32, copy=False)

    out = torch.empty(tuple(mmap.shape), dtype=torch.float32, device=device)
    n = mmap.shape[0]
    for start in range(0, n, chunk_windows):
        end = min(start + chunk_windows, n)
        chunk = np.array(mmap[start:end], dtype=np.float32, copy=True)
        out[start:end] = torch.from_numpy(chunk).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return out


class MRANCDataset(Dataset):
    """Dataset for MRANC. Tensors live on CPU or CUDA based on `storage`."""

    def __init__(
        self,
        data_dir: str | Path = "processed_data",
        storage: str = "cuda",
        chunk_windows: int = 2048,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"processed_data directory not found: {self.data_dir}")

        if storage not in ("cpu", "cuda"):
            raise ValueError(f"storage must be 'cpu' or 'cuda', got {storage!r}")
        if storage == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("storage='cuda' requested but CUDA is not available.")

        self.storage = storage
        self.device = torch.device("cuda" if storage == "cuda" else "cpu")

        files = {
            "mix": "mix.npy",
            "ref_eog": "ref_eog.npy",
            "ref_emg": "ref_emg.npy",
            "ref_ecg": "ref_ecg.npy",
        }

        if storage == "cuda":
            logger.info("Loading processed_data to GPU VRAM (chunked, low CPU RAM peak)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("Loading processed_data to CPU RAM...")

        self.mix = _npy_to_device(self.data_dir / files["mix"], self.device, chunk_windows)
        self.ref_eog = _npy_to_device(self.data_dir / files["ref_eog"], self.device, chunk_windows)
        self.ref_emg = _npy_to_device(self.data_dir / files["ref_emg"], self.device, chunk_windows)
        self.ref_ecg = _npy_to_device(self.data_dir / files["ref_ecg"], self.device, chunk_windows)

        self._validate_shapes()

        if storage == "cuda" and torch.cuda.is_available():
            alloc_gb = torch.cuda.memory_allocated() / 1e9
            logger.info("Dataset on VRAM | cuda allocated ~%.2f GB", alloc_gb)

    def _validate_shapes(self) -> None:
        arrays = {
            "mix": self.mix,
            "ref_eog": self.ref_eog,
            "ref_emg": self.ref_emg,
            "ref_ecg": self.ref_ecg,
        }

        for name, arr in arrays.items():
            if arr.ndim != 3:
                raise ValueError(f"{name} must be 3D (N, C, T). Got shape {tuple(arr.shape)}")

        n, _, t = self.mix.shape
        for name, arr in arrays.items():
            if arr.shape[0] != n or arr.shape[2] != t:
                raise ValueError(f"{name}: N/T mismatch vs mix ({n},*,{t}) vs {tuple(arr.shape)}")

        if self.mix.shape[1] != 32:
            raise ValueError(f"mix channels must be 32, got {self.mix.shape[1]}")
        if self.ref_eog.shape[1] != 2:
            raise ValueError(f"ref_eog channels must be 2, got {self.ref_eog.shape[1]}")
        if self.ref_emg.shape[1] != 2:
            raise ValueError(f"ref_emg channels must be 2, got {self.ref_emg.shape[1]}")
        if self.ref_ecg.shape[1] != 1:
            raise ValueError(f"ref_ecg channels must be 1, got {self.ref_ecg.shape[1]}")

    def __len__(self) -> int:
        return self.mix.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "mix": self.mix[idx],
            "ref_eog": self.ref_eog[idx],
            "ref_emg": self.ref_emg[idx],
            "ref_ecg": self.ref_ecg[idx],
        }


def _dataset_storage_cuda(dataset: Dataset) -> bool:
    """True if underlying MRANCDataset keeps tensors on CUDA (handles random_split Subset)."""
    if isinstance(dataset, MRANCDataset):
        return dataset.storage == "cuda"
    if isinstance(dataset, torch.utils.data.Subset):
        return isinstance(dataset.dataset, MRANCDataset) and dataset.dataset.storage == "cuda"
    return False


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    on_gpu = _dataset_storage_cuda(dataset)
    if on_gpu and num_workers > 0:
        raise ValueError("num_workers must be 0 when dataset storage='cuda'")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and not on_gpu

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_train_val_dataloaders(
    data_dir: str | Path = "processed_data",
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    storage: str = "cuda",
    chunk_windows: int = 2048,
    dataset: str | None = None,
) -> tuple[DataLoader, DataLoader, MRANCDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    full = MRANCDataset(data_dir=data_dir, storage=storage, chunk_windows=chunk_windows)
    train_idx, val_idx, group_ids, policy = train_val_indices_for_dir(
        data_dir,
        val_fraction=val_fraction,
        split_seed=seed,
        dataset=dataset,
    )
    audit_split_leakage(
        group_ids,
        train_idx,
        val_idx,
        split_policy=policy,
        label=str(data_dir),
    )
    train_ds = Subset(full, train_idx.tolist())
    val_ds = Subset(full, val_idx.tolist())

    pin = storage != "cuda"
    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, full
