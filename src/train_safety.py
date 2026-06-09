"""
Hardware safety checks for MRANC training (RTX 3070 8GB class GPUs).

Goals:
  - Do not allocate 100% VRAM (leave headroom for OS / display)
  - Preflight VRAM/RAM before loading full dataset
  - Cap batch size to safe maximum
  - Optional GPU temperature guard via nvidia-smi
  - Graceful handling of NaN loss and CUDA OOM
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# DEAP processed_data float32 footprint (approximate)
DATASET_VRAM_GB = 3.05
MODEL_TRAIN_OVERHEAD_GB = 1.6
PER_BATCH_ACTIVATION_GB = 0.018  # AMP fp16 estimate; scales ~linearly with batch @ 32x256


@dataclass
class SafetyConfig:
    memory_fraction: float = 0.85
    vram_reserve_gb: float = 1.0
    min_free_ram_gb: float = 2.0
    max_batch_size: int = 128
    min_batch_size: int = 8
    max_gpu_temp_c: float = 86.0
    abort_gpu_temp_c: float = 92.0
    gradient_clip_norm: float = 1.0
    check_temperature: bool = True


def cuda_mem_gb() -> tuple[float, float, float]:
    """Return (free_gb, total_gb, allocated_gb)."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info()
    alloc_b = torch.cuda.memory_allocated()
    return free_b / 1e9, total_b / 1e9, alloc_b / 1e9


def system_ram_free_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return None


def estimate_dataset_bytes(data_dir: Path) -> int:
    total = 0
    for name in ("mix.npy", "ref_eog.npy", "ref_emg.npy", "ref_ecg.npy"):
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        total += p.stat().st_size
    return total


def gpu_temperature_c() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        return float(out.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def apply_cuda_limits(cfg: SafetyConfig) -> None:
    if not torch.cuda.is_available():
        return
    frac = max(0.5, min(0.95, cfg.memory_fraction))
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info(
        "Safety: CUDA memory fraction capped at %.0f%% of GPU VRAM (reserve for OS/display)",
        frac * 100,
    )


def check_gpu_temperature(cfg: SafetyConfig) -> None:
    if not cfg.check_temperature:
        return
    temp = gpu_temperature_c()
    if temp is None:
        logger.info("Safety: GPU temperature check skipped (nvidia-smi unavailable)")
        return
    logger.info("Safety: GPU temperature %.0f C", temp)
    if temp >= cfg.abort_gpu_temp_c:
        raise RuntimeError(
            f"GPU temperature {temp:.0f}C >= {cfg.abort_gpu_temp_c:.0f}C — stopping to protect hardware. "
            "Let GPU cool and retry."
        )
    if temp >= cfg.max_gpu_temp_c:
        logger.warning(
            "Safety: GPU temperature %.0fC is high (limit %.0fC). Consider improving cooling.",
            temp,
            cfg.max_gpu_temp_c,
        )


def recommend_batch_size(cfg: SafetyConfig, storage: str) -> int:
    if storage != "cuda" or not torch.cuda.is_available():
        return min(cfg.max_batch_size, 64)

    free_gb, total_gb, _ = cuda_mem_gb()
    usable = max(0.0, free_gb - cfg.vram_reserve_gb)
    if total_gb > 0:
        cap = total_gb * cfg.memory_fraction - DATASET_VRAM_GB - MODEL_TRAIN_OVERHEAD_GB - cfg.vram_reserve_gb
        usable = min(usable, cap)

    if usable <= 0:
        return cfg.min_batch_size

    max_safe = int(usable / PER_BATCH_ACTIVATION_GB)
    max_safe = max(cfg.min_batch_size, min(cfg.max_batch_size, max_safe))
    return max_safe


def preflight(
    cfg: SafetyConfig,
    data_dir: Path,
    storage: str,
    batch_size: int,
) -> tuple[str, int]:
    """
    Validate hardware before training. Returns (storage, batch_size) possibly adjusted.
    """
    apply_cuda_limits(cfg)
    check_gpu_temperature(cfg)

    data_gb = estimate_dataset_bytes(data_dir) / 1e9
    logger.info("Safety: dataset files on disk ~%.2f GB", data_gb)

    if storage == "cuda":
        free_gb, total_gb, _ = cuda_mem_gb()
        need_gb = data_gb + MODEL_TRAIN_OVERHEAD_GB + batch_size * PER_BATCH_ACTIVATION_GB + cfg.vram_reserve_gb
        logger.info(
            "Safety: VRAM free %.2f / %.2f GB | estimated need ~%.2f GB (data + model + batch %d)",
            free_gb,
            total_gb,
            need_gb,
            batch_size,
        )
        if free_gb < need_gb and total_gb > 0:
            safe_batch = recommend_batch_size(cfg, "cuda")
            if safe_batch < batch_size:
                logger.warning(
                    "Safety: reducing batch_size %d -> %d to fit VRAM",
                    batch_size,
                    safe_batch,
                )
                batch_size = safe_batch
            need_gb = data_gb + MODEL_TRAIN_OVERHEAD_GB + batch_size * PER_BATCH_ACTIVATION_GB + cfg.vram_reserve_gb
            if free_gb < need_gb * 0.95:
                raise RuntimeError(
                    "Safety: VRAM still tight and CPU fallback is disabled. "
                    f"Reduce --batch-size (current batch_size={batch_size}) or use a GPU with more VRAM. "
                    f"(free_gb={free_gb:.2f}, need_gb={need_gb:.2f}, total_gb={total_gb:.2f})"
                )
    else:
        ram_free = system_ram_free_gb()
        if ram_free is not None:
            logger.info("Safety: system RAM free ~%.2f GB", ram_free)
            if ram_free < data_gb + cfg.min_free_ram_gb:
                raise RuntimeError(
                    f"Not enough free RAM ({ram_free:.1f} GB) to load dataset (~{data_gb:.1f} GB). "
                    "Close other apps or use --storage cuda if VRAM allows."
                )

    if batch_size > cfg.max_batch_size:
        logger.warning("Safety: batch_size capped at %d", cfg.max_batch_size)
        batch_size = cfg.max_batch_size

    safe_max = recommend_batch_size(cfg, storage)
    if batch_size > safe_max:
        logger.warning("Safety: batch_size %d -> %d (hardware limit)", batch_size, safe_max)
        batch_size = safe_max

    return storage, batch_size


def after_epoch_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def assert_finite_loss(loss: torch.Tensor, epoch: int) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at epoch {epoch} — stopping to avoid unstable training.")


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg
