"""
MRANC training — CUDA-only, physics-locked losses, AMP.

total_loss = loss_stems + 0.05 * loss_ortho + 0.01 * loss_tv + scale_penalty
loss_stems = loss_eog + loss_emg + loss_ecg + loss_basenoise
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from runtime import setup_src_path
from torch.utils.data import DataLoader, Dataset, random_split

setup_src_path()

from clinical_sanitize import sanitize_clinical_batch
from dataset import build_train_val_dataloaders
from model import DEAP_SEQ_LEN, MRANC
from validation_metrics import DEFAULT_SAMPLE_RATE, run_supervised_val_quality_metrics
from checkpoint_paths import new_run_id, versioned_weight_path
from paths import (
    ARTIFACT_BENCHMARK_CHECKPOINT,
    DEFAULT_CHECKPOINT,
    ARTIFACT_BENCHMARK_PROCESSED,
    PROCESSED_CLINICAL,
    PROCESSED_DEAP,
    PROCESSED_SEED,
    PROJECT_ROOT,
)
from train_safety import (
    SafetyConfig,
    after_epoch_cleanup,
    apply_cuda_limits,
    check_gpu_temperature,
    is_oom_error,
    cuda_mem_gb,
    system_ram_free_gb,
    preflight,
    MODEL_TRAIN_OVERHEAD_GB,
    PER_BATCH_ACTIVATION_GB,
)

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_weights_compat(model: MRANC, state: dict) -> bool:
    """
    Load weights allowing new modules (e.g. ms_attention) absent in older checkpoints.

    Returns True when ms_attention was absent and freshly initialized (caller should reset optimizer).
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    adapter_was_missing = False
    if missing:
        ms_missing = [k for k in missing if k.startswith("ms_attention.")]
        other = [k for k in missing if not k.startswith("ms_attention.")]
        adapter_was_missing = len(ms_missing) > 0
        if ms_missing:
            logger.info(
                "Checkpoint missing %d ms_attention keys; using freshly initialized adapter.",
                len(ms_missing),
            )
        if other:
            logger.warning("Checkpoint missing keys: %s", other[:10])
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:10])
    return adapter_was_missing

ORTHO_WEIGHT = 0.05
TV_WEIGHT = 0.01
SCALE_PENALTY_WEIGHT = 0.1

# artifact_benchmark anti-collapse safeguards
ARTIFACT_BENCHMARK_ORTHO_MULT = 10.0
ARTIFACT_BENCHMARK_TV_MULT = 2.0
ARTIFACT_VAR_MIN_THRESHOLD = 0.1
ARTIFACT_VAR_PENALTY_WEIGHT = 0.5
ARTIFACT_CLEAN_VAR_RATIO_THRESHOLD = 6.0
ARTIFACT_CLEAN_VAR_PENALTY_WEIGHT = 0.25
ARTIFACT_CLEAN_VAR_PENALTY_MULT = 5.0
ARTIFACT_TV_EOG_WEIGHT = 0.8
ARTIFACT_TV_EMG_WEIGHT = 0.3
ARTIFACT_TV_ECG_WEIGHT = 1.0
ARTIFACT_PSD_DEV_WEIGHT = 2.0
ARTIFACT_PUSH_WEIGHT = 0.25
ARTIFACT_CLEAN_PRESERVE_WEIGHT = 0.15
ARTIFACT_SCALE_MATCH_WEIGHT = 0.2

# Clinical / Stage-4 anti-identity-collapse (physics-only, no hardware refs)
CLINICAL_ORTHO_MULT = 8.0
CLINICAL_TV_MULT = 5.0
CLINICAL_BASE_ORTHO_COEF = 0.5
CLINICAL_BASE_TV_COEF = 0.1
CLINICAL_STEM_VAR_RATIO_MIN = 0.015
CLINICAL_STEM_VAR_HINGE_POWER = 2.0
CLINICAL_STEM_VAR_PENALTY_WEIGHT = 2.5
CLINICAL_STEM_XCORR_WEIGHT = 0.75
CLINICAL_ART_PUSH_WEIGHT = 0.35
CLINICAL_PSD_DEV_WEIGHT = 1.5
CLINICAL_TV_EOG_WEIGHT = 0.8
CLINICAL_TV_EMG_WEIGHT = 0.35
CLINICAL_TV_ECG_WEIGHT = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MRANC (CUDA + AMP)")
    p.add_argument("--data-dir", type=str, default="processed_data")
    p.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=("auto", "deap", "seed", "clinical", "artifact_benchmark"),
        help=(
            "Dataset preset: auto uses --data-dir. "
            "deap/seed map to processed_data/processed_data_seed. "
            "clinical maps to processed_clinical_data. "
            "artifact_benchmark maps to data/artifact_benchmark/processed."
        ),
    )
    p.add_argument("--storage", type=str, default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--chunk-windows", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--feature-channels", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 when storage=cuda (required); 2 when storage=cpu fallback")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--noise-coeff-start", type=float, default=0.001)
    p.add_argument("--noise-coeff-end", type=float, default=0.01)
    p.add_argument("--noise-warmup-epochs", type=int, default=5)
    p.add_argument("--memory-fraction", type=float, default=0.85)
    p.add_argument("--max-batch-size", type=int, default=128)
    p.add_argument("--gradient-clip", type=float, default=1.0, help="Deprecated: clipping is fixed to max_norm=1.0")
    p.add_argument("--max-gpu-temp", type=float, default=86.0)
    p.add_argument("--abort-gpu-temp", type=float, default=92.0)
    p.add_argument("--no-temp-check", action="store_true")
    p.add_argument(
        "--resume",
        "--resume_weights",
        dest="resume_weights",
        type=str,
        default=None,
        help="Checkpoint weights path (supports raw state_dict or checkpoint['model_state_dict'])",
    )
    p.add_argument(
        "--checkpoint-out",
        type=str,
        default=None,
        help="Override best-checkpoint save path (default: dataset-specific path in checkpoints/)",
    )
    p.add_argument(
        "--save-new",
        action="store_true",
        help="Use a new versioned checkpoint filename (never reuse the default path)",
    )
    return p.parse_args()


def safety_config_from_args(args: argparse.Namespace) -> SafetyConfig:
    return SafetyConfig(
        memory_fraction=args.memory_fraction,
        max_batch_size=args.max_batch_size,
        max_gpu_temp_c=args.max_gpu_temp,
        abort_gpu_temp_c=args.abort_gpu_temp,
        gradient_clip_norm=1.0,
        check_temperature=not args.no_temp_check,
    )


def pearson_r(pred: torch.Tensor, ref: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    pred = pred.float()
    ref = ref.float()
    pred_c = pred - pred.mean(dim=dim, keepdim=True)
    ref_c = ref - ref.mean(dim=dim, keepdim=True)
    cov = (pred_c * ref_c).mean(dim=dim)
    denom = pred_c.pow(2).mean(dim=dim).sqrt() * ref_c.pow(2).mean(dim=dim).sqrt() + eps
    return torch.clamp(cov / denom, -1.0, 1.0)


def combined_ref_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Option B: log1p(MSE) + (1 - Pearson r) in fp32 for stability."""
    pred = pred.float()
    ref = ref.float()
    mse = F.mse_loss(pred, ref)
    return torch.log1p(mse) + (1.0 - pearson_r(pred, ref, dim=-1)).mean()


def polarity_aligned_ref_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Min loss over pred and -pred so inverted hardware sign does not penalize training."""
    ref = ref.float()
    pred = pred.float()
    loss_pos = combined_ref_loss(pred, ref)
    loss_neg = combined_ref_loss(-pred, ref)
    return torch.minimum(loss_pos, loss_neg)


def hardware_ref_loss(pred: torch.Tensor, ref: torch.Tensor, seq_len: int) -> torch.Tensor:
    """DEAP (T=256): polarity-aligned min loss; SEED/other T: standard combined ref loss."""
    if seq_len == DEAP_SEQ_LEN:
        return polarity_aligned_ref_loss(pred, ref)
    return combined_ref_loss(pred, ref)


def pearson_r_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    cov = (a * b).mean(dim=1)
    denom = a.pow(2).mean(dim=1).sqrt() * b.pow(2).mean(dim=1).sqrt() + eps
    return torch.clamp(cov / denom, -1.0, 1.0)


def orthogonality_loss_pearson(pred_eeg: torch.Tensor, stems: list[torch.Tensor]) -> torch.Tensor:
    eeg = pred_eeg.reshape(pred_eeg.shape[0], -1)
    penalties = []
    for stem in stems:
        s = stem.reshape(stem.shape[0], -1)
        r = pearson_r_flat(eeg, s)
        penalties.append((r**2).mean())
    return torch.stack(penalties).mean()


def per_stem_variance_hinge_loss(
    mix: torch.Tensor,
    stems: list[torch.Tensor],
    min_ratio: float,
    *,
    hinge_power: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Hinge penalty when any artifact stem variance is below min_ratio * mix variance (per window).
  Non-linear (power) hinge forces active decomposition instead of near-zero stems.
    """
    mix_var = mix.float().var(dim=(-2, -1), unbiased=False)
    deficits = []
    for stem in stems:
        stem_var = stem.float().var(dim=(-2, -1), unbiased=False)
        ratio = stem_var / (mix_var + eps)
        deficit = torch.relu(min_ratio - ratio)
        if hinge_power != 1.0:
            deficit = deficit.pow(hinge_power)
        deficits.append(deficit)
    return torch.stack(deficits, dim=0).mean()


def stem_cross_correlation_penalty(
    stems: list[torch.Tensor],
    *,
    eps: float = 1e-5,
    include_cross_channel: bool = True,
) -> torch.Tensor:
    """
    Penalize high Pearson correlation between different artifact stems.

    - Same scalp channel, temporal waveform (mirrored collapse across stems).
    - Cross-channel (e.g. Fp1 vs Cz) correlation of stem tensors.
    """
    if len(stems) < 2:
        return torch.tensor(0.0, device=stems[0].device, dtype=stems[0].float().dtype)
    penalties: list[torch.Tensor] = []
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a = stems[i].float()
            b = stems[j].float()
            r_same_ch = pearson_r(a, b, dim=-1)
            penalties.append((r_same_ch**2).mean())
            if include_cross_channel:
                a_n = (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + eps)
                b_n = (b - b.mean(dim=-1, keepdim=True)) / (b.std(dim=-1, keepdim=True) + eps)
                cross = torch.einsum("bct,bdt->bcd", a_n, b_n) / a.shape[-1]
                penalties.append((cross**2).mean())
    return torch.stack(penalties).mean()


def psd_deviation_from_mixture_loss(mix: torch.Tensor, pred_eeg: torch.Tensor) -> torch.Tensor:
    """Anti-identity: penalize cosine similarity of full-band PSD (clean vs raw mix)."""
    x_raw = mix.float()
    x_clean = pred_eeg.float()
    raw_psd = torch.abs(torch.fft.rfft(x_raw, dim=-1)) ** 2
    clean_psd = torch.abs(torch.fft.rfft(x_clean, dim=-1)) ** 2
    raw_flat = raw_psd.reshape(raw_psd.shape[0], -1)
    clean_flat = clean_psd.reshape(clean_psd.shape[0], -1)
    return F.cosine_similarity(raw_flat, clean_flat, dim=-1).mean()


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:] - x[..., :-1]).abs().mean()


def tv_physiological_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        total_variation(out["pred_eog"])
        + total_variation(out["pred_emg"])
        + total_variation(out["pred_ecg"])
    ) / 3.0


def noise_coeff(epoch: int, args: argparse.Namespace) -> float:
    if epoch <= args.noise_warmup_epochs:
        return args.noise_coeff_start
    if args.epochs <= args.noise_warmup_epochs:
        return args.noise_coeff_end
    t = (epoch - args.noise_warmup_epochs) / (args.epochs - args.noise_warmup_epochs)
    return args.noise_coeff_start + t * (args.noise_coeff_end - args.noise_coeff_start)


def resolve_num_workers(storage: str, requested: int) -> int:
    if storage == "cuda":
        if requested != 0:
            logger.info("Safety: num_workers forced to 0 (dataset preloaded on GPU)")
        return 0
    workers = 2 if requested <= 0 else requested
    logger.info("Safety: CPU storage — num_workers=%d, pin_memory=True", workers)
    return workers


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if v.device != device else v for k, v in batch.items()}


def compute_losses(
    model: MRANC,
    out: dict[str, torch.Tensor],
    ref_eog: torch.Tensor,
    ref_emg: torch.Tensor,
    ref_ecg: torch.Tensor,
    noise_mult: float,
) -> dict[str, torch.Tensor]:
    # Losses in fp32 (Pearson / log1p unstable in fp16 under AMP).
    seq_len = out["pred_eog"].shape[-1]
    proj_eog, proj_emg, proj_ecg = model.project_hardware_refs(
        out["pred_eog"].float(),
        out["pred_emg"].float(),
        out["pred_ecg"].float(),
        seq_len=seq_len,
    )
    loss_eog = hardware_ref_loss(proj_eog, ref_eog.float(), seq_len)
    loss_emg = hardware_ref_loss(proj_emg, ref_emg.float(), seq_len)
    loss_ecg = hardware_ref_loss(proj_ecg, ref_ecg.float(), seq_len)
    loss_basenoise = noise_mult * torch.mean(out["pred_basenoise"].float() ** 2)

    stems = [out["pred_eog"].float(), out["pred_emg"].float(), out["pred_ecg"].float(), out["pred_basenoise"].float()]
    loss_ortho = orthogonality_loss_pearson(out["pred_eeg"].float(), stems)
    loss_tv = tv_physiological_loss({k: v.float() for k, v in out.items() if k.startswith("pred_")})
    loss_stems = loss_eog + loss_emg + loss_ecg + loss_basenoise
    scale_penalty = SCALE_PENALTY_WEIGHT * (
        torch.mean(out["pred_eog"].float() ** 2)
        + torch.mean(out["pred_emg"].float() ** 2)
        + torch.mean(out["pred_ecg"].float() ** 2)
    )
    total_loss = loss_stems + ORTHO_WEIGHT * loss_ortho + TV_WEIGHT * loss_tv + scale_penalty

    return {
        "loss_eog": loss_eog,
        "loss_emg": loss_emg,
        "loss_ecg": loss_ecg,
        "loss_basenoise": loss_basenoise,
        "loss_stems": loss_stems,
        "loss_ortho": loss_ortho,
        "loss_tv": loss_tv,
        "loss_scale": scale_penalty,
        "total_loss": total_loss,
    }


def compute_losses_clinical(
    out: dict[str, torch.Tensor],
    *,
    mix: torch.Tensor,
    dataset_mode: str,
) -> dict[str, torch.Tensor]:
    """
    Clinical (CHB-MIT) UDA loss: physics-only, no hardware reference channels.
    """
    stems = [out["pred_eog"], out["pred_emg"], out["pred_ecg"], out["pred_basenoise"]]
    loss_ortho = orthogonality_loss_pearson(out["pred_eeg"].float(), [s.float() for s in stems])
    loss_tv = tv_physiological_loss({k: v.float() for k, v in out.items() if k.startswith("pred_")})

    # Artifact-only minimal cleaning penalty (no basenoise term).
    pred_eog2 = torch.mean(out["pred_eog"].float() ** 2)
    pred_emg2 = torch.mean(out["pred_emg"].float() ** 2)
    pred_ecg2 = torch.mean(out["pred_ecg"].float() ** 2)
    loss_min_clean = pred_eog2 + pred_emg2 + pred_ecg2

    loss_artifact_var = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_stem_xcorr = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_clean_spatial_var = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_psd_dev = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_art_push = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_clean_preserve = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    loss_scale_match = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)
    ortho_mult = 1.0
    tv_mult = 1.0
    artifact_stems = [out["pred_eog"], out["pred_emg"], out["pred_ecg"]]

    if dataset_mode == "artifact_benchmark":
        ortho_mult = ARTIFACT_BENCHMARK_ORTHO_MULT
        tv_mult = ARTIFACT_BENCHMARK_TV_MULT
        # Allow jagged movement/manifold routing by relaxing EMG (and slightly EOG) smoothness.
        tv_eog = total_variation(out["pred_eog"].float())
        tv_emg = total_variation(out["pred_emg"].float())
        tv_ecg = total_variation(out["pred_ecg"].float())
        tv_wsum = ARTIFACT_TV_EOG_WEIGHT + ARTIFACT_TV_EMG_WEIGHT + ARTIFACT_TV_ECG_WEIGHT
        loss_tv = (
            ARTIFACT_TV_EOG_WEIGHT * tv_eog
            + ARTIFACT_TV_EMG_WEIGHT * tv_emg
            + ARTIFACT_TV_ECG_WEIGHT * tv_ecg
        ) / tv_wsum

        artifact_total = out["pred_eog"] + out["pred_emg"] + out["pred_ecg"] + out["pred_basenoise"]
        # Per-window variance floor: penalize vanishing artifact manifolds.
        artifact_var = artifact_total.float().var(dim=(-2, -1), unbiased=False)
        loss_artifact_var = torch.relu(ARTIFACT_VAR_MIN_THRESHOLD - artifact_var).mean()
        # Threshold-free artifact variance push.
        loss_art_push = torch.exp(-artifact_var.mean())

        # Penalize single-channel dominance in cleaned EEG (localized spike bleed-through).
        clean_var_ch = out["pred_eeg"].float().var(dim=-1, unbiased=False)  # (B, C)
        clean_var_med = clean_var_ch.median(dim=1).values
        clean_var_max = clean_var_ch.max(dim=1).values
        clean_ratio = clean_var_max / (clean_var_med + 1e-8)
        loss_clean_spatial_var = torch.relu(clean_ratio - ARTIFACT_CLEAN_VAR_RATIO_THRESHOLD).mean()

        loss_psd_dev = psd_deviation_from_mixture_loss(mix, out["pred_eeg"])

        # Anti-attenuation barrier: prevent collapsing clean EEG variance to zero.
        clean_var = out["pred_eeg"].float().var(unbiased=False)
        loss_clean_preserve = -torch.log(clean_var + 1e-6)

        # Variance conservation boundary: keep separated-track amplitude physically realistic.
        x_raw = mix.float()
        x_clean = out["pred_eeg"].float()
        raw_var = x_raw.var(unbiased=False)
        pred_var = x_clean.var(unbiased=False)
        art_var_total = artifact_total.float().var(unbiased=False)
        loss_scale_match = torch.abs((pred_var + art_var_total) - raw_var)

        total_loss = (
            CLINICAL_BASE_ORTHO_COEF * ortho_mult * loss_ortho
            + CLINICAL_BASE_TV_COEF * tv_mult * loss_tv
            + ARTIFACT_VAR_PENALTY_WEIGHT * loss_artifact_var
            + (ARTIFACT_CLEAN_VAR_PENALTY_WEIGHT * ARTIFACT_CLEAN_VAR_PENALTY_MULT) * loss_clean_spatial_var
            + ARTIFACT_PSD_DEV_WEIGHT * loss_psd_dev
            + ARTIFACT_PUSH_WEIGHT * loss_art_push
            + ARTIFACT_CLEAN_PRESERVE_WEIGHT * loss_clean_preserve
            + ARTIFACT_SCALE_MATCH_WEIGHT * loss_scale_match
        )
    elif dataset_mode == "clinical":
        ortho_mult = CLINICAL_ORTHO_MULT
        tv_mult = CLINICAL_TV_MULT
        tv_eog = total_variation(out["pred_eog"].float())
        tv_emg = total_variation(out["pred_emg"].float())
        tv_ecg = total_variation(out["pred_ecg"].float())
        tv_wsum = CLINICAL_TV_EOG_WEIGHT + CLINICAL_TV_EMG_WEIGHT + CLINICAL_TV_ECG_WEIGHT
        loss_tv = (
            CLINICAL_TV_EOG_WEIGHT * tv_eog
            + CLINICAL_TV_EMG_WEIGHT * tv_emg
            + CLINICAL_TV_ECG_WEIGHT * tv_ecg
        ) / tv_wsum

        mix_f = mix.float()
        loss_artifact_var = per_stem_variance_hinge_loss(
            mix_f,
            artifact_stems,
            CLINICAL_STEM_VAR_RATIO_MIN,
            hinge_power=CLINICAL_STEM_VAR_HINGE_POWER,
        )
        loss_stem_xcorr = stem_cross_correlation_penalty(artifact_stems)
        mix_var_w = mix_f.var(dim=(-2, -1), unbiased=False)
        push_terms = []
        for stem in artifact_stems:
            stem_var_w = stem.float().var(dim=(-2, -1), unbiased=False)
            push_terms.append(torch.exp(-stem_var_w / (mix_var_w + 1e-8)))
        loss_art_push = torch.stack(push_terms, dim=0).mean()
        loss_psd_dev = psd_deviation_from_mixture_loss(mix, out["pred_eeg"])
        loss_min_clean = torch.tensor(0.0, device=out["pred_eeg"].device, dtype=out["pred_eeg"].dtype)

        total_loss = (
            CLINICAL_BASE_ORTHO_COEF * ortho_mult * loss_ortho
            + CLINICAL_BASE_TV_COEF * tv_mult * loss_tv
            + CLINICAL_STEM_VAR_PENALTY_WEIGHT * loss_artifact_var
            + CLINICAL_STEM_XCORR_WEIGHT * loss_stem_xcorr
            + CLINICAL_ART_PUSH_WEIGHT * loss_art_push
            + CLINICAL_PSD_DEV_WEIGHT * loss_psd_dev
        )
    else:
        raise ValueError(f"Unsupported dataset_mode for clinical losses: {dataset_mode!r}")

    losses = {
        "loss_ortho": loss_ortho,
        "loss_tv": loss_tv,
        "loss_min_clean": loss_min_clean,
        "loss_artifact_var": loss_artifact_var,
        "loss_stem_xcorr": loss_stem_xcorr,
        "loss_clean_spatial_var": loss_clean_spatial_var,
        "loss_psd_dev": loss_psd_dev,
        "loss_art_push": loss_art_push,
        "loss_clean_preserve": loss_clean_preserve,
        "loss_scale_match": loss_scale_match,
        "total_loss": total_loss,
    }
    if dataset_mode in ("artifact_benchmark", "clinical"):
        losses["ortho_mult"] = torch.tensor(
            ortho_mult,
            device=out["pred_eeg"].device,
            dtype=out["pred_eeg"].dtype,
        )
        losses["tv_mult"] = torch.tensor(
            tv_mult,
            device=out["pred_eeg"].device,
            dtype=out["pred_eeg"].dtype,
        )
    return losses


def run_epoch(
    model: MRANC,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    coeff = noise_coeff(epoch, args)

    is_physics_only = getattr(args, "dataset", None) in ("clinical", "artifact_benchmark")
    if is_physics_only:
        keys = [
            "loss_ortho",
            "loss_tv",
            "loss_min_clean",
            "loss_artifact_var",
            "loss_stem_xcorr",
            "loss_clean_spatial_var",
            "loss_psd_dev",
            "loss_art_push",
            "loss_clean_preserve",
            "loss_scale_match",
            "total_loss",
        ]
    else:
        keys = [
            "loss_eog",
            "loss_emg",
            "loss_ecg",
            "loss_basenoise",
            "loss_ortho",
            "loss_tv",
            "loss_scale",
            "total_loss",
        ]

    running = {k: 0.0 for k in keys}
    n_batches = 0
    skipped = 0

    for step, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch["mix"])
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if is_physics_only:
                losses = compute_losses_clinical(
                    out,
                    mix=batch["mix"],
                    dataset_mode=getattr(args, "dataset", "auto"),
                )
            else:
                losses = compute_losses(
                    model,
                    out,
                    batch["ref_eog"],
                    batch["ref_emg"],
                    batch["ref_ecg"],
                    coeff,
                )

        total = losses["total_loss"]
        if not torch.isfinite(total):
            skipped += 1
            logger.warning("Skipping batch with non-finite loss (epoch %d)", epoch)
            continue

        if getattr(args, "dataset", None) == "artifact_benchmark" and step % 10 == 0:
            print(
                f"Step [{step}] | loss_total: {float(total.detach().item()):.4f} "
                f"| loss_psd_dev: {float(losses['loss_psd_dev'].detach().item()):.4f} "
                f"| loss_clean_spatial_var: {float(losses['loss_clean_spatial_var'].detach().item()):.4f} "
                f"| loss_clean_preserve: {float(losses['loss_clean_preserve'].detach().item()):.4f} "
                f"| loss_scale_match: {float(losses['loss_scale_match'].detach().item()):.4f}"
            )

        if train_mode:
            assert optimizer is not None and scaler is not None
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            # Safety guard against exploding gradients on extreme artifact windows.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()

        n_batches += 1
        for k in keys:
            running[k] += losses[k].item()

    if skipped:
        logger.warning("Epoch %d: skipped %d non-finite batches", epoch, skipped)

    n = max(1, n_batches)
    m = {k: v / n for k, v in running.items()}
    m["noise_coeff"] = coeff
    return m


def log_epoch_metrics(split: str, epoch: int, epochs: int, m: dict[str, float]) -> None:
    is_clinical = "loss_eog" not in m
    if is_clinical:
        logger.info(
            "Epoch %d/%d [%s] ortho=%.4f tv=%.4f art_var=%.4f stem_xcorr=%.4f psd_dev=%.4f "
            "art_push=%.4f total=%.4f",
            epoch,
            epochs,
            split,
            m["loss_ortho"],
            m["loss_tv"],
            m["loss_artifact_var"],
            m.get("loss_stem_xcorr", 0.0),
            m["loss_psd_dev"],
            m["loss_art_push"],
            m["total_loss"],
        )
    else:
        logger.info(
            "Epoch %d/%d [%s] eog=%.4f emg=%.4f ecg=%.4f basenoise=%.4f ortho=%.4f tv=%.4f scale=%.4f total=%.4f",
            epoch,
            epochs,
            split,
            m["loss_eog"],
            m["loss_emg"],
            m["loss_ecg"],
            m["loss_basenoise"],
            m["loss_ortho"],
            m["loss_tv"],
            m["loss_scale"],
            m["total_loss"],
        )


def _npy_to_device(path: Path, device: torch.device, chunk_windows: int = 2048) -> torch.Tensor:
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


class ClinicalMixDataset(Dataset):
    """
    Clinical dataset: only `mix.npy` (B, 32, 1000). No hardware reference wires exist.
    """

    def __init__(
        self,
        data_dir: str | Path,
        storage: str = "cuda",
        chunk_windows: int = 2048,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"processed_clinical_data directory not found: {self.data_dir}")

        if storage not in ("cpu", "cuda"):
            raise ValueError(f"storage must be 'cpu' or 'cuda', got {storage!r}")
        if storage == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("storage='cuda' requested but CUDA is not available.")

        self.storage = storage
        self.device = torch.device("cuda" if storage == "cuda" else "cpu")

        mix_path = self.data_dir / "mix.npy"
        if not mix_path.is_file():
            raise FileNotFoundError(f"Missing clinical mix.npy: {mix_path}")

        if storage == "cuda":
            logger.info("Loading processed_clinical_data to GPU VRAM (chunked)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("Loading processed_clinical_data to CPU RAM...")

        self.mix = _npy_to_device(mix_path, self.device, chunk_windows=chunk_windows)
        # Clinical sanitization: hard-clip and dead-channel replacement.
        # This runs once at dataset-load time to ensure all batches are clean
        # before they reach the GPU/model.
        self.mix = sanitize_clinical_batch(self.mix)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.mix.ndim != 3:
            raise ValueError(f"mix must be 3D (N, 32, T). Got shape {tuple(self.mix.shape)}")
        if self.mix.shape[1] != 32:
            raise ValueError(f"mix channels must be 32, got {self.mix.shape[1]}")

    def __len__(self) -> int:
        return int(self.mix.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"mix": self.mix[idx]}


def build_clinical_train_val_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
    storage: str,
    chunk_windows: int,
) -> tuple[DataLoader, DataLoader, ClinicalMixDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    full = ClinicalMixDataset(data_dir=data_dir, storage=storage, chunk_windows=chunk_windows)
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val

    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if storage == "cuda" and num_workers > 0:
        raise ValueError("num_workers must be 0 when clinical dataset storage='cuda'")

    pin_memory = torch.cuda.is_available() and storage != "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, full


class ArtifactBenchmarkMixDataset(Dataset):
    """
    Artifact benchmark dataset: only `mix.npy` (N, 32, T). No hardware reference wires.

    Unlike clinical, no dataset-specific sanitization is applied by default.
    """

    def __init__(
        self,
        data_dir: str | Path,
        storage: str = "cuda",
        chunk_windows: int = 2048,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"artifact_benchmark processed directory not found: {self.data_dir}")

        if storage not in ("cpu", "cuda"):
            raise ValueError(f"storage must be 'cpu' or 'cuda', got {storage!r}")
        if storage == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("storage='cuda' requested but CUDA is not available.")

        self.storage = storage
        self.device = torch.device("cuda" if storage == "cuda" else "cpu")

        mix_path = self.data_dir / "mix.npy"
        if not mix_path.is_file():
            raise FileNotFoundError(f"Missing artifact_benchmark mix.npy: {mix_path}")

        if storage == "cuda":
            logger.info("Loading artifact_benchmark to GPU VRAM (chunked)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("Loading artifact_benchmark to CPU RAM...")

        self.mix = _npy_to_device(mix_path, self.device, chunk_windows=chunk_windows)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.mix.ndim != 3:
            raise ValueError(f"mix must be 3D (N, 32, T). Got shape {tuple(self.mix.shape)}")
        if self.mix.shape[1] != 32:
            raise ValueError(f"mix channels must be 32, got {self.mix.shape[1]}")

    def __len__(self) -> int:
        return int(self.mix.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"mix": self.mix[idx]}


def build_artifact_benchmark_train_val_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    val_fraction: float,
    seed: int,
    num_workers: int,
    storage: str,
    chunk_windows: int,
) -> tuple[DataLoader, DataLoader, ArtifactBenchmarkMixDataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    full = ArtifactBenchmarkMixDataset(data_dir=data_dir, storage=storage, chunk_windows=chunk_windows)
    n_val = max(1, int(len(full) * val_fraction))
    n_train = len(full) - n_val

    train_ds, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    if storage == "cuda" and num_workers > 0:
        raise ValueError("num_workers must be 0 when artifact_benchmark storage='cuda'")

    pin_memory = torch.cuda.is_available() and storage != "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, full


def clinical_preflight(
    cfg: SafetyConfig,
    data_dir: Path,
    storage: str,
    batch_size: int,
) -> tuple[str, int]:
    """
    Safety preflight for clinical: only `mix.npy` exists (no ref_eog/ref_emg/ref_ecg).
    Adjust VRAM batch-size estimate by T/DEAP_SEQ_LEN.
    """
    apply_cuda_limits(cfg)
    check_gpu_temperature(cfg)

    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        raise FileNotFoundError(f"Missing clinical mix.npy: {mix_path}")

    data_gb = mix_path.stat().st_size / 1e9
    t = int(np.load(mix_path, mmap_mode="r").shape[2])
    act_scale = float(t) / float(DEAP_SEQ_LEN)

    batch_size = min(batch_size, cfg.max_batch_size)

    if storage == "cuda":
        free_gb, total_gb, _ = cuda_mem_gb()
        need_gb = data_gb + MODEL_TRAIN_OVERHEAD_GB + batch_size * PER_BATCH_ACTIVATION_GB * act_scale + cfg.vram_reserve_gb

        logger.info(
            "Safety(clinical): VRAM free %.2f / %.2f GB | mix=%.2f GB | T=%d => act_scale=%.3f | need ~%.2f GB (batch=%d)",
            free_gb,
            total_gb,
            data_gb,
            t,
            act_scale,
            need_gb,
            batch_size,
        )

        if free_gb < need_gb and total_gb > 0:
            usable = free_gb - cfg.vram_reserve_gb - data_gb - MODEL_TRAIN_OVERHEAD_GB
            denom = PER_BATCH_ACTIVATION_GB * act_scale
            safe_batch = int(usable / denom) if denom > 0 else cfg.min_batch_size
            safe_batch = max(cfg.min_batch_size, min(cfg.max_batch_size, safe_batch))

            if free_gb < need_gb * 0.95 and safe_batch <= cfg.min_batch_size:
                raise RuntimeError(
                    "Safety(clinical): VRAM too tight and CPU fallback is disabled. "
                    f"Reduce --batch-size (current batch_size={batch_size}) or use a GPU with more VRAM. "
                    f"(free_gb={free_gb:.2f}, need_gb={need_gb:.2f})"
                )
            else:
                if safe_batch < batch_size:
                    logger.warning("Safety(clinical): reducing batch_size %d -> %d", batch_size, safe_batch)
                    batch_size = safe_batch
    else:
        ram_free = system_ram_free_gb()
        if ram_free is not None:
            logger.info("Safety(clinical): system RAM free ~%.2f GB", ram_free)
            if ram_free < data_gb + cfg.min_free_ram_gb:
                raise RuntimeError(
                    f"Not enough free RAM ({ram_free:.1f} GB) to load clinical dataset (~{data_gb:.1f} GB). "
                    f"Close other apps or reduce batch-size."
                )

    return storage, batch_size


def artifact_benchmark_preflight(
    cfg: SafetyConfig,
    data_dir: Path,
    storage: str,
    batch_size: int,
) -> tuple[str, int]:
    """
    Safety preflight for artifact_benchmark: only `mix.npy` exists (no refs).
    Mirrors clinical preflight logic.
    """
    return clinical_preflight(cfg, data_dir, storage, batch_size)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No GPU detected — aborting.")

    safety = safety_config_from_args(args)

    dataset_mode = getattr(args, "dataset", "auto")
    is_physics_only = dataset_mode in ("clinical", "artifact_benchmark")

    # For artifact benchmark, run longer by default unless user explicitly sets --epochs.
    epochs_explicit = any(arg == "--epochs" or arg.startswith("--epochs=") for arg in sys.argv[1:])
    if dataset_mode == "artifact_benchmark" and not epochs_explicit and args.epochs < 100:
        args.epochs = 100
        logger.info("artifact_benchmark defaulting to longer run: epochs=%d", args.epochs)

    # If resuming artifact_benchmark and user did not pass --lr explicitly,
    # lower LR for safer fine-tuning convergence.
    lr_explicit = any(arg == "--lr" or arg.startswith("--lr=") for arg in sys.argv[1:])
    if dataset_mode == "artifact_benchmark" and args.resume_weights and not lr_explicit:
        args.lr = 1e-4
        logger.info(
            "artifact_benchmark resume detected with implicit LR; using safer fine-tune lr=%.6f",
            args.lr,
        )

    if dataset_mode == "auto":
        data_dir = PROJECT_ROOT / args.data_dir
        storage, batch_size = preflight(safety, data_dir, args.storage, args.batch_size)
    elif dataset_mode == "deap":
        data_dir = PROCESSED_DEAP
        storage, batch_size = preflight(safety, data_dir, args.storage, args.batch_size)
    elif dataset_mode == "seed":
        data_dir = PROCESSED_SEED
        storage, batch_size = preflight(safety, data_dir, args.storage, args.batch_size)
    elif dataset_mode == "clinical":
        data_dir = PROCESSED_CLINICAL
        storage, batch_size = clinical_preflight(safety, data_dir, args.storage, args.batch_size)
    elif dataset_mode == "artifact_benchmark":
        data_dir = ARTIFACT_BENCHMARK_PROCESSED
        storage, batch_size = artifact_benchmark_preflight(safety, data_dir, args.storage, args.batch_size)
    else:
        raise ValueError(f"Unknown dataset_mode={dataset_mode!r}")

    args.storage = storage
    args.batch_size = batch_size
    args.num_workers = resolve_num_workers(args.storage, args.num_workers)

    device = torch.device("cuda")
    scaler = torch.amp.GradScaler("cuda")

    try:
        if dataset_mode == "clinical":
            train_loader, val_loader, dataset = build_clinical_train_val_dataloaders(
                data_dir=data_dir,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                seed=args.split_seed,
                num_workers=args.num_workers,
                storage=args.storage,
                chunk_windows=args.chunk_windows,
            )
        elif dataset_mode == "artifact_benchmark":
            train_loader, val_loader, dataset = build_artifact_benchmark_train_val_dataloaders(
                data_dir=data_dir,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                seed=args.split_seed,
                num_workers=args.num_workers,
                storage=args.storage,
                chunk_windows=args.chunk_windows,
            )
        else:
            train_loader, val_loader, dataset = build_train_val_dataloaders(
                data_dir=data_dir,
                batch_size=args.batch_size,
                val_fraction=args.val_fraction,
                seed=args.split_seed,
                num_workers=args.num_workers,
                storage=args.storage,
                chunk_windows=args.chunk_windows,
            )
        model = MRANC(feature_channels=args.feature_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8)
    except RuntimeError as exc:
        if is_oom_error(exc):
            raise RuntimeError(
                "CUDA out of memory during setup. Retry with --batch-size 64 or --storage cpu"
            ) from exc
        raise

    if args.checkpoint_out:
        ckpt_path = Path(args.checkpoint_out)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if ckpt_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite existing checkpoint at {ckpt_path}. "
                "Use a new --checkpoint-out path."
            )
    else:
        base_ckpt = (
            ARTIFACT_BENCHMARK_CHECKPOINT
            if dataset_mode == "artifact_benchmark"
            else DEFAULT_CHECKPOINT
        )
        if args.save_new:
            ckpt_path = versioned_weight_path(base_ckpt, new_run_id())
            logger.info("Versioned checkpoint path: %s", ckpt_path)
        else:
            ckpt_path = base_ckpt
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume_weights:
        resume_path = Path(args.resume_weights)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logger.info("🔄 Safely loading existing weights from %s...", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        adapter_reset = False
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            adapter_reset = load_model_weights_compat(model, ckpt["model_state_dict"])
            has_optimizer_state = "optimizer_state_dict" in ckpt
            has_scaler_state = "scaler_state_dict" in ckpt
        elif isinstance(ckpt, dict):
            adapter_reset = load_model_weights_compat(model, ckpt)
            has_optimizer_state = False
            has_scaler_state = False
        else:
            raise ValueError(
                f"Unsupported checkpoint format at {resume_path}: expected dict or checkpoint with model_state_dict"
            )
        transfer_phase = bool(args.checkpoint_out and args.resume_weights)
        if adapter_reset:
            logger.info(
                "Adapter was not in checkpoint; using fresh optimizer/scaler and starting at epoch 1."
            )
            best_val = float("inf")
            start_epoch = 1
        elif transfer_phase:
            logger.info(
                "Transfer phase: weights from %s only; epoch 1..%d, saving to %s",
                resume_path,
                args.epochs,
                ckpt_path,
            )
            best_val = float("inf")
            start_epoch = 1
        elif is_physics_only:
            best_val = float("inf")
            start_epoch = 1
            if dataset_mode == "artifact_benchmark":
                logger.info(
                    "Loaded artifact_benchmark weights from %s with clean optimizer/scaler reset "
                    "(legacy momentum cleared). Starting at epoch %d.",
                    resume_path,
                    start_epoch,
                )
            else:
                logger.info(
                    "Loaded physics-only weights from %s (fresh optimizer/scaler). Starting at epoch %d.",
                    resume_path,
                    start_epoch,
                )
        else:
            if has_optimizer_state:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if has_scaler_state:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val_total_loss", float("inf")))
            logger.info(
                "Resumed from %s at epoch %d (best_val=%.6f)",
                resume_path,
                start_epoch,
                best_val,
            )
    else:
        best_val = float("inf")

    if start_epoch > args.epochs:
        logger.warning(
            "start_epoch=%d exceeds --epochs=%d; resetting to epoch 1 for this training phase.",
            start_epoch,
            args.epochs,
        )
        start_epoch = 1
        best_val = float("inf")

    logger.info(
        "MRANC | AMP=fp16 | dataset=%s | storage=%s | N=%d train=%d val=%d | batch=%d | workers=%d | cudnn.benchmark=True",
        dataset_mode,
        args.storage,
        len(dataset),
        len(train_loader.dataset),
        len(val_loader.dataset),
        args.batch_size,
        args.num_workers,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        check_gpu_temperature(safety)
        try:
            tr = run_epoch(model, train_loader, device, epoch, args, optimizer, scaler)
            va = run_epoch(model, val_loader, device, epoch, args, None, None)
        except RuntimeError as exc:
            if is_oom_error(exc):
                raise RuntimeError(
                    f"CUDA OOM at epoch {epoch}. Retry with --batch-size 64 or --storage cpu "
                    f"(current batch={args.batch_size}, storage={args.storage})."
                ) from exc
            raise

        if not (math.isfinite(tr["total_loss"]) and math.isfinite(va["total_loss"])):
            logger.error(
                "Non-finite epoch averages at epoch %d (train=%s val=%s). "
                "Best checkpoint kept at %s — resume with --resume_weights.",
                epoch, tr["total_loss"], va["total_loss"], ckpt_path,
            )
            break

        log_epoch_metrics("train", epoch, args.epochs, tr)
        log_epoch_metrics("val", epoch, args.epochs, va)

        if dataset_mode in ("seed", "deap"):
            fs = DEFAULT_SAMPLE_RATE[dataset_mode]
            max_batches = 32 if dataset_mode == "deap" else None
            qm = run_supervised_val_quality_metrics(
                model,
                val_loader,
                device,
                dataset_mode,
                fs,
                max_batches=max_batches,
            )
            logger.info(
                "Epoch %d/%d [val_quality] SNR_improve=%.3f dB artifact_RMSE=%.4f "
                "PSD_corr=%.4f (n=%.0f windows)",
                epoch,
                args.epochs,
                qm["snr_improvement_db"],
                qm["artifact_magnitude_rmse"],
                qm["psd_alpha_beta_correlation"],
                qm["n_windows"],
            )

        if va["total_loss"] < best_val:
            best_val = va["total_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_total_loss": best_val,
                    "config": vars(args),
                },
                ckpt_path,
            )
            logger.info("Saved %s (val_total=%.6f)", ckpt_path, best_val)

        after_epoch_cleanup()


if __name__ == "__main__":
    main()
