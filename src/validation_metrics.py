"""
GPU validation metrics shared by evaluate_metrics.py and train.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from metric_alignment import ARTIFACT_KEYS, prepare_supervised_eval_tensors
from model import MRANC

PSD_BAND_LOW_HZ = 8.0
PSD_BAND_HIGH_HZ = 30.0
EPS = 1e-12

DEFAULT_SAMPLE_RATE = {
    "seed": 200.0,
    "deap": 128.0,
    "clinical": 200.0,
    "artifact_benchmark": 200.0,
}


def _pearson_corr_torch_1d(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.float()
    b = b.float()
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    num = (a * b).sum(dim=-1)
    den = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1) + eps
    return num / den


def _alpha_beta_psd_profile(x: torch.Tensor, fs: float) -> torch.Tensor:
    """
    Channel-averaged PSD in 8-30 Hz, shape (B, n_bins).

    Spatial-mean traces often have near-zero band power on DEAP (T=256); averaging
    per-channel PSD avoids degenerate flat profiles and bogus correlations.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    x = x.float()
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1.0 / float(fs)).to(x.device)
    band = (freqs >= PSD_BAND_LOW_HZ) & (freqs <= PSD_BAND_HIGH_HZ)
    if not bool(band.any().item()):
        raise ValueError("No FFT bins in alpha/beta band for this window length and sample rate.")
    spec = torch.fft.rfft(x, dim=-1)
    psd = (spec.abs() ** 2)[..., band]
    profile = psd.mean(dim=1)
    profile = profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return profile


def psd_alpha_beta_correlation_torch(
    mix: torch.Tensor,
    clean: torch.Tensor,
    fs: float,
) -> torch.Tensor:
    """Normalized spectral-shape correlation over 8-30 Hz; flat profiles return NaN."""
    psd_mix = _alpha_beta_psd_profile(mix, fs)
    psd_clean = _alpha_beta_psd_profile(clean, fs)

    std_mix = psd_mix.std(dim=-1)
    std_clean = psd_clean.std(dim=-1)
    ok = (std_mix >= 1e-9) & (std_clean >= 1e-9)
    corr = _pearson_corr_torch_1d(psd_mix, psd_clean, eps=EPS)
    return torch.where(ok, corr, torch.full_like(corr, float("nan")))


def compute_batch_metrics_torch(
    mix: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor,
    fs: float,
    artifact_for_snr: torch.Tensor | None = None,
    clean_for_psd: torch.Tensor | None = None,
    mix_for_psd: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    reconstructed = clean + artifact
    loss_mse = (mix - reconstructed).pow(2).mean(dim=(1, 2))

    mix_sp = mix.mean(dim=1)
    clean_sp = clean.mean(dim=1)
    clean_psd_input = clean_for_psd if clean_for_psd is not None else clean
    residual_sp = (mix - clean - artifact).mean(dim=1)
    snr_art = artifact if artifact_for_snr is None else artifact_for_snr
    snr_art_sp = snr_art.mean(dim=1)

    var_floor = 1e-6
    p_mix = mix_sp.var(dim=1).clamp_min(var_floor)
    p_noise_mix = (mix - clean).mean(dim=1).var(dim=1).clamp_min(var_floor)
    p_signal_clean = clean_sp.var(dim=1).clamp_min(var_floor)
    p_noise_art = snr_art_sp.var(dim=1).clamp_min(var_floor)
    snr_mix = 10.0 * torch.log10(p_mix / p_noise_mix)
    snr_clean = 10.0 * torch.log10(p_signal_clean / p_noise_art)
    snr_improvement_db = snr_clean - snr_mix

    physics_residual_rmse = residual_sp.pow(2).mean(dim=1).sqrt()
    artifact_magnitude_rmse = (mix_sp - clean_sp).pow(2).mean(dim=1).sqrt()
    psd_mix = mix_for_psd if mix_for_psd is not None else mix
    psd_corr = psd_alpha_beta_correlation_torch(psd_mix, clean_psd_input, fs)

    mix_ch = mix.reshape(mix.shape[0], mix.shape[1], -1)
    clean_ch = clean.reshape(clean.shape[0], clean.shape[1], -1)
    residual_ch = (mix - clean - artifact).reshape(mix.shape[0], mix.shape[1], -1)
    raw_noise_ch = (mix - clean).reshape(mix.shape[0], mix.shape[1], -1)
    snr_art_ch = snr_art.reshape(mix.shape[0], mix.shape[1], -1)
    p_mix_ch = mix_ch.var(dim=-1).clamp_min(var_floor)
    p_noise_mix_ch = raw_noise_ch.var(dim=-1).clamp_min(var_floor)
    p_signal_clean_ch = clean_ch.var(dim=-1).clamp_min(var_floor)
    p_noise_art_ch = snr_art_ch.var(dim=-1).clamp_min(var_floor)
    snr_mix_ch = 10.0 * torch.log10(p_mix_ch / p_noise_mix_ch)
    snr_clean_ch = 10.0 * torch.log10(p_signal_clean_ch / p_noise_art_ch)
    snr_ch = snr_clean_ch - snr_mix_ch

    per_channel_mean = {
        "snr_improvement_db": snr_ch.mean(dim=1),
        "physics_residual_rmse": residual_ch.pow(2).mean(dim=-1).sqrt().mean(dim=1),
        "artifact_magnitude_rmse": (mix_ch - clean_ch).pow(2).mean(dim=-1).sqrt().mean(dim=1),
    }

    metrics = {
        "loss_mse": loss_mse,
        "snr_improvement_db": snr_improvement_db,
        "psd_alpha_beta_correlation": psd_corr,
        "physics_residual_rmse": physics_residual_rmse,
        "artifact_magnitude_rmse": artifact_magnitude_rmse,
    }
    return metrics, per_channel_mean


def aggregate_metrics(per_window: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = (
        "loss_mse",
        "snr_improvement_db",
        "psd_alpha_beta_correlation",
        "physics_residual_rmse",
        "artifact_magnitude_rmse",
    )
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        raw = [w[key] for w in per_window if w.get(key) is not None]
        vals = np.array(raw, dtype=np.float64) if raw else np.array([], dtype=np.float64)
        out[key] = {
            "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
            "std": float(np.nanstd(vals)) if vals.size else float("nan"),
            "n_valid": int(np.isfinite(vals).sum()) if vals.size else 0,
        }
    return out


def run_supervised_val_quality_metrics(
    model: MRANC,
    loader: DataLoader,
    device: torch.device,
    dataset: str,
    fs: float,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Validation-loop SNR / RMSE / PSD using a live model(mix) forward pass.
    """
    was_training = model.training
    model.eval()
    zscore_distance = dataset == "seed"
    sums: dict[str, float] = {
        "snr_improvement_db": 0.0,
        "artifact_magnitude_rmse": 0.0,
        "psd_alpha_beta_correlation": 0.0,
        "physics_residual_rmse": 0.0,
    }
    counts: dict[str, int] = {k: 0 for k in sums}

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            mix = batch["mix"].to(device, non_blocking=True)
            out = model(mix)
            mix_eval, clean_t, artifact_t, clean_for_psd, artifact_for_snr = prepare_supervised_eval_tensors(
                mix,
                out,
                dataset=dataset,
                zscore_distance=zscore_distance,
            )
            batch_metrics, _ = compute_batch_metrics_torch(
                mix_eval,
                clean_t,
                artifact_t,
                fs,
                artifact_for_snr=artifact_for_snr,
                clean_for_psd=clean_for_psd,
                mix_for_psd=mix.float() if zscore_distance else None,
            )
            for key in sums:
                vals = batch_metrics[key].detach().cpu().numpy()
                for v in vals:
                    if np.isfinite(v):
                        sums[key] += float(v)
                        counts[key] += 1

    if was_training:
        model.train()

    out_metrics: dict[str, float] = {}
    for key in sums:
        out_metrics[key] = sums[key] / max(counts[key], 1)
    out_metrics["n_windows"] = float(counts["snr_improvement_db"])
    return out_metrics
