"""
Amplitude alignment and sign correction for supervised MRANC validation metrics.
"""

from __future__ import annotations

import torch

ARTIFACT_KEYS = ("pred_eog", "pred_emg", "pred_ecg", "pred_basenoise")
SUPERVISED_METRIC_DATASETS = frozenset({"seed", "deap", "clinical", "artifact_benchmark"})


def amplitude_align_clean_artifact_to_mix(
    mix: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor,
    *,
    eps: float = 1e-8,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match temporal std of clean to mix (per-channel or per-window RMS), then artifact = mix - clean.
    """
    if per_channel:
        ref_std = mix.std(dim=-1, keepdim=True).clamp_min(eps)
        clean_std = clean.std(dim=-1, keepdim=True).clamp_min(eps)
    else:
        ref_std = mix.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
        clean_std = clean.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
    scale = ref_std / clean_std
    clean_aligned = clean * scale
    artifact_aligned = mix - clean_aligned
    return clean_aligned, artifact_aligned


def scale_artifact_for_snr(
    mix: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale artifact stems so residual (mix - clean) and artifact share the same RMS."""
    residual_std = (mix - clean).std(dim=(1, 2), keepdim=True).clamp_min(eps)
    artifact_std = artifact.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    return artifact * (residual_std / artifact_std)


def align_artifacts_for_subtraction_torch(
    stems: dict[str, torch.Tensor],
    mix: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Per-channel Pearson sign alignment of artifact stems to mix."""
    aligned: dict[str, torch.Tensor] = {}
    mix_zm = mix - mix.mean(dim=-1, keepdim=True)
    mix_std = mix_zm.std(dim=-1, keepdim=True).clamp_min(eps)

    for key in ARTIFACT_KEYS:
        s = stems[key]
        s_zm = s - s.mean(dim=-1, keepdim=True)
        s_std = s_zm.std(dim=-1, keepdim=True).clamp_min(eps)
        r = (s_zm * mix_zm).mean(dim=-1, keepdim=True) / (s_std * mix_std)
        s_aligned = torch.where(r < 0.0, -s_zm, s_zm)
        aligned[key] = s_aligned

    aligned["pred_eeg"] = mix - sum(aligned[k] for k in ARTIFACT_KEYS)
    return aligned


def total_artifact_torch(stems: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(stems[k] for k in ARTIFACT_KEYS)


def zscore_traces_for_distance_metrics(
    mix: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window z-score along time (per channel) before RMSE/SNR distance metrics."""
    def _z(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
        return (x - mu) / sd

    return _z(mix), _z(clean), _z(artifact)


def prepare_supervised_eval_tensors(
    mix: torch.Tensor,
    model_out: dict[str, torch.Tensor],
    *,
    dataset: str,
    zscore_distance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Build clean/artifact tensors from a live model forward pass for metric evaluation.

    Always uses sign-aligned stems and amplitude matching for supervised corpora (including DEAP).
    Optionally z-scores traces for SEED distance metrics to remove residual scale bias.
    """
    mix_f = mix.float()
    stems = {k: model_out[k].float() for k in ARTIFACT_KEYS}

    if dataset in SUPERVISED_METRIC_DATASETS:
        aligned = align_artifacts_for_subtraction_torch(stems, mix_f)
        clean_pre = aligned["pred_eeg"]
        artifact_pre = total_artifact_torch(aligned)
        clean_for_psd = clean_pre
        clean_t, artifact_t = amplitude_align_clean_artifact_to_mix(mix_f, clean_pre, artifact_pre)
    else:
        clean_for_psd = None
        clean_t = model_out["pred_eeg"].float()
        artifact_t = total_artifact_torch(stems)

    if zscore_distance:
        mix_z, clean_t, _ = zscore_traces_for_distance_metrics(mix_f, clean_t, artifact_t)
        artifact_t = mix_z - clean_t
        mix_f = mix_z

    artifact_for_snr = scale_artifact_for_snr(mix_f, clean_t, artifact_t)
    return mix_f, clean_t, artifact_t, clean_for_psd, artifact_for_snr
