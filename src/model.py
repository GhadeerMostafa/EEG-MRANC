"""
MRANC: Multi-Reference Adaptive Noise Canceller (production).

Physics (locked):
  - forward(mix) ONLY — raw scalp (B, 32, T); never ref_eog/ref_emg/ref_ecg.
  - Four 32-ch artifact stems: pred_eog, pred_emg, pred_ecg, pred_basenoise.
  - Clean estimate:
      pred_eeg = mix - (pred_eog + pred_emg + pred_ecg + pred_basenoise)
  - Projection heads map 32-ch stems -> 2/2/1 hardware refs for loss only.
  - Hardware ref gain: T=256 (DEAP) -> 20x on projected refs; T=1000 (SEED) -> 1x.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

DEFAULT_KERNEL_SIZE = 5
SCALP_CHANNELS = 32
EOG_REF_CHANNELS = 2
EMG_REF_CHANNELS = 2
ECG_REF_CHANNELS = 1

DEAP_SEQ_LEN = 256
SEED_SEQ_LEN = 1000
DEAP_HARDWARE_REF_GAIN = 20.0
DEFAULT_REF_GAIN = 1.0


class MRANCOutput(NamedTuple):
    pred_eeg: torch.Tensor
    pred_eog: torch.Tensor
    pred_emg: torch.Tensor
    pred_ecg: torch.Tensor
    pred_basenoise: torch.Tensor
    pred_noise: torch.Tensor


class SamePadConv1d(nn.Module):
    """Length-preserving 1D conv (padding='same', odd kernels, stride=1)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"SamePadConv1d requires odd kernel_size, got {kernel_size}")
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    """Symmetric encoder/decoder block: same-pad conv -> BN -> GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = DEFAULT_KERNEL_SIZE) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SamePadConv1d(in_ch, out_ch, kernel_size=kernel_size),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalSpatialStemDecoder(nn.Module):
    """
    Artifact stem head: shared encoder context, then per-channel temporal convolutions.

    A plain 1x1 projection from shared feature maps forces every scalp channel to share
    the same temporal waveform (only scaled/inverted). Depthwise-separable convolutions
    (groups=out_ch) give each channel independent temporal filters.
    """

    def __init__(
        self,
        feat_ch: int,
        out_ch: int = SCALP_CHANNELS,
        temporal_kernel: int = 7,
    ) -> None:
        super().__init__()
        mid = max(feat_ch // 2, out_ch)
        self.shared = nn.Sequential(
            ConvBlock(feat_ch, feat_ch),
            ConvBlock(feat_ch, mid),
            nn.Conv1d(mid, out_ch, kernel_size=1),
        )
        self.per_channel_temporal = DepthwiseSeparableConv1d(out_ch, kernel_size=temporal_kernel)
        self.out_norm = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        h = self.per_channel_temporal(h)
        return self.out_norm(h)


# Backward-compatible alias used by MRANC and stacking docs.
StemDecoder = TemporalSpatialStemDecoder


class ReferenceProjectionHead(nn.Module):
    """32-ch stem -> hardware ref channels (loss evaluation only)."""

    def __init__(self, in_ch: int = SCALP_CHANNELS, out_ch: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, stem_32: torch.Tensor) -> torch.Tensor:
        return self.proj(stem_32)


class DepthwiseSeparableConv1d(nn.Module):
    """1D depthwise separable convolution (depthwise groups=channels, then pointwise 1x1)."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"DepthwiseSeparableConv1d requires odd kernel_size, got {kernel_size}")
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale temporal adapter with squeeze-and-excitation recalibration.

    Three parallel depthwise-separable branches (kernels 5, 11, 21) fuse into an SE
    gate. The final projection is zero-initialized so output = input before training.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = channels
        self.latest_attention_weights: torch.Tensor | None = None

        self.branch5 = DepthwiseSeparableConv1d(channels, kernel_size=5)
        self.branch11 = DepthwiseSeparableConv1d(channels, kernel_size=11)
        self.branch21 = DepthwiseSeparableConv1d(channels, kernel_size=21)
        self.fuse = nn.Conv1d(channels, channels, kernel_size=1)

        hidden = max(channels // reduction, 4)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _temporal_attention_map(self, gated: torch.Tensor, excitation: torch.Tensor) -> torch.Tensor:
        """Per-sample normalized temporal saliency in [0, 1], shape (B, 1, T)."""
        weighted = gated * excitation
        tmap = weighted.abs().mean(dim=1, keepdim=True)
        t_min = tmap.amin(dim=-1, keepdim=True)
        t_max = tmap.amax(dim=-1, keepdim=True)
        denom = (t_max - t_min).clamp_min(1e-8)
        return (tmap - t_min) / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch5(x)
        b2 = self.branch11(x)
        b3 = self.branch21(x)
        fused = self.fuse(b1 + b2 + b3)

        pooled = self.se_pool(fused)
        excitation = self.se_mlp(pooled)
        gated = fused * excitation
        self.latest_attention_weights = self._temporal_attention_map(gated, excitation)

        adapter_delta = self.out_proj(gated)
        return x + adapter_delta


class MRANC(nn.Module):
    """
    Multi-Reference Adaptive Noise Canceller.

    forward(mix) accepts only raw noisy scalp EEG (B, 32, T).
    Returns dict with pred_eeg, four artifact stems, and pred_noise (= pred_basenoise).
    """

    def __init__(self, in_channels: int = SCALP_CHANNELS, feature_channels: int = 128) -> None:
        super().__init__()
        if in_channels != SCALP_CHANNELS:
            raise ValueError(f"MRANC expects in_channels={SCALP_CHANNELS}, got {in_channels}")

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, feature_channels),
            ConvBlock(feature_channels, feature_channels),
            ConvBlock(feature_channels, feature_channels),
        )

        self.ms_attention = MultiScaleAttentionBlock(feature_channels)

        self.eog_head = StemDecoder(feature_channels, SCALP_CHANNELS)
        self.emg_head = StemDecoder(feature_channels, SCALP_CHANNELS)
        self.ecg_head = StemDecoder(feature_channels, SCALP_CHANNELS)
        self.basenoise_head = StemDecoder(feature_channels, SCALP_CHANNELS)

        self.project_eog = ReferenceProjectionHead(SCALP_CHANNELS, EOG_REF_CHANNELS)
        self.project_emg = ReferenceProjectionHead(SCALP_CHANNELS, EMG_REF_CHANNELS)
        self.project_ecg = ReferenceProjectionHead(SCALP_CHANNELS, ECG_REF_CHANNELS)

    @staticmethod
    def sum_artifact_stems(
        pred_eog: torch.Tensor,
        pred_emg: torch.Tensor,
        pred_ecg: torch.Tensor,
        pred_basenoise: torch.Tensor,
    ) -> torch.Tensor:
        return pred_eog + pred_emg + pred_ecg + pred_basenoise

    @staticmethod
    def isolate_eeg(
        mix: torch.Tensor,
        pred_eog: torch.Tensor,
        pred_emg: torch.Tensor,
        pred_ecg: torch.Tensor,
        pred_basenoise: torch.Tensor,
    ) -> torch.Tensor:
        """pred_eeg = mix - sum(artifact stems)."""
        return mix - MRANC.sum_artifact_stems(pred_eog, pred_emg, pred_ecg, pred_basenoise)

    @staticmethod
    def hardware_ref_gain(
        seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Scalar gain for projected hardware refs (DEAP T=256 only). No extra state_dict keys."""
        gain = DEAP_HARDWARE_REF_GAIN if seq_len == DEAP_SEQ_LEN else DEFAULT_REF_GAIN
        return torch.tensor(gain, device=device, dtype=dtype)

    def project_hardware_refs(
        self,
        pred_eog: torch.Tensor,
        pred_emg: torch.Tensor,
        pred_ecg: torch.Tensor,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 32-ch stems to hardware ref dims with dataset-specific gain."""
        t_len = seq_len if seq_len is not None else pred_eog.shape[-1]
        gain = self.hardware_ref_gain(t_len, device=pred_eog.device, dtype=pred_eog.dtype)
        return (
            self.project_eog(pred_eog) * gain,
            self.project_emg(pred_emg) * gain,
            self.project_ecg(pred_ecg) * gain,
        )

    def forward(self, mix: torch.Tensor) -> dict[str, torch.Tensor]:
        if mix.dim() != 3 or mix.shape[1] != SCALP_CHANNELS:
            raise ValueError(f"mix must be (B, {SCALP_CHANNELS}, T), got {tuple(mix.shape)}")

        features = self.encoder(mix)
        features = self.ms_attention(features)

        pred_eog = self.eog_head(features)
        pred_emg = self.emg_head(features)
        pred_ecg = self.ecg_head(features)
        pred_basenoise = self.basenoise_head(features)

        pred_eeg = self.isolate_eeg(mix, pred_eog, pred_emg, pred_ecg, pred_basenoise)
        pred_noise = pred_basenoise

        seq_len = mix.shape[-1]
        gain = self.hardware_ref_gain(seq_len, device=mix.device, dtype=pred_eog.dtype)
        proj_eog = self.project_eog(pred_eog) * gain
        proj_emg = self.project_emg(pred_emg) * gain
        proj_ecg = self.project_ecg(pred_ecg) * gain

        return {
            "pred_eeg": pred_eeg,
            "pred_eog": pred_eog,
            "pred_emg": pred_emg,
            "pred_ecg": pred_ecg,
            "pred_basenoise": pred_basenoise,
            "pred_noise": pred_noise,
            "proj_eog": proj_eog,
            "proj_emg": proj_emg,
            "proj_ecg": proj_ecg,
        }

    def forward_tuple(self, mix: torch.Tensor) -> MRANCOutput:
        out = self.forward(mix)
        return MRANCOutput(
            pred_eeg=out["pred_eeg"],
            pred_eog=out["pred_eog"],
            pred_emg=out["pred_emg"],
            pred_ecg=out["pred_ecg"],
            pred_basenoise=out["pred_basenoise"],
            pred_noise=out["pred_noise"],
        )
