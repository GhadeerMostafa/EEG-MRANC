"""
1D FCN autoencoder baseline (EEGdenoiseNet-style) without physics lock or attention adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SCALP_CHANNELS = 32


class EEGDenoiseNetFCN(nn.Module):
    """Encoder-decoder 1D CNN mapping noisy (B, 32, T) to denoised (B, 32, T)."""

    def __init__(self, in_channels: int = SCALP_CHANNELS) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1]:
            out = nn.functional.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        return out
