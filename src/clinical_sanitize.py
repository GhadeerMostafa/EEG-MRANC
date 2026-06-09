"""Clinical batch sanitization (clip + dead-channel repair)."""

from __future__ import annotations

import numpy as np
import torch


def sanitize_clinical_batch(
    data: torch.Tensor | np.ndarray,
    clip_uV: float = 500.0,
    dead_std_uV: float = 1e-3,
) -> torch.Tensor | np.ndarray:
    """
    Sanitize a clinical window/batch before it reaches the model.

    - Hard clip to +/- clip_uV
    - Dead-channel detection: std over time < dead_std_uV
    - Replacement: replace dead channels with mean of non-dead channels
    """
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        x = data
        orig_2d = x.ndim == 2
        if orig_2d:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[1] != 32:
            raise ValueError(f"sanitize_clinical_batch: expected (B,32,T) or (32,T), got {tuple(x.shape)}")

        with torch.no_grad():
            x = x.clamp(-clip_uV, clip_uV)
            if torch.isnan(x).any():
                raise RuntimeError("sanitize_clinical_batch: NaNs detected after clipping")

            stds = x.std(dim=-1)
            dead = stds < dead_std_uV
            keep = (~dead).to(x.dtype)
            counts_safe = torch.clamp(keep.sum(dim=1, keepdim=True), min=1.0)
            sum_keep = (x * keep.unsqueeze(-1)).sum(dim=1)
            mean_keep = sum_keep / counts_safe
            mean_keep_expanded = mean_keep.unsqueeze(1).expand(-1, 32, -1)
            x = torch.where(dead.unsqueeze(-1), mean_keep_expanded, x)

            if x.abs().max() > clip_uV * 1.00001:
                raise RuntimeError("sanitize_clinical_batch: amplitude exceeds clip_uV after sanitization")

        return x.squeeze(0) if orig_2d else x

    x_np = np.asarray(data)
    orig_2d = x_np.ndim == 2
    if orig_2d:
        x_np = x_np[None, ...]
    if x_np.ndim != 3 or x_np.shape[1] != 32:
        raise ValueError(f"sanitize_clinical_batch: expected (B,32,T) or (32,T), got {tuple(x_np.shape)}")

    x_np = np.clip(x_np, -clip_uV, clip_uV).astype(x_np.dtype, copy=False)
    if np.isnan(x_np).any():
        raise RuntimeError("sanitize_clinical_batch: NaNs detected after clipping")

    stds = x_np.std(axis=-1)
    dead = stds < dead_std_uV
    keep = (~dead).astype(x_np.dtype)
    counts_safe = np.maximum(keep.sum(axis=1, keepdims=True), 1.0)
    sum_keep = (x_np * keep[..., None]).sum(axis=1)
    mean_keep = sum_keep / counts_safe
    mean_keep_expanded = mean_keep[:, None, :]
    x_np = np.where(dead[..., None], mean_keep_expanded, x_np)

    if np.max(np.abs(x_np)) > clip_uV * 1.00001:
        raise RuntimeError("sanitize_clinical_batch: amplitude exceeds clip_uV after sanitization")

    return x_np[0] if orig_2d else x_np
