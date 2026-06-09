"""
Shared helpers for MRANC critical-figures and real-world evaluation scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split

from clinical_sanitize import sanitize_clinical_batch
from model import MRANC, SCALP_CHANNELS
from paths import MAIN_CHECKPOINT, PROJECT_ROOT, resolve_processed_dir

DATASETS = ("deap", "seed", "clinical", "artifact_benchmark")
ARTIFACT_KEYS = ("pred_eog", "pred_emg", "pred_ecg", "pred_basenoise")
DEAP_32_FALLBACK = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz", "P3",
    "PO3", "O1", "Oz", "O2", "PO4", "P4", "CP2", "CP6", "T8", "C8", "C4", "FC2",
    "FC6", "F4", "AF4", "Fp2", "Fz", "Cz", "CPz", "POz",
]
DEAP_TO_UV_SCALE = 1e6
DEFAULT_DPI = 300

STEM_SPECS = (
    ("pred_eog", "EOG Stem (ocular / blink)", "#9467bd"),
    ("pred_emg", "EMG Stem (muscle bursts)", "#ff7f0e"),
    ("pred_ecg", "ECG Stem (cardiovascular pulse)", "#d62728"),
    ("pred_basenoise", "Base Environmental Noise Stem", "#8c564b"),
)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(choice)


def resolve_checkpoint(path_arg: str | None) -> Path:
    if path_arg:
        p = Path(path_arg)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return MAIN_CHECKPOINT


def resolve_data_dir(dataset: str, data_dir: str | None) -> Path:
    if data_dir:
        p = Path(data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return resolve_processed_dir(dataset)


def load_deap_channel_names(data_dir: Path) -> list[str]:
    map_path = data_dir / "channel_map.json"
    if map_path.is_file():
        data = json.loads(map_path.read_text(encoding="utf-8"))
        names = data.get("deap_channels")
        if names:
            return list(names)
    return list(DEAP_32_FALLBACK)


def norm_ch(name: str) -> str:
    return name.strip().upper()


def resolve_channel_indices(names: list[str], deap_channels: list[str]) -> list[int]:
    index = {norm_ch(n): i for i, n in enumerate(deap_channels)}
    resolved: list[int] = []
    for name in names:
        key = norm_ch(name)
        if key not in index:
            raise ValueError(f"Unknown channel {name!r}")
        resolved.append(index[key])
    return resolved


def safe_channel_slug(name: str) -> str:
    slug = "".join(c for c in name if c.isalnum())
    return slug or "ch"


def val_indices(n_total: int, val_fraction: float, split_seed: int) -> list[int]:
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    placeholder = TensorDataset(torch.zeros(n_total))
    _, val_ds = random_split(
        placeholder,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(split_seed),
    )
    return list(val_ds.indices)


def parse_window_indices(spec: str, n_val: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(n_val))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= n_val:
            raise ValueError(f"Window index {idx} out of val range [0, {n_val})")
        out.append(idx)
    if not out:
        raise ValueError("No window indices parsed")
    return out


def load_mix_window(data_dir: Path, global_idx: int) -> np.ndarray:
    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        raise FileNotFoundError(f"Missing {mix_path}")
    mix = np.array(np.load(mix_path, mmap_mode="r")[global_idx], dtype=np.float32)
    if mix.ndim != 2 or mix.shape[0] != SCALP_CHANNELS:
        raise ValueError(f"Expected mix shape (32, T), got {mix.shape}")
    return mix


def unit_scale_for_dataset(dataset: str) -> float:
    return DEAP_TO_UV_SCALE if dataset == "deap" else 1.0


def zero_mean_per_channel(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def pearson_per_channel(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).mean(axis=-1)
    den = a.std(axis=-1) * b.std(axis=-1) + eps
    return num / den


def align_artifacts_for_subtraction(
    stems: dict[str, np.ndarray],
    mix: np.ndarray,
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for key in ARTIFACT_KEYS:
        centered = zero_mean_per_channel(stems[key])
        r = pearson_per_channel(centered, mix)
        flip = r < 0.0
        centered = centered.copy()
        centered[flip] *= -1.0
        aligned[key] = centered
    aligned["pred_eeg"] = mix - sum(aligned[k] for k in ARTIFACT_KEYS)
    return aligned


def total_artifact(stems: dict[str, np.ndarray]) -> np.ndarray:
    return sum(stems[k] for k in ARTIFACT_KEYS)


def max_recon_error(mix: np.ndarray, stems: dict[str, np.ndarray]) -> float:
    recon = stems["pred_eeg"] + total_artifact(stems)
    return float(np.max(np.abs(mix - recon)))


def load_model(checkpoint: Path, device: torch.device) -> MRANC:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = MRANC(feature_channels=int(config.get("feature_channels", 128))).to(device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def upsample_attention(attn: torch.Tensor, target_len: int) -> torch.Tensor:
    if attn.shape[-1] == target_len:
        return attn
    if attn.dim() == 2:
        attn = attn.unsqueeze(1)
    return F.interpolate(attn.float(), size=target_len, mode="linear", align_corners=False)


@torch.inference_mode()
def run_window_inference(
    model: MRANC,
    mix: np.ndarray,
    device: torch.device,
    dataset: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    x = torch.from_numpy(mix[np.newaxis].astype(np.float32)).to(device)
    if dataset == "clinical":
        x = sanitize_clinical_batch(x)
        mix_used = x[0].detach().cpu().numpy()
    else:
        mix_used = mix.copy()

    out = model(x)
    raw_stems = {k: out[k][0].detach().cpu().numpy() for k in ("pred_eeg", *ARTIFACT_KEYS)}

    if dataset == "deap":
        stems = dict(raw_stems)
    else:
        stems = align_artifacts_for_subtraction(
            {k: raw_stems[k] for k in ARTIFACT_KEYS},
            mix_used,
        )

    attn = model.ms_attention.latest_attention_weights
    if attn is None:
        raise RuntimeError("latest_attention_weights was not set during forward pass")
    attn_np = upsample_attention(attn, x.shape[-1])[0, 0].detach().cpu().numpy()

    scale = unit_scale_for_dataset(dataset)
    if scale != 1.0:
        mix_used = mix_used * scale
        stems = {k: v * scale for k, v in stems.items()}

    return stems, attn_np, mix_used


def overlay_panel_title(dataset: str) -> str:
    if dataset == "clinical":
        return "Clinical Overlay (Raw vs MRANC Denoised)"
    return "Input Overlay (Raw vs MRANC Denoised)"


def dataset_display_name(dataset: str) -> str:
    names = {
        "deap": "DEAP",
        "seed": "SEED",
        "clinical": "Clinical",
        "artifact_benchmark": "Artifact Benchmark",
    }
    return names.get(dataset, dataset)


def output_file_prefix(dataset: str) -> str:
    """Filesystem prefix for figure outputs (clinical real-world runs use tuh)."""
    return "tuh" if dataset == "clinical" else dataset


def setup_publication_style(dpi: int = DEFAULT_DPI) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )


def shared_ylim_pair(a: np.ndarray, b: np.ndarray, pad_ratio: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([a.ravel(), b.ravel()])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    span = hi - lo
    if span <= 1e-12:
        base = max(abs(lo), 1e-3)
        return (lo - 0.1 * base, hi + 0.1 * base)
    pad = pad_ratio * span
    return (lo - pad, hi + pad)


def plot_channel_decomposition_ultrawide(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    attn: np.ndarray,
    channel_idx: int,
    channel_name: str,
    dataset: str,
    global_idx: int,
    val_slot: int,
    cmap: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
) -> None:
    """One 16x12 inch figure: overlay, four artifact stems, attention heatmap (sharex=True)."""
    t = np.arange(mix.shape[-1])
    raw_ch = mix[channel_idx]
    clean_ch = stems["pred_eeg"][channel_idx]

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True)

    ax0 = axes[0]
    ylim = shared_ylim_pair(raw_ch, clean_ch)
    if dataset == "clinical":
        raw_label = "Raw Clinical Input"
        clean_label = "MRANC Denoised Output"
    else:
        raw_label = "Raw Input"
        clean_label = "MRANC Denoised Output"
    ax0.plot(t, raw_ch, color="#5B7C99", linewidth=1.0, alpha=0.45, label=raw_label)
    ax0.plot(t, clean_ch, color="#2ca02c", linewidth=1.4, label=clean_label)
    ax0.set_ylim(ylim)
    ax0.set_ylabel("Amplitude (uV)")
    ax0.set_title(overlay_panel_title(dataset))
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.25)

    for ax, (key, title, color) in zip(axes[1:5], STEM_SPECS):
        ax.plot(t, stems[key][channel_idx], color=color, linewidth=1.1)
        ax.set_ylabel("Amplitude (uV)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    ax5 = axes[5]
    attn_2d = attn[np.newaxis, :]
    im = ax5.imshow(
        attn_2d,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[0, len(t) - 1, 0, 1],
    )
    ax5.set_ylabel("Attention")
    ax5.set_xlabel("Sample index")
    ax5.set_title("Normalized Multi-Scale Attention Weights")
    cbar = fig.colorbar(im, ax=ax5, fraction=0.02, pad=0.02)
    cbar.set_label("Attention (0 to 1)")

    fig.suptitle(
        f"{dataset_display_name(dataset)} Decomposition | window {global_idx} "
        f"(val slot {val_slot}) | {channel_name} | T={len(t)}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def resolve_decomposition_output_dir(
    output_root: Path,
    dataset: str,
) -> Path:
    """Always write under output_root/{dataset}/ (no flat clinical root)."""
    out = output_root / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out


def decomposition_output_path(
    output_dir: Path,
    dataset: str,
    window_tag: int,
    channel_name: str,
) -> Path:
    prefix = output_file_prefix(dataset)
    slug = safe_channel_slug(channel_name)
    return output_dir / f"{prefix}_eval_window{window_tag}_{slug}_decomposition.png"


def save_per_channel_decomposition_figures(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    attn: np.ndarray,
    dataset: str,
    channel_names: list[str],
    channel_indices: list[int],
    channel_labels: list[str],
    global_idx: int,
    val_slot: int,
    output_dir: Path,
    *,
    filename_index: str = "val",
    cmap: str = "viridis",
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    window_tag = val_slot if filename_index == "val" else global_idx
    saved: list[Path] = []
    for ch_idx, ch_name in zip(channel_indices, channel_labels):
        out_path = decomposition_output_path(output_dir, dataset, window_tag, ch_name)
        plot_channel_decomposition_ultrawide(
            mix,
            stems,
            attn,
            ch_idx,
            ch_name,
            dataset,
            global_idx,
            val_slot,
            cmap,
            out_path,
            dpi,
        )
        saved.append(out_path)
        print(f"Saved {out_path}")
    return saved
