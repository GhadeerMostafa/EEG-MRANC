"""
MRANC artifact subtraction report.

--dataset seed (default): processed_data_seed, 32-ch spatial map, 3-panel layout (no hardware refs).
--dataset deap: processed_data, 4-panel layout with pred vs hardware ref_eog/emg/ecg.
--dataset clinical: processed_clinical_data, SEED-style 3-panel layout (no hardware refs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from runtime import setup_src_path
from scipy.signal import detrend
from torch.utils.data import TensorDataset, random_split

setup_src_path()

from figure_common import (
    load_deap_channel_names,
    load_mix_window,
    load_model,
    max_recon_error,
    resolve_channel_indices,
    resolve_checkpoint,
    resolve_decomposition_output_dir,
    resolve_device as resolve_device_common,
    run_window_inference,
    save_per_channel_decomposition_figures,
    setup_publication_style,
    val_indices,
)
from model import MRANC
from paths import (
    DATASET_DIRS,
    CRITICAL_FIGURES_DIR,
    FIGURES_DATASET_DIRS,
    MAIN_CHECKPOINT,
    PROJECT_ROOT,
)

REF_FILES = {
    "ref_eog": "ref_eog.npy",
    "ref_emg": "ref_emg.npy",
    "ref_ecg": "ref_ecg.npy",
}
REF_KEYS = ("ref_eog", "ref_emg", "ref_ecg")
PROJ_KEYS = ("pred_eog", "pred_emg", "pred_ecg")

DEAP_32_FALLBACK = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz", "P3",
    "PO3", "O1", "Oz", "O2", "PO4", "P4", "CP2", "CP6", "T8", "C8", "C4", "FC2",
    "FC6", "F4", "AF4", "Fp2", "Fz", "Cz", "CPz", "POz",
]

ARTIFACT_KEYS = ("pred_eog", "pred_emg", "pred_ecg", "pred_basenoise")
DEAP_TO_UV_SCALE = 1e6

REF_EOG_CH = 0
REF_EMG_CH = 0
REF_ECG_CH = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRANC visualization (SEED, DEAP, or clinical)")
    p.add_argument(
        "--dataset",
        choices=("seed", "deap", "clinical", "artifact_benchmark"),
        default="seed",
        help=(
            "Dataset preset: "
            "seed -> processed_data_seed, "
            "deap -> processed_data, "
            "clinical -> processed_clinical_data, "
            "artifact_benchmark -> data/artifact_benchmark/processed"
        ),
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override dataset folder (default from --dataset)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: checkpoints/best_mranc_artifact_benchmark_weights.pth)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: outputs/figures/<dataset>/artifact_subtraction_report.png)",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Window index within validation set (default) or full dataset (--no-val-split)",
    )
    p.add_argument(
        "--highlight-channels",
        type=str,
        default="Fp1,Cz,O1",
        help="Comma-separated DEAP scalp channel names to highlight",
    )
    p.add_argument(
        "--no-val-split",
        action="store_true",
        help="Load --window-index from full mix.npy instead of the validation subset",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device (default: cuda if available)",
    )
    p.add_argument(
        "--skip-critical-figures",
        action="store_true",
        help="Skip ultra-wide per-channel decomposition PNGs in critical_figures/",
    )
    p.add_argument(
        "--critical-figures-dir",
        type=str,
        default=str(CRITICAL_FIGURES_DIR),
        help="Output folder for per-channel 6-row decomposition PNGs",
    )
    p.add_argument(
        "--legacy-spatial-rms",
        action="store_true",
        help="Also write legacy spatial-RMS report under outputs/figures/",
    )
    p.add_argument("--cmap", type=str, default="viridis", choices=("viridis", "plasma"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(choice)


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    return DATASET_DIRS[args.dataset]


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
            raise ValueError(f"Unknown channel {name!r}; expected one of {deap_channels}")
        resolved.append(index[key])
    return resolved


def channel_color_map(highlight_idx: list[int]) -> dict[int, tuple[float, float, float, float]]:
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(highlight_idx), 1)))
    return {ch: colors[i] for i, ch in enumerate(highlight_idx)}


def diversity_sample_t(mix: np.ndarray) -> int:
    return min(500, mix.shape[1] - 1)


def dataset_length(data_dir: Path) -> int:
    mix_path = data_dir / "mix.npy"
    if not mix_path.exists():
        raise FileNotFoundError(f"Missing {mix_path}")
    return int(np.load(mix_path, mmap_mode="r").shape[0])


def val_indices(
    n_total: int,
    val_fraction: float,
    split_seed: int,
) -> list[int]:
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    placeholder = TensorDataset(torch.zeros(n_total))
    _, val_ds = random_split(
        placeholder,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(split_seed),
    )
    return list(val_ds.indices)


def select_window(
    data_dir: Path,
    window_index: int,
    use_val_split: bool,
    val_fraction: float,
    split_seed: int,
) -> tuple[int, int | None, np.ndarray]:
    """Return (global_index, val_index_or_None, mix_32xT)."""
    n_total = dataset_length(data_dir)
    if use_val_split:
        val_idx = val_indices(n_total, val_fraction, split_seed)
        if window_index < 0 or window_index >= len(val_idx):
            raise ValueError(
                f"--window-index {window_index} out of val range [0, {len(val_idx)})"
            )
        global_idx = val_idx[window_index]
        val_pos = window_index
    else:
        if window_index < 0 or window_index >= n_total:
            raise ValueError(f"--window-index {window_index} out of range [0, {n_total})")
        global_idx = window_index
        val_pos = None

    mix = np.array(np.load(data_dir / "mix.npy", mmap_mode="r")[global_idx], dtype=np.float32)
    if mix.ndim != 2 or mix.shape[0] != 32:
        raise ValueError(f"Expected mix shape (32, T), got {mix.shape}")
    return global_idx, val_pos, mix


def load_refs_window(data_dir: Path, global_idx: int) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for key, fname in REF_FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"DEAP mode requires {path}")
        mmap = np.load(path, mmap_mode="r")
        if global_idx < 0 or global_idx >= mmap.shape[0]:
            raise ValueError(f"{fname}: index {global_idx} out of range [0, {mmap.shape[0]})")
        refs[key] = np.array(mmap[global_idx], dtype=np.float32)
    return refs


def spatial_mean(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0)


def spatial_rms_trace(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Spatial RMS over channel dimension.

    - (B, C, T) -> (B, T) using dim/axis=1
    - (C, T) -> (T,) using dim/axis=0
    """
    if isinstance(x, torch.Tensor):
        if x.ndim == 3:
            return torch.sqrt(torch.mean(x**2, dim=1))
        if x.ndim == 2:
            return torch.sqrt(torch.mean(x**2, dim=0))
        raise ValueError(f"spatial_rms_trace expected 2D/3D tensor, got {tuple(x.shape)}")

    x_np = np.asarray(x)
    if x_np.ndim == 3:
        return np.sqrt(np.mean(x_np**2, axis=1))
    if x_np.ndim == 2:
        return np.sqrt(np.mean(x_np**2, axis=0))
    raise ValueError(f"spatial_rms_trace expected 2D/3D array, got {x_np.shape}")


def unit_scale_for_dataset(dataset: str) -> float:
    # DEAP is commonly stored in Volts; display all DEAP traces in microvolts.
    return DEAP_TO_UV_SCALE if dataset == "deap" else 1.0


def apply_unit_scale_to_stems(stems: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    if scale == 1.0:
        return stems
    return {k: (v * scale) for k, v in stems.items()}


def shared_ylim_with_padding(*traces: np.ndarray, pad_ratio: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(t).ravel() for t in traces])
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-1.0, 1.0)
    span = hi - lo
    if span <= 1e-12:
        # near-constant trace: create a small symmetric window around the value
        base = max(abs(lo), 1e-3)
        return (lo - 0.1 * base, hi + 0.1 * base)
    pad = pad_ratio * span
    return (lo - pad, hi + pad)


def load_checkpoint(path: Path, device: torch.device) -> tuple[MRANC, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = MRANC(feature_channels=int(config.get("feature_channels", 128))).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


@torch.inference_mode()
def run_inference(model: MRANC, mix_window: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    x = torch.from_numpy(mix_window[np.newaxis]).to(device)
    out = model(x)
    return {k: out[k][0].detach().cpu().numpy() for k in ("pred_eeg", *ARTIFACT_KEYS)}


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


def verify_aligned_subtraction(mix: np.ndarray, stems: dict[str, np.ndarray], tol: float = 1e-5) -> float:
    recomputed = mix - total_artifact(stems)
    err = float(np.max(np.abs(stems["pred_eeg"] - recomputed)))
    if err > tol:
        raise RuntimeError(f"Aligned subtraction check failed: max error = {err:.6e}")
    return err


def preprocess_reference_channels(refs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in REF_KEYS:
        x = detrend(refs[key].astype(np.float64), axis=-1, type="linear")
        x = x - x.mean(axis=-1, keepdims=True)
        out[key] = x.astype(np.float32)
    print("Preprocessed references: scipy detrend (linear) + zero-mean per channel")
    return out


@torch.inference_mode()
def project_stems(model: MRANC, stems: dict[str, np.ndarray], device: torch.device) -> dict[str, np.ndarray]:
    pred_eog = torch.from_numpy(stems["pred_eog"]).unsqueeze(0).to(device)
    pred_emg = torch.from_numpy(stems["pred_emg"]).unsqueeze(0).to(device)
    pred_ecg = torch.from_numpy(stems["pred_ecg"]).unsqueeze(0).to(device)
    proj_eog, proj_emg, proj_ecg = model.project_hardware_refs(
        pred_eog,
        pred_emg,
        pred_ecg,
        seq_len=stems["pred_eog"].shape[-1],
    )
    return {
        "pred_eog": proj_eog[0].cpu().numpy(),
        "pred_emg": proj_emg[0].cpu().numpy(),
        "pred_ecg": proj_ecg[0].cpu().numpy(),
    }


def fix_polarity_vs_ref(pred: np.ndarray, ref: np.ndarray, name: str) -> np.ndarray:
    out = pred.copy()
    r = pearson_per_channel(out, ref)
    flip = r < 0.0
    out[flip] *= -1.0
    n_flip = int(flip.sum())
    if n_flip:
        print(f"  {name}: flipped {n_flip}/{pred.shape[0]} projected channel(s) vs ref")
    return out


def align_projected_vs_ref(
    proj: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    print("Aligning projected preds (zero-mean + polarity vs ref)...")
    aligned: dict[str, np.ndarray] = {}
    for key, ref_key in zip(PROJ_KEYS, REF_KEYS):
        centered = zero_mean_per_channel(proj[key])
        aligned[key] = fix_polarity_vs_ref(centered, refs[ref_key], key)
    return aligned


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def window_description(global_idx: int, val_pos: int | None) -> str:
    if val_pos is not None:
        return f"val window {val_pos} (global {global_idx})"
    return f"global window {global_idx}"


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")
    backend = plt.get_backend().lower()
    if backend != "agg":
        plt.show()
    plt.close(fig)


def plot_raw_mix_panel(
    ax: plt.Axes,
    mix: np.ndarray,
    highlight_idx: list[int],
    channel_labels: list[str],
    colors: dict[int, tuple[float, float, float, float]],
) -> None:
    t = np.arange(mix.shape[1])
    ax.plot(t, spatial_mean(mix), color="black", linewidth=2.0, label="Spatial mean", zorder=5)

    for ch_idx in highlight_idx:
        name = channel_labels[ch_idx]
        ax.plot(
            t,
            mix[ch_idx],
            color=colors[ch_idx],
            linewidth=1.1,
            alpha=0.9,
            label=name,
            zorder=6,
        )

    ax.set_title("Panel 1: Raw mixed EEG (32-ch spatial map)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_highlight_artifact_panel(
    ax: plt.Axes,
    art: np.ndarray,
    highlight_idx: list[int],
    channel_labels: list[str],
    colors: dict[int, tuple[float, float, float, float]],
) -> None:
    t = np.arange(art.shape[1])
    ax.plot(
        t,
        spatial_mean(art),
        color="0.55",
        linewidth=1.0,
        alpha=0.25,
        label="Spatial mean (artifact)",
        zorder=3,
    )

    for ch_idx in highlight_idx:
        name = channel_labels[ch_idx]
        ax.plot(
            t,
            art[ch_idx],
            color=colors[ch_idx],
            linewidth=1.1,
            alpha=0.9,
            label=name,
            zorder=6,
        )

    ax.set_title("Panel 2: Isolated artifact predictions (highlight channels)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_decomposition_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    global_idx: int,
    val_pos: int | None,
    epoch: object,
    val_loss: float,
    output_path: Path,
    *,
    title_prefix: str,
) -> None:
    """
    5-row scientific decomposition layout (spatial-RMS focused):
      1) Raw input spatial RMS
      2) Cleaned EEG spatial RMS
      3) Raw vs cleaned spatial RMS overlay
      4) Artifact total spatial mean
      5) Raw vs reconstructed spatial RMS overlay (+ MSE title)
    """
    clean = stems["pred_eeg"]
    artifacts = total_artifact(stems)
    reconstructed = clean + artifacts

    if mix.ndim != 2:
        raise ValueError(f"Expected mix to have shape (C,T), got {mix.shape}")
    if clean.shape != mix.shape or artifacts.shape != mix.shape or reconstructed.shape != mix.shape:
        raise ValueError(
            f"Shape mismatch: mix={mix.shape}, clean={clean.shape}, artifacts={artifacts.shape}, "
            f"reconstructed={reconstructed.shape}"
        )

    raw_spatial = np.asarray(spatial_rms_trace(mix), dtype=np.float32)
    clean_spatial = np.asarray(spatial_rms_trace(clean), dtype=np.float32)
    artifact_spatial_rms = np.asarray(spatial_rms_trace(artifacts), dtype=np.float32)
    reconstructed_spatial = np.asarray(spatial_rms_trace(reconstructed), dtype=np.float32)
    mse_spatial = float(np.mean((raw_spatial - reconstructed_spatial) ** 2))
    shared_ylim = shared_ylim_with_padding(raw_spatial, clean_spatial)

    t_len = mix.shape[1]
    t = np.arange(t_len)
    palette = {
        "raw": "#1f77b4",
        "clean": "#2ca02c",
        "artifact": "#ff7f0e",
        "reconstructed": "#9467bd",
    }

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    fig.suptitle(
        f"{title_prefix} | {window_description(global_idx, val_pos)} | "
        f"epoch={epoch} val_loss={val_loss:.4f} | T={t_len}",
        fontsize=11,
    )

    axes[0].plot(t, raw_spatial, color=palette["raw"], linewidth=1.8, label="Raw spatial RMS")
    axes[0].set_title("Row 1: Raw Input Signal (Spatial RMS across channels)")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].set_ylim(*shared_ylim)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, clean_spatial, color=palette["clean"], linewidth=1.8, label="Clean spatial RMS")
    axes[1].set_title("Row 2: Cleaned EEG (Spatial RMS across channels)")
    axes[1].set_ylabel("Amplitude (uV)")
    axes[1].set_ylim(*shared_ylim)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, raw_spatial, color=palette["raw"], linewidth=1.6, label="Raw spatial RMS")
    axes[2].plot(t, clean_spatial, color=palette["clean"], linewidth=1.6, label="Clean spatial RMS")
    axes[2].set_title("Row 3: Global Spatial RMS Comparison (Raw vs Cleaned)")
    axes[2].set_ylabel("Amplitude (uV)")
    axes[2].set_ylim(*shared_ylim)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(
        t,
        artifact_spatial_rms,
        color=palette["artifact"],
        linewidth=1.8,
        label="Artifact spatial RMS",
    )
    axes[3].set_title("Row 4: Artifacts (Total Sum, Spatial RMS across channels)")
    axes[3].set_ylabel("Amplitude (uV)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(t, raw_spatial, color=palette["raw"], linewidth=1.6, label="Raw spatial RMS")
    axes[4].plot(
        t,
        reconstructed_spatial,
        color=palette["reconstructed"],
        linewidth=1.6,
        label="Reconstructed spatial RMS",
    )
    axes[4].set_title(
        "Row 5: Reconstruction Verification "
        f"(MSE spatial RMS={mse_spatial:.6e})"
    )
    axes[4].set_ylabel("Amplitude (uV)")
    axes[4].set_xlabel("Sample index")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)


def with_output_suffix(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def with_checkpoint_label(output_path: Path, checkpoint_path: Path) -> Path:
    """
    Add a checkpoint-derived label to output filename for traceability.
    Example:
      artifact_subtraction_report.png ->
      artifact_subtraction_report__best_mranc_artifact_benchmark_weights.png
    """
    ckpt_label = checkpoint_path.stem.replace(" ", "_")
    return output_path.with_name(f"{output_path.stem}__{ckpt_label}{output_path.suffix}")


def plot_highlight_channels_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    channel_labels: list[str],
    output_path: Path,
    *,
    channel_names: tuple[str, ...] = ("Fp1", "Cz", "O1"),
) -> None:
    artifacts = total_artifact(stems)
    clean = stems["pred_eeg"]
    reconstructed = clean + artifacts
    t = np.arange(mix.shape[1])

    selected: list[tuple[str, int]] = []
    label_map = {norm_ch(name): i for i, name in enumerate(channel_labels)}
    for name in channel_names:
        idx = label_map.get(norm_ch(name))
        if idx is not None:
            selected.append((name, idx))
    if not selected:
        print("Skipped channel trace figure: none of Fp1/Cz/O1 available in channel map.")
        return

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.6 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    palette = {
        "raw": "#1f77b4",
        "clean": "#2ca02c",
        "artifact": "#ff7f0e",
        "reconstructed": "#9467bd",
    }
    for ax, (name, idx) in zip(axes, selected):
        ax.plot(t, mix[idx], color=palette["raw"], linewidth=1.2, label="Raw")
        ax.plot(t, clean[idx], color=palette["clean"], linewidth=1.2, label="Clean")
        ax.plot(t, artifacts[idx], color=palette["artifact"], linewidth=1.0, label="Artifact sum")
        ax.plot(t, reconstructed[idx], color=palette["reconstructed"], linewidth=1.0, label="Reconstructed")
        ax.set_title(f"Channel Trace: {name}")
        ax.set_ylabel("Amplitude (uV)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=4)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Highlighted Channel Traces (Fp1, Cz, O1)", fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_recovered_eeg_panel(
    ax: plt.Axes,
    pred_eeg: np.ndarray,
    highlight_idx: list[int],
    channel_labels: list[str],
    colors: dict[int, tuple[float, float, float, float]],
    *,
    panel_title: str = "Panel 3: Recovered clean EEG (32-ch, mix − aligned stems)",
    set_xlabel: bool = True,
    deap_hard_ylim: tuple[float, float] | None = None,
) -> None:
    t = np.arange(pred_eeg.shape[1])
    n_ch = pred_eeg.shape[0]

    for ch in range(n_ch):
        if ch in colors:
            continue
        ax.plot(t, pred_eeg[ch], color="0.65", linewidth=0.5, alpha=0.12, zorder=1)

    ax.plot(
        t,
        spatial_mean(pred_eeg),
        color="black",
        linewidth=2.0,
        label="Spatial mean (clean)",
        zorder=5,
    )

    for ch_idx in highlight_idx:
        name = channel_labels[ch_idx]
        ax.plot(
            t,
            pred_eeg[ch_idx],
            color=colors[ch_idx],
            linewidth=1.1,
            alpha=0.95,
            label=name,
            zorder=6,
        )

    ax.set_title(panel_title)
    if set_xlabel:
        ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    if deap_hard_ylim is not None:
        # DEAP sometimes has single-channel drifts that dominate autoscaling.
        # Clamp to a stable microvolt window so local oscillations stay visible.
        ax.set_ylim(*deap_hard_ylim)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_pred_vs_ref_panel(
    ax: plt.Axes,
    proj_aligned: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
) -> None:
    t = np.arange(refs["ref_eog"].shape[1])
    pairs = (
        ("pred_eog", "ref_eog", "C0", REF_EOG_CH),
        ("pred_emg", "ref_emg", "C1", REF_EMG_CH),
        ("pred_ecg", "ref_ecg", "C2", REF_ECG_CH),
    )
    for pred_key, ref_key, color, ch in pairs:
        ax.plot(
            t,
            proj_aligned[pred_key][ch],
            color=color,
            linewidth=0.9,
            label=f"pred_{pred_key[5:]}",
            alpha=0.95,
        )
        ax.plot(
            t,
            refs[ref_key][ch],
            color=color,
            linewidth=0.9,
            linestyle="--",
            label=f"ref_{ref_key[4:]}",
            alpha=0.55,
        )

    ax.set_title("Panel 2: Predicted vs reference artifacts")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_residual_panel(
    ax: plt.Axes,
    proj_aligned: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
) -> None:
    t = np.arange(refs["ref_eog"].shape[1])
    pairs = (
        ("pred_eog", "ref_eog", "C0", REF_EOG_CH, "residual_eog"),
        ("pred_emg", "ref_emg", "C1", REF_EMG_CH, "residual_emg"),
        ("pred_ecg", "ref_ecg", "C2", REF_ECG_CH, "residual_ecg"),
    )
    for pred_key, ref_key, color, ch, label in pairs:
        ax.plot(
            t,
            refs[ref_key][ch] - proj_aligned[pred_key][ch],
            color=color,
            linewidth=0.8,
            label=label,
            alpha=0.9,
        )

    ax.set_title("Panel 3: Isolation residual (ref − pred)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_seed_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    highlight_idx: list[int],
    channel_labels: list[str],
    global_idx: int,
    val_pos: int | None,
    epoch: object,
    val_loss: float,
    output_path: Path,
    *,
    title_prefix: str = "MRANC SEED",
) -> None:
    colors = channel_color_map(highlight_idx)
    art = total_artifact(stems)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"{title_prefix} | {window_description(global_idx, val_pos)} | "
        f"epoch={epoch} val_loss={val_loss:.4f}",
        fontsize=11,
    )

    plot_raw_mix_panel(axes[0], mix, highlight_idx, channel_labels, colors)
    plot_highlight_artifact_panel(axes[1], art, highlight_idx, channel_labels, colors)
    plot_recovered_eeg_panel(axes[2], stems["pred_eeg"], highlight_idx, channel_labels, colors)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_deap_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    proj_aligned: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    highlight_idx: list[int],
    channel_labels: list[str],
    global_idx: int,
    val_pos: int | None,
    epoch: object,
    val_loss: float,
    output_path: Path,
) -> None:
    colors = channel_color_map(highlight_idx)
    t_len = mix.shape[1]

    print(f"Projection RMSE - EOG ch{REF_EOG_CH}: {rmse(proj_aligned['pred_eog'][REF_EOG_CH], refs['ref_eog'][REF_EOG_CH]):.4f}")
    print(f"Projection RMSE - EMG ch{REF_EMG_CH}: {rmse(proj_aligned['pred_emg'][REF_EMG_CH], refs['ref_emg'][REF_EMG_CH]):.4f}")
    print(f"Projection RMSE - ECG ch{REF_ECG_CH}: {rmse(proj_aligned['pred_ecg'][REF_ECG_CH], refs['ref_ecg'][REF_ECG_CH]):.4f}")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(
        f"MRANC DEAP | {window_description(global_idx, val_pos)} | "
        f"epoch={epoch} val_loss={val_loss:.4f} | T={t_len}",
        fontsize=11,
    )

    plot_raw_mix_panel(axes[0], mix, highlight_idx, channel_labels, colors)
    plot_pred_vs_ref_panel(axes[1], proj_aligned, refs)
    plot_residual_panel(axes[2], proj_aligned, refs)
    plot_recovered_eeg_panel(
        axes[3],
        stems["pred_eeg"],
        highlight_idx,
        channel_labels,
        colors,
        panel_title="Panel 4: Recovered clean EEG (32-ch, mix − aligned stems)",
        set_xlabel=True,
        deap_hard_ylim=(-30.0, 30.0),
    )

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_artifact_benchmark_8subplot_report(
    mix: np.ndarray,
    stems: dict[str, np.ndarray],
    channel_labels: list[str],
    global_idx: int,
    val_pos: int | None,
    epoch: object,
    val_loss: float,
    output_path: Path,
) -> None:
    clean = stems["pred_eeg"]
    recon = clean + total_artifact(stems)

    # Macro spatial RMS traces
    raw_rms = np.asarray(spatial_rms_trace(mix), dtype=np.float32)
    clean_rms = np.asarray(spatial_rms_trace(clean), dtype=np.float32)
    recon_rms = np.asarray(spatial_rms_trace(recon), dtype=np.float32)

    # Stem manifold spatial RMS traces
    ecg_rms = np.asarray(spatial_rms_trace(stems["pred_ecg"]), dtype=np.float32)
    eog_rms = np.asarray(spatial_rms_trace(stems["pred_eog"]), dtype=np.float32)
    emg_rms = np.asarray(spatial_rms_trace(stems["pred_emg"]), dtype=np.float32)
    base_rms = np.asarray(spatial_rms_trace(stems["pred_basenoise"]), dtype=np.float32)

    t = np.arange(mix.shape[1])
    # Wider canvas to prevent waveform squeezing across the 2-column grid.
    fig, axes = plt.subplots(4, 2, figsize=(24, 12), sharex=True)
    fig.suptitle(
        f"MRANC Artifact Benchmark | {window_description(global_idx, val_pos)} | "
        f"epoch={epoch} val_loss={val_loss:.4f} | T={mix.shape[1]}",
        fontsize=11,
    )

    # Shared y-limits by logical groups.
    macro_ylim = shared_ylim_with_padding(raw_rms, clean_rms, recon_rms)
    artifact_ylim = shared_ylim_with_padding(ecg_rms, eog_rms, emg_rms, base_rms)

    # Subplot 1
    axes[0, 0].plot(t, raw_rms, color="#1f77b4", linewidth=1.7)
    axes[0, 0].set_title("1. Raw EEG (Spatial RMS)")
    axes[0, 0].set_ylabel("Amplitude (uV)")
    axes[0, 0].set_ylim(*macro_ylim)
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2
    axes[0, 1].plot(t, ecg_rms, color="#d62728", linewidth=1.7)
    axes[0, 1].set_title("2. Isolated ECG (Heart Artifact Spatial RMS)")
    axes[0, 1].set_ylabel("Amplitude (uV)")
    axes[0, 1].set_ylim(*artifact_ylim)
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3
    axes[1, 0].plot(t, clean_rms, color="#2ca02c", linewidth=1.7)
    axes[1, 0].set_title("3. Cleaned EEG (Spatial RMS)")
    axes[1, 0].set_ylabel("Amplitude (uV)")
    axes[1, 0].set_ylim(*macro_ylim)
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4
    axes[1, 1].plot(t, eog_rms, color="#9467bd", linewidth=1.7)
    axes[1, 1].set_title("4. Isolated EOG (Ocular Artifact Spatial RMS)")
    axes[1, 1].set_ylabel("Amplitude (uV)")
    axes[1, 1].set_ylim(*artifact_ylim)
    axes[1, 1].grid(True, alpha=0.3)

    # Subplot 5
    axes[2, 0].plot(t, recon_rms, color="#17becf", linewidth=1.7)
    axes[2, 0].set_title("5. Reconstructed Sum (Clean + Artifacts Spatial RMS)")
    axes[2, 0].set_ylabel("Amplitude (uV)")
    axes[2, 0].set_ylim(*macro_ylim)
    axes[2, 0].grid(True, alpha=0.3)

    # Subplot 6
    axes[2, 1].plot(t, emg_rms, color="#ff7f0e", linewidth=1.7)
    axes[2, 1].set_title("6. Isolated EMG (Muscle Artifact Spatial RMS)")
    axes[2, 1].set_ylabel("Amplitude (uV)")
    axes[2, 1].set_ylim(*artifact_ylim)
    axes[2, 1].grid(True, alpha=0.3)

    # Subplot 8 (Row 4, Col 1): channel-local microscopic view
    label_map = {norm_ch(name): i for i, name in enumerate(channel_labels)}
    selected = []
    for name in ("Fp1", "Cz", "O1"):
        idx = label_map.get(norm_ch(name))
        if idx is not None:
            selected.append((name, idx))

    for name, idx in selected:
        axes[3, 0].plot(t, clean[idx], marker=".", markevery=30, linewidth=1.0, label=name)
    local_mean = np.mean(clean, axis=0)
    axes[3, 0].plot(t, local_mean, color="black", linewidth=1.5, label="Local spatial mean")
    axes[3, 0].set_title("8. Local Channel Spatial Breakdown (Fp1, Cz, O1)")
    axes[3, 0].set_ylabel("Amplitude (uV)")
    axes[3, 0].set_xlabel("Sample index")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend(loc="upper right", fontsize=8)

    # Subplot 7
    axes[3, 1].plot(t, base_rms, color="#8c564b", linewidth=1.7)
    axes[3, 1].set_title("7. Isolated Baseline Noise (Spatial RMS)")
    axes[3, 1].set_ylabel("Amplitude (uV)")
    axes[3, 1].set_ylim(*artifact_ylim)
    axes[3, 1].set_xlabel("Sample index")
    axes[3, 1].grid(True, alpha=0.3)

    # Keep sample index label present for all panels.
    for r in range(4):
        for c in range(2):
            if r < 3:
                axes[r, c].set_xlabel("Sample index")

    plt.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    setup_publication_style(args.dpi)

    data_dir = resolve_data_dir(args)
    device = resolve_device_common(args.device)
    use_val_split = not args.no_val_split
    is_deap = args.dataset == "deap"
    is_clinical = args.dataset == "clinical"
    is_artifact_benchmark = args.dataset == "artifact_benchmark"

    scalp_channels = load_deap_channel_names(data_dir)
    highlight_names = [s.strip() for s in args.highlight_channels.split(",") if s.strip()]
    highlight_idx = resolve_channel_indices(highlight_names, scalp_channels)

    ckpt_path = resolve_checkpoint(args.checkpoint)

    if args.output is None:
        if args.dataset == "artifact_benchmark":
            output_path = FIGURES_DATASET_DIRS[args.dataset] / "artifact_benchmark_8subplot_rms.jpg"
        else:
            output_path = FIGURES_DATASET_DIRS[args.dataset] / "artifact_subtraction_report.png"
        if args.checkpoint is not None:
            output_path = with_checkpoint_label(output_path, ckpt_path)
    else:
        output_path = Path(args.output)

    model = load_model(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    val_fraction = float(config.get("val_fraction", args.val_fraction))
    split_seed = int(config.get("split_seed", args.split_seed))

    global_idx, val_pos, mix = select_window(
        data_dir,
        args.window_index,
        use_val_split,
        val_fraction,
        split_seed,
    )
    val_slot = val_pos if val_pos is not None else global_idx

    split_mode = "validation" if use_val_split else "full dataset"
    t_div = diversity_sample_t(mix)
    print(
        f"Dataset: {args.dataset} | dir={data_dir} | split={split_mode} | "
        f"highlights={highlight_names}"
    )
    print(f"mix shape {mix.shape} | NaNs={np.isnan(mix).any()}")
    print(f"Channel diversity @ t={t_div}: std across channels = {float(np.std(mix[:, t_div])):.4f}")

    stems, attn, mix_plot = run_window_inference(model, mix, device, args.dataset)
    max_err = max_recon_error(mix_plot, stems)
    print(f"Reconstruction OK: max error = {max_err:.6e}")

    art_plot = total_artifact(stems)
    mix_rms = float(np.sqrt(np.mean(mix_plot**2)))
    art_rms = float(np.sqrt(np.mean(art_plot**2)))
    print(f"Artifact / mix RMS ratio: {art_rms / max(mix_rms, 1e-8):.4f}")

    if not args.skip_critical_figures:
        crit_root = Path(args.critical_figures_dir)
        if not crit_root.is_absolute():
            crit_root = PROJECT_ROOT / crit_root
        crit_dir = resolve_decomposition_output_dir(crit_root, args.dataset)
        save_per_channel_decomposition_figures(
            mix_plot,
            stems,
            attn,
            args.dataset,
            scalp_channels,
            highlight_idx,
            highlight_names,
            global_idx,
            val_slot,
            crit_dir,
            cmap=args.cmap,
            dpi=args.dpi,
        )

    if not args.legacy_spatial_rms:
        return

    stems_plot = stems
    if is_deap:
        raw_spatial_rms = float(np.sqrt(np.mean(spatial_rms_trace(mix_plot) ** 2)))
        clean_spatial_rms = float(np.sqrt(np.mean(spatial_rms_trace(stems_plot["pred_eeg"]) ** 2)))
        recon_spatial_rms = float(np.sqrt(np.mean(spatial_rms_trace(stems_plot["pred_eeg"] + art_plot) ** 2)))
        print(
            "DEAP scale diagnostics | "
            f"raw_spatial_rms={raw_spatial_rms:.6e} "
            f"clean_spatial_rms={clean_spatial_rms:.6e} "
            f"recon_spatial_rms={recon_spatial_rms:.6e}"
        )
        if raw_spatial_rms > 0.0:
            ratio = clean_spatial_rms / raw_spatial_rms
            if ratio < 1e-3 or ratio > 1e3:
                print(
                    "WARNING: DEAP clean/raw spatial RMS ratio is extreme "
                    f"({ratio:.3e}). No visual rescaling applied."
                )

    epoch = ckpt.get("epoch", "?")
    val_loss = float(ckpt.get("best_val_total_loss", float("nan")))

    if is_deap:
        prefix = "MRANC DEAP"
    elif is_clinical:
        prefix = "MRANC Clinical"
    elif is_artifact_benchmark:
        prefix = "MRANC Artifact Benchmark"
    else:
        prefix = "MRANC SEED"

    if is_artifact_benchmark:
        plot_artifact_benchmark_8subplot_report(
            mix_plot,
            stems_plot,
            scalp_channels,
            global_idx,
            val_pos,
            epoch,
            val_loss,
            output_path,
        )
    else:
        plot_decomposition_report(
            mix_plot,
            stems_plot,
            global_idx,
            val_pos,
            epoch,
            val_loss,
            output_path,
            title_prefix=prefix,
        )
    plot_highlight_channels_report(
        mix_plot,
        stems_plot,
        scalp_channels,
        with_output_suffix(output_path, "_channels"),
    )


if __name__ == "__main__":
    main()
