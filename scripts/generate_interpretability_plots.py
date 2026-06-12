"""
Generate MRANC critical figures: ultra-wide per-channel 6-row decomposition PNGs.

Optional legacy 2-panel summary figures (--with-summary):
  {prefix}_denoising_fidelity.png
  {prefix}_attention_map_heatmap.png

Default checkpoint: checkpoints/best_mranc_artifact_benchmark_weights.pth
Override with --checkpoint path/to/weights.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from runtime import setup_src_path
from scipy.signal import detrend

setup_src_path()

from figure_common import (
    DATASETS,
    dataset_display_name,
    load_deap_channel_names,
    load_mix_window,
    load_model,
    max_recon_error,
    output_file_prefix,
    parse_window_indices,
    resolve_channel_indices,
    resolve_checkpoint,
    resolve_data_dir,
    resolve_decomposition_output_dir,
    resolve_device,
    run_window_inference,
    save_per_channel_decomposition_figures,
    setup_publication_style,
    val_indices,
)
from paths import CRITICAL_FIGURES_DIR, PROJECT_ROOT

REF_FILES = ("ref_eog.npy", "ref_emg.npy", "ref_ecg.npy")
HIGHLIGHT_CHANNELS = ("Fp1", "Cz", "O1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRANC critical figure generation")
    p.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="clinical",
        help="Dataset preset or all four datasets",
    )
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model weights (default: checkpoints/best_mranc_artifact_benchmark_weights.pth)",
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--window-index",
        type=int,
        default=None,
        help="Single validation holdout index (deprecated; use --window-indices)",
    )
    p.add_argument(
        "--window-indices",
        type=str,
        default=None,
        help="all or comma list e.g. 0,1 (default: 0 or --window-index)",
    )
    p.add_argument(
        "--highlight-channels",
        type=str,
        default=",".join(HIGHLIGHT_CHANNELS),
    )
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--output-dir", type=str, default=str(CRITICAL_FIGURES_DIR))
    p.add_argument("--cmap", type=str, default="viridis", choices=("viridis", "plasma"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--with-summary",
        action="store_true",
        help="Also write legacy 2-panel denoising + attention summary PNGs",
    )
    p.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip per-channel 6-row decomposition PNGs (summary-only mode)",
    )
    p.add_argument(
        "--filename-index",
        choices=("val", "global"),
        default="val",
        help="Window index in filenames: val=holdout list index, global=mix.npy row",
    )
    return p.parse_args()


def resolve_window_indices_arg(args: argparse.Namespace, n_val: int) -> list[int]:
    if args.window_indices is not None:
        return parse_window_indices(args.window_indices, n_val)
    if args.window_index is not None:
        return parse_window_indices(str(args.window_index), n_val)
    return [0]


def spatial_rms_trace(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return np.sqrt(np.mean(x**2, axis=1))[0]
    if x.ndim == 2:
        return np.sqrt(np.mean(x**2, axis=0))
    raise ValueError(f"spatial_rms_trace expected 2D/3D array, got {x.shape}")


def shared_ylim_with_padding(*traces: np.ndarray, pad_ratio: float = 0.05) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(t).ravel() for t in traces])
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


def load_legacy_target_eeg(root: Path) -> np.ndarray | None:
    path = root / "dataset" / "val" / "target_eeg.npy"
    if not path.is_file():
        return None
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2:
        return np.array(arr, dtype=np.float32)
    if arr.ndim == 3:
        return np.array(arr[0], dtype=np.float32)
    return None


def load_refs_window(data_dir: Path, global_idx: int) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for fname in REF_FILES:
        path = data_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        refs[fname.replace(".npy", "")] = np.array(
            np.load(path, mmap_mode="r")[global_idx], dtype=np.float32
        )
    return {
        "ref_eog": refs["ref_eog"],
        "ref_emg": refs["ref_emg"],
        "ref_ecg": refs["ref_ecg"],
    }


def proxy_clean_from_refs(mix: np.ndarray, refs: dict[str, np.ndarray]) -> np.ndarray:
    artifact = np.zeros_like(mix, dtype=np.float32)
    for key in ("ref_eog", "ref_emg", "ref_ecg"):
        ref = refs[key]
        if ref.ndim == 1:
            ref = ref[np.newaxis, :]
        for c in range(ref.shape[0]):
            r = ref[c]
            r = r - r.mean()
            std_r = r.std() + 1e-8
            for ch in range(mix.shape[0]):
                m = mix[ch]
                m_c = m - m.mean()
                scale = float(np.dot(m_c, r) / (std_r * std_r * len(r) + 1e-8))
                artifact[ch] += scale * r
    return mix - artifact


def plot_denoising_fidelity(
    mix: np.ndarray,
    clean_gt: np.ndarray | None,
    pred_eeg: np.ndarray,
    channel_names: list[str],
    highlight_idx: list[int],
    dataset: str,
    output_path: Path,
    dpi: int,
) -> None:
    t = np.arange(mix.shape[-1])
    n_cols = 3 if clean_gt is not None else 2
    fig, axes = plt.subplots(
        len(highlight_idx) + 1,
        n_cols,
        figsize=(4.2 * n_cols, 2.2 * (len(highlight_idx) + 1)),
        sharex=True,
        squeeze=False,
    )

    col_titles = ["Raw Input", "MRANC Denoised Output"]
    col_data = [mix, pred_eeg]
    if clean_gt is not None:
        col_titles = ["Raw Input", "Ground Truth Clean", "MRANC Denoised Output"]
        col_data = [mix, clean_gt, pred_eeg]

    palette = {"raw": "#1f77b4", "gt": "#ff7f0e", "clean": "#2ca02c"}

    for row, ch in enumerate(highlight_idx):
        name = channel_names[ch]
        for col in range(n_cols):
            ax = axes[row, col]
            trace = col_data[col][ch]
            if col == 0:
                color = palette["raw"]
            elif clean_gt is not None and col == 1:
                color = palette["gt"]
            else:
                color = palette["clean"]
            ax.plot(t, trace, color=color, linewidth=0.9)
            if row == 0:
                ax.set_title(col_titles[col])
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)

    raw_rms = spatial_rms_trace(mix)
    clean_rms = spatial_rms_trace(pred_eeg)
    rms_traces = [raw_rms, clean_rms]
    rms_labels = ["Raw spatial RMS", "Denoised spatial RMS"]
    if clean_gt is not None:
        rms_traces = [raw_rms, spatial_rms_trace(clean_gt), clean_rms]
        rms_labels = ["Raw spatial RMS", "GT spatial RMS", "Denoised spatial RMS"]

    ylim = shared_ylim_with_padding(*rms_traces)
    ax_rms_row = len(highlight_idx)
    for col in range(n_cols):
        ax = axes[ax_rms_row, col]
        if col == 0:
            ax.plot(t, rms_traces[0], color=palette["raw"], linewidth=1.5, label=rms_labels[0])
        elif clean_gt is not None and col == 1:
            ax.plot(t, rms_traces[1], color=palette["gt"], linewidth=1.5, label=rms_labels[1])
        else:
            idx = 2 if clean_gt is not None else 1
            ax.plot(t, rms_traces[idx], color=palette["clean"], linewidth=1.5, label=rms_labels[idx])
        ax.set_ylim(ylim)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Spatial RMS")
        ax.set_title("Spatial RMS preservation")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{dataset_display_name(dataset)} Denoising Fidelity", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_attention_heatmap(
    mix: np.ndarray,
    attn: np.ndarray,
    dataset: str,
    cmap: str,
    output_path: Path,
    dpi: int,
) -> None:
    t = np.arange(mix.shape[-1])
    eeg_trace = detrend(mix.mean(axis=0), type="linear")

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})

    axes[0].plot(t, eeg_trace, color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].set_title(f"{dataset_display_name(dataset)} Spatial Mean EEG Trace (Raw Input)")
    axes[0].grid(True, alpha=0.25)

    attn_2d = attn[np.newaxis, :]
    im = axes[1].imshow(
        attn_2d,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[0, len(t) - 1, 0, 1],
    )
    axes[1].set_ylabel("Attention")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Normalized Multi-Scale Attention Weights")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.02)
    cbar.set_label("Attention (0 to 1)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    model,
) -> int:
    data_dir = resolve_data_dir(dataset, args.data_dir if args.dataset != "all" else None)
    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        print(f"SKIP {dataset}: missing {mix_path}")
        return 0

    n_total = int(np.load(mix_path, mmap_mode="r").shape[0])
    val_global = val_indices(
        n_total, args.val_fraction, args.split_seed, data_dir=data_dir, dataset=dataset
    )
    val_slots = resolve_window_indices_arg(args, len(val_global))

    channel_names = load_deap_channel_names(data_dir)
    ch_list = [s.strip() for s in args.highlight_channels.split(",") if s.strip()]
    ch_indices = resolve_channel_indices(ch_list, channel_names)

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    decomp_dir = resolve_decomposition_output_dir(out_root, dataset)
    summary_dir = out_root / dataset
    summary_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_file_prefix(dataset)

    clean_gt: np.ndarray | None = load_legacy_target_eeg(PROJECT_ROOT)

    png_count = 0
    for val_slot in val_slots:
        global_idx = val_global[val_slot]
        mix = load_mix_window(data_dir, global_idx)
        stems, attn, mix_used = run_window_inference(model, mix, device, dataset)
        recon_err = max_recon_error(mix_used, stems)
        print(
            f"{dataset} window global={global_idx} val_slot={val_slot} "
            f"recon_max_err={recon_err:.6e}"
        )

        if not args.skip_decomposition:
            saved = save_per_channel_decomposition_figures(
                mix_used,
                stems,
                attn,
                dataset,
                channel_names,
                ch_indices,
                ch_list,
                global_idx,
                val_slot,
                decomp_dir,
                filename_index=args.filename_index,
                cmap=args.cmap,
                dpi=args.dpi,
            )
            png_count += len(saved)

        if args.with_summary and val_slot == val_slots[0]:
            window_clean_gt = clean_gt
            if window_clean_gt is None and dataset in ("seed", "deap"):
                refs = load_refs_window(data_dir, global_idx)
                window_clean_gt = proxy_clean_from_refs(mix_used, refs)

            fidelity_path = summary_dir / f"{prefix}_denoising_fidelity.png"
            heatmap_path = summary_dir / f"{prefix}_attention_map_heatmap.png"
            plot_denoising_fidelity(
                mix_used,
                window_clean_gt,
                stems["pred_eeg"],
                channel_names,
                ch_indices,
                dataset,
                fidelity_path,
                args.dpi,
            )
            plot_attention_heatmap(mix_used, attn, dataset, args.cmap, heatmap_path, args.dpi)
            print(f"Saved {fidelity_path}")
            print(f"Saved {heatmap_path}")

    return png_count


def main() -> int:
    args = parse_args()
    setup_publication_style(args.dpi)

    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    print(f"Using checkpoint: {checkpoint}")
    model = load_model(checkpoint, device)

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    total = 0
    for ds in datasets:
        total += run_dataset(ds, args, device, model)

    print(f"Done. Total decomposition PNG files written: {total}")
    if total > 0 or args.with_summary:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
