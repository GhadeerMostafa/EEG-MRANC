"""
Real-world MRANC evaluation and critical-figure generation.

Runs validation-window inference, optional metrics (evaluate_metrics.py),
optional summary interpretability plots (generate_interpretability_plots.py),
and ultra-wide per-channel 6-row decomposition PNGs.

Default checkpoint: checkpoints/best_mranc_artifact_benchmark_weights.pth
Override with --checkpoint path/to/weights.pth

Example:
  py scripts/run_real_world_pipeline.py --dataset all --max-windows 1
  py scripts/run_real_world_pipeline.py --dataset clinical --window-indices 0 --run-metrics --with-summary-figures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from runtime import setup_src_path

setup_src_path()

from figure_common import (
    DATASETS,
    load_deap_channel_names,
    load_mix_window,
    load_model,
    max_recon_error,
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
from paths import CRITICAL_FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR

DEFAULT_CHANNELS = ("Fp1", "Cz", "O1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MRANC real-world evaluation and per-channel decomposition pipeline",
    )
    p.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="clinical",
        help="Dataset preset or all four datasets",
    )
    p.add_argument("--data-dir", type=str, default=None, help="Override processed mix.npy folder")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model weights (default: checkpoints/best_mranc_artifact_benchmark_weights.pth)",
    )
    p.add_argument("--output-dir", type=str, default=str(CRITICAL_FIGURES_DIR))
    p.add_argument("--channels", type=str, default=",".join(DEFAULT_CHANNELS))
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--window-indices", type=str, default="all", help="all or comma list e.g. 0,1,2")
    p.add_argument("--max-windows", type=int, default=0, help="Cap val windows (0 = no cap)")
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda",),
        help="CUDA device for inference (VRAM)",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", type=str, default="viridis", choices=("viridis", "plasma"))
    p.add_argument(
        "--run-metrics",
        action="store_true",
        help="Run evaluate_metrics.py per dataset and save JSON under outputs/reports/{dataset}/",
    )
    p.add_argument(
        "--with-summary-figures",
        action="store_true",
        help="Also run generate_interpretability_plots.py per dataset (2 summary PNGs)",
    )
    p.add_argument(
        "--skip-decomposition",
        action="store_true",
        help="Skip per-channel 6-row decomposition PNGs",
    )
    p.add_argument(
        "--filename-index",
        choices=("val", "global"),
        default="val",
        help="Window index in filenames: val=holdout list index (window0), global=mix.npy row",
    )
    return p.parse_args()


def run_metrics_subprocess(
    dataset: str,
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    report_path = output_dir / dataset / f"evaluation_report_{dataset}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_metrics.py"),
        "--dataset",
        dataset,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(report_path),
        "--val-fraction",
        str(args.val_fraction),
        "--split-seed",
        str(args.split_seed),
        "--device",
        args.device,
    ]
    if args.data_dir and args.dataset != "all":
        cmd.extend(["--data-dir", args.data_dir])
    print(f"Running metrics: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def run_summary_figures_subprocess(
    dataset: str,
    checkpoint: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_interpretability_plots.py"),
        "--dataset",
        dataset,
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--window-indices",
        "0",
        "--val-fraction",
        str(args.val_fraction),
        "--split-seed",
        str(args.split_seed),
        "--device",
        args.device,
        "--cmap",
        args.cmap,
        "--with-summary",
        "--skip-decomposition",
    ]
    if args.data_dir and args.dataset != "all":
        cmd.extend(["--data-dir", args.data_dir])
    print(f"Running summary figures: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def write_window_manifest(manifest_dir: Path, entries: list[dict]) -> None:
    manifest_path = manifest_dir / "decomposition_manifest.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote manifest {manifest_path}")


def run_dataset_pipeline(
    dataset: str,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: Path,
    model,
) -> int:
    data_dir = resolve_data_dir(dataset, args.data_dir if args.dataset != "all" else None)
    mix_path = data_dir / "mix.npy"
    if not mix_path.is_file():
        print(f"SKIP {dataset}: missing {mix_path}")
        return 0

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_dir = resolve_decomposition_output_dir(out_root, dataset)

    if args.run_metrics:
        run_metrics_subprocess(dataset, checkpoint, REPORTS_DIR, args)

    if args.with_summary_figures:
        run_summary_figures_subprocess(dataset, checkpoint, out_root, args)

    if args.skip_decomposition:
        return 0

    n_total = int(np.load(mix_path, mmap_mode="r").shape[0])
    val_global = val_indices(
        n_total, args.val_fraction, args.split_seed, data_dir=data_dir, dataset=dataset
    )
    val_slots = parse_window_indices(args.window_indices, len(val_global))
    if args.max_windows > 0:
        val_slots = val_slots[: args.max_windows]

    channel_names = load_deap_channel_names(data_dir)
    ch_list = [s.strip() for s in args.channels.split(",") if s.strip()]
    ch_indices = resolve_channel_indices(ch_list, channel_names)

    print(f"Dataset={dataset} dir={data_dir} val_windows={len(val_slots)} checkpoint={checkpoint}")

    png_count = 0
    manifest: list[dict] = []
    for val_slot in val_slots:
        global_idx = val_global[val_slot]
        mix = load_mix_window(data_dir, global_idx)
        stems, attn, mix_used = run_window_inference(model, mix, device, dataset)
        recon_err = max_recon_error(mix_used, stems)
        print(f"  window global={global_idx} val_slot={val_slot} recon_max_err={recon_err:.6e}")

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
            out_dir,
            filename_index=args.filename_index,
            cmap=args.cmap,
            dpi=args.dpi,
        )
        png_count += len(saved)
        manifest.append(
            {
                "dataset": dataset,
                "global_window_index": global_idx,
                "val_slot": val_slot,
                "recon_max_error": recon_err,
                "files": [p.name for p in saved],
            }
        )

    if manifest:
        write_window_manifest(out_dir, manifest)

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

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    model = load_model(checkpoint, device) if not args.skip_decomposition else None

    total_png = 0
    for ds in datasets:
        total_png += run_dataset_pipeline(ds, args, device, checkpoint, model)

    print(f"Done. Total decomposition PNG files written: {total_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
