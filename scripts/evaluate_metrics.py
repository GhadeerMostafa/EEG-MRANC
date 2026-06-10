"""
MRANC validation metrics — SNR improvement, PSD preservation (8–30 Hz), RMSE.

Loads checkpoints/best_mranc_artifact_benchmark_weights.pth by default, evaluates all validation windows for
--dataset seed|deap|clinical, prints a summary table, and writes a JSON report
under outputs/reports/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from runtime import setup_src_path
from torch.utils.data import TensorDataset, random_split

setup_src_path()

from metric_alignment import (
    ARTIFACT_KEYS,
    SUPERVISED_METRIC_DATASETS,
    prepare_supervised_eval_tensors,
)
from model import MRANC
from checkpoint_paths import resolve_stacking_latest
from paths import (
    ARTIFACT_BENCHMARK_CHECKPOINT,
    CHECKPOINT_PHASE2_DEAP,
    CHECKPOINT_PHASE3_SEED,
    DATASET_DIRS,
    MAIN_CHECKPOINT,
    PROJECT_ROOT,
    REPORTS_DATASET_DIRS,
    resolve_clinical_weights,
)

ORTHO_WEIGHT = 0.05
TV_WEIGHT = 0.01
SCALE_PENALTY_WEIGHT = 0.1

from validation_metrics import (
    DEFAULT_SAMPLE_RATE,
    PSD_BAND_HIGH_HZ,
    PSD_BAND_LOW_HZ,
    aggregate_metrics,
    compute_batch_metrics_torch,
)

EVAL_DATASETS = ("clinical", "seed", "deap", "artifact_benchmark")


def list_eval_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(EVAL_DATASETS)
    if dataset_arg not in EVAL_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_arg!r}; choose from {', '.join(EVAL_DATASETS)} or all"
        )
    return [dataset_arg]


def pearson_r(pred: torch.Tensor, ref: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    pred = pred.float()
    ref = ref.float()
    pred_c = pred - pred.mean(dim=dim, keepdim=True)
    ref_c = ref - ref.mean(dim=dim, keepdim=True)
    cov = (pred_c * ref_c).mean(dim=dim)
    denom = pred_c.pow(2).mean(dim=dim).sqrt() * ref_c.pow(2).mean(dim=dim).sqrt() + eps
    return torch.clamp(cov / denom, -1.0, 1.0)


def combined_ref_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """log1p(MSE) + (1 - Pearson r) in fp32 for stability."""
    pred = pred.float()
    ref = ref.float()
    mse = torch.nn.functional.mse_loss(pred, ref)
    return torch.log1p(mse) + (1.0 - pearson_r(pred, ref, dim=-1)).mean()


def polarity_aligned_ref_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Min loss over pred and -pred so inverted hardware sign does not penalize loss."""
    ref = ref.float()
    pred = pred.float()
    loss_pos = combined_ref_loss(pred, ref)
    loss_neg = combined_ref_loss(-pred, ref)
    return torch.minimum(loss_pos, loss_neg)


def hardware_ref_loss(pred: torch.Tensor, ref: torch.Tensor, seq_len: int) -> torch.Tensor:
    """DEAP (T=256): polarity-aligned min loss; other T: standard combined ref loss."""
    # Import kept local to avoid a hard dependency on model internals here.
    from model import DEAP_SEQ_LEN  # noqa: WPS433

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


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:] - x[..., :-1]).abs().mean()


def tv_physiological_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        total_variation(out["pred_eog"])
        + total_variation(out["pred_emg"])
        + total_variation(out["pred_ecg"])
    ) / 3.0


def compute_val_total_loss(
    model: MRANC,
    out: dict[str, torch.Tensor],
    *,
    dataset: str,
    noise_mult: float,
    ref_eog: torch.Tensor | None,
    ref_emg: torch.Tensor | None,
    ref_ecg: torch.Tensor | None,
) -> torch.Tensor:
    """
    Recompute a training-consistent total_loss on validation windows.

    - deap/seed: supervised hardware-ref losses + orth/tv + scale penalty
    - clinical: physics-only loss
    """
    if dataset in ("clinical", "artifact_benchmark"):
        stems = [out["pred_eog"], out["pred_emg"], out["pred_ecg"], out["pred_basenoise"]]
        loss_ortho = orthogonality_loss_pearson(out["pred_eeg"].float(), [s.float() for s in stems])
        loss_tv = tv_physiological_loss({k: v.float() for k, v in out.items() if k.startswith("pred_")})
        pred_eog2 = torch.mean(out["pred_eog"].float() ** 2)
        pred_emg2 = torch.mean(out["pred_emg"].float() ** 2)
        pred_ecg2 = torch.mean(out["pred_ecg"].float() ** 2)
        loss_min_clean = pred_eog2 + pred_emg2 + pred_ecg2
        return 0.5 * loss_ortho + 0.1 * loss_tv + 0.01 * loss_min_clean

    if ref_eog is None or ref_emg is None or ref_ecg is None:
        raise ValueError("ref tensors are required for non-clinical val_total_loss")

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

    stems = [
        out["pred_eog"].float(),
        out["pred_emg"].float(),
        out["pred_ecg"].float(),
        out["pred_basenoise"].float(),
    ]
    loss_ortho = orthogonality_loss_pearson(out["pred_eeg"].float(), stems)
    loss_tv = tv_physiological_loss({k: v.float() for k, v in out.items() if k.startswith("pred_")})
    loss_stems = loss_eog + loss_emg + loss_ecg + loss_basenoise
    scale_penalty = SCALE_PENALTY_WEIGHT * (
        torch.mean(out["pred_eog"].float() ** 2)
        + torch.mean(out["pred_emg"].float() ** 2)
        + torch.mean(out["pred_ecg"].float() ** 2)
    )
    return loss_stems + ORTHO_WEIGHT * loss_ortho + TV_WEIGHT * loss_tv + scale_penalty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRANC validation metrics (SEED, DEAP, or clinical)")
    p.add_argument(
        "--dataset",
        choices=("seed", "deap", "clinical", "artifact_benchmark", "all"),
        default="seed",
        help="Corpus to evaluate, or all for clinical, seed, deap, and artifact_benchmark",
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
    p.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Hz for PSD (default: 200 seed, 128 deap)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: outputs/reports/<dataset>/evaluation_report_<dataset>.json)",
    )
    p.add_argument(
        "--disable-msab",
        action="store_true",
        help="Bypass Multi-Scale Attention Block (identity pass-through) at inference",
    )
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(choice)


def resolve_data_dir(dataset: str, args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir) if Path(args.data_dir).is_absolute() else PROJECT_ROOT / args.data_dir
    return DATASET_DIRS[dataset]


def resolve_checkpoint_path(dataset: str, override: str | None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else PROJECT_ROOT / p
    if dataset == "clinical":
        return resolve_clinical_weights()
    phase_by_dataset = {
        "artifact_benchmark": ("phase1_artifact_benchmark", ARTIFACT_BENCHMARK_CHECKPOINT),
        "deap": ("phase2_deap", CHECKPOINT_PHASE2_DEAP),
        "seed": ("phase3_seed", CHECKPOINT_PHASE3_SEED),
    }
    if dataset in phase_by_dataset:
        phase_key, fallback = phase_by_dataset[dataset]
        latest = resolve_stacking_latest(phase_key, fallback)
        if latest is not None:
            return latest
    return MAIN_CHECKPOINT


def resolve_sample_rate(dataset: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    return DEFAULT_SAMPLE_RATE[dataset]


def dataset_length(data_dir: Path) -> int:
    mix_path = data_dir / "mix.npy"
    if not mix_path.exists():
        raise FileNotFoundError(f"Missing {mix_path}")
    return int(np.load(mix_path, mmap_mode="r").shape[0])


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


def _metric_scalar(value: torch.Tensor | float) -> float | None:
    v = float(value) if not isinstance(value, torch.Tensor) else float(value.detach().cpu().item())
    return v if math.isfinite(v) else None


def load_checkpoint(path: Path, device: torch.device) -> tuple[MRANC, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = MRANC(feature_channels=int(config.get("feature_channels", 128))).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def print_summary_table(
    dataset: str,
    n_val: int,
    seq_len: int,
    fs: float,
    metrics: dict[str, dict[str, float]],
    val_total_loss: float,
) -> None:
    header = (
        f"MRANC Evaluation - dataset={dataset} | val windows={n_val} | "
        f"T={seq_len} | fs={fs:.0f} Hz"
    )
    print(header)
    print(f"val_total_loss: {val_total_loss:.6f}")
    print(f"mse : val_total_loss = {metrics['loss_mse']['mean']:.6f} : {val_total_loss:.6f}")
    print("+" + "-" * 33 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'Metric':<31} | {'Mean':>10} | {'Std':>10} |")
    print("+" + "-" * 33 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    rows = [
        ("Reconstruction MSE (loss_mse)", "loss_mse"),
        ("SNR improvement (dB)", "snr_improvement_db"),
        ("PSD corr (alpha/beta 8-30Hz)", "psd_alpha_beta_correlation"),
        ("Physics residual RMSE", "physics_residual_rmse"),
        ("Artifact magnitude RMSE", "artifact_magnitude_rmse"),
    ]
    for label, key in rows:
        m = metrics[key]["mean"]
        s = metrics[key]["std"]
        print(f"| {label:<31} | {m:10.4f} | {s:10.4f} |")
    print("+" + "-" * 33 + "+" + "-" * 12 + "+" + "-" * 12 + "+")


def evaluate_dataset_report(dataset: str, args: argparse.Namespace) -> dict:
    data_dir = resolve_data_dir(dataset, args)
    checkpoint_path = resolve_checkpoint_path(dataset, args.checkpoint)
    device = resolve_device(args.device)

    n_total = dataset_length(data_dir)
    mix_mmap = np.load(data_dir / "mix.npy", mmap_mode="r")
    seq_len = int(mix_mmap.shape[2])

    model, ckpt = load_checkpoint(checkpoint_path, device)
    config = ckpt.get("config", {})
    val_fraction = float(config.get("val_fraction", args.val_fraction))
    split_seed = int(config.get("split_seed", args.split_seed))
    fs = resolve_sample_rate(dataset, args.sample_rate)
    noise_mult = float(config.get("noise_coeff_end", 0.01))

    val_idx = val_indices(n_total, val_fraction, split_seed)
    per_window: list[dict] = []
    total_loss_sum = 0.0
    total_loss_count = 0

    for start in range(0, len(val_idx), args.batch_size):
        batch_ids = val_idx[start : start + args.batch_size]
        mix_batch = np.array([mix_mmap[i] for i in batch_ids], dtype=np.float32)

        x = torch.from_numpy(mix_batch).to(device)
        with torch.no_grad():
            out = model(x, disable_msab=args.disable_msab)

        ref_eog_b = ref_emg_b = ref_ecg_b = None
        if dataset not in ("clinical", "artifact_benchmark"):
            ref_eog_m = np.load(data_dir / "ref_eog.npy", mmap_mode="r")
            ref_emg_m = np.load(data_dir / "ref_emg.npy", mmap_mode="r")
            ref_ecg_m = np.load(data_dir / "ref_ecg.npy", mmap_mode="r")
            ref_eog_b = torch.from_numpy(np.array([ref_eog_m[i] for i in batch_ids], dtype=np.float32)).to(device)
            ref_emg_b = torch.from_numpy(np.array([ref_emg_m[i] for i in batch_ids], dtype=np.float32)).to(device)
            ref_ecg_b = torch.from_numpy(np.array([ref_ecg_m[i] for i in batch_ids], dtype=np.float32)).to(device)

        batch_total = compute_val_total_loss(
            model,
            out,
            dataset=dataset,
            noise_mult=noise_mult,
            ref_eog=ref_eog_b,
            ref_emg=ref_emg_b,
            ref_ecg=ref_ecg_b,
        )
        total_loss_sum += float(batch_total.detach().mean().cpu().item())
        total_loss_count += 1

        mix_eval, clean_t, artifact_t, clean_for_psd_t, artifact_for_snr_t = prepare_supervised_eval_tensors(
            x,
            out,
            dataset=dataset,
            zscore_distance=(dataset == "seed"),
        )

        batch_metrics, batch_pcm = compute_batch_metrics_torch(
            mix_eval,
            clean_t,
            artifact_t,
            fs,
            artifact_for_snr=artifact_for_snr_t,
            clean_for_psd=clean_for_psd_t,
            mix_for_psd=x.float() if dataset == "seed" else None,
        )

        for j, global_i in enumerate(batch_ids):
            per_window.append(
                {
                    "global_index": int(global_i),
                    "loss_mse": _metric_scalar(batch_metrics["loss_mse"][j]),
                    "snr_improvement_db": _metric_scalar(batch_metrics["snr_improvement_db"][j]),
                    "psd_alpha_beta_correlation": _metric_scalar(
                        batch_metrics["psd_alpha_beta_correlation"][j]
                    ),
                    "physics_residual_rmse": _metric_scalar(batch_metrics["physics_residual_rmse"][j]),
                    "artifact_magnitude_rmse": _metric_scalar(
                        batch_metrics["artifact_magnitude_rmse"][j]
                    ),
                    "per_channel_mean": {
                        "snr_improvement_db": _metric_scalar(batch_pcm["snr_improvement_db"][j]),
                        "physics_residual_rmse": _metric_scalar(batch_pcm["physics_residual_rmse"][j]),
                        "artifact_magnitude_rmse": _metric_scalar(
                            batch_pcm["artifact_magnitude_rmse"][j]
                        ),
                    },
                }
            )

    metrics = aggregate_metrics(per_window)
    val_total_loss = float(total_loss_sum / max(total_loss_count, 1))

    report = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint_path),
        "n_val_windows": len(val_idx),
        "seq_len": seq_len,
        "sample_rate_hz": fs,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "val_total_loss": val_total_loss,
        "mse : val_total_loss": f"{metrics['loss_mse']['mean']:.6f} : {val_total_loss:.6f}",
        "val_total_loss_noise_mult": noise_mult,
        "psd_band_hz": [PSD_BAND_LOW_HZ, PSD_BAND_HIGH_HZ],
        "amplitude_aligned_metrics": dataset in SUPERVISED_METRIC_DATASETS,
        "distance_zscore_metrics": dataset == "seed",
        "disable_msab": args.disable_msab,
        "metrics": metrics,
        "per_window": per_window,
    }

    return report


def evaluate_dataset(dataset: str, args: argparse.Namespace) -> None:
    report = evaluate_dataset_report(dataset, args)
    dataset = report["dataset"]
    if args.output is None:
        output_path = REPORTS_DATASET_DIRS[dataset] / f"evaluation_report_{dataset}.json"
    else:
        output_path = Path(args.output)
    metrics = report["metrics"]
    print_summary_table(
        dataset,
        report["n_val_windows"],
        report["seq_len"],
        report["sample_rate_hz"],
        metrics,
        report["val_total_loss"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    if args.dataset == "all":
        if args.data_dir:
            raise ValueError("--data-dir cannot be used with --dataset all")
        if args.output:
            raise ValueError("--output cannot be used with --dataset all")
        if args.checkpoint:
            raise ValueError(
                "--checkpoint cannot be used with --dataset all; "
                "each corpus uses its stacking phase from checkpoints/stacking_latest.json"
            )
    try:
        datasets = list_eval_datasets(args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for dataset in datasets:
        evaluate_dataset(dataset, args)


if __name__ == "__main__":
    main()
