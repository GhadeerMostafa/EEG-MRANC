"""
Clinical ablation study: stacking phases and MSAB bypass on held-out windows.

Compares four inference variants on the clinical validation split using
checkpoints from stacking_latest.json.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from checkpoint_paths import resolve_stacking_latest
from evaluate_metrics import evaluate_dataset_report, resolve_device
from paths import (
    ARTIFACT_BENCHMARK_CHECKPOINT,
    CHECKPOINT_PHASE3_SEED,
    PROJECT_ROOT,
    REPORTS_DATASET_DIRS,
    resolve_clinical_weights,
)

ABLATION_VARIANTS = (
    {
        "key": "full_mranc",
        "label": "Full MRANC (Phase 4 + adapter)",
        "phase_key": "phase4_clinical_attention",
        "checkpoint_fallback": None,
        "disable_msab": False,
    },
    {
        "key": "no_attention_adapter",
        "label": "w/o Attention Adapter (Phase 3 SEED)",
        "phase_key": "phase3_seed",
        "checkpoint_fallback": CHECKPOINT_PHASE3_SEED,
        "disable_msab": False,
    },
    {
        "key": "no_stacking",
        "label": "w/o Sequential Stacking (Phase 1 only)",
        "phase_key": "phase1_artifact_benchmark",
        "checkpoint_fallback": ARTIFACT_BENCHMARK_CHECKPOINT,
        "disable_msab": False,
    },
    {
        "key": "no_msab",
        "label": "w/o MSAB (Phase 4, attention bypass)",
        "phase_key": "phase4_clinical_attention",
        "checkpoint_fallback": None,
        "disable_msab": True,
    },
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MRANC clinical ablation evaluation")
    p.add_argument("--dataset", default="clinical", choices=("clinical",))
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON (default: artifacts/reports/clinical/ablation_report_clinical.json)",
    )
    return p.parse_args()


def resolve_variant_checkpoint(variant: dict) -> Path:
    if variant["phase_key"] == "phase4_clinical_attention":
        return resolve_clinical_weights()
    latest = resolve_stacking_latest(variant["phase_key"], variant["checkpoint_fallback"])
    if latest is not None:
        return latest
    if variant["checkpoint_fallback"] is not None:
        return variant["checkpoint_fallback"]
    raise FileNotFoundError(f"No checkpoint for variant {variant['key']}")


def _delta(full: float, variant: float) -> float:
    return variant - full


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else REPORTS_DATASET_DIRS["clinical"] / "ablation_report_clinical.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    base_args = Namespace(
        data_dir=None,
        checkpoint=None,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        device=args.device,
        batch_size=args.batch_size,
        sample_rate=None,
        output=None,
        disable_msab=False,
    )

    rows: list[dict] = []
    full_metrics: dict[str, float] | None = None

    print(f"Clinical ablation | device={resolve_device(args.device)}")
    print("+" + "-" * 72 + "+")
    print(f"| {'Variant':<40} | {'SNR dB':>8} | {'PSD':>8} | {'Art RMSE':>8} |")
    print("+" + "-" * 72 + "+")

    for variant in ABLATION_VARIANTS:
        ckpt = resolve_variant_checkpoint(variant)
        run_args = Namespace(**vars(base_args))
        run_args.checkpoint = str(ckpt)
        run_args.disable_msab = variant["disable_msab"]

        report = evaluate_dataset_report(args.dataset, run_args)
        m = report["metrics"]
        snr = m["snr_improvement_db"]["mean"]
        psd = m["psd_alpha_beta_correlation"]["mean"]
        art = m["artifact_magnitude_rmse"]["mean"]

        if variant["key"] == "full_mranc":
            full_metrics = {"snr": snr, "psd": psd, "artifact_rmse": art}

        row = {
            "key": variant["key"],
            "label": variant["label"],
            "checkpoint": str(ckpt),
            "disable_msab": variant["disable_msab"],
            "n_val_windows": report["n_val_windows"],
            "snr_improvement_db": snr,
            "snr_improvement_db_std": m["snr_improvement_db"]["std"],
            "psd_alpha_beta_correlation": psd,
            "psd_alpha_beta_correlation_std": m["psd_alpha_beta_correlation"]["std"],
            "artifact_magnitude_rmse": art,
            "artifact_magnitude_rmse_std": m["artifact_magnitude_rmse"]["std"],
            "physics_residual_rmse": m["physics_residual_rmse"]["mean"],
            "loss_mse": m["loss_mse"]["mean"],
        }
        if full_metrics is not None and variant["key"] != "full_mranc":
            row["delta_snr_db"] = _delta(full_metrics["snr"], snr)
            row["delta_psd"] = _delta(full_metrics["psd"], psd)
            row["delta_artifact_rmse"] = _delta(full_metrics["artifact_rmse"], art)
        rows.append(row)

        print(
            f"| {variant['label']:<40} | {snr:8.4f} | {psd:8.4f} | {art:8.4f} |"
        )

    print("+" + "-" * 72 + "+")

    payload = {
        "dataset": args.dataset,
        "reference_variant": "full_mranc",
        "variants": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
