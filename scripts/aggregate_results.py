"""
Aggregate dataset evaluation JSON reports into paper-ready tables.

Reads:
  - outputs/reports/seed/evaluation_report_seed.json
  - outputs/reports/clinical/evaluation_report_clinical.json
  - outputs/reports/deap/evaluation_report_deap.json

Prints:
  - Console summary table
  - LaTeX table via pandas.to_latex()
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from runtime import setup_src_path

setup_src_path()

from paths import PROJECT_ROOT, REPORTS_DIR

REQUIRED_DATASETS = ("seed", "clinical", "deap")

METRIC_MAP = {
    "loss_mse": "Reconstruction MSE",
    "snr_improvement_db": "SNR Improvement (dB)",
    "psd_alpha_beta_correlation": "PSD Correlation",
    "artifact_magnitude_rmse": "Artifact Magnitude RMSE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate MRANC evaluation reports for paper table")
    p.add_argument(
        "--reports-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory containing evaluation_report_<dataset>.json files",
    )
    p.add_argument(
        "--float-format",
        type=str,
        default="%.6f",
        help="Float format for LaTeX table output",
    )
    return p.parse_args()


def report_path(reports_dir: Path, dataset: str) -> Path:
    return reports_dir / dataset / f"evaluation_report_{dataset}.json"


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing report file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataframe(reports: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for dataset in REQUIRED_DATASETS:
        report = reports[dataset]
        metrics = report.get("metrics", {})
        row: dict[str, float | str] = {"dataset": dataset.upper()}
        for key in METRIC_MAP:
            if key not in metrics:
                raise KeyError(f"{dataset}: missing metrics['{key}']")
            row[f"{key}_mean"] = float(metrics[key]["mean"])
            row[f"{key}_std"] = float(metrics[key]["std"])
        rows.append(row)
    return pd.DataFrame(rows)


def pretty_console_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    view = view.rename(
        columns={
            "dataset": "Dataset",
            "loss_mse_mean": "Recon MSE (mean)",
            "loss_mse_std": "Recon MSE (std)",
            "snr_improvement_db_mean": "SNR dB (mean)",
            "snr_improvement_db_std": "SNR dB (std)",
            "psd_alpha_beta_correlation_mean": "PSD Corr (mean)",
            "psd_alpha_beta_correlation_std": "PSD Corr (std)",
            "artifact_magnitude_rmse_mean": "Art RMSE (mean)",
            "artifact_magnitude_rmse_std": "Art RMSE (std)",
        }
    )
    return view


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    reports: dict[str, dict] = {}
    for ds in REQUIRED_DATASETS:
        path = report_path(reports_dir, ds)
        reports[ds] = load_report(path)

    df = build_dataframe(reports)
    console_df = pretty_console_table(df)

    print("MRANC Benchmark Summary (SEED / Clinical / DEAP)")
    print(console_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nLaTeX table (copy into your paper):")
    print(df.to_latex(index=False, float_format=args.float_format))


if __name__ == "__main__":
    main()
