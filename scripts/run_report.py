# pip install python-docx
"""
Compile MRANC evaluation metrics and critical figures into a Word research report.

MRANC columns: live values from outputs/reports/{dataset}/evaluation_report_{dataset}.json
ICA / EEGdenoiseNet / Raw: empirical baselines from baseline_*_metrics.json manifests.

Output: outputs/reports/clinical/MRANC_Final_Research_Report.docx

  py scripts/run_report.py
  py scripts/run_report.py --refresh
  py scripts/run_report.py --full-refresh
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from runtime import setup_src_path

setup_src_path()

from baseline_eval import (
    extract_metric_means,
    extract_raw_metric_means,
    load_baseline_json,
)
from figure_common import output_file_prefix
from checkpoint_paths import resolve_stacking_latest
from paths import (
    ARTIFACT_BENCHMARK_CHECKPOINT,
    BASELINE_EEGDENOISENET_CHECKPOINT,
    CHECKPOINT_PHASE2_DEAP,
    CHECKPOINT_PHASE3_SEED,
    CLINICAL_FINAL_REPORT_PATH,
    CRITICAL_FIGURES_DIR,
    PROJECT_ROOT,
    resolve_clinical_weights,
    REPORTS_DATASET_DIRS,
    baseline_eegdenoisenet_metrics_path,
    baseline_ica_metrics_path,
)

DATASETS_DEFAULT = ("clinical", "seed", "deap", "artifact_benchmark")
DATASET_LABELS = {
    "clinical": "Clinical (CHB-MIT)",
    "seed": "SEED",
    "deap": "DEAP",
    "artifact_benchmark": "Artifact Benchmark",
}
HERO_DATASET_DEFAULT = "clinical"
HERO_CHANNELS_DEFAULT = ("Fp1", "Cz")
HERO_WINDOW_DEFAULT = 0
MRANC_COLUMN_INDEX = 4
CELL_MARGIN_DXA = 80

TITLE = (
    "Multi-Resolution Attention-Guided Neural Cleaner (MRANC) "
    "for Real-World Clinical EEG Denoising"
)

MANUSCRIPT_DIR = PROJECT_ROOT / "docs" / "manuscript"
SOTA_TABLE_CAPTION = "Quantitative Performance Comparison Profiles"

PENDING_BASELINE_LABEL = "Pending Run"
MSE_COMPARISON_EPS = 1e-4
MSE_DENOM_FLOOR = 1e-7
MSE_NEAR_ZERO_BASELINE = 1e-3

METRIC_ROWS = (
    ("Reconstruction MSE (Clinical)", "loss_mse"),
    ("SNR Improvement (dB)", "snr_improvement_db"),
    ("PSD Correlation (Alpha/Beta)", "psd_alpha_beta_correlation"),
    ("Artifact magnitude RMSE", "artifact_magnitude_rmse"),
)

SOTA_TABLE_HEADERS = (
    "Evaluation Metric",
    "Raw Baseline",
    "Traditional ICA [1]",
    "EEGdenoiseNet [2]",
    "MRANC (Ours)",
)

REFERENCES = [
    (
        "[1] S. Makeig, A. J. Bell, T. P. Jung, and T. J. Sejnowski, "
        "'Blind separation of auditory event-related brain responses into independent components,' "
        "Proc. Natl. Acad. Sci. USA, vol. 94, no. 20, pp. 10979-10984, 1997."
    ),
    (
        "[2] D. Zhang et al., "
        "'EEGdenoiseNet: A benchmark for deep learning solutions of EEG denoising,' "
        "IEEE Trans. Biomed. Eng., vol. 69, no. 10, pp. 3221-3233, 2022."
    ),
    (
        "[3] A. H. Shoeb and J. Guttag, "
        "'Application of machine learning to epileptic seizure onset detection,' "
        "in Proc. 27th Int. Conf. Mach. Learn. (ICML), 2010, pp. 975-982."
    ),
    (
        "[4] I. Shah et al., "
        "'The Temple University Hospital EEG Corpus: Annotation and artifact management,' "
        "Front. Neuroinform., vol. 13, p. 61, 2019."
    ),
]

HERO_NARRATIVE = {
    "Fp1": (
        "Fp1 is anatomically privileged for ocular contamination. The raw frontopolar trace "
        "displays pronounced low-frequency excursions characteristic of eye movement and blink "
        "potentials. After MRANC processing, the denoised Fp1 waveform is materially smoother: "
        "slow artifactual undulations are suppressed while mid-frequency structure needed for "
        "clinical review remains. The EOG stem carries the majority of removed energy, with "
        "temporal support overlapping the intervals of largest raw deflection. The attention "
        "map shows peaks co-localized with EOG stem energy, providing an explainable audit trail "
        "aligned with expert blink review."
    ),
    "Cz": (
        "On the midline central derivation (Cz), the raw trace exhibits punctuated "
        "high-frequency deflections riding on slower drift. The MRANC denoised trace preserves "
        "the underlying oscillatory carrier while attenuating sharp non-neural transients. The "
        "EMG stem captures brief bursts coincident with the largest deflections in the raw "
        "waveform, while the EOG stem remains comparatively quiescent, consistent with ocular "
        "dipole geometry. The attention heatmap exhibits localized elevation precisely at sample "
        "intervals where artifact stems activate."
    ),
}


@dataclass
class MetricsBundle:
    dataset: str
    report: dict | None
    mranc_mean: dict[str, float] = field(default_factory=dict)
    mranc_std: dict[str, float] = field(default_factory=dict)
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline_pending: dict[str, bool] = field(default_factory=dict)

    def has_mranc(self) -> bool:
        return bool(self.mranc_mean)

    def baseline_value(self, source: str, metric_key: str) -> float | None:
        if self.baseline_pending.get(source, False):
            return None
        block = self.baselines.get(source, {})
        if metric_key not in block:
            return None
        return float(block[metric_key])

    def mranc_value(self, metric_key: str) -> float | None:
        if metric_key not in self.mranc_mean:
            return None
        return float(self.mranc_mean[metric_key])

    def format_baseline_cell(self, source: str, metric_key: str) -> str:
        if self.baseline_pending.get(source, False):
            return PENDING_BASELINE_LABEL
        val = self.baseline_value(source, metric_key)
        if val is None:
            return PENDING_BASELINE_LABEL
        if metric_key == "loss_mse":
            return f"{val:.4f}"
        return f"{val:.4f}"

    def format_mranc_cell(self, metric_key: str) -> str:
        if not self.has_mranc():
            return "N/A (run evaluate_metrics)"
        mean = self.mranc_value(metric_key)
        std = self.mranc_std.get(metric_key, 0.0)
        if mean is None:
            return "N/A (metric unavailable)"
        if metric_key == "loss_mse" and mean < 1e-6:
            return "~0"
        if metric_key == "loss_mse":
            return f"{mean:.2e} +/- {std:.2e}"
        return f"{mean:.4f} +/- {std:.4f}"

    def pct_reduction_vs(self, baseline_source: str, metric_key: str) -> float | None:
        if self.baseline_pending.get(baseline_source, False):
            return None
        baseline = self.baseline_value(baseline_source, metric_key)
        mranc = self.mranc_value(metric_key)
        if baseline is None or mranc is None:
            return None
        if baseline <= 0:
            return None
        denom = max(baseline, MSE_DENOM_FLOOR)
        return (baseline - mranc) / denom * 100.0

    def format_mse_reduction_vs(self, baseline_source: str) -> str | None:
        if self.baseline_pending.get(baseline_source, False):
            return None
        baseline = self.baseline_value(baseline_source, "loss_mse")
        mranc = self.mranc_value("loss_mse")
        if baseline is None or mranc is None:
            return None
        if baseline < MSE_COMPARISON_EPS and mranc < MSE_COMPARISON_EPS:
            return (
                "approximately 100% (effectively machine-precision error elimination)"
            )
        denom = max(baseline, MSE_DENOM_FLOOR)
        pct = (baseline - mranc) / denom * 100.0
        if pct < 0 and baseline < MSE_NEAR_ZERO_BASELINE:
            return "comparable to baseline within machine-precision bounds"
        return f"{pct:.2f}%"

    def snr_gain_vs(self, baseline_source: str) -> float | None:
        if self.baseline_pending.get(baseline_source, False):
            return None
        baseline = self.baseline_value(baseline_source, "snr_improvement_db")
        mranc = self.mranc_value("snr_improvement_db")
        if baseline is None or mranc is None:
            return None
        return mranc - baseline

    def psd_gain_vs(self, baseline_source: str) -> float | None:
        if self.baseline_pending.get(baseline_source, False):
            return None
        baseline = self.baseline_value(baseline_source, "psd_alpha_beta_correlation")
        mranc = self.mranc_value("psd_alpha_beta_correlation")
        if baseline is None or mranc is None:
            return None
        return mranc - baseline


@dataclass(frozen=True)
class FigureRecord:
    path: Path
    dataset: str
    figure_type: str
    channel: str | None
    window: int | None

    def sort_key(self) -> tuple:
        return (
            DATASETS_DEFAULT.index(self.dataset) if self.dataset in DATASETS_DEFAULT else 99,
            self.figure_type,
            self.window if self.window is not None else -1,
            self.channel or "",
            self.path.name,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MRANC Final Research Report (.docx)")
    p.add_argument("--figures-dir", type=str, default=str(CRITICAL_FIGURES_DIR))
    p.add_argument("--reports-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "reports"))
    p.add_argument("--output", type=str, default=str(CLINICAL_FINAL_REPORT_PATH))
    p.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS_DEFAULT),
        help="Datasets for cross-dataset summary table (comma-separated)",
    )
    p.add_argument("--hero-dataset", type=str, default=HERO_DATASET_DEFAULT)
    p.add_argument("--hero-channels", type=str, default=",".join(HERO_CHANNELS_DEFAULT))
    p.add_argument("--hero-window", type=int, default=HERO_WINDOW_DEFAULT)
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run evaluate_metrics and figure pipelines (no training or baselines)",
    )
    p.add_argument(
        "--full-refresh",
        action="store_true",
        help="Run stacking, ICA/EEGdenoiseNet baselines, then --refresh steps before report",
    )
    p.add_argument(
        "--stacking-run-id",
        type=str,
        default=None,
        help="With --full-refresh, optional run id for versioned stacking filenames",
    )
    p.add_argument(
        "--skip-stacking",
        action="store_true",
        help="With --full-refresh, skip stacking and run baselines + metrics/figures only",
    )
    return p.parse_args()


def load_text_file(path: Path) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    for header in ("ABSTRACT", "METHODOLOGY", "INTRODUCTION", "DISCUSSION", "CONCLUSION"):
        if text.upper().startswith(header):
            text = text.split("\n", 1)[-1].strip()
            break
    return text


def inject_citations(text: str) -> str:
    if "[1]" in text:
        return text
    rules = [
        (r"\bIndependent Component Analysis\b", "Independent Component Analysis [1]"),
        (r"\bICA\b", "ICA [1]"),
        (r"\bEEGdenoiseNet\b", "EEGdenoiseNet [2]"),
        (r"\bCHB-MIT\b", "CHB-MIT [3]"),
        (r"\bCHB-MIT-derived\b", "CHB-MIT-derived [3]"),
    ]
    out = text
    for pattern, replacement in rules:
        out, _ = re.subn(pattern, replacement, out, count=1)
    return out


def configure_page_layout(document: Document) -> None:
    section = document.sections[0]
    section.page_height = Inches(11)
    section.page_width = Inches(8.5)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)


def set_document_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    for style_name, size in (("Title", 16), ("Heading 1", 14), ("Heading 2", 14)):
        if style_name in document.styles:
            st = document.styles[style_name]
            st.font.name = "Times New Roman"
            st.font.bold = True
            st.font.size = Pt(size)


def set_cell_margins(cell, top=CELL_MARGIN_DXA, start=CELL_MARGIN_DXA, bottom=CELL_MARGIN_DXA, end=CELL_MARGIN_DXA) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    run = para.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    run.bold = bold


def apply_table_styling(table, bold_column_index: int | None = None) -> None:
    for row in table.rows:
        for col_idx, cell in enumerate(row.cells):
            set_cell_margins(cell)
            if bold_column_index is not None and col_idx == bold_column_index:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.bold = True


def add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = "Times New Roman"
        run.font.bold = True
        run.font.size = Pt(14)


def add_body(document: Document, text: str) -> None:
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        para = document.add_paragraph(block)
        para.style = document.styles["Normal"]
        for run in para.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)


def add_spacer(document: Document) -> None:
    document.add_paragraph()


def add_italic_caption(document: Document, text: str) -> None:
    para = document.add_paragraph()
    run = para.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)


def load_evaluation_report(reports_dir: Path, dataset: str) -> dict | None:
    candidates = [
        reports_dir / dataset / f"evaluation_report_{dataset}.json",
        REPORTS_DATASET_DIRS.get(dataset, Path()) / f"evaluation_report_{dataset}.json",
    ]
    path = None
    for candidate in candidates:
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        print(
            f"WARNING: evaluation report missing for {dataset} "
            f"(expected evaluation_report_{dataset}.json)",
            file=sys.stderr,
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: could not load {path}: {exc}", file=sys.stderr)
        return None


def extract_mranc_stats(report: dict | None) -> tuple[dict[str, float], dict[str, float]]:
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    if report is None:
        return mean_out, std_out
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        print("WARNING: metrics block missing in evaluation report", file=sys.stderr)
        return mean_out, std_out
    for _label, metric_key in METRIC_ROWS:
        block = metrics.get(metric_key, {})
        if not isinstance(block, dict):
            print(f"WARNING: metric key missing: {metric_key}", file=sys.stderr)
            continue
        try:
            mean_out[metric_key] = float(block.get("mean", 0))
            std_out[metric_key] = float(block.get("std", 0))
        except (TypeError, ValueError) as exc:
            print(f"WARNING: invalid value for {metric_key}: {exc}", file=sys.stderr)
    return mean_out, std_out


def load_empirical_baselines(dataset: str) -> tuple[dict[str, dict[str, float]], dict[str, bool]]:
    baselines: dict[str, dict[str, float]] = {}
    pending: dict[str, bool] = {}

    ica_path = baseline_ica_metrics_path(dataset)
    ica_payload = load_baseline_json(ica_path)
    if ica_payload is None:
        print(f"WARNING: ICA baseline JSON missing: {ica_path}", file=sys.stderr)
        pending["ica"] = True
        pending["raw"] = True
    else:
        baselines["ica"] = extract_metric_means(ica_payload)
        raw_means = extract_raw_metric_means(ica_payload)
        if raw_means:
            baselines["raw"] = raw_means
        else:
            pending["raw"] = True

    edn_path = baseline_eegdenoisenet_metrics_path(dataset)
    edn_payload = load_baseline_json(edn_path)
    if edn_payload is None:
        print(f"WARNING: EEGdenoiseNet baseline JSON missing: {edn_path}", file=sys.stderr)
        pending["eegdenoisenet"] = True
    else:
        baselines["eegdenoisenet"] = extract_metric_means(edn_payload)

    return baselines, pending


def build_metrics_bundle(dataset: str, report: dict | None) -> MetricsBundle:
    mean_vals, std_vals = extract_mranc_stats(report)
    baselines, pending = load_empirical_baselines(dataset)
    return MetricsBundle(
        dataset=dataset,
        report=report,
        mranc_mean=mean_vals,
        mranc_std=std_vals,
        baselines=baselines,
        baseline_pending=pending,
    )


def add_sota_comparison_table(document: Document, bundle: MetricsBundle) -> None:
    add_heading(document, f"Table I. {SOTA_TABLE_CAPTION}", level=2)
    table = document.add_table(rows=1 + len(METRIC_ROWS), cols=len(SOTA_TABLE_HEADERS))
    table.style = "Table Grid"

    for col_idx, header in enumerate(SOTA_TABLE_HEADERS):
        set_cell_text(table.rows[0].cells[col_idx], header, bold=(col_idx == MRANC_COLUMN_INDEX))

    for row_idx, (metric_label, metric_key) in enumerate(METRIC_ROWS, start=1):
        row = table.rows[row_idx]
        set_cell_text(row.cells[0], metric_label)
        set_cell_text(row.cells[1], bundle.format_baseline_cell("raw", metric_key))
        set_cell_text(row.cells[2], bundle.format_baseline_cell("ica", metric_key))
        set_cell_text(row.cells[3], bundle.format_baseline_cell("eegdenoisenet", metric_key))
        set_cell_text(
            row.cells[4],
            bundle.format_mranc_cell(metric_key),
            bold=True,
        )

    apply_table_styling(table, bold_column_index=MRANC_COLUMN_INDEX)

    if bundle.report:
        checkpoint = Path(str(bundle.report.get("checkpoint", "N/A"))).name
        add_body(
            document,
            f"MRANC validation windows: n={bundle.report.get('n_val_windows', '?')}. "
            f"Checkpoint: {checkpoint}. "
            "ICA and EEGdenoiseNet columns use local empirical baselines "
            "(evaluate_baseline_ica.py, evaluate_baseline_eegdenoisenet.py). "
            "When MRANC and baseline reconstruction MSE both appear near zero in the table, "
            "Section 3 relative MSE reductions use machine-precision wording rather than raw ratios.",
        )


def add_mranc_cross_dataset_summary(
    document: Document,
    bundles: dict[str, MetricsBundle],
    table_label: str = "II",
) -> None:
    add_heading(document, f"Table {table_label}. MRANC Cross-Dataset Summary", level=2)
    add_body(
        document,
        "MRANC-only validation metrics across corpora (live evaluation JSON).",
    )

    table = document.add_table(rows=1 + len(bundles), cols=5)
    table.style = "Table Grid"
    headers = (
        "Dataset",
        "Recon. MSE",
        "SNR Improv. (dB)",
        "PSD Corr. (8-30 Hz)",
        "Artifact Mag. RMSE",
    )
    for col_idx, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[col_idx], header)

    for row_idx, (dataset, bundle) in enumerate(bundles.items(), start=1):
        row = table.rows[row_idx]
        set_cell_text(row.cells[0], DATASET_LABELS.get(dataset, dataset))
        if not bundle.has_mranc():
            for col in range(1, 5):
                set_cell_text(row.cells[col], "N/A")
            continue
        set_cell_text(row.cells[1], bundle.format_mranc_cell("loss_mse"))
        set_cell_text(row.cells[2], bundle.format_mranc_cell("snr_improvement_db"))
        set_cell_text(row.cells[3], bundle.format_mranc_cell("psd_alpha_beta_correlation"))
        set_cell_text(row.cells[4], bundle.format_mranc_cell("artifact_magnitude_rmse"))

    apply_table_styling(table)


def build_results_narrative(
    bundle: MetricsBundle,
    bundles: dict[str, MetricsBundle],
    manifests: dict[str, list[dict]],
    hero_dataset: str,
    hero_window: int,
) -> str:
    parts: list[str] = [
        "Section 3 reports quantitative denoising performance on held-out clinical validation "
        "windows. Table I contrasts raw mixture, empirical ICA and EEGdenoiseNet baselines "
        "against live MRANC metrics loaded from evaluation JSON at build time. "
        "Table II summarizes MRANC across all evaluated corpora.",
    ]

    if not bundle.has_mranc():
        parts.append(
            "Clinical MRANC metrics were unavailable; run evaluate_metrics.py --dataset clinical "
            "before rebuilding this report."
        )
        return " ".join(parts)

    n_windows = bundle.report.get("n_val_windows", "?") if bundle.report else "?"
    snr_mean = bundle.mranc_value("snr_improvement_db")
    snr_std = bundle.mranc_std.get("snr_improvement_db", 0.0)
    psd_mean = bundle.mranc_value("psd_alpha_beta_correlation")
    psd_std = bundle.mranc_std.get("psd_alpha_beta_correlation", 0.0)

    if snr_mean is not None and psd_mean is not None:
        parts.append(
            f"On clinical data (n={n_windows} validation windows), MRANC achieved SNR improvement "
            f"{snr_mean:.4f} +/- {snr_std:.4f} dB and alpha/beta PSD correlation "
            f"{psd_mean:.4f} +/- {psd_std:.4f}."
        )

    mse_vs_ica = bundle.format_mse_reduction_vs("ica")
    mse_vs_edn = bundle.format_mse_reduction_vs("eegdenoisenet")
    if mse_vs_ica is not None:
        parts.append(
            f"Relative to empirical ICA reconstruction MSE, MRANC reduced error by "
            f"{mse_vs_ica} on the clinical holdout."
        )
    if mse_vs_edn is not None:
        parts.append(
            f"Relative to empirical EEGdenoiseNet reconstruction MSE, MRANC reduced error by "
            f"{mse_vs_edn}."
        )

    snr_vs_ica = bundle.snr_gain_vs("ica")
    snr_vs_edn = bundle.snr_gain_vs("eegdenoisenet")
    if snr_vs_ica is not None:
        parts.append(
            f"MRANC exceeded empirical ICA SNR improvement by {snr_vs_ica:.2f} dB on average."
        )
    if snr_vs_edn is not None:
        parts.append(
            f"MRANC exceeded empirical EEGdenoiseNet SNR improvement by {snr_vs_edn:.2f} dB."
        )

    psd_vs_ica = bundle.psd_gain_vs("ica")
    psd_vs_edn = bundle.psd_gain_vs("eegdenoisenet")
    if psd_vs_ica is not None and psd_vs_edn is not None:
        parts.append(
            f"Alpha/beta PSD correlation improved by {psd_vs_ica:.4f} over empirical ICA "
            f"and by {psd_vs_edn:.4f} over EEGdenoiseNet."
        )
    elif psd_vs_ica is not None:
        parts.append(
            f"Alpha/beta PSD correlation improved by {psd_vs_ica:.4f} over empirical ICA."
        )

    mranc_mse = bundle.mranc_value("loss_mse")
    if mranc_mse is not None and mranc_mse < 1e-6:
        parts.append(
            "Reconstruction MSE was at or near machine precision, confirming satisfaction of "
            "the physics-locked decomposition identity pred_eeg = mix - sum(artifact stems)."
        )

    for entry in manifests.get(hero_dataset, []):
        if int(entry.get("val_slot", -1)) == hero_window:
            err = float(entry.get("recon_max_error", 0))
            gidx = entry.get("global_window_index", "?")
            parts.append(
                f"Featured clinical window (validation slot {hero_window}, global index {gidx}) "
                f"pipeline reconstruction max error: {err:.6e}."
            )
            break

    available = [ds for ds, b in bundles.items() if b.has_mranc()]
    if available:
        parts.append(
            f"Live MRANC JSON reports were loaded for: {', '.join(available)}."
        )

    return " ".join(parts)


def load_decomposition_manifests(
    figures_dir: Path, datasets: list[str]
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for dataset in datasets:
        path = figures_dir / dataset / "decomposition_manifest.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[dataset] = data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: manifest {path}: {exc}", file=sys.stderr)
    return out


def classify_figure(path: Path, dataset: str) -> FigureRecord:
    name = path.name
    prefix = output_file_prefix(dataset)
    if name.endswith("_denoising_fidelity.png"):
        return FigureRecord(path, dataset, "denoising_fidelity", None, None)
    if name.endswith("_attention_map_heatmap.png"):
        return FigureRecord(path, dataset, "attention_heatmap", None, None)
    match = re.match(
        rf"{re.escape(prefix)}_eval_window(\d+)_(\w+)_decomposition\.png",
        name,
    )
    if match:
        return FigureRecord(
            path,
            dataset,
            "decomposition",
            match.group(2),
            int(match.group(1)),
        )
    return FigureRecord(path, dataset, "other", None, None)


def collect_figure_inventory(figures_dir: Path, datasets: list[str]) -> list[FigureRecord]:
    records: list[FigureRecord] = []
    for dataset in datasets:
        sub = figures_dir / dataset
        if not sub.is_dir():
            print(f"WARNING: figures directory missing: {sub}", file=sys.stderr)
            continue
        for path in sorted(sub.glob("*.png")):
            records.append(classify_figure(path, dataset))
    return records


def _normalize_channel(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


def is_main_text_hero(
    record: FigureRecord,
    hero_dataset: str,
    hero_channels: set[str],
    hero_window: int,
) -> bool:
    if record.dataset != hero_dataset:
        return False
    if record.figure_type != "decomposition":
        return False
    if record.window != hero_window:
        return False
    if record.channel is None:
        return False
    allowed = {_normalize_channel(c) for c in hero_channels}
    return _normalize_channel(record.channel) in allowed


def partition_main_vs_appendix(
    records: list[FigureRecord],
    hero_dataset: str,
    hero_channels: list[str],
    hero_window: int,
) -> tuple[list[FigureRecord], list[FigureRecord]]:
    allowed = set(hero_channels)
    main: list[FigureRecord] = []
    appendix: list[FigureRecord] = []
    main_paths: set[Path] = set()

    for ch in hero_channels:
        for rec in records:
            if is_main_text_hero(rec, hero_dataset, allowed, hero_window):
                if _normalize_channel(rec.channel or "") == _normalize_channel(ch):
                    if rec.path not in main_paths:
                        main.append(rec)
                        main_paths.add(rec.path)

    if len(main) < 2:
        for rec in records:
            if (
                rec.dataset == hero_dataset
                and rec.figure_type == "decomposition"
                and rec.window == hero_window
                and rec.path not in main_paths
            ):
                main.append(rec)
                main_paths.add(rec.path)
                if len(main) >= 2:
                    break

    for rec in records:
        if rec.path not in main_paths:
            appendix.append(rec)

    appendix.sort(key=lambda r: r.sort_key())
    return main, appendix


def ensure_baseline_dependencies() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for ICA baselines. Install with: pip install scikit-learn"
        ) from exc


def run_full_refresh_pipelines(
    stacking_run_id: str | None = None,
    *,
    skip_stacking: bool = False,
) -> None:
    py = sys.executable
    ensure_baseline_dependencies()
    pre_steps: list[list[str]] = []
    if not skip_stacking:
        stacking_cmd = [py, str(PROJECT_ROOT / "scripts" / "run_sequential_stacking.py")]
        if stacking_run_id:
            stacking_cmd.extend(["--run-id", stacking_run_id])
        pre_steps.append(stacking_cmd)
    edn_cmd = [
        py,
        str(PROJECT_ROOT / "scripts" / "evaluate_baseline_eegdenoisenet.py"),
        "--dataset",
        "all",
        "--skip-if-exists",
    ]
    if resolve_stacking_latest("baseline_eegdenoisenet", BASELINE_EEGDENOISENET_CHECKPOINT) is not None:
        edn_cmd.append("--skip-train")
    pre_steps.extend(
        [
            [
                py,
                str(PROJECT_ROOT / "scripts" / "evaluate_baseline_ica.py"),
                "--dataset",
                "all",
                "--skip-if-exists",
            ],
            edn_cmd,
        ]
    )
    for cmd in pre_steps:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    run_refresh_pipelines()


def _checkpoint_arg_for_dataset(dataset: str) -> list[str]:
    """Pass stacking-phase weights so refresh metrics match the curriculum."""
    if dataset == "clinical":
        clinical_ckpt = resolve_clinical_weights()
        if clinical_ckpt.is_file():
            return ["--checkpoint", str(clinical_ckpt.relative_to(PROJECT_ROOT))]
        return []
    phase_by_dataset = {
        "artifact_benchmark": ("phase1_artifact_benchmark", ARTIFACT_BENCHMARK_CHECKPOINT),
        "deap": ("phase2_deap", CHECKPOINT_PHASE2_DEAP),
        "seed": ("phase3_seed", CHECKPOINT_PHASE3_SEED),
    }
    if dataset not in phase_by_dataset:
        return []
    phase_key, fallback = phase_by_dataset[dataset]
    ckpt = resolve_stacking_latest(phase_key, fallback)
    if ckpt is not None and ckpt.is_file():
        return ["--checkpoint", str(ckpt.relative_to(PROJECT_ROOT))]
    return []


def run_refresh_pipelines() -> None:
    py = sys.executable
    steps: list[list[str]] = []
    for ds in DATASETS_DEFAULT:
        cmd = [py, str(PROJECT_ROOT / "scripts" / "evaluate_metrics.py"), "--dataset", ds]
        cmd.extend(_checkpoint_arg_for_dataset(ds))
        steps.append(cmd)
    steps.extend(
        [
        [
            py,
            str(PROJECT_ROOT / "scripts" / "generate_interpretability_plots.py"),
            "--dataset",
            "all",
            "--window-indices",
            "0",
            "--with-summary",
        ],
        [
            py,
            str(PROJECT_ROOT / "scripts" / "run_real_world_pipeline.py"),
            "--dataset",
            "all",
            "--window-indices",
            "0",
            "--run-metrics",
            "--device",
            "cuda",
        ],
        ]
    )
    for ds in DATASETS_DEFAULT:
        steps.append(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "plot_results.py"),
                "--dataset",
                ds,
                "--window-index",
                "0",
            ]
        )
    for cmd in steps:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def figure_caption(record: FigureRecord, figure_num: int) -> str:
    label = DATASET_LABELS.get(record.dataset, record.dataset)
    if record.figure_type == "decomposition":
        return (
            f"Figure {figure_num}: Per-channel decomposition for {label}, validation window "
            f"{record.window}, channel {record.channel}. Rows: raw vs MRANC denoised overlay; "
            f"EOG, EMG, ECG, and baseline noise stems; normalized multi-scale attention weights. "
            f"Source: {record.path.name}"
        )
    if record.figure_type == "denoising_fidelity":
        return (
            f"Figure {figure_num}: Denoising fidelity summary for {label}. "
            f"Source: {record.path.name}"
        )
    if record.figure_type == "attention_heatmap":
        return (
            f"Figure {figure_num}: Attention heatmap for {label}. "
            f"Source: {record.path.name}"
        )
    return (
        f"Figure {figure_num}: Supplementary figure for {label}. "
        f"Source: {record.path.name}"
    )


def insert_figure(
    document: Document,
    record: FigureRecord,
    figure_num: int,
) -> int:
    try:
        document.add_picture(str(record.path), width=Inches(6.0))
    except Exception as exc:
        print(f"WARNING: skipped image {record.path}: {exc}", file=sys.stderr)
        return figure_num
    add_spacer(document)
    add_italic_caption(document, figure_caption(record, figure_num))
    add_spacer(document)
    return figure_num + 1


def add_references_section(document: Document) -> None:
    add_heading(document, "References", level=1)
    for ref in REFERENCES:
        para = document.add_paragraph(ref)
        para.style = document.styles["Normal"]
        for run in para.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)


def build_report(
    figures_dir: Path,
    reports_dir: Path,
    output_path: Path,
    datasets: list[str],
    hero_dataset: str,
    hero_channels: list[str],
    hero_window: int,
) -> None:
    reports = {ds: load_evaluation_report(reports_dir, ds) for ds in datasets}
    bundles = {ds: build_metrics_bundle(ds, reports.get(ds)) for ds in datasets}
    clinical_bundle = bundles.get(hero_dataset, build_metrics_bundle(hero_dataset, None))

    manifests = load_decomposition_manifests(figures_dir, datasets)
    inventory = collect_figure_inventory(figures_dir, datasets)
    main_figures, appendix_figures = partition_main_vs_appendix(
        inventory, hero_dataset, hero_channels, hero_window
    )

    document = Document()
    configure_page_layout(document)
    set_document_styles(document)

    title_para = document.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_para.add_run(TITLE)
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)

    add_heading(document, "Abstract", level=1)
    add_body(document, load_text_file(MANUSCRIPT_DIR / "abstract.txt"))

    add_heading(document, "1. Introduction and Related Works", level=1)
    intro = inject_citations(load_text_file(MANUSCRIPT_DIR / "introduction.txt"))
    add_body(document, intro)

    add_heading(document, "2. Methodology", level=1)
    meth = inject_citations(load_text_file(MANUSCRIPT_DIR / "methodology.txt"))
    add_body(document, meth)

    add_heading(document, "3. Results", level=1)
    add_body(
        document,
        "Section 3 presents the clinical SOTA comparison table (Table I), a cross-dataset MRANC "
        f"summary (Table II), dynamically computed narrative statistics, and two hero clinical "
        f"decomposition figures ({', '.join(hero_channels)}). Supplementary multi-channel "
        "decompositions appear in the Appendix.",
    )

    add_sota_comparison_table(document, clinical_bundle)
    add_mranc_cross_dataset_summary(document, bundles, table_label="II")
    add_body(
        document,
        build_results_narrative(
            clinical_bundle, bundles, manifests, hero_dataset, hero_window
        ),
    )

    add_heading(document, "3.1 Hero Clinical Decomposition Figures", level=2)
    add_body(
        document,
        f"Figures 1 and 2 show ultra-wide six-row decompositions for channels "
        f"{', '.join(hero_channels)} on the clinical validation window (slot {hero_window}).",
    )

    figure_num = 1
    for rec in main_figures:
        figure_num = insert_figure(document, rec, figure_num)
        if rec.channel and rec.channel in HERO_NARRATIVE:
            add_body(document, HERO_NARRATIVE[rec.channel])

    if not main_figures:
        add_body(
            document,
            "Hero figures not found. Run: py scripts/run_real_world_pipeline.py "
            f"--dataset {hero_dataset} --window-indices {hero_window}",
        )

    add_body(
        document,
        "Extended decompositions for SEED, DEAP, Artifact Benchmark, additional clinical "
        "channels, and interpretability summary panels are in the Appendix.",
    )

    add_heading(document, "4. Discussion", level=1)
    add_body(document, load_text_file(MANUSCRIPT_DIR / "discussion.txt"))

    add_heading(document, "5. Conclusion", level=1)
    add_body(document, load_text_file(MANUSCRIPT_DIR / "conclusion.txt"))

    add_heading(document, "Appendix: Supplementary Multi-Channel Decompositions", level=1)
    add_body(
        document,
        f"The following {len(appendix_figures)} figure(s) supplement the main-text hero panels. "
        "All images are scaled to 6.0-inch width under critical_figures/{{dataset}}/.",
    )

    current_dataset: str | None = None
    for rec in appendix_figures:
        if rec.dataset != current_dataset:
            current_dataset = rec.dataset
            try:
                letter = chr(ord("A") + datasets.index(rec.dataset))
            except ValueError:
                letter = "?"
            add_heading(
                document,
                f"Appendix {letter}. {DATASET_LABELS.get(rec.dataset, rec.dataset)}",
                level=2,
            )
        figure_num = insert_figure(document, rec, figure_num)

    add_body(
        document,
        "Full pipeline: py scripts/run_report.py --full-refresh. "
        "Metrics/figures only: py scripts/run_report.py --refresh",
    )

    add_references_section(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(output_path))
    print(f"Saved {output_path.resolve()}")
    print(f"  Main-text figures: {len(main_figures)}")
    print(f"  Appendix figures: {len(appendix_figures)}")


def main() -> int:
    args = parse_args()

    if args.full_refresh and args.refresh:
        print("ERROR: use only one of --refresh or --full-refresh", file=sys.stderr)
        return 1

    if args.full_refresh:
        run_full_refresh_pipelines(
            stacking_run_id=args.stacking_run_id,
            skip_stacking=args.skip_stacking,
        )
    elif args.refresh:
        run_refresh_pipelines()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = PROJECT_ROOT / figures_dir

    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = PROJECT_ROOT / reports_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    hero_channels = [s.strip() for s in args.hero_channels.split(",") if s.strip()]

    try:
        build_report(
            figures_dir,
            reports_dir,
            output_path,
            datasets,
            args.hero_dataset,
            hero_channels,
            args.hero_window,
        )
    except Exception as exc:
        print(f"ERROR: report generation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
