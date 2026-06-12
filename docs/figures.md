# Critical figures and research report

Generated artifacts are **local only** (gitignored):

- Figure PNGs → `critical_figures/{dataset}/`
- Metrics JSON → `outputs/reports/{dataset}/evaluation_report_{dataset}.json`
- Word report → `outputs/reports/clinical/MRANC_Final_Research_Report.docx`
- LaTeX export → `outputs/reports/latex/MRANC_Final_Research_Report.tex`

Source prose for the report (`docs/manuscript/*.txt`) **is** in the repository so `run_report.py` works after clone.

See [getting-started.md](getting-started.md) for the ordered workflow: train → `evaluate_metrics.py` → `plot_results.py` → `run_report.py`.

## Scripts

| Script | Purpose | Key outputs |
|--------|---------|-------------|
| `generate_interpretability_plots.py` | Summary panels per dataset | `{prefix}_denoising_fidelity.png`, `{prefix}_attention_map_heatmap.png` |
| `run_real_world_pipeline.py` | Per-channel 6-row decompositions + optional metrics | `{prefix}_eval_window{N}_{Ch}_decomposition.png`, `decomposition_manifest.json` |
| `evaluate_metrics.py` | Validation metrics only | `outputs/reports/{dataset}/evaluation_report_{dataset}.json` |
| `plot_results.py` | Decomposition PNGs (default) | `critical_figures/{dataset}/` |
| `run_sequential_stacking.py` | 4-phase MRANC stacking | `weights/mranc_final_attention.pth` |
| `evaluate_baseline_ica.py` | FastICA empirical baseline | `outputs/reports/{dataset}/baseline_ica_metrics.json` |
| `evaluate_baseline_eegdenoisenet.py` | FCN autoencoder baseline | `outputs/reports/{dataset}/baseline_eegdenoisenet_metrics.json` |
| `run_report.py` | Word research report | `outputs/reports/clinical/MRANC_Final_Research_Report.docx` |
| `transform_report.py` | Word to IEEEtran LaTeX | `outputs/reports/latex/MRANC_Final_Research_Report.tex` |

`prefix` is `clinical` for CHB-MIT clinical runs, otherwise the dataset name.

## Output paths (no overlap)

| Artifact | Path | Produced by |
|----------|------|-------------|
| Word manuscript | `outputs/reports/clinical/MRANC_Final_Research_Report.docx` | `run_report.py` |
| LaTeX manuscript | `outputs/reports/latex/MRANC_Final_Research_Report.tex` | `transform_report.py` |
| LaTeX image cache | `outputs/reports/latex/latex_assets/` | `transform_report.py` |

`transform_report.py` never writes to the `.docx` path. Use `--output` only with a `.tex` file.

## Research report structure (`run_report.py`)

| Section | Content |
|---------|---------|
| Title | MRANC multi-dataset denoising (centered, 16pt) |
| Abstract | `docs/manuscript/abstract.txt` |
| 1. Introduction and Related Works | `introduction.txt` + IEEE citations [1]-[3] |
| 2. Methodology | `methodology.txt` + citations |
| 3. Results | Table I (clinical SOTA), Table II (MRANC all datasets), dynamic narrative |
| 3.1 Hero figures | Clinical Fp1 and Cz decomposition only (validation window 0) |
| 4. Discussion | `discussion.txt` |
| 5. Conclusion | `conclusion.txt` |
| Appendix: Supplementary Multi-Channel Decompositions | All other PNGs (O1, summaries, seed/deap/benchmark) |
| References | IEEE bibliography [1]-[3] |

**Table I** columns: Evaluation Metric, Raw Baseline, Traditional ICA [1], EEGdenoiseNet [2], MRANC (Ours).
MRANC cells load from evaluation JSON; ICA/EEGdenoiseNet/raw load from baseline JSON manifests.
Missing baseline files show **Pending Run** in Table I. Results paragraphs compute percentage
and dB deltas from the same live values.

Layout: 1.0 inch margins, Times New Roman, 6.0 inch figure width, padded tables, bold MRANC column.

## LaTeX transform (`transform_report.py`)

Reads the Word report and emits:

- IEEEtran preamble (`\documentclass[journal,twocolumn]{IEEEtran}`)
- `\begin{abstract}`, `\section{}`, `\subsection{}`
- `table*` environments with `booktabs` for both tables
- `figure*` environments (`\includegraphics[width=\linewidth]{...}`) for all figures
- `\begin{thebibliography}` from References

Figure paths prefer `critical_figures/{dataset}/` via `Source: filename.png` in captions.

Default input (first file found):

1. `outputs/reports/clinical/MRANC_Final_Research_Report.docx`
2. `outputs/reports/MRANC_Final_Research_Report.docx`

## End-to-end workflow

```bash
pip install python-docx

# Full refresh: train stack, baselines, metrics, figures, Word, LaTeX
py scripts/run_report.py --full-refresh
py scripts/transform_report.py

# Metrics and figures only
py scripts/run_report.py --refresh
py scripts/transform_report.py
```

Step by step:

```bash
# 1. Metrics for all datasets
py scripts/evaluate_metrics.py --dataset clinical
py scripts/evaluate_metrics.py --dataset seed
py scripts/evaluate_metrics.py --dataset deap
py scripts/evaluate_metrics.py --dataset artifact_benchmark

# 2. Figures (summary + per-channel decompositions)
py scripts/generate_interpretability_plots.py --dataset all --window-indices 0 --with-summary
py scripts/run_real_world_pipeline.py --dataset all --window-indices 0 --run-metrics

# 3. Word report
py scripts/run_report.py

# 4. LaTeX (does not modify the .docx)
py scripts/transform_report.py
```

Compile LaTeX (from repo root, with `pdflatex` on PATH):

```bash
cd outputs/reports/latex
pdflatex MRANC_Final_Research_Report.tex
```

Flags: see [commands.md](commands.md#research-report-word) and
[commands.md](commands.md#latex-transform-full-system-report).
