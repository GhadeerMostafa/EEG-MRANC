# MRANC Project Features and Command Reference

**Developed by: Ghadeer Mostafa**  
**Repository:** [github.com/GhadeerMostafa/EEG-MRANC](https://github.com/GhadeerMostafa/EEG-MRANC)

This document is the comprehensive capability map for the MRANC repository. Every major feature includes the command needed to run it from the **repository root**. Commands use Windows `py`; on Linux or macOS, use `python3` instead.

---

## What this project can do

MRANC is an end-to-end research pipeline for 32-channel EEG artifact decomposition and clinical denoising. From a single repository you can:

| Capability | Summary |
|------------|---------|
| **Data ingestion** | Convert DEAP, SEED, CHB-MIT clinical, and SSVEP artifact-benchmark corpora into unified `.npy` tensors |
| **Deep learning training** | Train MRANC on one dataset or run the full 4-phase Sequential Knowledge-Stacking curriculum |
| **Clinical adaptation** | Fine-tune a zero-initialized Multi-Scale Attention adapter on real CHB-MIT windows |
| **Quantitative evaluation** | Compute SNR improvement, PSD correlation, reconstruction MSE, and artifact RMSE |
| **Baseline comparison** | Benchmark against empirical ICA and EEGdenoiseNet on the same validation splits |
| **Publication figures** | Generate ultra-wide 6-row decomposition PNGs, attention heatmaps, and denoising fidelity panels |
| **Research paper generation** | Build a Word report from live metrics and figures, export IEEEtran LaTeX, and compile a PDF manuscript |
| **Live report refresh** | Re-run metrics and figures after any model or data change, then regenerate the paper automatically |

---

## 0. Install all dependencies

### Python packages

```bash
pip install -r requirements.txt
```

### CUDA PyTorch (Windows, offline wheels)

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### LaTeX / pdflatex (required for PDF manuscript)

`pdflatex` is not a Python package. Install via the LaTeX requirements file:

```powershell
.\scripts\install_latex_local.ps1
```

Or manually:

```powershell
winget install MiKTeX.MiKTeX --accept-package-agreements --accept-source-agreements
```

Verify:

```powershell
py scripts/check_dependencies.py
```

### Create folder structure

```bash
py src/setup_folders.py
py scripts/migrate_project_layout.py --apply   # if upgrading from pre-v2 layout
```

---

## 1. Data preparation

Convert raw EEG into processed `(N_windows, 32, T)` float32 arrays.

| Dataset | Place raw data | Command | Output folder |
|---------|----------------|---------|---------------|
| DEAP | `data/raw/deap/data_preprocessed_python/*.dat` | `py scripts/deap/convert_batch_2.py` | `data/processed/deap/` |
| SEED | `data/raw/seed/` or `.seeda_workspace/seeda.zip` | `py scripts/seed/extract_and_convert_seed.py` | `data/processed/seed/` |
| Clinical (CHB-MIT) | `data/raw/clinical/*.edf` | `py scripts/clinical/download_clinical.py` then `py scripts/clinical/preprocess_clinical.py` | `data/processed/clinical/` |
| Artifact benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` then `py scripts/preprocess_artifact_benchmark.py` | `data/processed/artifact_benchmark/` |

Validate DEAP conversion:

```bash
py scripts/validate_processed_data.py
```

Inspect SEED archive structure:

```bash
py scripts/seed/inspect_seed_structure.py
```

---

## 2. Model training

### Single-dataset training

Train MRANC on one corpus with configurable epochs, batch size, and resume weights.

```bash
py scripts/train.py --dataset deap --storage cuda --epochs 20 --batch-size 128
py scripts/train.py --dataset seed --storage cuda --epochs 20 --batch-size 64
py scripts/train.py --dataset clinical --storage cuda --epochs 50 --batch-size 32 --resume_weights checkpoints/best_mranc.pth
py scripts/train.py --dataset artifact_benchmark --storage cuda --epochs 50 --batch_size 32
```

**Output:** versioned `.pth` checkpoints under `artifacts/models/checkpoints/` and `artifacts/models/weights/`.

### Full 4-phase Sequential Knowledge-Stacking (recommended)

Artifact benchmark pretrain, DEAP transfer, SEED transfer, then clinical attention adapter.

```bash
py scripts/run_sequential_stacking.py
```

Preview commands without running:

```bash
py scripts/run_sequential_stacking.py --dry-run
```

**Output:** `artifacts/models/checkpoints/stacking_latest.json` records the latest weight path for each phase.

### Clinical attention adapter only

```bash
py scripts/train_attention_adapter.py --resume-weights checkpoints/best_mranc_phase3_seed.pth
```

---

## 3. Evaluation and baselines

### MRANC metrics (SNR, PSD, reconstruction MSE, artifact RMSE)

```bash
py scripts/evaluate_metrics.py --dataset clinical
py scripts/evaluate_metrics.py --dataset seed
py scripts/evaluate_metrics.py --dataset deap
py scripts/evaluate_metrics.py --dataset artifact_benchmark
```

**Output:** `artifacts/reports/{dataset}/evaluation_report_{dataset}.json`

### Empirical ICA baseline

```bash
py scripts/evaluate_baseline_ica.py --dataset all
```

**Output:** `artifacts/reports/{dataset}/baseline_ica_metrics.json`

### EEGdenoiseNet baseline

```bash
py scripts/evaluate_baseline_eegdenoisenet.py --dataset all
```

**Output:** `artifacts/reports/{dataset}/baseline_eegdenoisenet_metrics.json`

### Aggregate results across datasets

```bash
py scripts/aggregate_results.py
```

---

## 4. Visualization and interpretability

### Per-dataset decomposition figures (default: ultra-wide 6-row PNGs)

```bash
py scripts/plot_results.py --dataset clinical
py scripts/plot_results.py --dataset seed --window-index 0
py scripts/plot_results.py --dataset deap --legacy-spatial-rms
```

**Output:** `artifacts/figures/critical/{dataset}/` (default) or `artifacts/figures/legacy/` (legacy mode)

### Interpretability summary panels (fidelity + attention heatmap)

```bash
py scripts/generate_interpretability_plots.py --dataset all
py scripts/generate_interpretability_plots.py --dataset clinical --window-index 0
```

**Output:**
- `artifacts/figures/critical/{dataset}/{prefix}_denoising_fidelity.png`
- `artifacts/figures/critical/{dataset}/{prefix}_attention_map_heatmap.png`

### Full real-world pipeline (metrics + optional summary figures + per-channel decompositions)

```bash
py scripts/run_real_world_pipeline.py --dataset all --window-indices 0 --run-metrics --with-summary-figures --device cuda
```

**Output:** metrics JSON, summary figures, and per-channel decomposition PNGs under `artifacts/figures/critical/{dataset}/`

---

## 5. Research paper generation (Word, LaTeX, PDF)

The pipeline can produce a complete research paper that updates automatically whenever you re-run metrics, regenerate figures, or retrain the model.

### Step A: Build the Word report

Uses live evaluation JSON, `artifacts/figures/critical/`, baseline manifests, and prose from `docs/manuscript/*.txt`.

```bash
py scripts/run_report.py
```

**Output:** `artifacts/reports/clinical/MRANC_Final_Research_Report.docx`

### Step B: Refresh metrics and figures, then rebuild the report

Run this after any model, checkpoint, or evaluation change:

```bash
py scripts/run_report.py --refresh
```

This re-runs metrics, figure generation, and Word report compilation in one step.

### Step C: Full baseline extraction (retrain + baselines + metrics + figures + report)

Long GPU job that rebuilds everything from scratch:

```bash
py scripts/run_report.py --full-refresh
```

### Step D: Export IEEEtran LaTeX manuscript and sync submission figures

```bash
py scripts/transform_report.py
```

**Output:**
- `docs/manuscript/MRANC_Final_Research_Report.tex` (tracked in git)
- `artifacts/figures/manuscript/{dataset}/` (PNG exports for journal upload)
- `artifacts/reports/latex/MRANC_Final_Research_Report.tex` (local build copy)

### Step E: Compile PDF for journal submission

Requires `pdflatex` (install with `.\scripts\install_latex_local.ps1`):

```powershell
cd docs\manuscript
pdflatex -interaction=nonstopmode MRANC_Final_Research_Report.tex
```

**Output:** `docs/manuscript/MRANC_Final_Research_Report.pdf`

### One-shot paper refresh workflow

After any code, checkpoint, or data change:

```bash
py scripts/run_report.py --refresh
py scripts/transform_report.py
cd docs\manuscript
pdflatex -interaction=nonstopmode MRANC_Final_Research_Report.tex
```

---

## 6. Utilities

| Feature | Command | Purpose |
|---------|---------|---------|
| Dependency check | `py scripts/check_dependencies.py` | Verify Python, CUDA, and pdflatex |
| Output layout migration | `py scripts/migrate_output_layout.py` | Dry-run misplaced output files |
| Apply migration | `py scripts/migrate_output_layout.py --apply` | Move misplaced output files |
| DEAP training launcher | `.\scripts\start_training.ps1` | Validate DEAP data and start training |

---

## 7. Output artifact map

| Artifact | Path |
|----------|------|
| Model checkpoints | `artifacts/models/checkpoints/`, `artifacts/models/weights/` |
| Metrics JSON | `artifacts/reports/{dataset}/evaluation_report_{dataset}.json` |
| Baseline JSON | `artifacts/reports/{dataset}/baseline_ica_metrics.json` |
| Decomposition PNGs | `artifacts/figures/critical/{dataset}/` |
| Word research report | `artifacts/reports/clinical/MRANC_Final_Research_Report.docx` |
| LaTeX manuscript | `docs/manuscript/MRANC_Final_Research_Report.tex` |
| Submission figure PNGs | `artifacts/figures/manuscript/{dataset}/` |
| Compiled PDF | `docs/manuscript/MRANC_Final_Research_Report.pdf` |

---

## 8. Further reading

| Document | Content |
|----------|---------|
| [README.md](README.md) | Repository landing page and license |
| [docs/getting-started.md](docs/getting-started.md) | Clone-to-report walkthrough |
| [docs/commands.md](docs/commands.md) | Full CLI flag reference |
| [docs/training.md](docs/training.md) | Losses, stacking, checkpoints |
| [docs/datasets.md](docs/datasets.md) | Per-corpus preprocessing details |
| [docs/figures.md](docs/figures.md) | Figure types and report workflow |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | v2 folder layout and migration |
| [requirements-latex.txt](requirements-latex.txt) | LaTeX / pdflatex system dependency |

---

## License

Copyright (c) 2026 Ghadeer Mostafa. Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
