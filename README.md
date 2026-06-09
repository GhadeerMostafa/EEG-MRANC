<meta name="google-site-verification" content="TU1_WKv5q71kFTqop14Gav7YHIKZi_LUnN3BDedGbp4" />
<p align="center">
  <strong>Multi-Resolution Attention-Guided Neural Cleaner (MRANC)<br>for Real-World Clinical EEG Denoising</strong>
</p>

<p align="center">
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0">
  </a>
</p>

<p align="center">
  <strong>Developed by: Ghadeer Mostafa</strong><br>
  <a href="https://github.com/GhadeerMostafa/EEG-MRANC">github.com/GhadeerMostafa/EEG-MRANC</a>
</p>

---

## Abstract

MRANC (Multi-Resolution Attention-Guided Neural Cleaner) is a physics-locked deep decomposition framework for 32-channel scalp EEG. The model separates each window into four interpretable artifact stems (EOG, EMG, ECG, baseline noise) plus a recovered neural trace while preserving an exact additive reconstruction identity. Training follows **Sequential Knowledge-Stacking**: weights are transferred progressively from the SSVEP Artifact Benchmark through DEAP and SEED before parameter-efficient adaptation on CHB-MIT clinical windows. A zero-initialized Multi-Scale Attention Module supplies temporal saliency maps for explainable artifact localization without destabilizing stacked representations at initialization.

This repository ships the complete PyTorch training, evaluation, figure-generation, and manuscript compilation pipeline required to reproduce quantitative benchmarks and the accompanying research report.

## What is in this repository?

| Included in GitHub | Generated or downloaded locally |
|--------------------|----------------------------------|
| Model, training, evaluation, plotting, and report scripts | Raw EEG files (`.dat`, `.edf`, benchmark archives) |
| `docs/` guides and `docs/manuscript/` prose and LaTeX template | Processed `.npy` tensors |
| Empty data and output folder placeholders (`.gitkeep`) | Trained `.pth` / `.pt` weights |
| `requirements.txt`, `LICENSE`, `checkpoints/stacking_latest.json` | `critical_figures/`, `figures/`, `outputs/reports/` |
| `figures/` layout for journal submission assets | Word/LaTeX build artifacts |

**Comprehensive feature map (all commands):** [FEATURES.md](FEATURES.md)  
**Full walkthrough:** [docs/getting-started.md](docs/getting-started.md)

## Quick start

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
pip install -r requirements.txt
py src/setup_folders.py
py scripts/check_dependencies.py
```

CUDA PyTorch on Windows (offline wheels):

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

LaTeX / PDF manuscript (`pdflatex`):

```powershell
.\scripts\install_latex_local.ps1
```

See [requirements-latex.txt](requirements-latex.txt) for Linux and macOS install notes.

## Core workflow

Run all commands from the **repository root**. CUDA is required for training and default evaluation.

```text
1. Place raw data       ->  data/raw/{deap,seed,clinical,artifact_benchmark}/
2. Convert / preprocess ->  dataset-specific scripts under scripts/
3. Train (stacking)     ->  py scripts/run_sequential_stacking.py
4. Metrics              ->  py scripts/evaluate_metrics.py --dataset <name>
5. Figures              ->  py scripts/plot_results.py --dataset <name>
6. Report + manuscript  ->  py scripts/run_report.py [--refresh | --full-refresh]
```

### Data preparation

| Dataset | Raw folder | Conversion command | Processed output |
|---------|------------|--------------------|------------------|
| DEAP | `data/raw/deap/` | `py scripts/deap/convert_batch_2.py` | `data/processed/deap/` |
| SEED | `data/raw/seed/` | `py scripts/seed/extract_and_convert_seed.py` | `data/processed/seed/` |
| Clinical | `data/raw/clinical/` | `py scripts/clinical/download_clinical.py` then `preprocess_clinical.py` | `data/processed/clinical/` |
| Artifact benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` then `py scripts/preprocess_artifact_benchmark.py` | `data/processed/artifact_benchmark/` |

Details: [docs/datasets.md](docs/datasets.md)

### Training (4-phase sequential stacking)

```bash
py scripts/run_sequential_stacking.py
```

Phases: artifact benchmark pretrain, DEAP transfer, SEED transfer, clinical attention adapter. Versioned weights are saved under `artifacts/models/checkpoints/` and `artifacts/models/weights/`; latest paths are recorded in `artifacts/models/checkpoints/stacking_latest.json`.

### Validation benchmarks and report regeneration

**Regenerate metrics, figures, Word report, and LaTeX manuscript from existing checkpoints:**

```bash
py scripts/run_report.py --refresh
py scripts/transform_report.py
```

**Full baseline extraction (retrain stack, baselines, metrics, figures, and report):**

```bash
py scripts/run_report.py --full-refresh
py scripts/transform_report.py
```

| Output | Path |
|--------|------|
| Word report | `artifacts/reports/clinical/MRANC_Final_Research_Report.docx` |
| LaTeX manuscript (submission) | `docs/manuscript/MRANC_Final_Research_Report.tex` |
| Figure assets (relative paths) | `artifacts/figures/manuscript/{dataset}/` |
| LaTeX build copy | `artifacts/reports/latex/MRANC_Final_Research_Report.tex` |

### Evaluation metrics (standalone)

```bash
py scripts/evaluate_metrics.py --dataset clinical
py scripts/evaluate_metrics.py --dataset seed
py scripts/evaluate_metrics.py --dataset deap
py scripts/evaluate_metrics.py --dataset artifact_benchmark
```

Writes `artifacts/reports/{dataset}/evaluation_report_{dataset}.json`.

### Visualization

```bash
py scripts/plot_results.py --dataset clinical
py scripts/run_real_world_pipeline.py --dataset all --window-indices 0 --run-metrics --with-summary-figures --device cuda
```

Decomposition PNGs are written to `artifacts/figures/critical/{dataset}/`. Legacy 5-row spatial-RMS plots: add `--legacy-spatial-rms` to `plot_results.py`.

## Project layout

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full v2 map. Summary:

```text
EEG-MRANC/
|-- README.md, FEATURES.md, PROJECT_STRUCTURE.md
|-- src/, scripts/, docs/manuscript/
|-- data/raw/{deap,seed,clinical,artifact_benchmark}/
|-- data/processed/{deap,seed,clinical,artifact_benchmark}/
`-- artifacts/
    |-- models/{checkpoints,weights}/
    |-- figures/{critical,manuscript,legacy}/
    `-- reports/
```

Upgrade from the old scattered root folders:

```bash
py scripts/migrate_project_layout.py --apply
```

## Documentation

| Doc | Description |
|-----|-------------|
| [FEATURES.md](FEATURES.md) | **All features and commands** |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | **v2 folder layout** |
| [docs/getting-started.md](docs/getting-started.md) | Clone to full report |
| [docs/setup.md](docs/setup.md) | Python, CUDA, folder setup |
| [docs/datasets.md](docs/datasets.md) | Raw to processed pipelines |
| [docs/training.md](docs/training.md) | Losses, stacking, checkpoints |
| [docs/commands.md](docs/commands.md) | Full CLI reference |
| [docs/clinical.md](docs/clinical.md) | CHB-MIT download and UDA |
| [docs/figures.md](docs/figures.md) | Figures and report workflow |

## Tensor layout

Processed `.npy` files: **`(N_windows, 32, T)`** float32 (microvolts). Batches: **`[B, 32, T]`**.

| Dataset | T | Sample rate |
|---------|---|-------------|
| DEAP | 256 | 128 Hz |
| SEED | 1000 | 200 Hz |
| Clinical | 1000 | 200 Hz |
| Artifact benchmark | 1000 | 200 Hz |

## Core modules

| Path | Role |
|------|------|
| `src/model.py` | MRANC encoder, attention, decoders |
| `src/dataset.py` | `.npy` DataLoader |
| `src/paths.py` | Project path constants |
| `scripts/train.py` | Single-dataset training |
| `scripts/run_sequential_stacking.py` | 4-phase stacking pipeline |
| `scripts/evaluate_metrics.py` | SNR, PSD, RMSE metrics |
| `scripts/plot_results.py` | Decomposition figures |
| `scripts/run_real_world_pipeline.py` | Metrics and figures pipeline |
| `scripts/run_report.py` | Word research report |
| `scripts/transform_report.py` | Word to IEEEtran LaTeX export |

## Windows notes

- DataLoader defaults to `num_workers=0`.
- Run scripts as `py scripts/<name>.py` from the repo root.
- Training requires **CUDA**; CPU fallback is not used by default.

---

## License and Academic Usage

Copyright (c) 2026 **Ghadeer Mostafa**.

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE](LICENSE).

**Non-Commercial:** Commercial use or financial exploitation of this software, model weights, manuscript text, or derivative research artifacts is strictly prohibited.

**ShareAlike:** Any derivative frameworks, modified source code, adapted model architectures, or redistributed research outputs must be distributed under the exact same CC BY-NC-SA 4.0 license terms.

**Attribution:** Explicit credit must be given to the original creator, **Ghadeer Mostafa**, with a link to this repository and to the license whenever the work is shared, cited in academic publications, or incorporated into downstream non-commercial research.
