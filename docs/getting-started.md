# Getting started (GitHub clone → full local workflow)

This repository ships **source code, scripts, folder placeholders, and manuscript text**
used by `run_report.py`. It does **not** include raw EEG files, processed `.npy` tensors,
trained `.pth` weights, figures, metrics JSON, or generated Word/LaTeX reports. Those are
created on your machine after you add datasets and run the pipelines below.

## 1. Clone and install

```bash
git clone https://github.com/GhadeerMostafa/EEG-MRANC.git
cd EEG-MRANC
pip install -r requirements.txt
pip install python-docx mne scikit-learn
py src/setup_folders.py
```

**CUDA is required** for training and for the default evaluation/plotting scripts.

Windows PyTorch (offline wheels):

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Place raw data in the repo folders

Empty directories are kept in git via `.gitkeep`. Copy or download your data into:

| Dataset | Put raw files here | Then run |
|---------|-------------------|----------|
| DEAP | `data/raw/deap/data_preprocessed_python/*.dat` | `py scripts/deap/convert_batch_2.py` |
| SEED | `data/raw/seed/` (or `.seeda_workspace/seeda.zip`) | `py scripts/seed/extract_and_convert_seed.py` |
| Clinical (CHB-MIT) | `data/raw/clinical/*.edf` | `py scripts/clinical/download_clinical.py` then `py scripts/clinical/preprocess_clinical.py` |
| Artifact benchmark | `data/raw/artifact_benchmark/` | `bash scripts/download_artifact_benchmark.sh` then `py scripts/preprocess_artifact_benchmark.py` |

Processed outputs land in:

| Dataset | Processed folder |
|---------|------------------|
| DEAP | `data/processed/deap/` |
| SEED | `data/processed/seed/` |
| Clinical | `data/processed/clinical/` |
| Artifact benchmark | `data/processed/artifact_benchmark/` |

Validate DEAP arrays after conversion:

```bash
py scripts/validate_processed_data.py
```

## 3. Train (4-phase stacking pipeline)

Recommended end-to-end training sequence (artifact benchmark → DEAP → SEED → clinical adapter):

```bash
py scripts/run_sequential_stacking.py
```

This writes versioned checkpoints under `checkpoints/` and `weights/`, and updates
`checkpoints/stacking_latest.json` with the latest paths for each phase.

Preview commands without running:

```bash
py scripts/run_sequential_stacking.py --dry-run
```

Or train one dataset at a time — see [training.md](training.md) and [commands.md](commands.md).

## 4. Evaluate metrics

After you have checkpoints (from step 3 or your own training):

```bash
py scripts/evaluate_metrics.py --dataset clinical
py scripts/evaluate_metrics.py --dataset seed
py scripts/evaluate_metrics.py --dataset deap
py scripts/evaluate_metrics.py --dataset artifact_benchmark
```

JSON reports are written to `outputs/reports/{dataset}/evaluation_report_{dataset}.json`
(generated locally; not in git).

## 5. Plot results and critical figures

**Per-dataset decomposition plots** (default: ultra-wide PNGs under `critical_figures/`):

```bash
py scripts/plot_results.py --dataset clinical
py scripts/plot_results.py --dataset deap
py scripts/plot_results.py --dataset seed
py scripts/plot_results.py --dataset artifact_benchmark
```

**Legacy 5-row spatial-RMS layout** (under `outputs/figures/`):

```bash
py scripts/plot_results.py --dataset clinical --legacy-spatial-rms
```

**Full figure + optional metrics pipeline**:

```bash
py scripts/run_real_world_pipeline.py --dataset all --window-indices 0 --run-metrics --with-summary-figures
```

## 6. Build the Word research report (same as local)

`run_report.py` reads live metrics JSON, critical figures, baseline manifests, and prose from
`docs/manuscript/*.txt` (included in the repo). Generated `.docx` / `.tex` files stay local.

**Metrics and figures already up to date:**

```bash
py scripts/run_report.py
py scripts/transform_report.py
```

**Refresh metrics + figures, then compile report:**

```bash
py scripts/run_report.py --refresh
py scripts/transform_report.py
```

**Full refresh** (retrain stack, baselines, metrics, figures, then Word — long GPU job):

```bash
py scripts/run_report.py --full-refresh
py scripts/transform_report.py
```

| Output | Path |
|--------|------|
| Word report | `outputs/reports/clinical/MRANC_Final_Research_Report.docx` |
| LaTeX export | `outputs/reports/latex/MRANC_Final_Research_Report.tex` |

## What git includes vs excludes

| Included in GitHub | You add locally |
|--------------------|-----------------|
| `src/`, `scripts/`, `docs/`, `requirements.txt` | Raw EEG in `*_raw/` folders |
| `docs/manuscript/` text for `run_report.py` | Processed `mix.npy` / refs |
| `.gitkeep` folder placeholders | `checkpoints/*.pth`, `weights/*.pth` |
| `checkpoints/stacking_latest.json` (when present) | `critical_figures/`, `outputs/` |
| `LICENSE` (CC BY-NC-SA 4.0) | Generated `.docx` / `.tex` reports |

Cursor/IDE metadata (`.cursor/`, `*.plan.md`) is never committed.
