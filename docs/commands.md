# Command reference

All commands assume the current directory is the **repo root**.

## End-to-end workflow (summary)

```bash
# 1. Setup
pip install -r requirements.txt && pip install python-docx mne scikit-learn
py src/setup_folders.py

# 2. Place raw data in data/raw/{deap,seed,clinical,artifact_benchmark}/
# 3. Convert (see Data preparation below)

# 4. Train full stack
py scripts/run_sequential_stacking.py

# 5. Metrics
py scripts/evaluate_metrics.py --dataset clinical   # repeat per dataset

# 6. Figures
py scripts/plot_results.py --dataset clinical

# 7. Word + LaTeX report
py scripts/run_report.py
py scripts/transform_report.py
```

Full narrative: [getting-started.md](getting-started.md)

## Setup

| Command | Description |
|---------|-------------|
| `py src/setup_folders.py` | Create v2 data/artifacts folder layout |
| `py scripts/migrate_project_layout.py --apply` | Migrate legacy root folders to v2 layout |
| `pip install -r requirements.txt` | Install Python dependencies |
| `.\scripts\install_torch_local.ps1` | Download/install PyTorch cu121 wheels (Windows) |
| `.\scripts\start_training.ps1` | Validate DEAP data and start DEAP training |

## Data preparation

| Command | Description |
|---------|-------------|
| `py scripts/deap/convert_batch_2.py` | Convert DEAP `.dat` → `processed_data/` |
| `py scripts/deap/convert_batch_2.py --raw-dir deap_raw --out-dir processed_data` | Custom DEAP paths |
| `py scripts/seed/extract_and_convert_seed.py` | SEEDA zip → `processed_data_seed/` |
| `py scripts/seed/download_and_convert_seeda.py --skip-download` | Legacy SEEDA Mendeley pipeline |
| `py scripts/seed/inspect_seed_structure.py` | Inspect variables inside SEEDA `.mat` zip |
| `py scripts/clinical/download_clinical.py` | Download CHB-MIT chb01 EDF files |
| `py scripts/clinical/preprocess_clinical.py` | Preprocess EDF → `processed_clinical_data/mix.npy` |
| `bash scripts/download_artifact_benchmark.sh` | Download and extract SSVEP artifact benchmark archive |
| `py scripts/preprocess_artifact_benchmark.py` | Build `data/artifact_benchmark/processed/mix.npy` from raw SSVEP data |
| `py scripts/validate_processed_data.py` | Validate DEAP `processed_data/*.npy` shapes |

## Training

```bash
py scripts/train.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `auto` | `auto`, `deap`, `seed`, `clinical`, `artifact_benchmark` |
| `--data-dir` | `processed_data` | Used when `--dataset auto` |
| `--storage` | `cuda` | `cuda` (required for training) |
| `--epochs` | `50` | Number of training epochs |
| `--batch-size` / `--batch_size` | `32` | Batch size (auto-reduced if VRAM tight) |
| `--lr` | `0.001` | Adam learning rate |
| `--val-fraction` | `0.1` | Validation split fraction |
| `--split-seed` | `42` | Random split seed |
| `--memory-fraction` | `0.85` | Max GPU memory fraction |
| `--resume` / `--resume_weights` | `None` | Weights path (supports raw state_dict or checkpoint with `model_state_dict`) |

Examples:

```bash
py scripts/train.py --dataset deap --storage cuda --epochs 20 --batch-size 128 --lr 1e-3
py scripts/train.py --dataset seed --storage cuda --epochs 20 --batch-size 64 --lr 1e-3
py scripts/train.py --dataset clinical --storage cuda --epochs 50 --batch-size 32 --lr 1e-4 --resume_weights checkpoints/best_mranc.pth
py scripts/train.py --dataset artifact_benchmark --storage cuda --epochs 50 --batch_size 32 --resume_weights checkpoints/best_mranc_artifact_benchmark_weights.pth
```

## Visualization

```bash
py scripts/plot_results.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `seed` | `seed`, `deap`, `clinical`, `artifact_benchmark` |
| `--checkpoint` | `checkpoints/best_mranc_artifact_benchmark_weights.pth` | Override model weights |
| `--output` | dataset-specific | Output PNG (`artifact_benchmark` defaults to `outputs/figures/artifact_benchmark/artifact_benchmark_8subplot_rms.jpg`) |
| `--window-index` | `0` | Window index in val set |
| `--no-val-split` | off | Use full dataset index |
| `--highlight-channels` | `Fp1,Cz,O1` | Channels to highlight |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--skip-critical-figures` | off | Skip per-channel decomposition PNGs |
| `--legacy-spatial-rms` | off | Write legacy 5-row report to `outputs/figures/` |

Default (without `--legacy-spatial-rms`): writes ultra-wide decomposition PNGs to
`artifacts/figures/critical/{dataset}/`.

Output behavior (with `--legacy-spatial-rms`):
- Main output (`--output`):
  - `seed/deap/clinical`: 5-row Spatial RMS decomposition figure
  - `artifact_benchmark`: 8-subplot Spatial RMS matrix (4x2)
- Companion output: `<output>_channels.png` for highlighted channel traces (`Fp1`, `Cz`, `O1`)
- Rows 1–3 use shared y-axis limits for direct comparison
- Row 4 uses artifact spatial RMS across channels
- Row 5 title includes Spatial RMS reconstruction MSE
- DEAP prints scale diagnostics (no visual rescaling)

## Interpretability summary figures

```bash
py scripts/generate_interpretability_plots.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `clinical` | `deap`, `seed`, `clinical`, `artifact_benchmark`, or `all` |
| `--checkpoint` | `checkpoints/best_mranc_artifact_benchmark_weights.pth` | Override model weights |
| `--data-dir` | dataset preset | Override processed folder |
| `--output-dir` | `critical_figures/` | Root; writes under `{output_dir}/{dataset}/` |
| `--window-index` | `0` | Index into validation holdout list |
| `--val-fraction` | `0.1` | Validation fraction |
| `--split-seed` | `42` | Split seed |
| `--highlight-channels` | `Fp1,Cz,O1` | Channels in fidelity figure |
| `--cmap` | `viridis` | Attention heatmap colormap (`viridis` or `plasma`) |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |

Outputs per dataset (`prefix` is `tuh` for clinical):
- `critical_figures/{dataset}/{prefix}_denoising_fidelity.png`
- `critical_figures/{dataset}/{prefix}_attention_map_heatmap.png`

Examples:

```bash
py scripts/generate_interpretability_plots.py --dataset all
py scripts/generate_interpretability_plots.py --dataset seed --checkpoint weights/mranc_final_attention.pth
```

## Real-world decomposition pipeline

```bash
py scripts/run_real_world_pipeline.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `clinical` | `deap`, `seed`, `clinical`, `artifact_benchmark`, or `all` |
| `--checkpoint` | `checkpoints/best_mranc_artifact_benchmark_weights.pth` | Override model weights |
| `--data-dir` | dataset preset | Override processed folder |
| `--output-dir` | `critical_figures/` | Root; writes under `{output_dir}/{dataset}/` |
| `--channels` | `Fp1,Cz,O1` | One 6-row figure per channel per window |
| `--window-indices` | `all` | `all` or comma list `0,1,2` (validation list indices) |
| `--max-windows` | `0` | Cap windows plotted (0 = no cap) |
| `--val-fraction` | `0.1` | Validation fraction |
| `--split-seed` | `42` | Split seed |
| `--dpi` | `300` | PNG resolution |
| `--cmap` | `viridis` | Attention heatmap colormap |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |

Each output file uses an ultra-wide 6-row layout (`figsize=(16, 12)`, shared x-axis):
1. Raw vs denoised overlay
2. EOG stem
3. EMG stem
4. ECG stem
5. Base noise stem
6. Normalized multi-scale attention heatmap

Output pattern:
`critical_figures/{dataset}/{prefix}_eval_window{G}_{Channel}_decomposition.png`
where `prefix` is `tuh` for clinical and the dataset name otherwise.

| Flag | Description |
|------|-------------|
| `--run-metrics` | Also run `evaluate_metrics.py` and save JSON under `artifacts/reports/{dataset}/` |
| `--with-summary-figures` | Also run `generate_interpretability_plots.py` per dataset |
| `--skip-decomposition` | Skip per-channel 6-row PNGs (metrics/summary only) |

Examples:

```bash
py scripts/run_real_world_pipeline.py --dataset all
py scripts/run_real_world_pipeline.py --dataset clinical --window-indices 0
py scripts/run_real_world_pipeline.py --dataset deap --checkpoint checkpoints/best_mranc.pth
```

## Sequential stacking (4-phase)

```bash
py scripts/run_sequential_stacking.py [--run-id YYYYMMDD_HHMMSS] [--dry-run]
```

Runs artifact_benchmark pretrain, DEAP transfer, SEED transfer, and clinical PEFT adapter training.
Each phase saves a **new versioned file** (`{stem}_{run_id}.pth`) and never overwrites older weights.
Latest paths are recorded in `checkpoints/stacking_latest.json` for evaluation and reporting.

## Empirical baselines

```bash
py scripts/evaluate_baseline_ica.py --dataset all
py scripts/evaluate_baseline_eegdenoisenet.py --dataset all [--force] [--skip-train]
```

Writes `outputs/reports/{clinical,deap,seed}/baseline_ica_metrics.json` and
`baseline_eegdenoisenet_metrics.json`.

## Attention adapter training

```bash
py scripts/train_attention_adapter.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--resume-weights` | `checkpoints/best_mranc_phase3_seed.pth` | Base stacked weights |
| `--output` | `weights/mranc_final_attention.pth` | Saved adapter checkpoint |
| `--adapter-epochs` | `15` | Adapter-only epochs (backbone frozen) |
| `--adapter-lr` | `1e-5` | Learning rate (adapter only) |
| `--stabilize-epochs` | deprecated | Ignored; full-model unfreeze disabled |

## Evaluation

```bash
py scripts/evaluate_metrics.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `seed` | `seed`, `deap`, `clinical`, `artifact_benchmark` |
| `--checkpoint` | auto | Clinical uses `weights/mranc_final_attention.pth` when present; else artifact benchmark ckpt |
| `--output` | `outputs/reports/<dataset>/evaluation_report_<dataset>.json` | JSON report |
| `--batch-size` | `32` | Inference batch size |
| `--sample-rate` | auto | PSD sample rate (200 seed/clinical/artifact_benchmark, 128 deap) |

Metrics reported: reconstruction MSE (**loss_mse**), validation total loss (`val_total_loss`), and string pair `mse : val_total_loss`, plus SNR improvement (dB), PSD alpha/beta correlation (8–30 Hz), physics residual RMSE, artifact magnitude RMSE.

## Aggregation (paper table)

```bash
py scripts/aggregate_results.py
```

Expected inputs under `outputs/reports/`:
- `seed/evaluation_report_seed.json`
- `clinical/evaluation_report_clinical.json`
- `deap/evaluation_report_deap.json`
- `artifact_benchmark/evaluation_report_artifact_benchmark.json` (optional row when present)

Outputs:
- Console benchmark dataframe (publication summary)
- LaTeX table block from `pandas.DataFrame.to_latex()`

## Research report (Word)

```bash
py scripts/run_report.py [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--figures-dir` | `critical_figures/` | Root for per-dataset decomposition PNGs |
| `--reports-dir` | `outputs/reports/` | Root for evaluation JSON reports |
| `--output` | `outputs/reports/clinical/MRANC_Final_Research_Report.docx` | Output Word document |
| `--datasets` | `clinical,seed,deap,artifact_benchmark` | Datasets in Table II cross-summary |
| `--hero-dataset` | `clinical` | Dataset for main-text hero figures |
| `--hero-channels` | `Fp1,Cz` | Hero channels (Figure 1, Figure 2, ...) |
| `--hero-window` | `0` | Validation window index in filenames |
| `--refresh` | off | Re-run evaluate_metrics and figure pipelines (no training) |
| `--full-refresh` | off | Stacking + ICA/EEGdenoiseNet baselines + `--refresh`, then build report |
| `--force-stacking` | off | With `--full-refresh`, pass `--force` to phase 1 stacking |

Requires `python-docx` and `scikit-learn` for full baseline pipeline. The report includes:

- Introduction and Methodology with IEEE citation brackets [1]-[4]
- Results: Table I (clinical SOTA comparison), Table II (MRANC cross-dataset summary), dynamic narrative
- Main text: two hero clinical decomposition figures (Fp1, Cz) only
- Appendix: Supplementary Multi-Channel Decompositions with all remaining PNGs at 6.0-inch width
- References section (IEEE style) at document end

MRANC table cells use `outputs/reports/{dataset}/evaluation_report_{dataset}.json`. ICA and
EEGdenoiseNet columns use `baseline_ica_metrics.json` and `baseline_eegdenoisenet_metrics.json`
(missing files show **Pending Run** in the table). Use `--full-refresh` for an end-to-end update.

## LaTeX transform (full system report)

```bash
pip install python-docx
py scripts/transform_report.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | clinical or root `MRANC_Final_Research_Report.docx` | Source Word file |
| `--output` | `outputs/reports/latex/MRANC_Final_Research_Report.tex` | IEEEtran LaTeX (does not overwrite `.docx`) |
| `--assets-dir` | `outputs/reports/latex/latex_assets/` | Extracted embedded images |

Transforms the **entire** Word report: Table I (clinical SOTA), Table II (all datasets), hero figures, appendix decompositions for clinical/seed/deap/artifact\_benchmark, and References. Figure paths resolve under `artifacts/figures/critical/{dataset}/` across all corpora.

`transform_report.py` only writes under `outputs/reports/latex/`; it will refuse `--output` paths that point at a `.docx` file.

## End-to-end: updated Word report and LaTeX

Run from the **repo root**.

```bash
pip install python-docx
```

**Recommended full pipeline (train, baselines, metrics, figures, Word, LaTeX):**

```bash
py scripts/run_report.py --full-refresh
py scripts/transform_report.py
```

**Metrics and figures only (no retraining):**

```bash
py scripts/run_report.py --refresh
py scripts/transform_report.py
```

**When `outputs/reports/{dataset}/evaluation_report_*.json` and `critical_figures/` are already current:**

```bash
py scripts/run_report.py
py scripts/transform_report.py
```

**Outputs:**

| File | Description |
|------|-------------|
| `outputs/reports/clinical/MRANC_Final_Research_Report.docx` | Full system Word report |
| `outputs/reports/latex/MRANC_Final_Research_Report.tex` | IEEEtran LaTeX (full document) |
| `outputs/reports/latex/latex_assets/` | Embedded images extracted from Word when needed |

**Optional: compile PDF**

```bash
cd outputs/reports/latex
pdflatex MRANC_Final_Research_Report.tex
```

**Manual steps (same result as `--refresh` without re-running `run_report` twice):**

```bash
py scripts/evaluate_metrics.py --dataset clinical
py scripts/evaluate_metrics.py --dataset seed
py scripts/evaluate_metrics.py --dataset deap
py scripts/evaluate_metrics.py --dataset artifact_benchmark
py scripts/generate_interpretability_plots.py --dataset all --window-indices 0 --with-summary
py scripts/run_real_world_pipeline.py --dataset all --window-indices 0 --run-metrics
py scripts/run_report.py
py scripts/transform_report.py
```

## Output layout migration

```bash
py scripts/migrate_output_layout.py          # dry-run
py scripts/migrate_output_layout.py --apply  # move misplaced files
```

Moves flat `critical_figures/tuh_*.png` into `critical_figures/clinical/`, relocates
misplaced evaluation JSON, and archives legacy flat reports under `outputs/reports/_archive/`.
