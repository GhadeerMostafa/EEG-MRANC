# Datasets

Place downloaded corpora in the **raw folders** below, then run the conversion script for each dataset. Processed `.npy` files are **not** in git; they are created locally and gitignored.

| Corpus | Raw folder (you populate) | Processed folder (script output) |
|--------|---------------------------|----------------------------------|
| DEAP | `deap_raw/` | `processed_data/` |
| SEED | `seed_raw/` or `.seeda_workspace/` | `processed_data_seed/` |
| Clinical | `raw_clinical_data/` | `processed_clinical_data/` |
| Artifact benchmark | `data/artifact_benchmark/raw/` | `data/artifact_benchmark/processed/` |

After all conversions you need for your experiment, continue with [training.md](training.md) or `py scripts/run_sequential_stacking.py`.

## DEAP (`processed_data/`)

**Source:** `deap_raw/data_preprocessed_python/s*.dat`

**Conversion:**

```bash
py scripts/deap/convert_batch_2.py
```

**Output files:**

| File | Shape | Description |
|------|-------|-------------|
| `mix.npy` | (N, 32, 256) | Scalp EEG mix |
| `ref_eog.npy` | (N, 2, 256) | Hardware EOG references |
| `ref_emg.npy` | (N, 2, 256) | Hardware EMG references |
| `ref_ecg.npy` | (N, 1, 256) | Hardware ECG reference |

**Sample rate:** 128 Hz, window T=256 (2 s).

**Training:**

```bash
py scripts/train.py --dataset deap --storage cuda
```

**Visualization:** 5-row Spatial RMS decomposition + companion channel plot (`*_channels.png`).

**Critical figures:** `py scripts/run_real_world_pipeline.py --dataset deap`

---

## SEED / SEEDA (`processed_data_seed/`)

**Source:** `.seeda_workspace/seeda.zip` or `seed_raw/`

**Conversion:**

```bash
py scripts/seed/extract_and_convert_seed.py
```

Maps 19-channel SEEDA montage to 32-channel DEAP layout via spatial interpolation.

**Output:** Same file names as DEAP but T=1000 at 200 Hz.

**Training:**

```bash
py scripts/train.py --dataset seed --storage cuda
```

**Visualization:** 5-row Spatial RMS decomposition + companion channel plot (`*_channels.png`).

**Critical figures:** `py scripts/run_real_world_pipeline.py --dataset seed`

---

## Clinical CHB-MIT (`processed_clinical_data/`)

**Source:** PhysioNet CHB-MIT scalp EEG (patient chb01)

**Pipeline:**

```bash
py scripts/clinical/download_clinical.py
py scripts/clinical/preprocess_clinical.py
```

See [clinical.md](clinical.md) for bipolar-to-reference reconstruction details.

**Output:** `mix.npy` only — shape (N, 32, 1000), no hardware reference channels.

**Training:** Unsupervised domain adaptation with physics-only losses.

```bash
py scripts/train.py --dataset clinical --storage cuda --epochs 50 --resume_weights checkpoints/best_mranc.pth
```

**Visualization:** 5-row Spatial RMS decomposition + companion channel plot (`*_channels.png`).

**Critical figures:** `py scripts/run_real_world_pipeline.py --dataset clinical`

---

## SSVEP Artifact Benchmark (`data/artifact_benchmark/processed/`)

**Source:** Open-source MNE SSVEP benchmark archive (OSF mirror used by MNE)

**Pipeline:**

```bash
bash scripts/download_artifact_benchmark.sh
py scripts/preprocess_artifact_benchmark.py
```

Preprocessing converts the raw recording to FIF, reloads with `mne.io.read_raw_fif()`,
keeps EEG channels, maps/pads to the standard 32-channel layout, resamples (default 200 Hz),
and windows into `mix.npy` with shape `(N, 32, 1000)` by default.

**Training:** Physics-only objective (no hardware refs), isolated checkpoint output.

```bash
py scripts/train.py --dataset artifact_benchmark --storage cuda --epochs 50 --batch_size 32 --resume_weights checkpoints/best_mranc_artifact_benchmark_weights.pth
```

**Visualization/Evaluation:**

```bash
py scripts/plot_results.py --dataset artifact_benchmark --output outputs/figures/artifact_benchmark/artifact_benchmark_8subplot_rms.jpg
py scripts/evaluate_metrics.py --dataset artifact_benchmark
```

**Critical figures:** `py scripts/run_real_world_pipeline.py --dataset artifact_benchmark`

---

## Critical figures (all datasets)

Default inference weights: `checkpoints/best_mranc_artifact_benchmark_weights.pth`

Override with `--checkpoint` on either script.

```bash
# Summary XAI pair per dataset
py scripts/generate_interpretability_plots.py --dataset all

# Per-channel 6-row ultra-wide decomposition (validation windows)
py scripts/run_real_world_pipeline.py --dataset all
```

Outputs under `critical_figures/{dataset}/`.

---

## Validation

```bash
py scripts/validate_processed_data.py          # DEAP processed_data
py scripts/seed/inspect_seed_structure.py      # SEEDA zip structure
```
