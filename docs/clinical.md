# Clinical workflow (CHB-MIT)

Unsupervised domain adaptation (UDA) on clinical scalp EEG without hardware EOG/EMG/ECG reference wires.

## 1. Download raw EDF files

```bash
py scripts/clinical/download_clinical.py
```

Downloads `chb01_01.edf`, `chb01_02.edf`, `chb01_03.edf` into `raw_clinical_data/` using `curl` with resume and retry.

You can also place EDF files manually in `raw_clinical_data/`.

## 2. Preprocess

```bash
py scripts/clinical/preprocess_clinical.py
```

Steps performed per recording:

1. Load EDF with MNE, resample to **200 Hz**
2. Reconstruct reference-based electrodes from **bipolar pairs** via pseudo-inverse (`pinv(B) @ bipolar_data`)
3. Map recovered electrodes to the standard **32-channel DEAP/SEED grid** (zeros for unmapped)
4. Per-channel normalization to **15 µV std** baseline
5. Window into non-overlapping segments of **T=1000** samples
6. Save `processed_clinical_data/mix.npy`

## 3. Data sanitization

During training, `ClinicalMixDataset` applies `sanitize_clinical_batch` from `src/clinical_sanitize.py`:

- Hard clip to **±500 µV**
- Dead channels (std < 1e-3 µV) replaced with mean of live channels

## 4. Fine-tune (UDA)

Clinical training uses **physics-only** losses (no hardware-ref MSE):

```text
loss = 0.5 * orthogonality + 0.1 * total_variation + 0.01 * minimal_cleaning
```

```bash
py scripts/train.py --dataset clinical --storage cuda --epochs 50 --batch-size 32 --lr 1e-4 --resume_weights checkpoints/best_mranc.pth
```

When `--dataset clinical` and `--resume` are set:

- Only **model weights** are loaded from the checkpoint
- Optimizer/scaler state is reset
- Training starts at epoch 1 with fresh fine-tuning

## 5. Visualize and evaluate

```bash
py scripts/plot_results.py --dataset clinical --output outputs/figures/clinical/clinical_5row.png
py scripts/evaluate_metrics.py --dataset clinical --output outputs/reports/clinical/evaluation_report_clinical.json

# Critical figures (default checkpoint: best_mranc_artifact_benchmark_weights.pth)
py scripts/generate_interpretability_plots.py --dataset clinical
py scripts/run_real_world_pipeline.py --dataset clinical --window-indices 0

# Attention adapter fine-tune (optional, saves weights/mranc_final_attention.pth)
py scripts/train_attention_adapter.py
```

Clinical plotting uses the same 5-row spatial-mean layout as other datasets:
- Rows 1–3 share identical y-axis limits
- Row 4 uses artifact spatial RMS across channels
- Row 5 overlays raw vs reconstructed spatial means with MSE in title
- Companion channel figure is saved as `<output>_channels.png` (Fp1/Cz/O1)

Clinical evaluation uses the same unsupervised metrics as SEED (including `loss_mse`, `val_total_loss`, and `mse : val_total_loss`) since no ground-truth clean EEG exists.
