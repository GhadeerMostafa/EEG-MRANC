# Training

## Sequential stacking (recommended)

After processed data exists for artifact benchmark, DEAP, SEED, and clinical, run the full four-phase pipeline:

```bash
py scripts/run_sequential_stacking.py
```

| Phase | Dataset | Output (versioned) |
|-------|---------|-------------------|
| 1 | `artifact_benchmark` | `checkpoints/best_mranc_artifact_benchmark_weights_<run_id>.pth` |
| 2 | `deap` | `checkpoints/best_mranc_phase2_deap_<run_id>.pth` |
| 3 | `seed` | `checkpoints/best_mranc_phase3_seed_<run_id>.pth` |
| 4 | clinical adapter | `weights/mranc_final_attention_<run_id>.pth` |

Latest paths are written to `checkpoints/stacking_latest.json` for `evaluate_metrics.py`, `plot_results.py`, and `run_report.py`.

Dry-run (print commands only):

```bash
py scripts/run_sequential_stacking.py --dry-run
```

## Overview

MRANC separates 32-channel EEG into:

| Stem | Key |
|------|-----|
| Clean EEG | `pred_eeg` |
| EOG | `pred_eog` |
| EMG | `pred_emg` |
| ECG | `pred_ecg` |
| Baseline noise | `pred_basenoise` |

## Supervised loss (DEAP / SEED)

```text
total_loss = loss_stems + 0.05 * ortho + 0.01 * tv + scale_penalty
loss_stems = MSE(pred_eog, ref_eog) + MSE(pred_emg, ref_emg) + MSE(pred_ecg, ref_ecg) + MSE(pred_basenoise, 0)
```

## Physics-only loss (Clinical + Artifact Benchmark)

Hardware reference losses are disabled. Only physics terms:

Clinical uses:

```text
loss = 0.5 * spatial_orthogonality + 0.1 * total_variation + 0.01 * minimal_cleaning
```

Artifact benchmark uses anti-collapse safeguards:

```text
loss = 0.5 * (10x orthogonality) + 0.1 * (2x total_variation) + 0.01 * minimal_cleaning
     + variance_floor_penalty(artifacts_total)
```

This helps prevent identity-mapping behavior where artifact stems collapse toward zero.

## GPU / VRAM safety

`src/train_safety.py` runs a preflight check before training:

- Estimates VRAM needed for data + activations
- Auto-reduces batch size if needed
- **Raises an error** if GPU memory is insufficient (no CPU fallback)

Clinical and artifact_benchmark preflight scale activation budget for long-T windows.

## Checkpoints

Best model checkpoints:

```text
checkpoints/best_mranc.pth
checkpoints/best_mranc_artifact_benchmark_weights.pth
weights/mranc_final_attention.pth
```

Contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `config`, `best_val_total_loss` (training checkpoints).

Primary stacked weights for inference and critical figures:
`checkpoints/best_mranc_artifact_benchmark_weights.pth`

Override at runtime with `--checkpoint` on `generate_interpretability_plots.py` or `run_real_world_pipeline.py`.

## Parameter-efficient attention adapter

Fine-tune the multi-scale attention block on clinical data without destabilizing stacked weights:

```bash
py scripts/train_attention_adapter.py
```

- Phase 1 (15 epochs, lr=1e-5): freeze encoder and stem decoders; train adapter only with feature distillation MSE plus clinical physics losses.
- Phase 2 (5 epochs, lr=1e-6): unfreeze full model for stabilization.
- Default resume: `checkpoints/best_mranc_artifact_benchmark_weights.pth`
- Output: `weights/mranc_final_attention.pth`

## Recommended settings

| Dataset | Epochs | Batch size | LR | Sequence T |
|---------|--------|------------|-----|------------|
| DEAP | 20 | 128 | 1e-3 | 256 |
| SEED | 20 | 64 | 1e-3 | 1000 |
| Clinical | 50 | 32 | 1e-4 | 1000 |
| Artifact Benchmark | 50 | 32 | 1e-4 (auto on resume if --lr omitted) | 1000 |

## Monitoring

Training logs per-epoch train and validation losses. For clinical runs, validation uses the same physics-only loss as training.

Resume handling supports both:
- checkpoint dictionaries (`model_state_dict`)
- raw state dictionaries

using `--resume_weights` (alias: `--resume`) with safe `map_location`.

After training:

```bash
py scripts/plot_results.py --dataset <seed|deap|clinical|artifact_benchmark>
py scripts/evaluate_metrics.py --dataset <seed|deap|clinical|artifact_benchmark>
```

Reports are written to `outputs/reports/`; figures to `outputs/figures/`.

Evaluation metrics additionally report **loss_mse**, **val_total_loss**, and the formatted pair `mse : val_total_loss`.
