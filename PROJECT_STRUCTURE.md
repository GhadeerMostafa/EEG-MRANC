# MRANC Project Structure (v2)

Developed by **Ghadeer Mostafa**. Canonical layout introduced to consolidate scattered root folders.

## Top-level layout

```text
EEG-MRANC/
|-- README.md, FEATURES.md, PROJECT_STRUCTURE.md
|-- requirements.txt, requirements-latex.txt, LICENSE
|-- src/                          # Core library (model, paths, dataset)
|-- scripts/                      # Runnable pipelines and dataset tools
|-- docs/                         # Guides and manuscript sources
|   `-- manuscript/               # LaTeX submission template (.tex tracked)
|-- data/                         # All EEG inputs and processed tensors
|   |-- raw/
|   |   |-- deap/
|   |   |-- seed/
|   |   |-- clinical/
|   |   `-- artifact_benchmark/
|   `-- processed/
|       |-- deap/
|       |-- seed/
|       |-- clinical/
|       `-- artifact_benchmark/
`-- artifacts/                    # All generated outputs
    |-- models/
    |   |-- checkpoints/          # Stacking checkpoints + stacking_latest.json
    |   `-- weights/                # Clinical attention adapter weights
    |-- figures/
    |   |-- critical/               # Ultra-wide decomposition PNGs
    |   |-- manuscript/             # Journal submission figure exports
    |   `-- legacy/                 # Legacy spatial-RMS plots
    `-- reports/                    # Metrics JSON, Word report, LaTeX build copy
```

## Legacy path mapping

| Legacy (pre-v2) | Canonical (v2) |
|-----------------|----------------|
| `deap_raw/` | `data/raw/deap/` |
| `seed_raw/` | `data/raw/seed/` |
| `raw_clinical_data/` | `data/raw/clinical/` |
| `data/artifact_benchmark/raw/` | `data/raw/artifact_benchmark/` |
| `processed_data/` | `data/processed/deap/` |
| `processed_data_seed/` | `data/processed/seed/` |
| `processed_clinical_data/` | `data/processed/clinical/` |
| `data/artifact_benchmark/processed/` | `data/processed/artifact_benchmark/` |
| `checkpoints/` | `artifacts/models/checkpoints/` |
| `weights/` | `artifacts/models/weights/` |
| `critical_figures/` | `artifacts/figures/critical/` |
| `figures/` | `artifacts/figures/manuscript/` |
| `outputs/figures/` | `artifacts/figures/legacy/` |
| `outputs/reports/` | `artifacts/reports/` |

Code resolves legacy paths automatically until you migrate:

```bash
py scripts/migrate_project_layout.py --apply
```

## Setup commands

```bash
py src/setup_folders.py
py scripts/migrate_project_layout.py --apply   # if upgrading from pre-v2 layout
py scripts/check_dependencies.py
```
