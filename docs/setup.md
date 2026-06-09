# Setup

## Requirements

- Python 3.12 (recommended on Windows)
- NVIDIA GPU with CUDA 12.1 support for training and default evaluation
- `curl` on PATH (clinical download script)
- Git clone: `git clone https://github.com/GhadeerMostafa/EEG-MRANC.git`

## Install dependencies

From the repo root:

```bash
pip install -r requirements.txt
pip install python-docx mne scikit-learn
py src/setup_folders.py
```

`setup_folders.py` creates checkpoint, data, output, and figure directories. Empty folders are tracked in git via `.gitkeep` so you can place raw data immediately after clone.

## PyTorch with CUDA (Windows)

```powershell
.\scripts\install_torch_local.ps1
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Resume interrupted downloads:

```powershell
.\scripts\install_torch_local.ps1 -ResumeTorch -DownloadOnly
.\scripts\install_torch_local.ps1 -InstallOnly
```

## Folder structure after setup

| Path | Purpose |
|------|---------|
| `data/raw/{deap,seed,clinical,artifact_benchmark}/` | **You** place raw EEG here |
| `data/processed/{deap,seed,clinical,artifact_benchmark}/` | Created by convert/preprocess scripts |
| `artifacts/models/{checkpoints,weights}/` | Training outputs (`.pth` gitignored) |
| `artifacts/figures/critical/{dataset}/` | Generated decomposition PNGs |
| `artifacts/reports/{dataset}/` | Metrics JSON + Word report |
| `docs/manuscript/` | Report prose (in git; used by `run_report.py`) |

## What is not committed

- Raw and processed numpy tensors (`*.npy`)
- Model weights (`*.pth`, `*.pt`)
- Generated figures, metrics JSON, Word/LaTeX reports
- `.cursor/`, IDE config, virtual environments

See [getting-started.md](getting-started.md) for the full workflow.

## Environment notes

- Training **requires CUDA** (`--storage cuda`).
- DataLoader uses `num_workers=0` on Windows.
- Run scripts as `py scripts/<name>.py` from the repo root so `scripts/runtime.py` adds `src/` to the import path.
