# Run after PyTorch cu121 install finishes.
# Usage (from repo root): .\scripts\start_training.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "Checking PyTorch + CUDA..."
py -3.12 -c @"
import torch
print('torch', torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA not available - install torch cu121 first.')
print('GPU:', torch.cuda.get_device_name(0))
"@

Write-Host "Installing remaining deps (numpy, scipy, requests, psutil)..."
py -3.12 -m pip install "numpy>=1.24" "scipy>=1.10" "requests>=2.28" "psutil>=5.9" --default-timeout=300

Write-Host "Validating processed_data..."
py -3.12 scripts/validate_processed_data.py

Write-Host "Starting MRANC training (CUDA-only, hardware safety limits enabled)..."
py -3.12 scripts/train.py --dataset deap --storage cuda --epochs 20 --batch-size 128 --lr 1e-3 --memory-fraction 0.85
