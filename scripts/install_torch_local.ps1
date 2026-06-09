# Download PyTorch cu121 wheels ONCE, then install from disk.
# Usage (from repo root):
#   .\scripts\install_torch_local.ps1                 # download deps + torch, then install
#   .\scripts\install_torch_local.ps1 -DownloadOnly   # download only
#   .\scripts\install_torch_local.ps1 -InstallOnly    # install from existing wheels
#   .\scripts\install_torch_local.ps1 -ResumeTorch -DownloadOnly

param(
    [switch]$InstallOnly,
    [switch]$DownloadOnly,
    [switch]$ResumeTorch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$WheelDir = Join-Path $RepoRoot "wheels"
$IndexUrl = "https://download.pytorch.org/whl/cu121"
$TorchWheelName = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"
$TorchUrl = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl"
$ExpectedTorchBytes = 2449303700

New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

function Test-TorchWheelComplete {
    $path = Join-Path $WheelDir $TorchWheelName
    if (-not (Test-Path $path)) { return $false }
    $len = (Get-Item $path).Length
    if ($len -lt ($ExpectedTorchBytes * 0.99)) {
        $gb = [math]::Round($len / 1GB, 2)
        Write-Host "Partial torch wheel: ${gb} GB (need ~2.45 GB)"
        return $false
    }
    return $true
}

function Invoke-Pip {
    param([string[]]$PipArgs)
    & py -3.12 -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed with exit code $LASTEXITCODE"
    }
}

if ($ResumeTorch) {
    $out = Join-Path $WheelDir $TorchWheelName
    Write-Host "Resuming torch download with curl -> $out"
    curl.exe -L -C - -o $out $TorchUrl
    if ($LASTEXITCODE -ne 0) { throw "curl download failed" }
    if (-not (Test-TorchWheelComplete)) { throw "Torch wheel still incomplete after curl" }
    Write-Host "Torch wheel complete."
    if ($DownloadOnly) { exit 0 }
}

if (-not $InstallOnly) {
    if (-not $ResumeTorch) {
        Write-Host "Step 1/2: Download small dependencies..."
        try {
            Invoke-Pip @(
                "download", "filelock", "typing-extensions", "networkx", "jinja2", "fsspec", "setuptools", "sympy==1.13.1",
                "--dest", $WheelDir,
                "--index-url", $IndexUrl,
                "--default-timeout", "1000",
                "--retries", "15"
            )
        } catch {
            Write-Warning "Some deps may already exist in wheels; continuing..."
        }

        if (Test-TorchWheelComplete) {
            Write-Host "Torch wheel already complete in wheels - skipping download."
        } else {
            Write-Host "Step 2/2: Download torch wheel (~2.45 GB) - use stable network; 20-40 min..."
            Write-Host "If this fails, run: .\scripts\install_torch_local.ps1 -ResumeTorch -DownloadOnly"
            try {
                Invoke-Pip @(
                    "download", "torch",
                    "--dest", $WheelDir,
                    "--no-deps",
                    "--index-url", $IndexUrl,
                    "--default-timeout", "1000",
                    "--retries", "15"
                )
            } catch {
                Write-Host ""
                Write-Host "If pip download failed, try resumable download:"
                Write-Host "  .\scripts\install_torch_local.ps1 -ResumeTorch -DownloadOnly"
                exit 1
            }
        }
    }

    if (-not (Test-TorchWheelComplete)) {
        Write-Host "ERROR: torch wheel missing or incomplete in $WheelDir"
        exit 1
    }
    Write-Host "All wheels ready in: $WheelDir"
}

if ($DownloadOnly) { exit 0 }

$torchWhl = Get-ChildItem $WheelDir -Filter "torch*.whl" -ErrorAction SilentlyContinue
if (-not $torchWhl) {
    Write-Host "ERROR: No torch .whl in $WheelDir. Run download first."
    exit 1
}

Write-Host "Installing from local wheels (no network)..."
Invoke-Pip @("install", "--no-index", "--find-links", $WheelDir, "torch")

Write-Host "Verifying..."
$verifyCmd = "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
py -3.12 -c $verifyCmd
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Done. Wheels kept in: $WheelDir"
