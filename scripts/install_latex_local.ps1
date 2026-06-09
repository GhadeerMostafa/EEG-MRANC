# Install MiKTeX so pdflatex is available for manuscript PDF compilation.
# Usage (from repo root):
#   .\scripts\install_latex_local.ps1
#   .\scripts\install_latex_local.ps1 -VerifyOnly

param(
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Find-PdfLatex {
    $cmd = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    return $null
}

$pdflatex = Find-PdfLatex
if ($pdflatex) {
    Write-Host "pdflatex already available: $pdflatex"
    & $pdflatex --version
    if ($VerifyOnly) { exit 0 }
    exit 0
}

if ($VerifyOnly) {
    Write-Host "ERROR: pdflatex not found. Run .\scripts\install_latex_local.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "Installing MiKTeX via winget..."
winget install MiKTeX.MiKTeX --accept-package-agreements --accept-source-agreements

$pdflatex = Find-PdfLatex
if (-not $pdflatex) {
    Write-Host "ERROR: MiKTeX installed but pdflatex not found. Restart the terminal and run:" -ForegroundColor Red
    Write-Host "  py scripts/check_dependencies.py"
    exit 1
}

$miktexBin = Split-Path -Parent $pdflatex
$initexmf = Join-Path $miktexBin "initexmf.exe"
if (Test-Path $initexmf) {
    & $initexmf --set-config-value "[MPM]AutoInstall=1"
    & $initexmf --set-config-value "[MPM]CheckForUpdates=0"
}

Write-Host "pdflatex ready: $pdflatex"
& $pdflatex --version
Write-Host ""
Write-Host "Compile the manuscript with:"
Write-Host "  cd docs\manuscript"
Write-Host "  pdflatex -interaction=nonstopmode MRANC_Final_Research_Report.tex"
