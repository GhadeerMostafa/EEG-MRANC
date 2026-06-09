"""
Verify Python, CUDA, and LaTeX dependencies for the MRANC pipeline.

  py scripts/check_dependencies.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

REQUIRED_PYTHON = (
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "docx",
    "sklearn",
    "mne",
)


def check_python_packages() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_PYTHON:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def check_cuda() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "CUDA not available (training requires a GPU)"
    return True, f"CUDA available ({torch.cuda.get_device_name(0)})"


def check_pdflatex() -> tuple[bool, str]:
    path = shutil.which("pdflatex")
    if path is None:
        local = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
        if local.is_file():
            path = str(local)
        else:
            return False, "pdflatex not found (run .\\scripts\\install_latex_local.ps1)"
    try:
        proc = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"pdflatex check failed: {exc}"
    first_line = (proc.stdout or proc.stderr or "").strip().splitlines()
    detail = first_line[0] if first_line else str(path)
    return proc.returncode == 0, detail


def main() -> int:
    print("MRANC dependency check")
    print("=" * 40)

    missing = check_python_packages()
    if missing:
        print(f"FAIL  Python packages missing: {', '.join(missing)}")
        print("      pip install -r requirements.txt")
    else:
        print("OK    Python packages")

    cuda_ok, cuda_msg = check_cuda()
    print(f"{'OK' if cuda_ok else 'WARN'}  {cuda_msg}")

    latex_ok, latex_msg = check_pdflatex()
    print(f"{'OK' if latex_ok else 'FAIL'}  {latex_msg}")
    if not latex_ok:
        print("      See requirements-latex.txt or run .\\scripts\\install_latex_local.ps1")

    if missing or not latex_ok:
        return 1
    print("All required dependencies satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
