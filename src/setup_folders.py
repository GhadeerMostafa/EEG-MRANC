"""
Create the canonical v2 folder structure for the MRANC project.
Run: py src/setup_folders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from paths import PROJECT_ROOT  # noqa: E402

DATASETS = ("deap", "seed", "clinical", "artifact_benchmark")

REL_FOLDERS = (
    "docs/manuscript",
    "scripts/deap",
    "scripts/seed",
    "scripts/clinical",
    "data/raw/deap",
    "data/raw/seed",
    "data/raw/clinical",
    "data/raw/artifact_benchmark",
    "data/processed/deap",
    "data/processed/seed",
    "data/processed/clinical",
    "data/processed/artifact_benchmark",
    "artifacts/models/checkpoints",
    "artifacts/models/weights",
    "artifacts/figures/critical/deap",
    "artifacts/figures/critical/seed",
    "artifacts/figures/critical/clinical",
    "artifacts/figures/critical/artifact_benchmark",
    "artifacts/figures/manuscript/deap",
    "artifacts/figures/manuscript/seed",
    "artifacts/figures/manuscript/clinical",
    "artifacts/figures/manuscript/artifact_benchmark",
    "artifacts/figures/manuscript/misc",
    "artifacts/figures/legacy/deap",
    "artifacts/figures/legacy/seed",
    "artifacts/figures/legacy/clinical",
    "artifacts/figures/legacy/artifact_benchmark",
    "artifacts/reports/deap",
    "artifacts/reports/seed",
    "artifacts/reports/clinical",
    "artifacts/reports/artifact_benchmark",
    "artifacts/reports/latex",
    "dataset/train",
    "dataset/val",
)

MANUSCRIPT_FIGURES_README = """MRANC Journal Submission Figures
================================

PNG exports referenced by docs/manuscript/MRANC_Final_Research_Report.tex.

Populate by running:

  py scripts/run_report.py --refresh
  py scripts/transform_report.py

Subdirectories: clinical, seed, deap, artifact_benchmark, misc
PNG files are gitignored. Only this README and .gitkeep placeholders are tracked.
"""


def create_folders(root: Path | None = None) -> list[Path]:
    root = root or PROJECT_ROOT
    created: list[Path] = []

    for name in REL_FOLDERS:
        path = root / name
        if path.exists():
            print(f"  exists: {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
            print(f"  created: {path}")

    readme = root / "artifacts" / "figures" / "manuscript" / "README.txt"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(MANUSCRIPT_FIGURES_README, encoding="utf-8")
    print(f"  updated: {readme}")

    return created


if __name__ == "__main__":
    print("Setting up project folders (v2 layout)...")
    create_folders()
    print("Done.")
    print("Migrate existing data from legacy paths:")
    print("  py scripts/migrate_project_layout.py --apply")
