MRANC Journal Submission Figures
================================

This folder holds PNG exports referenced by the submission LaTeX manuscript at
docs/manuscript/MRANC_Final_Research_Report.tex.

Populate by running:

  py scripts/run_report.py --refresh
  py scripts/transform_report.py

Each dataset has its own subdirectory:

  figures/clinical/
  figures/seed/
  figures/deap/
  figures/artifact_benchmark/

PNG files are gitignored. Only this README and .gitkeep placeholders are tracked.
