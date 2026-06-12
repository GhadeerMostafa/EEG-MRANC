"""
One-time migration to per-dataset output layout.

Moves misplaced critical figures and evaluation JSON into canonical paths.
Dry-run by default; pass --apply to execute moves.

  py scripts/migrate_output_layout.py
  py scripts/migrate_output_layout.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from paths import CRITICAL_FIGURES_DIR, PROJECT_ROOT, REPORTS_DATASET_DIRS

DATASETS = ("clinical", "deap", "seed", "artifact_benchmark")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate outputs to per-dataset layout")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute moves (default is dry-run only)",
    )
    return p.parse_args()


def safe_move(src: Path, dst: Path, apply: bool, log: list[str]) -> None:
    if not src.is_file():
        return
    if dst.exists():
        log.append(f"SKIP (exists): {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if apply:
        shutil.move(str(src), str(dst))
        log.append(f"MOVED: {src} -> {dst}")
    else:
        log.append(f"WOULD MOVE: {src} -> {dst}")


def migrate_critical_figures_root(apply: bool, log: list[str]) -> None:
    clinical_dir = CRITICAL_FIGURES_DIR / "clinical"
    clinical_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("tuh_*.png", "clinical_*.png"):
        for png in CRITICAL_FIGURES_DIR.glob(pattern):
            safe_move(png, clinical_dir / png.name, apply, log)

    root_manifest = CRITICAL_FIGURES_DIR / "decomposition_manifest.json"
    if root_manifest.is_file():
        safe_move(root_manifest, clinical_dir / "decomposition_manifest.json", apply, log)


def migrate_evaluation_from_critical_figures(apply: bool, log: list[str]) -> None:
    misplaced = CRITICAL_FIGURES_DIR / "clinical" / "evaluation_report_clinical.json"
    if not misplaced.is_file():
        return
    target = REPORTS_DATASET_DIRS["clinical"] / "evaluation_report_clinical.json"
    if target.is_file():
        if apply:
            misplaced.unlink()
            log.append(f"REMOVED duplicate: {misplaced} (canonical at {target})")
        else:
            log.append(f"WOULD REMOVE duplicate: {misplaced} (canonical exists at {target})")
        return
    safe_move(misplaced, target, apply, log)


def archive_flat_reports(apply: bool, log: list[str]) -> None:
    reports_root = PROJECT_ROOT / "outputs" / "reports"
    archive = reports_root / "_archive"
    if not reports_root.is_dir():
        return

    for path in sorted(reports_root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix != ".json":
            continue
        dst = archive / path.name
        safe_move(path, dst, apply, log)


def main() -> int:
    args = parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"migrate_output_layout.py [{mode}]")
    log: list[str] = []

    migrate_critical_figures_root(args.apply, log)
    migrate_evaluation_from_critical_figures(args.apply, log)
    archive_flat_reports(args.apply, log)

    if not log:
        print("Nothing to migrate.")
    else:
        for line in log:
            print(line)

    print(f"Done ({len(log)} actions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
