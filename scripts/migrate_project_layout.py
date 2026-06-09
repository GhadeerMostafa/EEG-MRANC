"""
One-time migration from legacy root-level folders to the v2 layout.

Canonical layout:
  data/raw/{dataset}/
  data/processed/{dataset}/
  artifacts/models/{checkpoints,weights}/
  artifacts/figures/{critical,manuscript,legacy}/
  artifacts/reports/

Dry-run by default. Pass --apply to execute moves.

  py scripts/migrate_project_layout.py
  py scripts/migrate_project_layout.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from paths import CHECKPOINTS_DIR, PROJECT_ROOT, stacking_manifest_path

DATASETS = ("deap", "seed", "clinical", "artifact_benchmark")

MOVE_MAP: list[tuple[Path, Path]] = [
    (PROJECT_ROOT / "deap_raw", PROJECT_ROOT / "data/raw/deap"),
    (PROJECT_ROOT / "seed_raw", PROJECT_ROOT / "data/raw/seed"),
    (PROJECT_ROOT / "raw_clinical_data", PROJECT_ROOT / "data/raw/clinical"),
    (PROJECT_ROOT / "data/artifact_benchmark/raw", PROJECT_ROOT / "data/raw/artifact_benchmark"),
    (PROJECT_ROOT / "processed_data", PROJECT_ROOT / "data/processed/deap"),
    (PROJECT_ROOT / "processed_data_seed", PROJECT_ROOT / "data/processed/seed"),
    (PROJECT_ROOT / "processed_clinical_data", PROJECT_ROOT / "data/processed/clinical"),
    (PROJECT_ROOT / "data/artifact_benchmark/processed", PROJECT_ROOT / "data/processed/artifact_benchmark"),
    (PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "artifacts/models/checkpoints"),
    (PROJECT_ROOT / "weights", PROJECT_ROOT / "artifacts/models/weights"),
    (PROJECT_ROOT / "critical_figures", PROJECT_ROOT / "artifacts/figures/critical"),
    (PROJECT_ROOT / "figures", PROJECT_ROOT / "artifacts/figures/manuscript"),
    (PROJECT_ROOT / "outputs/figures", PROJECT_ROOT / "artifacts/figures/legacy"),
    (PROJECT_ROOT / "outputs/reports", PROJECT_ROOT / "artifacts/reports"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Migrate legacy folders to v2 project layout")
    p.add_argument("--apply", action="store_true", help="Execute moves (default is dry-run)")
    return p.parse_args()


def merge_dir_contents(src: Path, dst: Path, apply: bool, log: list[str]) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == ".gitkeep":
            continue
        target = dst / item.name
        if item.is_dir():
            merge_dir_contents(item, target, apply, log)
            continue
        if target.exists():
            log.append(f"SKIP (exists): {target}")
            continue
        if apply:
            shutil.move(str(item), str(target))
            log.append(f"MOVED: {item} -> {target}")
        else:
            log.append(f"WOULD MOVE: {item} -> {target}")


def migrate_tree(src: Path, dst: Path, apply: bool, log: list[str]) -> None:
    if not src.exists():
        return
    if dst.exists() and any(dst.iterdir()):
        merge_dir_contents(src, dst, apply, log)
        if apply and src.is_dir() and not any(src.iterdir()):
            src.rmdir()
            log.append(f"REMOVED empty dir: {src}")
        return
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            merge_dir_contents(src, dst, apply, log)
            if src.is_dir():
                shutil.rmtree(src, ignore_errors=True)
        else:
            shutil.move(str(src), str(dst))
        log.append(f"MOVED tree: {src} -> {dst}")
    else:
        log.append(f"WOULD MOVE tree: {src} -> {dst}")


def rewrite_manifest_paths(apply: bool, log: list[str]) -> None:
    manifest = stacking_manifest_path()
    if not manifest.is_file():
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    replacements = {
        "checkpoints\\": "artifacts/models/checkpoints\\",
        "checkpoints/": "artifacts/models/checkpoints/",
        "weights\\": "artifacts/models/weights\\",
        "weights/": "artifacts/models/weights/",
    }

    def rewrite(value: str) -> str:
        out = value
        for old, new in replacements.items():
            out = out.replace(old, new)
        return out

    changed = False
    latest = data.get("latest", {})
    for key, path in list(latest.items()):
        new_path = rewrite(str(path))
        if new_path != path:
            latest[key] = new_path
            changed = True
    for entry in data.get("history", []):
        path = entry.get("path", "")
        new_path = rewrite(str(path))
        if new_path != path:
            entry["path"] = new_path
            changed = True

    if not changed:
        log.append("Manifest paths already canonical")
        return

    target = CHECKPOINTS_DIR / "stacking_latest.json"
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.append(f"UPDATED manifest: {target}")
    else:
        log.append(f"WOULD UPDATE manifest at {target}")


def main() -> int:
    args = parse_args()
    log: list[str] = []
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"migrate_project_layout.py ({mode})")

    for src, dst in MOVE_MAP:
        migrate_tree(src, dst, args.apply, log)

    rewrite_manifest_paths(args.apply, log)

    for line in log:
        print(line)
    if not log:
        print("Nothing to migrate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
