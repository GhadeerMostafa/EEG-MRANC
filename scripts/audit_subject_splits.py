"""
Audit subject/session-independent train/validation splits for all MRANC datasets.

Usage:
  py scripts/audit_subject_splits.py --dataset all
  py scripts/audit_subject_splits.py --dataset deap --val-fraction 0.1 --split-seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from group_splits import ALL_SPLIT_DATASETS, audit_dataset_split, load_window_groups
from paths import DATASET_DIRS, DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRACTION, resolve_processed_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify group-wise train/val splits have zero subject/session overlap",
    )
    p.add_argument(
        "--dataset",
        choices=(*ALL_SPLIT_DATASETS, "all"),
        default="all",
    )
    p.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--data-dir", type=str, default=None, help="Override processed data directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datasets = list(ALL_SPLIT_DATASETS) if args.dataset == "all" else [args.dataset]
    failed = False

    for name in datasets:
        data_dir = Path(args.data_dir) if args.data_dir else resolve_processed_dir(name)
        print(f"\n=== {name} ({data_dir}) ===")
        try:
            groups = load_window_groups(data_dir)
            print(f"windows={groups.shape[0]} unique_groups={len(set(groups.tolist()))}")
            audit_dataset_split(
                name,
                val_fraction=args.val_fraction,
                split_seed=args.split_seed,
                data_dir=data_dir,
            )
        except FileNotFoundError as exc:
            failed = True
            print(f"SKIP {name}: {exc}")
        except ValueError as exc:
            failed = True
            print(f"FAIL {name}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
