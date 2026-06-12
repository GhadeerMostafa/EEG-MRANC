"""
Backfill window_groups.npy for processed datasets that pre-date group-wise splits.

Prefer re-running full preprocessing scripts when raw sources are available.
This utility covers quick backfills from existing metadata where possible.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from runtime import setup_src_path

setup_src_path()

from paths import DATASET_DIRS


def backfill_artifact(root: Path) -> None:
    mix = np.load(root / "mix.npy", mmap_mode="r")
    groups = np.zeros(mix.shape[0], dtype=np.int32)
    np.save(root / "window_groups.npy", groups)
    (root / "split_policy.json").write_text(
        json.dumps({"split_policy": "contiguous"}, indent=2),
        encoding="utf-8",
    )
    print(f"artifact_benchmark: saved window_groups.npy shape={groups.shape}")


def backfill_seed(root: Path) -> None:
    meta_path = root / "conversion_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; run extract_and_convert_seed.py")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    per_sim = meta.get("sim_stats", {})
    groups: list[int] = []
    total_windows = int(meta.get("total_windows", 0))
    for key, info in per_sim.items():
        match = re.search(r"sim(\d+)", key, re.IGNORECASE)
        if not match:
            continue
        sim_id = int(match.group(1))
        n_win = int(info.get("windows", 0))
        groups.extend([sim_id] * n_win)
    if not groups and total_windows:
        raise ValueError("Could not reconstruct SEED groups from conversion_meta.json")
    arr = np.asarray(groups, dtype=np.int32)
    mix_n = int(np.load(root / "mix.npy", mmap_mode="r").shape[0])
    if arr.shape[0] != mix_n:
        raise ValueError(f"SEED groups length {arr.shape[0]} != mix windows {mix_n}")
    np.save(root / "window_groups.npy", arr)
    print(f"seed: saved window_groups.npy shape={arr.shape} unique={len(np.unique(arr))}")


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill window_groups.npy for processed datasets")
    p.add_argument(
        "--dataset",
        choices=("artifact_benchmark", "seed", "all"),
        default="all",
    )
    args = p.parse_args()
    targets = ["artifact_benchmark", "seed"] if args.dataset == "all" else [args.dataset]
    failed = False
    for name in targets:
        root = DATASET_DIRS[name]
        try:
            if name == "artifact_benchmark":
                backfill_artifact(root)
            elif name == "seed":
                backfill_seed(root)
        except (FileNotFoundError, ValueError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
