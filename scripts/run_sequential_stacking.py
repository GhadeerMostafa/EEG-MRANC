"""
Automated 4-phase sequential knowledge-stacking training pipeline.

Each run writes new versioned weight files (never overwrites prior checkpoints).
Latest paths are recorded in checkpoints/stacking_latest.json.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from checkpoint_paths import (
    new_run_id,
    record_stacking_checkpoint,
    resolve_stacking_latest_required,
    versioned_weight_path,
)
from paths import (
    ARTIFACT_BENCHMARK_CHECKPOINT,
    CHECKPOINT_PHASE2_DEAP,
    CHECKPOINT_PHASE3_SEED,
    FINAL_ATTENTION_WEIGHTS,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 4-phase MRANC sequential stacking")
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id for versioned filenames (default: timestamp)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return p.parse_args()


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or new_run_id()
    logger.info("Stacking run_id=%s (all phase outputs use new versioned filenames)", run_id)

    py = sys.executable
    train = str(SCRIPTS_DIR / "train.py")
    adapter = str(SCRIPTS_DIR / "train_attention_adapter.py")

    phase1_out = versioned_weight_path(ARTIFACT_BENCHMARK_CHECKPOINT, run_id)
    logger.info("Phase 1 output -> %s", phase1_out)
    if phase1_out.is_file() and not args.dry_run:
        logger.info("Phase 1 output already exists for run_id=%s; skipping training", run_id)
    else:
        run_cmd(
            [
                py,
                train,
                "--dataset",
                "artifact_benchmark",
                "--storage",
                "cuda",
                "--epochs",
                "50",
                "--batch-size",
                "32",
                "--checkpoint-out",
                rel_path(phase1_out),
            ],
            args.dry_run,
        )
    if not args.dry_run:
        if not phase1_out.is_file():
            raise FileNotFoundError(f"Phase 1 checkpoint was not created: {phase1_out}")
        record_stacking_checkpoint("phase1_artifact_benchmark", phase1_out, run_id)

    phase1_resume = phase1_out if not args.dry_run else resolve_stacking_latest_required(
        "phase1_artifact_benchmark", ARTIFACT_BENCHMARK_CHECKPOINT
    )

    phase2_out = versioned_weight_path(CHECKPOINT_PHASE2_DEAP, run_id)
    logger.info("Phase 2 output -> %s", phase2_out)
    if phase2_out.is_file() and not args.dry_run:
        logger.info("Phase 2 output already exists for run_id=%s; skipping training", run_id)
    else:
        run_cmd(
            [
                py,
                train,
                "--dataset",
                "deap",
                "--storage",
                "cuda",
                "--epochs",
                "20",
                "--batch-size",
                "128",
                "--lr",
                "1e-3",
                "--resume_weights",
                rel_path(phase1_resume),
                "--checkpoint-out",
                rel_path(phase2_out),
            ],
            args.dry_run,
        )
    if not args.dry_run:
        if not phase2_out.is_file():
            raise FileNotFoundError(f"Phase 2 checkpoint was not created: {phase2_out}")
        record_stacking_checkpoint("phase2_deap", phase2_out, run_id)

    phase2_resume = phase2_out if not args.dry_run else resolve_stacking_latest_required(
        "phase2_deap", CHECKPOINT_PHASE2_DEAP
    )

    phase3_out = versioned_weight_path(CHECKPOINT_PHASE3_SEED, run_id)
    logger.info("Phase 3 output -> %s", phase3_out)
    if phase3_out.is_file() and not args.dry_run:
        logger.info("Phase 3 output already exists for run_id=%s; skipping training", run_id)
    else:
        run_cmd(
            [
                py,
                train,
                "--dataset",
                "seed",
                "--storage",
                "cuda",
                "--epochs",
                "20",
                "--batch-size",
                "64",
                "--lr",
                "1e-3",
                "--resume_weights",
                rel_path(phase2_resume),
                "--checkpoint-out",
                rel_path(phase3_out),
            ],
            args.dry_run,
        )
    if not args.dry_run:
        if not phase3_out.is_file():
            raise FileNotFoundError(f"Phase 3 checkpoint was not created: {phase3_out}")
        record_stacking_checkpoint("phase3_seed", phase3_out, run_id)

    phase3_resume = phase3_out if not args.dry_run else resolve_stacking_latest_required(
        "phase3_seed", CHECKPOINT_PHASE3_SEED
    )

    phase4_out = versioned_weight_path(FINAL_ATTENTION_WEIGHTS, run_id)
    logger.info("Phase 4 output -> %s", phase4_out)
    if phase4_out.is_file() and not args.dry_run:
        logger.info("Phase 4 output already exists for run_id=%s; skipping training", run_id)
    else:
        run_cmd(
            [
                py,
                adapter,
                "--resume-weights",
                rel_path(phase3_resume),
                "--output",
                rel_path(phase4_out),
            ],
            args.dry_run,
        )
    if not args.dry_run:
        if not phase4_out.is_file():
            raise FileNotFoundError(f"Phase 4 weights were not created: {phase4_out}")
        record_stacking_checkpoint("phase4_clinical_attention", phase4_out, run_id)

    logger.info("Sequential stacking complete. run_id=%s manifest updated.", run_id)


if __name__ == "__main__":
    main()
