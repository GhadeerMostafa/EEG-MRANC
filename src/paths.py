"""Project root and canonical path constants (repo root = parent of src/)."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Canonical layout (v2) ---
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_PROCESSED = DATA_ROOT / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
WEIGHTS_DIR = MODELS_DIR / "weights"
FIGURES_ROOT = ARTIFACTS_DIR / "figures"
CRITICAL_FIGURES_DIR = FIGURES_ROOT / "critical"
MANUSCRIPT_FIGURES_DIR = FIGURES_ROOT / "manuscript"
FIGURES_DIR = FIGURES_ROOT / "legacy"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Raw data (canonical)
DEAP_RAW = DATA_RAW / "deap"
SEED_RAW = DATA_RAW / "seed"
RAW_CLINICAL = DATA_RAW / "clinical"
ARTIFACT_BENCHMARK_RAW = DATA_RAW / "artifact_benchmark"

# Processed data (canonical)
PROCESSED_DEAP = DATA_PROCESSED / "deap"
PROCESSED_SEED = DATA_PROCESSED / "seed"
PROCESSED_CLINICAL = DATA_PROCESSED / "clinical"
ARTIFACT_BENCHMARK_PROCESSED = DATA_PROCESSED / "artifact_benchmark"

# Legacy paths (pre-v2 layout; used for automatic fallback)
_LEGACY_DEAP_RAW = PROJECT_ROOT / "deap_raw"
_LEGACY_SEED_RAW = PROJECT_ROOT / "seed_raw"
_LEGACY_RAW_CLINICAL = PROJECT_ROOT / "raw_clinical_data"
_LEGACY_AB_RAW = PROJECT_ROOT / "data" / "artifact_benchmark" / "raw"
_LEGACY_PROCESSED_DEAP = PROJECT_ROOT / "processed_data"
_LEGACY_PROCESSED_SEED = PROJECT_ROOT / "processed_data_seed"
_LEGACY_PROCESSED_CLINICAL = PROJECT_ROOT / "processed_clinical_data"
_LEGACY_AB_PROCESSED = PROJECT_ROOT / "data" / "artifact_benchmark" / "processed"
_LEGACY_CHECKPOINTS = PROJECT_ROOT / "checkpoints"
_LEGACY_WEIGHTS = PROJECT_ROOT / "weights"
_LEGACY_CRITICAL_FIGURES = PROJECT_ROOT / "critical_figures"
_LEGACY_MANUSCRIPT_FIGURES = PROJECT_ROOT / "figures"
_LEGACY_FIGURES = PROJECT_ROOT / "outputs" / "figures"
_LEGACY_REPORTS = PROJECT_ROOT / "outputs" / "reports"

SEEDA_WORKSPACE = PROJECT_ROOT / ".seeda_workspace"
ARTIFACT_BENCHMARK_DIR = DATA_RAW / "artifact_benchmark"
MANUSCRIPT_DIR = PROJECT_ROOT / "docs" / "manuscript"
MANUSCRIPT_TEX_PATH = MANUSCRIPT_DIR / "MRANC_Final_Research_Report.tex"
LATEX_REPORT_DIR = REPORTS_DIR / "latex"
SYSTEM_REPORT_TEX_PATH = LATEX_REPORT_DIR / "MRANC_Final_Research_Report.tex"
SYSTEM_LATEX_ASSETS_DIR = LATEX_REPORT_DIR / "latex_assets"

OUTPUTS_DIR = ARTIFACTS_DIR
FINAL_ATTENTION_WEIGHTS = WEIGHTS_DIR / "mranc_final_attention.pth"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "best_mranc.pth"
ARTIFACT_BENCHMARK_CHECKPOINT = (
    CHECKPOINTS_DIR / "best_mranc_artifact_benchmark_weights.pth"
)
CHECKPOINT_PHASE2_DEAP = CHECKPOINTS_DIR / "best_mranc_phase2_deap.pth"
CHECKPOINT_PHASE3_SEED = CHECKPOINTS_DIR / "best_mranc_phase3_seed.pth"
BASELINE_EEGDENOISENET_CHECKPOINT = CHECKPOINTS_DIR / "baseline_eegdenoisenet.pth"
def stacking_manifest_path() -> Path:
    canonical = CHECKPOINTS_DIR / "stacking_latest.json"
    legacy = _LEGACY_CHECKPOINTS / "stacking_latest.json"
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    return canonical


STACKING_MANIFEST_PATH = CHECKPOINTS_DIR / "stacking_latest.json"
MAIN_CHECKPOINT = ARTIFACT_BENCHMARK_CHECKPOINT

BASELINE_DATASETS = ("clinical", "deap", "seed")
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 42

FINAL_REPORT_PATH = REPORTS_DIR / "MRANC_Final_Research_Report.docx"
CLINICAL_FINAL_REPORT_PATH = REPORTS_DIR / "clinical" / "MRANC_Final_Research_Report.docx"

FIGURES_DATASET_DIRS = {
    "seed": FIGURES_DIR / "seed",
    "deap": FIGURES_DIR / "deap",
    "clinical": FIGURES_DIR / "clinical",
    "artifact_benchmark": FIGURES_DIR / "artifact_benchmark",
}
REPORTS_DATASET_DIRS = {
    "seed": REPORTS_DIR / "seed",
    "deap": REPORTS_DIR / "deap",
    "clinical": REPORTS_DIR / "clinical",
    "artifact_benchmark": REPORTS_DIR / "artifact_benchmark",
}
CRITICAL_FIGURES_DATASET_DIRS = {
    "seed": CRITICAL_FIGURES_DIR / "seed",
    "deap": CRITICAL_FIGURES_DIR / "deap",
    "clinical": CRITICAL_FIGURES_DIR / "clinical",
    "artifact_benchmark": CRITICAL_FIGURES_DIR / "artifact_benchmark",
}
DATASET_DIRS = {
    "seed": PROCESSED_SEED,
    "deap": PROCESSED_DEAP,
    "clinical": PROCESSED_CLINICAL,
    "artifact_benchmark": ARTIFACT_BENCHMARK_PROCESSED,
}
RAW_DATASET_DIRS = {
    "seed": SEED_RAW,
    "deap": DEAP_RAW,
    "clinical": RAW_CLINICAL,
    "artifact_benchmark": ARTIFACT_BENCHMARK_RAW,
}


def resolve_existing_path(canonical: Path, *legacy: Path) -> Path:
    """Return the first path that exists on disk, else the canonical target."""
    for candidate in (canonical, *legacy):
        if candidate.exists():
            return candidate
    return canonical


def resolve_raw_dir(dataset: str) -> Path:
    legacy = {
        "deap": (_LEGACY_DEAP_RAW,),
        "seed": (_LEGACY_SEED_RAW,),
        "clinical": (_LEGACY_RAW_CLINICAL,),
        "artifact_benchmark": (_LEGACY_AB_RAW,),
    }
    return resolve_existing_path(RAW_DATASET_DIRS[dataset], *legacy.get(dataset, ()))


def resolve_processed_dir(dataset: str) -> Path:
    legacy = {
        "deap": (_LEGACY_PROCESSED_DEAP,),
        "seed": (_LEGACY_PROCESSED_SEED,),
        "clinical": (_LEGACY_PROCESSED_CLINICAL,),
        "artifact_benchmark": (_LEGACY_AB_PROCESSED,),
    }
    return resolve_existing_path(DATASET_DIRS[dataset], *legacy.get(dataset, ()))


def resolve_checkpoints_dir() -> Path:
    return resolve_existing_path(CHECKPOINTS_DIR, _LEGACY_CHECKPOINTS)


def resolve_weights_dir() -> Path:
    return resolve_existing_path(WEIGHTS_DIR, _LEGACY_WEIGHTS)


def resolve_critical_figures_dir() -> Path:
    return resolve_existing_path(CRITICAL_FIGURES_DIR, _LEGACY_CRITICAL_FIGURES)


def resolve_manuscript_figures_dir() -> Path:
    return resolve_existing_path(MANUSCRIPT_FIGURES_DIR, _LEGACY_MANUSCRIPT_FIGURES)


def resolve_reports_dir() -> Path:
    return resolve_existing_path(REPORTS_DIR, _LEGACY_REPORTS)


def resolve_clinical_weights() -> Path:
    """Latest phase-4 clinical adapter weights, else legacy final path if present."""
    from checkpoint_paths import resolve_stacking_latest

    latest = resolve_stacking_latest("phase4_clinical_attention", FINAL_ATTENTION_WEIGHTS)
    if latest is not None:
        return latest
    legacy_final = _LEGACY_WEIGHTS / "mranc_final_attention.pth"
    if legacy_final.is_file():
        return legacy_final
    return FINAL_ATTENTION_WEIGHTS


def evaluation_report_path(dataset: str) -> Path:
    return resolve_reports_dir() / dataset / f"evaluation_report_{dataset}.json"


def baseline_ica_metrics_path(dataset: str) -> Path:
    return resolve_reports_dir() / dataset / "baseline_ica_metrics.json"


def baseline_eegdenoisenet_metrics_path(dataset: str) -> Path:
    return resolve_reports_dir() / dataset / "baseline_eegdenoisenet_metrics.json"
