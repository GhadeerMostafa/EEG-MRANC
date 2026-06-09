"""
Versioned checkpoint paths and stacking manifest (never overwrite prior weight files).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from paths import CHECKPOINTS_DIR, PROJECT_ROOT, WEIGHTS_DIR, stacking_manifest_path

PHASE_KEYS = (
    "phase1_artifact_benchmark",
    "phase2_deap",
    "phase3_seed",
    "phase4_clinical_attention",
    "baseline_eegdenoisenet",
)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def versioned_weight_path(base: Path, run_id: str) -> Path:
    """
  Return a new path that does not exist yet: {stem}_{run_id}{suffix}, then {stem}_{run_id}_001, ...
    """
    parent = base.parent
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / f"{base.stem}_{run_id}{base.suffix}"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = parent / f"{base.stem}_{run_id}_{index:03d}{base.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _read_manifest() -> dict:
    manifest_path = stacking_manifest_path()
    if not manifest_path.is_file():
        return {"latest": {}, "history": []}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"latest": {}, "history": []}
    if not isinstance(data, dict):
        return {"latest": {}, "history": []}
    data.setdefault("latest", {})
    data.setdefault("history", [])
    return data


def _write_manifest(data: dict) -> None:
    manifest_path = CHECKPOINTS_DIR / "stacking_latest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not write manifest {manifest_path}: {exc}") from exc


def _resolve_path(stored: str) -> Path:
    p = Path(stored)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def record_stacking_checkpoint(phase_key: str, path: Path, run_id: str) -> None:
    if phase_key not in PHASE_KEYS:
        raise ValueError(f"Unknown phase key: {phase_key}")
    try:
        rel = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(path)
    data = _read_manifest()
    data["latest"][phase_key] = rel
    data["history"].append(
        {
            "run_id": run_id,
            "phase": phase_key,
            "path": rel,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _write_manifest(data)


def resolve_stacking_latest(phase_key: str, fallback: Path | None = None) -> Path | None:
    data = _read_manifest()
    stored = data.get("latest", {}).get(phase_key)
    if stored:
        path = _resolve_path(stored)
        if path.is_file():
            return path
    if fallback is not None and fallback.is_file():
        return fallback
    return None


def resolve_stacking_latest_required(phase_key: str, fallback: Path) -> Path:
    path = resolve_stacking_latest(phase_key, fallback)
    if path is not None and path.is_file():
        return path
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No weights found for {phase_key}. Run stacking or place weights at {fallback}"
    )
