"""
SEEDA 19-to-32 spatial mapping converter.

Extracts .seeda_workspace/seeda.zip -> seed_raw, loads Contaminated_Data.mat (sim*_con),
maps 19-channel 10-20 montage into DEAP-compatible 32-channel layout via direct placement
and nearest-neighbor interpolation, then windows (1000 samples), DEAP std scaling,
and saves processed_data_seed/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from runtime import setup_src_path
from scipy.io import loadmat
from scipy.signal import detrend

setup_src_path()

from paths import PROCESSED_DEAP, PROCESSED_SEED, PROJECT_ROOT, SEEDA_WORKSPACE, SEED_RAW

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCALP_CHANNELS = 32
EOG_REF_CHANNELS = 2
EMG_REF_CHANNELS = 2
ECG_REF_CHANNELS = 1

SIM_CON_PATTERN = re.compile(r"^sim(\d+)_con$", re.IGNORECASE)
VEOG_PATTERN = re.compile(r"^veog_(\d+)$", re.IGNORECASE)
HEOG_PATTERN = re.compile(r"^heog_(\d+)$", re.IGNORECASE)

# SEEDA contaminated EEG row order (Klados et al., 19 electrodes)
SEEDA_19_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

# DEAP 32-scalp montage (first 32 channels of DEAP 40-ch recording)
DEAP_32_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz", "P3",
    "PO3", "O1", "Oz", "O2", "PO4", "P4", "CP2", "CP6", "T8", "C8", "C4", "FC2",
    "FC6", "F4", "AF4", "Fp2", "Fz", "Cz", "CPz", "POz",
]

# Legacy SEEDA names -> DEAP 10-10 names for direct placement when present in DEAP_32
SEEDA_TO_DEAP_LABEL: dict[str, str] = {
    "FP1": "Fp1", "FP2": "Fp2", "F3": "F3", "F4": "F4",
    "F7": "F7", "F8": "F8", "C3": "C3", "C4": "C4",
    "P3": "P3", "P4": "P4", "O1": "O1", "O2": "O2",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

# Approximate 2D cap positions (unit circle, nose = +Y) for distance-based interpolation
CAP_XY: dict[str, tuple[float, float]] = {
    "Fp1": (-0.30, 0.95), "Fp2": (0.30, 0.95), "F3": (-0.45, 0.55), "F4": (0.45, 0.55),
    "F7": (-0.75, 0.45), "F8": (0.75, 0.45), "FC5": (-0.65, 0.35), "FC1": (-0.35, 0.35),
    "FC2": (0.35, 0.35), "FC6": (0.65, 0.35), "C3": (-0.50, 0.05), "C4": (0.50, 0.05),
    "T3": (-0.85, 0.00), "T4": (0.85, 0.00), "T7": (-0.85, 0.00), "T8": (0.85, 0.00),
    "T5": (-0.70, -0.25), "T6": (0.70, -0.25), "P7": (-0.70, -0.25), "P8": (0.70, -0.25),
    "CP5": (-0.55, -0.15), "CP1": (-0.25, -0.15), "CP2": (0.25, -0.15), "CP6": (0.55, -0.15),
    "CPz": (0.00, -0.20), "Pz": (0.00, -0.35), "P3": (-0.40, -0.45), "P4": (0.40, -0.45),
    "PO3": (-0.30, -0.70), "PO4": (0.30, -0.70), "POz": (0.00, -0.85), "PO7": (-0.45, -0.75),
    "PO8": (0.45, -0.75), "O1": (-0.25, -0.90), "O2": (0.25, -0.90), "Oz": (0.00, -0.95),
    "Fz": (0.00, 0.55), "Cz": (0.00, 0.05), "AF3": (-0.20, 0.75), "AF4": (0.20, 0.75),
    "C8": (0.75, 0.05),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEEDA 19-to-32 spatial converter")
    p.add_argument("--zip-path", type=str, default=str(SEEDA_WORKSPACE / "seeda.zip"))
    p.add_argument("--raw-dir", type=str, default=str(SEED_RAW))
    p.add_argument("--out-dir", type=str, default=str(PROCESSED_SEED))
    p.add_argument("--deap-data-dir", type=str, default=str(PROCESSED_DEAP))
    p.add_argument("--window-samples", type=int, default=1000)
    p.add_argument("--target-std", type=float, default=0.0, help="Override DEAP mix std (0=auto)")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--interpolation-neighbors", type=int, default=3)
    return p.parse_args()


def norm_name(name: str) -> str:
    return name.strip().upper()


def cap_position(label: str) -> np.ndarray:
    key = label if label in CAP_XY else label.upper()
    if key not in CAP_XY:
        for k, v in CAP_XY.items():
            if k.upper() == label.upper():
                return np.array(v, dtype=np.float64)
        raise KeyError(f"No cap coordinates for electrode {label}")
    return np.array(CAP_XY[key], dtype=np.float64)


class SpatialMapper19to32:
    """Map SEEDA (19, T) contaminated EEG to DEAP-order (32, T)."""

    def __init__(self, k_neighbors: int = 3) -> None:
        self.k_neighbors = k_neighbors
        self.deap_names = list(DEAP_32_NAMES)
        self.deap_index = {norm_name(n): i for i, n in enumerate(self.deap_names)}
        self.deap_xy = np.stack([cap_position(n) for n in self.deap_names], axis=0)
        self.fill_method: list[str] = ["empty"] * SCALP_CHANNELS
        self.channel_map: list[dict] = []
        self._build_seed_to_deap_index()

    def _build_seed_to_deap_index(self) -> None:
        self.seed_to_deap: list[int | None] = []
        for i, seed_name in enumerate(SEEDA_19_NAMES):
            target = SEEDA_TO_DEAP_LABEL.get(norm_name(seed_name), seed_name)
            idx = self.deap_index.get(norm_name(target))
            self.seed_to_deap.append(idx)
            self.channel_map.append(
                {
                    "seeda_index": i,
                    "seeda_name": seed_name,
                    "target_deap_name": target,
                    "deap_index": idx,
                    "placement": "pending",
                }
            )

    def _nearest_deap_index(self, label: str) -> int:
        pos = cap_position(label)
        dist = np.linalg.norm(self.deap_xy - pos, axis=1)
        return int(np.argmin(dist))

    def transform(self, eeg_19: np.ndarray) -> np.ndarray:
        if eeg_19.shape[0] != 19:
            raise ValueError(f"Expected 19 channels, got {eeg_19.shape[0]}")
        t_len = eeg_19.shape[1]
        out = np.full((SCALP_CHANNELS, t_len), np.nan, dtype=np.float32)
        filled = np.zeros(SCALP_CHANNELS, dtype=bool)

        for i in range(19):
            row = eeg_19[i].astype(np.float32)
            deap_idx = self.seed_to_deap[i]
            if deap_idx is None:
                target = SEEDA_TO_DEAP_LABEL.get(norm_name(SEEDA_19_NAMES[i]), SEEDA_19_NAMES[i])
                deap_idx = self._nearest_deap_index(target)
                self.channel_map[i]["placement"] = f"nearest_to_{target}"
            else:
                self.channel_map[i]["placement"] = "direct"
            self.channel_map[i]["deap_index"] = deap_idx

            if filled[deap_idx]:
                dist = np.linalg.norm(self.deap_xy - cap_position(SEEDA_19_NAMES[i]), axis=1)
                dist[filled] += 1e6
                deap_idx = int(np.argmin(dist))
                self.channel_map[i]["placement"] += "_relocated"
            out[deap_idx] = row
            filled[deap_idx] = True
            self.fill_method[deap_idx] = "seeda_direct"

        baseline = np.nanmean(out[filled], axis=0) if filled.any() else np.zeros(t_len, dtype=np.float32)
        filled_indices = np.where(filled)[0]
        empty_indices = np.where(~filled)[0]

        for j in empty_indices:
            if filled_indices.size == 0:
                out[j] = baseline
                self.fill_method[j] = "spatial_mean"
                continue

            dist = np.linalg.norm(self.deap_xy[filled_indices] - self.deap_xy[j], axis=1)
            k = min(self.k_neighbors, filled_indices.size)
            nn = filled_indices[np.argsort(dist)[:k]]
            w = 1.0 / (dist[np.argsort(dist)[:k]] + 1e-6)
            w /= w.sum()
            out[j] = np.average(out[nn], axis=0, weights=w)
            self.fill_method[j] = f"interp_k{k}"

        if np.isnan(out).any():
            raise RuntimeError("Spatial mapping left NaN values in 32-channel matrix")

        return out.astype(np.float32)

    def summary(self) -> dict:
        methods = {}
        for m in self.fill_method:
            methods[m] = methods.get(m, 0) + 1
        return {
            "deap_channels": self.deap_names,
            "fill_method_counts": methods,
            "channel_map": self.channel_map,
        }


def extract_zip(zip_path: Path, raw_dir: Path, force: bool) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    if raw_dir.exists() and any(raw_dir.iterdir()) and not force:
        logger.info("Using existing extraction: %s", raw_dir)
        return
    if raw_dir.exists() and force:
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", zip_path, raw_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)
        logger.info("Extracted %d file(s)", len(archive.namelist()))


def find_mat_file(raw_dir: Path, filename: str) -> Path:
    matches = list(raw_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {raw_dir}")
    return matches[0]


def _to_2d_recording(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            return arr
        return arr.T
    raise ValueError(f"Cannot coerce array with ndim={arr.ndim} to (C, T)")


def load_mat_dict(path: Path) -> dict[str, np.ndarray]:
    logger.info("Loading %s", path)
    mat_dict = loadmat(
        path,
        simplify_cells=True,
        squeeze_me=True,
        verify_compressed_data_integrity=False,
    )
    return {
        k: np.asarray(v)
        for k, v in mat_dict.items()
        if not k.startswith("__") and isinstance(v, np.ndarray)
    }


def sim_con_recordings(mat: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, arr in mat.items():
        match = SIM_CON_PATTERN.match(key)
        if not match:
            continue
        sim_id = int(match.group(1))
        rec = _to_2d_recording(arr).astype(np.float32)
        if rec.shape[0] != 19:
            raise ValueError(f"{key}: expected 19 channels, got {rec.shape[0]}")
        out[sim_id] = rec
    if not out:
        raise ValueError("No sim*_con variables found in Contaminated_Data.mat")
    logger.info("Contaminated_Data: %d simulations (sim%d..sim%d)", len(out), min(out), max(out))
    return dict(sorted(out.items()))


def eog_by_sim(mat: dict[str, np.ndarray], pattern: re.Pattern[str]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, arr in mat.items():
        match = pattern.match(key)
        if not match:
            continue
        out[int(match.group(1))] = np.squeeze(arr).astype(np.float32).reshape(-1)
    return out


def preprocess_channels(x: np.ndarray) -> np.ndarray:
    """Per-channel linear detrend + zero-mean. x: (C, T)."""
    out = np.empty_like(x, dtype=np.float64)
    for c in range(x.shape[0]):
        row = detrend(x[c].astype(np.float64), type="linear")
        out[c] = row - row.mean()
    return out.astype(np.float32)


def preprocess_1d(x: np.ndarray) -> np.ndarray:
    row = detrend(x.astype(np.float64), type="linear")
    return (row - row.mean()).astype(np.float32)


def window_starts(n_samples: int, window_samples: int) -> list[int]:
    return [w * window_samples for w in range(n_samples // window_samples)]


def deap_target_std(deap_dir: Path, override: float) -> float:
    if override > 0:
        return override
    mix_path = deap_dir / "mix.npy"
    if not mix_path.exists():
        raise FileNotFoundError(f"DEAP mix not found: {mix_path}")
    mmap = np.load(mix_path, mmap_mode="r")
    step = max(1, mmap.shape[0] // 500)
    sample = np.array(mmap[::step], dtype=np.float64)
    std = float(np.std(sample))
    logger.info("DEAP target std (subsampled): %.6f from %s", std, mix_path)
    return std


def apply_global_scale(arrays: list[np.ndarray], target_std: float) -> tuple[list[np.ndarray], float, float]:
    stacked = np.stack(arrays, axis=0)
    current_std = float(np.std(stacked))
    scale = target_std / max(current_std, 1e-8)
    return [(a * scale).astype(np.float32) for a in arrays], current_std, scale


def convert_all(
    contaminated: dict[int, np.ndarray],
    veog: dict[int, np.ndarray],
    heog: dict[int, np.ndarray],
    window_samples: int,
    k_neighbors: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict, SpatialMapper19to32]:
    mix_windows: list[np.ndarray] = []
    eog_windows: list[np.ndarray] = []
    emg_windows: list[np.ndarray] = []
    ecg_windows: list[np.ndarray] = []
    stats: dict = {"per_sim": {}, "total_windows": 0}

    mapper = SpatialMapper19to32(k_neighbors=k_neighbors)

    for sim_id, eeg_19 in contaminated.items():
        if sim_id not in veog or sim_id not in heog:
            logger.warning("Skipping sim%d: missing VEOG/HEOG", sim_id)
            continue

        min_t = min(eeg_19.shape[1], veog[sim_id].shape[0], heog[sim_id].shape[0])
        eeg_19 = eeg_19[:, :min_t]
        veog_1d = preprocess_1d(veog[sim_id][:min_t])
        heog_1d = preprocess_1d(heog[sim_id][:min_t])

        mix_32 = mapper.transform(eeg_19[:, :min_t])
        mix_32 = preprocess_channels(mix_32)

        starts = window_starts(min_t, window_samples)
        n_win = len(starts)
        dropped = min_t - n_win * window_samples
        stats["per_sim"][f"sim{sim_id}_con"] = {
            "time_samples": min_t,
            "windows": n_win,
            "dropped_trailing": dropped,
        }

        for start in starts:
            end = start + window_samples
            mix_windows.append(mix_32[:, start:end].copy())
            eog_windows.append(
                np.stack([veog_1d[start:end], heog_1d[start:end]], axis=0).astype(np.float32)
            )
            emg_windows.append(np.zeros((EMG_REF_CHANNELS, window_samples), dtype=np.float32))
            ecg_windows.append(np.zeros((ECG_REF_CHANNELS, window_samples), dtype=np.float32))

        logger.info("sim%d: T=%d -> %d windows (dropped %d)", sim_id, min_t, n_win, dropped)

    stats["total_windows"] = len(mix_windows)
    if not mix_windows:
        raise ValueError("No windows produced")

    return mix_windows, eog_windows, emg_windows, ecg_windows, stats, mapper


def save_outputs(
    out_dir: Path,
    mix: np.ndarray,
    ref_eog: np.ndarray,
    ref_emg: np.ndarray,
    ref_ecg: np.ndarray,
    meta: dict,
    channel_map: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mix.npy", mix)
    np.save(out_dir / "ref_eog.npy", ref_eog)
    np.save(out_dir / "ref_emg.npy", ref_emg)
    np.save(out_dir / "ref_ecg.npy", ref_ecg)
    (out_dir / "conversion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "channel_map.json").write_text(json.dumps(channel_map, indent=2), encoding="utf-8")
    logger.info("Saved %s/mix.npy shape=%s", out_dir, mix.shape)
    logger.info("Saved %s/ref_eog.npy shape=%s", out_dir, ref_eog.shape)


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    deap_dir = Path(args.deap_data_dir)

    extract_zip(zip_path, raw_dir, args.force_extract)

    contaminated_path = find_mat_file(raw_dir, "Contaminated_Data.mat")
    veog_path = find_mat_file(raw_dir, "VEOG.mat")
    heog_path = find_mat_file(raw_dir, "HEOG.mat")

    contaminated = sim_con_recordings(load_mat_dict(contaminated_path))
    veog = eog_by_sim(load_mat_dict(veog_path), VEOG_PATTERN)
    heog = eog_by_sim(load_mat_dict(heog_path), HEOG_PATTERN)

    mix_list, eog_list, emg_list, ecg_list, sim_stats, mapper = convert_all(
        contaminated,
        veog,
        heog,
        args.window_samples,
        args.interpolation_neighbors,
    )

    target_std = deap_target_std(deap_dir, args.target_std)
    mix_list, cur_std, scale = apply_global_scale(mix_list, target_std)
    eog_list, _, _ = apply_global_scale(eog_list, target_std)
    emg_list, _, _ = apply_global_scale(emg_list, target_std)
    ecg_list, _, _ = apply_global_scale(ecg_list, target_std)

    mix = np.stack(mix_list, axis=0)
    ref_eog = np.stack(eog_list, axis=0)
    ref_emg = np.stack(emg_list, axis=0)
    ref_ecg = np.stack(ecg_list, axis=0)

    mapper_info = mapper.summary()
    logger.info("Spatial fill methods: %s", mapper_info["fill_method_counts"])

    meta = {
        "source_zip": str(zip_path),
        "raw_dir": str(raw_dir),
        "mapping": "19_to_32_spatial",
        "window_samples": args.window_samples,
        "target_std": target_std,
        "pre_scale_std": cur_std,
        "scale_factor": scale,
        "post_scale_std_mix": float(np.std(mix)),
        "sim_stats": sim_stats,
        "shapes": {
            "mix": list(mix.shape),
            "ref_eog": list(ref_eog.shape),
            "ref_emg": list(ref_emg.shape),
            "ref_ecg": list(ref_ecg.shape),
        },
        "spatial_fill_method_counts": mapper_info["fill_method_counts"],
    }

    save_outputs(out_dir, mix, ref_eog, ref_emg, ref_ecg, meta, mapper_info)

    ch0 = mix[0, 0]
    ch1 = mix[0, 1]
    diversity = float(np.mean(np.abs(ch0 - ch1)))
    logger.info(
        "Done. N=%d windows | mix std=%.4f | ch0-ch1 mean abs diff=%.4f (non-broadcast check)",
        mix.shape[0],
        float(np.std(mix)),
        diversity,
    )


if __name__ == "__main__":
    main()
