"""
Convert DEAP preprocessed .dat subject files into windowed numpy arrays.

Expected DEAP shape per file:
    data: (40 videos, 40 channels, 8064 time points)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from runtime import setup_src_path

setup_src_path()

from paths import DEAP_RAW, PROCESSED_DEAP, PROJECT_ROOT

SAMPLE_RATE = 128
WINDOW_SIZE = 256  # 2 seconds at 128 Hz
HOP_SIZE = 128  # 50% overlap


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DEAP .dat files into windowed channel-group arrays."
    )
    parser.add_argument("--raw-dir", type=str, default=str(DEAP_RAW))
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DEAP))
    return parser.parse_args()


def load_deap_file(dat_path: Path) -> np.ndarray:
    """Load one DEAP .dat file and return data array with shape (40, 40, 8064)."""
    with dat_path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")

    if not isinstance(payload, dict):
        raise ValueError(f"{dat_path.name}: expected dict payload, got {type(payload)}")
    if "data" not in payload:
        raise ValueError(f"{dat_path.name}: missing 'data' key")

    data = np.asarray(payload["data"])
    if data.ndim != 3:
        raise ValueError(
            f"{dat_path.name}: expected 3D data (videos, channels, time), got {data.shape}"
        )

    videos, channels, time_points = data.shape
    if channels != 40 or time_points != 8064:
        raise ValueError(
            f"{dat_path.name}: expected channels/time = (40, 8064), got ({channels}, {time_points})"
        )
    if videos != 40:
        logger.warning("%s: expected 40 videos, got %d", dat_path.name, videos)

    return data.astype(np.float32, copy=False)


def window_trial(trial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice one trial (40, 8064) into overlapping windows and channel groups.

    Returns
    -------
    mix : (W, 32, 256)
    ref_eog : (W, 2, 256)
    ref_emg : (W, 2, 256)
    ref_ecg : (W, 1, 256)
    """
    if trial.shape != (40, 8064):
        raise ValueError(f"trial shape must be (40, 8064), got {trial.shape}")

    starts = range(0, 8064 - WINDOW_SIZE + 1, HOP_SIZE)

    mix_windows: list[np.ndarray] = []
    eog_windows: list[np.ndarray] = []
    emg_windows: list[np.ndarray] = []
    ecg_windows: list[np.ndarray] = []

    for start in starts:
        end = start + WINDOW_SIZE
        segment = trial[:, start:end]  # (40, 256)
        mix_windows.append(segment[0:32, :])
        eog_windows.append(segment[32:34, :])
        emg_windows.append(segment[34:36, :])
        ecg_windows.append(segment[36:37, :])

    return (
        np.stack(mix_windows, axis=0),
        np.stack(eog_windows, axis=0),
        np.stack(emg_windows, axis=0),
        np.stack(ecg_windows, axis=0),
    )


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    dat_files = sorted(raw_dir.rglob("s*.dat"))
    if not dat_files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in {raw_dir}. Expected files like s01.dat, s02.dat."
        )

    logger.info("Found %d DEAP file(s) recursively in %s", len(dat_files), raw_dir)
    for dat_path in dat_files:
        logger.info("  discovered: %s", dat_path.relative_to(raw_dir))

    all_mix: list[np.ndarray] = []
    all_eog: list[np.ndarray] = []
    all_emg: list[np.ndarray] = []
    all_ecg: list[np.ndarray] = []

    for dat_path in dat_files:
        data = load_deap_file(dat_path)  # (40, 40, 8064)
        file_windows = 0

        for video_idx in range(data.shape[0]):
            trial = data[video_idx]  # (40, 8064)
            mix_w, eog_w, emg_w, ecg_w = window_trial(trial)
            all_mix.append(mix_w)
            all_eog.append(eog_w)
            all_emg.append(emg_w)
            all_ecg.append(ecg_w)
            file_windows += mix_w.shape[0]

        logger.info(
            "%s -> videos=%d windows=%d",
            dat_path.name,
            data.shape[0],
            file_windows,
        )

    mix = np.concatenate(all_mix, axis=0)
    ref_eog = np.concatenate(all_eog, axis=0)
    ref_emg = np.concatenate(all_emg, axis=0)
    ref_ecg = np.concatenate(all_ecg, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)

    mix_path = out_dir / "mix.npy"
    eog_path = out_dir / "ref_eog.npy"
    emg_path = out_dir / "ref_emg.npy"
    ecg_path = out_dir / "ref_ecg.npy"

    np.save(mix_path, mix)
    np.save(eog_path, ref_eog)
    np.save(emg_path, ref_emg)
    np.save(ecg_path, ref_ecg)

    logger.info("Saved %s shape=%s", mix_path, mix.shape)
    logger.info("Saved %s shape=%s", eog_path, ref_eog.shape)
    logger.info("Saved %s shape=%s", emg_path, ref_emg.shape)
    logger.info("Saved %s shape=%s", ecg_path, ref_ecg.shape)
    logger.info("Done.")


if __name__ == "__main__":
    main()

