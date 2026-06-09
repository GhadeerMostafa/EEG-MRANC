"""
Download, parse, window, and convert the SEEDA semi-simulated EEG/EOG dataset
into train/val .npy arrays for multi-stem source separation training.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from runtime import setup_src_path
from scipy.io import loadmat

setup_src_path()

from paths import PROJECT_ROOT, SEEDA_WORKSPACE
from setup_folders import create_folders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/wb6yvr725d/files/"
    "f98fa4c3-ff39-49c5-a6fc-f0c39a5ca2d9/file_downloaded"
)

REQUIRED_MAT_FILES = (
    "Contaminated_Data.mat",
    "Pure_Data.mat",
    "VEOG.mat",
    "HEOG.mat",
)

SFREQ = 200
WINDOW_SEC = 2
WINDOW_SAMPLES = SFREQ * WINDOW_SEC  # 400 time steps at 200 Hz
TRAIN_RATIO = 0.8

OUTPUT_STEMS = {
    "mix.npy": "mix",
    "target_eeg.npy": "eeg",
    "target_ecg.npy": "ecg",
    "target_emg.npy": "emg",
    "target_eog.npy": "eog",
    "target_noise.npy": "noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert SEEDA dataset to project .npy format"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=MENDELEY_URL,
        help="Direct download URL for the SEEDA zip archive",
    )
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--workspace", type=str, default=".seeda_workspace")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an existing workspace/seeda.zip and skip network download",
    )
    parser.add_argument(
        "--placeholder-std-scale",
        type=float,
        default=0.01,
        help="ECG/EMG Gaussian scale as fraction of per-recording mix std",
    )
    return parser.parse_args()


def download_zip(url: str, dest: Path, force: bool = False) -> Path:
    """Stream-download the SEEDA archive with progress logging."""
    if dest.exists() and not force:
        logger.info("Using existing zip: %s (%.2f MB)", dest, dest.stat().st_size / 1e6)
        return dest

    logger.info("Downloading SEEDA dataset from Mendeley...")
    logger.info("URL: %s", url)

    headers = {
        "User-Agent": "Mozilla/5.0 (SEEDA-downloader; +https://data.mendeley.com/)",
        "Accept": "*/*",
    }
    with requests.get(
        url,
        stream=True,
        timeout=300,
        headers=headers,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        content_type = response.headers.get("Content-Type", "")
        downloaded = 0
        chunk_size = 1024 * 1024

        with open(dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    logger.info(
                        "Download progress: %.1f%% (%.2f / %.2f MB)",
                        pct,
                        downloaded / 1e6,
                        total / 1e6,
                    )
                else:
                    logger.info("Downloaded %.2f MB...", downloaded / 1e6)

    size_bytes = dest.stat().st_size
    logger.info(
        "Download finished: %s bytes (Content-Type=%s, Content-Length=%s)",
        size_bytes,
        content_type or "unknown",
        total if total > 0 else "unknown",
    )

    # Validate that we really downloaded a zip archive. Mendeley direct links may
    # sometimes return HTML (or a rate-limit/empty response) even with 200 OK.
    if size_bytes < 1024 or not zipfile.is_zipfile(dest):
        # Mendeley sometimes returns an HTML landing page or an empty response when the
        # direct link is rate-limited / blocked. Fail fast with a clear message.
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded file is not a valid zip archive. "
            "The Mendeley direct link may have returned an empty/HTML response. "
            "Try again later, run with --url pointing to a fresh direct-download link, "
            "or download the zip manually from https://data.mendeley.com/datasets/wb6yvr725d/4 "
            "and place it at workspace/seeda.zip, then re-run without --force-download."
        )

    logger.info("Download complete: %s (%.2f MB)", dest, size_bytes / 1e6)
    return dest


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract archive into workspace/extracted."""
    if not zipfile.is_zipfile(zip_path):
        raise zipfile.BadZipFile(f"File is not a zip file: {zip_path}")
    if extract_dir.exists():
        logger.info("Removing previous extraction: %s", extract_dir)
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", zip_path, extract_dir)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
        logger.info("Extracted %d file(s)", len(archive.namelist()))

    return extract_dir


def find_mat_files(extract_dir: Path) -> dict[str, Path]:
    """Locate required .mat files anywhere under the extracted tree."""
    all_mats = {p.name: p for p in extract_dir.rglob("*.mat")}
    logger.info("Found %d .mat file(s) under %s", len(all_mats), extract_dir)

    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in REQUIRED_MAT_FILES:
        if name in all_mats:
            found[name] = all_mats[name]
            logger.info("  %s -> %s", name, all_mats[name])
        else:
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"Missing required MATLAB files: {missing}. "
            f"Available: {sorted(all_mats.keys())}"
        )
    return found


def _recording_sort_key(key: str) -> int:
    """Sort SEEDA variables like sim12_con, sim12_resampled, veog_12, heog_12."""
    match = re.search(r"(\d+)\s*$", key)
    if not match:
        match = re.search(r"(\d+)", key)
    return int(match.group(1)) if match else 0


def load_mat_recordings(path: Path, layout: str) -> list[np.ndarray]:
    """
    Load all per-recording variables from a SEEDA .mat file.

    SEEDA stores 54 simulations as separate variables (e.g. sim1_con, veog_1),
    not as a single cell array.

    Args:
        path: path to .mat file
        layout: 'eeg' for (channels, time) matrices, 'eog' for 1D time series
    """
    logger.info("Loading %s", path)
    mat_dict = loadmat(
        path,
        simplify_cells=True,
        squeeze_me=True,
        verify_compressed_data_integrity=False,
    )

    keys = sorted(
        [
            k
            for k, v in mat_dict.items()
            if not k.startswith("__") and isinstance(v, np.ndarray)
        ],
        key=_recording_sort_key,
    )
    if not keys:
        raise ValueError(f"No ndarray variables in {path}. Keys: {list(mat_dict.keys())}")

    logger.info(
        "  %d variable(s): first=%s last=%s",
        len(keys),
        keys[0],
        keys[-1],
    )

    recordings: list[np.ndarray] = []
    for key in keys:
        arr = np.asarray(mat_dict[key])
        if layout == "eeg":
            rec = _to_2d_recording(arr).astype(np.float32)
        elif layout == "eog":
            rec = np.squeeze(arr).astype(np.float32)
            if rec.ndim != 1:
                rec = rec.reshape(-1)
        else:
            raise ValueError(f"Unknown layout '{layout}' (use 'eeg' or 'eog')")
        recordings.append(rec)

    logger.info(
        "  %s: extracted %d recording(s), example shape=%s",
        path.name,
        len(recordings),
        recordings[0].shape,
    )
    return recordings


def _to_2d_recording(arr: np.ndarray) -> np.ndarray:
    """Coerce a single recording to float32 shape (channels, time)."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        # Prefer (channels, time): more channels than time -> already (C, T)
        if arr.shape[0] <= arr.shape[1]:
            return arr
        return arr.T
    raise ValueError(f"Cannot coerce array with ndim={arr.ndim} to (C, T)")


def iter_recordings(data: np.ndarray, source_name: str) -> list[np.ndarray]:
    """
    Flatten MATLAB cell arrays or stacked recordings into list of (C, T) arrays.

    Handles:
      - 2D (C, T): single recording
      - 3D (C, T, N): N recordings along last axis
      - object/cell dtype: walk cells in C-order
    """
    data = np.asarray(data)

    if data.dtype == object:
        recordings: list[np.ndarray] = []
        flat = data.ravel()
        for idx, cell in enumerate(flat):
            if cell is None:
                continue
            try:
                rec = _to_2d_recording(cell)
                recordings.append(rec.astype(np.float32))
            except ValueError as exc:
                logger.warning("Skipping cell %d in %s: %s", idx, source_name, exc)
        if not recordings:
            raise ValueError(f"No valid recordings extracted from {source_name}")
        logger.info(
            "  %s: %d recording(s) from cell array, example shape (C,T)=%s",
            source_name,
            len(recordings),
            recordings[0].shape,
        )
        return recordings

    if data.ndim == 1:
        logger.info("  %s: 1 EOG trace, shape (T,)=%s", source_name, data.shape)
        return [np.squeeze(data).astype(np.float32)]

    if data.ndim == 2:
        rec = _to_2d_recording(data).astype(np.float32)
        logger.info("  %s: 1 recording, shape (C,T)=%s", source_name, rec.shape)
        return [rec]

    if data.ndim == 3:
        # (C, T, N) recordings along last axis
        recordings = [
            data[:, :, i].astype(np.float32) for i in range(data.shape[2])
        ]
        logger.info(
            "  %s: %d recording(s) from 3D stack, shape (C,T)=%s",
            source_name,
            len(recordings),
            recordings[0].shape,
        )
        return recordings

    raise ValueError(
        f"Unsupported array layout in {source_name}: ndim={data.ndim}, dtype={data.dtype}"
    )


def align_eog_to_eeg(eog_1d: np.ndarray, n_channels: int, n_time: int) -> np.ndarray:
    """Broadcast summed VEOG+HEOG (1, T) to (C, T) across EEG channels."""
    eog_1d = np.squeeze(eog_1d).astype(np.float32)
    if eog_1d.ndim != 1:
        eog_1d = eog_1d.reshape(-1)
    if eog_1d.shape[0] != n_time:
        min_t = min(eog_1d.shape[0], n_time)
        eog_1d = eog_1d[:min_t]
        n_time = min_t
    return np.tile(eog_1d[np.newaxis, :], (n_channels, 1))


def trim_recording_pair(
    mix: np.ndarray,
    pure: np.ndarray,
    veog: np.ndarray,
    heog: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Trim all signals to common channel count (EEG) and min time length."""
    n_channels = mix.shape[0]
    min_t = min(mix.shape[1], pure.shape[1], veog.shape[-1], heog.shape[-1])

    mix_t = mix[:, :min_t]
    pure_t = pure[:, :min_t]

    veog_1d = np.squeeze(veog).astype(np.float32).reshape(-1)[:min_t]
    heog_1d = np.squeeze(heog).astype(np.float32).reshape(-1)[:min_t]
    eog_sum = veog_1d + heog_1d
    eog_ct = align_eog_to_eeg(eog_sum, n_channels, min_t)

    return mix_t, pure_t, eog_ct, eog_sum, min_t


def window_recording(
    mix: np.ndarray,
    pure: np.ndarray,
    eog: np.ndarray,
    rng: np.random.Generator,
    placeholder_scale: float,
) -> dict[str, list[np.ndarray]]:
    """
    Chop one (C, T) aligned recording into 2-second windows and build stem lists.

    Returns dict of stem_name -> list of (C, 400) windows.
    """
    n_channels, n_time = mix.shape
    n_windows = n_time // WINDOW_SAMPLES
    dropped = n_time - n_windows * WINDOW_SAMPLES

    if n_windows == 0:
        logger.warning(
            "Recording too short for windowing: T=%d, need %d",
            n_time,
            WINDOW_SAMPLES,
        )
        return {k: [] for k in ("mix", "eeg", "ecg", "emg", "eog", "noise")}

    if dropped > 0:
        logger.debug("Dropping %d trailing samples (T=%d)", dropped, n_time)

    mix_std = float(np.std(mix)) if np.std(mix) > 0 else 1.0
    placeholder_std = placeholder_scale * mix_std

    stems: dict[str, list[np.ndarray]] = {
        "mix": [],
        "eeg": [],
        "ecg": [],
        "emg": [],
        "eog": [],
        "noise": [],
    }

    for w in range(n_windows):
        start = w * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES

        mix_w = mix[:, start:end]
        eeg_w = pure[:, start:end]
        eog_w = eog[:, start:end]

        ecg_w = rng.normal(0.0, placeholder_std, size=(n_channels, WINDOW_SAMPLES)).astype(
            np.float32
        )
        emg_w = rng.normal(0.0, placeholder_std, size=(n_channels, WINDOW_SAMPLES)).astype(
            np.float32
        )
        noise_w = (mix_w - eeg_w - eog_w).astype(np.float32)

        stems["mix"].append(mix_w)
        stems["eeg"].append(eeg_w)
        stems["ecg"].append(ecg_w)
        stems["emg"].append(emg_w)
        stems["eog"].append(eog_w)
        stems["noise"].append(noise_w)

    return stems


def stack_stem_lists(stem_lists: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack lists of (C, T) windows into (N, C, T) arrays."""
    stacked: dict[str, np.ndarray] = {}
    for name, windows in stem_lists.items():
        if not windows:
            raise ValueError(f"No windows collected for stem '{name}'")
        stacked[name] = np.stack(windows, axis=0).astype(np.float32)
        logger.info("  stacked %s: shape=%s", name, stacked[name].shape)
    return stacked


def train_val_split(
    arrays: dict[str, np.ndarray],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Shuffle windows and split into train / val dicts."""
    n_total = arrays["mix"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_train = int(n_total * train_ratio)
    if n_train == 0 or n_train == n_total:
        raise ValueError(
            f"Invalid split: n_total={n_total}, n_train={n_train}. "
            "Need at least 2 windows for 80/20 split."
        )

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train = {k: v[train_idx] for k, v in arrays.items()}
    val = {k: v[val_idx] for k, v in arrays.items()}

    logger.info(
        "Split: train=%d val=%d (ratio=%.0f/%.0f)",
        train_idx.size,
        val_idx.size,
        train_ratio * 100,
        (1 - train_ratio) * 100,
    )
    return train, val


def save_split(split_name: str, arrays: dict[str, np.ndarray], out_dir: Path) -> None:
    """Write six .npy files for one split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {
        "mix": "mix.npy",
        "eeg": "target_eeg.npy",
        "ecg": "target_ecg.npy",
        "emg": "target_emg.npy",
        "eog": "target_eog.npy",
        "noise": "target_noise.npy",
    }
    for stem_key, filename in file_map.items():
        path = out_dir / filename
        np.save(path, arrays[stem_key])
        logger.info("Saved %s/%s shape=%s", split_name, filename, arrays[stem_key].shape)


def verify_conservation(arrays: dict[str, np.ndarray], split_name: str) -> None:
    """Spot-check mix ≈ eeg + eog + noise (ECG/EMG are near-zero placeholders)."""
    recon = arrays["eeg"] + arrays["eog"] + arrays["noise"]
    err = float(np.mean((arrays["mix"] - recon) ** 2))
    logger.info(
        "[%s] conservation MSE(mix, eeg+eog+noise) = %.6e (ECG/EMG excluded)",
        split_name,
        err,
    )


def convert_seeda(
    mat_paths: dict[str, Path],
    rng: np.random.Generator,
    placeholder_scale: float,
) -> dict[str, np.ndarray]:
    """Load all .mat files, align recordings, window, and stack."""
    mix_recs = load_mat_recordings(mat_paths["Contaminated_Data.mat"], layout="eeg")
    pure_recs = load_mat_recordings(mat_paths["Pure_Data.mat"], layout="eeg")
    veog_recs = load_mat_recordings(mat_paths["VEOG.mat"], layout="eog")
    heog_recs = load_mat_recordings(mat_paths["HEOG.mat"], layout="eog")

    n_recs = min(len(mix_recs), len(pure_recs), len(veog_recs), len(heog_recs))
    if n_recs == 0:
        raise ValueError("No recordings found in SEEDA .mat files")

    if len({len(mix_recs), len(pure_recs), len(veog_recs), len(heog_recs)}) > 1:
        logger.warning(
            "Recording count mismatch: mix=%d pure=%d veog=%d heog=%d; using min=%d",
            len(mix_recs),
            len(pure_recs),
            len(veog_recs),
            len(heog_recs),
            n_recs,
        )

    master_lists: dict[str, list[np.ndarray]] = {
        k: [] for k in ("mix", "eeg", "ecg", "emg", "eog", "noise")
    }
    total_windows = 0

    for i in range(n_recs):
        mix_t, pure_t, eog_ct, _, min_t = trim_recording_pair(
            mix_recs[i], pure_recs[i], veog_recs[i], heog_recs[i]
        )
        logger.info(
            "Recording %d/%d: C=%d T=%d (trimmed)",
            i + 1,
            n_recs,
            mix_t.shape[0],
            min_t,
        )

        rec_stems = window_recording(
            mix_t, pure_t, eog_ct, rng, placeholder_scale
        )
        for key in master_lists:
            master_lists[key].extend(rec_stems[key])
        total_windows += len(rec_stems["mix"])

    logger.info(
        "Total windows: %d (each %d samples @ %d Hz = %.1f s)",
        total_windows,
        WINDOW_SAMPLES,
        SFREQ,
        WINDOW_SEC,
    )

    return stack_stem_lists(master_lists)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dataset_root = PROJECT_ROOT / args.dataset_root
    workspace = PROJECT_ROOT / args.workspace
    zip_path = workspace / "seeda.zip"
    extract_dir = workspace / "extracted"

    create_folders(PROJECT_ROOT)

    workspace.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        if not zip_path.exists():
            raise FileNotFoundError(
                f"--skip-download was set but zip not found: {zip_path}. "
                "Download the SEEDA zip manually and place it at this path."
            )
        logger.info("Skipping download; using local zip: %s", zip_path)
    else:
        download_zip(args.url, zip_path, force=args.force_download)
    extract_zip(zip_path, extract_dir)

    if not args.keep_zip and not args.skip_download:
        logger.info("Removing zip archive (use --keep-zip to retain)")
        zip_path.unlink(missing_ok=True)

    mat_paths = find_mat_files(extract_dir)
    arrays = convert_seeda(mat_paths, rng, args.placeholder_std_scale)

    train_arrays, val_arrays = train_val_split(arrays, TRAIN_RATIO, args.seed)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    save_split("train", train_arrays, train_dir)
    save_split("val", val_arrays, val_dir)

    verify_conservation(train_arrays, "train")
    verify_conservation(val_arrays, "val")

    n, c, t = train_arrays["mix"].shape
    bytes_est = n * c * t * 4 * 6
    logger.info(
        "Done. Train shape (N,C,T)=(%d,%d,%d), ~%.2f MB per split (6 float32 arrays)",
        n,
        c,
        t,
        bytes_est / 1e6,
    )


if __name__ == "__main__":
    main()
