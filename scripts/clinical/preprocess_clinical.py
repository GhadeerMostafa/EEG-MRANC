"""
Preprocess CHB-MIT clinical EDF data into 32x1000 windows at 200 Hz.

CHB-MIT recordings use a bipolar montage (e.g. "FP1-F7", "F3-C3").  This
script reconstructs approximate reference-based per-electrode signals from
those bipolar pairs using a pseudo-inverse least-squares solve, then
projects the recovered electrodes into the standard 32-channel DEAP/SEED
scalp montage (zero-filling electrodes that cannot be recovered), resamples
to 200 Hz, windows into non-overlapping (32, 1000) segments, and saves
processed_clinical_data/mix.npy with shape (N_windows, 32, 1000) in float32.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from runtime import setup_src_path

setup_src_path()

from paths import PROCESSED_CLINICAL, RAW_CLINICAL

DEFAULT_INPUT_DIR = RAW_CLINICAL
DEFAULT_OUTPUT_DIR = PROCESSED_CLINICAL

TARGET_SFREQ = 200.0
WINDOW_SIZE = 1000  # samples

# Canonical 32-channel DEAP/SEED scalp montage (order must match DEAP_32_NAMES/DEAP_32_FALLBACK).
DEAP_32_NAMES: List[str] = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "Pz",
    "P3",
    "PO3",
    "O1",
    "Oz",
    "O2",
    "PO4",
    "P4",
    "CP2",
    "CP6",
    "T8",
    "C8",
    "C4",
    "FC2",
    "FC6",
    "F4",
    "AF4",
    "Fp2",
    "Fz",
    "Cz",
    "CPz",
    "POz",
]

# Known CHB-MIT label aliases → canonical DEAP_32_NAMES labels.
# Bipolar channel names in CHB-MIT sometimes use older/alternate 10-20 spellings.
_CHB_ALIAS: Dict[str, str] = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess CHB-MIT EDF recordings into 32x1000 windows at 200 Hz")
    p.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing raw clinical EDF files (e.g., chb01_01.edf)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for processed_clinical_data/mix.npy",
    )
    p.add_argument(
        "--sfreq",
        type=float,
        default=TARGET_SFREQ,
        help="Target sampling frequency in Hz (default: 200)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Window size in samples (default: 1000)",
    )
    return p.parse_args()


def find_edf_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.edf"))


def _normalise_electrode(raw: str) -> str:
    """
    Normalise CHB-MIT electrode token extracted from a bipolar pair.

    - Removes trailing disambiguation like "-0"/"-1" (e.g. "P8-0" -> "P8").
    - Drops artefact tokens like "0"/"1".
    - Applies alias mapping for known spelling differences.
    """
    name = re.sub(r"-\d+$", "", raw.strip())
    name = name.strip()
    if not name or name in {"0", "1"}:
        return ""
    upper = name.upper()
    return _CHB_ALIAS.get(upper, name.title())


def _parse_bipolar_pair(ch_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse a bipolar channel name such as "FP1-F7" or "T8-P8-0".
    Returns (anode, cathode) as normalised electrode names, or None if not bipolar.
    Only the first '-' that separates two electrode names is used as the split point.
    """
    parts = ch_name.strip().split("-", 1)
    if len(parts) != 2:
        return None
    anode = _normalise_electrode(parts[0])
    cathode = _normalise_electrode(parts[1])
    if not anode or not cathode:
        return None
    return anode, cathode


def reconstruct_reference_from_bipolar(
    ch_names: List[str],
    data: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Given bipolar montage channel names and data (n_bip, n_times), reconstruct
    approximate reference-based per-electrode signals via pseudo-inverse.

    Returns a dict {electrode_name: signal_1d} for all recoverable electrodes.

    The bipolar recording satisfies: b[i] = x[anode[i]] - x[cathode[i]]
    i.e.  B @ x = b   where B is the signed incidence matrix (n_bip x n_elec).
    We solve x = pinv(B) @ b (minimum-norm least-squares, zero-mean reference).
    """
    pairs = [_parse_bipolar_pair(ch) for ch in ch_names]
    valid_pairs = [(i, p) for i, p in enumerate(pairs) if p is not None]
    if not valid_pairs:
        return {}

    # Build sorted electrode list
    elec_set: dict[str, int] = {}
    for _, (anode, cathode) in valid_pairs:
        for e in (anode, cathode):
            if e not in elec_set:
                elec_set[e] = len(elec_set)
    elec_names = list(elec_set.keys())
    n_elec = len(elec_names)

    # Build incidence matrix B (n_valid_bip x n_elec)
    n_bip = len(valid_pairs)
    B = np.zeros((n_bip, n_elec), dtype=np.float64)
    bip_data = np.zeros((n_bip, data.shape[1]), dtype=np.float64)
    for row, (bip_idx, (anode, cathode)) in enumerate(valid_pairs):
        B[row, elec_set[anode]] = +1.0
        B[row, elec_set[cathode]] = -1.0
        bip_data[row] = data[bip_idx]

    # Pseudo-inverse: x = pinv(B) @ b  (minimum-norm solution)
    B_pinv = np.linalg.pinv(B)
    x = B_pinv @ bip_data  # (n_elec, n_times) reconstructed electrode signals

    # Robust per-channel normalization:
    # Bring reconstructed electrode amplitudes into the same variance scale the
    # model expects (microvolt baseline).
    # For each reconstructed electrode/channel: if std > 0, scale to 15.0.
    stds = x.std(axis=1)
    nonzero = stds > 0
    if np.any(nonzero):
        x[nonzero] = (x[nonzero] / stds[nonzero, None]) * 15.0

    return {name: x[i] for i, name in enumerate(elec_names)}


def build_channel_mapping_direct(raw_ch_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    For reference-montage EDF: map raw channel names to DEAP_32_NAMES positions.
    Returns (src_index_or_minus1[32], present_mask[32]).
    """
    index = {name.strip().upper(): i for i, name in enumerate(raw_ch_names)}
    src_idx = np.full(len(DEAP_32_NAMES), -1, dtype=np.int32)
    present = np.zeros(len(DEAP_32_NAMES), dtype=bool)
    for j, target_name in enumerate(DEAP_32_NAMES):
        key = target_name.strip().upper()
        if key in index:
            src_idx[j] = index[key]
            present[j] = True
    return src_idx, present


def project_to_32_channels(data: np.ndarray, src_idx: np.ndarray) -> np.ndarray:
    """Project raw data (n_src, n_times) into DEAP_32_NAMES order with zero-filling."""
    n_times = data.shape[1]
    mix_32 = np.zeros((len(DEAP_32_NAMES), n_times), dtype=np.float32)
    for j, src in enumerate(src_idx):
        if src >= 0:
            mix_32[j] = data[src].astype(np.float32, copy=False)
    return mix_32


def project_recovered_to_32_channels(
    recovered: Dict[str, np.ndarray],
    n_times: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map the dict of recovered reference-based electrode signals into the 32-channel layout.
    Returns (mix_32 float32 array, present_mask bool[32]).
    """
    # Build uppercase lookup
    index = {k.strip().upper(): v for k, v in recovered.items()}
    present = np.zeros(len(DEAP_32_NAMES), dtype=bool)
    mix_32 = np.zeros((len(DEAP_32_NAMES), n_times), dtype=np.float32)
    for j, target_name in enumerate(DEAP_32_NAMES):
        key = target_name.strip().upper()
        if key in index:
            mix_32[j] = index[key].astype(np.float32)
            present[j] = True
    return mix_32, present


def window_non_overlapping(mix_32: np.ndarray, window_size: int) -> np.ndarray:
    """Return windows with shape (n_windows, 32, window_size)."""
    n_ch, n_times = mix_32.shape
    n_windows = n_times // window_size
    if n_windows == 0:
        return np.zeros((0, n_ch, window_size), dtype=np.float32)
    trimmed = mix_32[:, : n_windows * window_size]
    windows = trimmed.reshape(n_ch, n_windows, window_size).transpose(1, 0, 2)
    return windows.astype(np.float32, copy=False)


def _is_bipolar_montage(ch_names: List[str]) -> bool:
    """Return True when most channel names look like bipolar pairs (contain '-')."""
    bipolar_count = sum(1 for ch in ch_names if _parse_bipolar_pair(ch) is not None)
    return bipolar_count > len(ch_names) / 2


def process_edf_file(path: Path, target_sfreq: float, window_size: int) -> np.ndarray:
    print(f"\nProcessing EDF: {path}")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    print(f"  Original sfreq: {raw.info['sfreq']:.3f} Hz | channels: {len(raw.ch_names)}")
    print(f"  EDF channels: {raw.ch_names}")

    if raw.info["sfreq"] != target_sfreq:
        raw = raw.copy().resample(target_sfreq)
        print(f"  Resampled to: {raw.info['sfreq']:.3f} Hz")

    data = raw.get_data()  # (n_channels, n_times) in Volts
    # Convert to microvolts for consistency with typical EEG scales.
    data = data * 1e6

    if _is_bipolar_montage(raw.ch_names):
        bipolar_channels = [ch for ch in raw.ch_names if _parse_bipolar_pair(ch) is not None]
        print(
            "  Detected bipolar montage — reconstructing reference signals via pseudo-inverse.\n"
            f"  Bipolar channels ({len(bipolar_channels)}): {bipolar_channels}"
        )
        recovered = reconstruct_reference_from_bipolar(raw.ch_names, data)
        print(f"  Recovered electrode tokens: {sorted(recovered.keys())}")
        mix_32, present = project_recovered_to_32_channels(recovered, data.shape[1])
    else:
        src_idx, present = build_channel_mapping_direct(raw.ch_names)
        mix_32 = project_to_32_channels(data, src_idx)

    present_names = [DEAP_32_NAMES[i] for i, ok in enumerate(present) if ok]
    missing_names = [DEAP_32_NAMES[i] for i, ok in enumerate(present) if not ok]
    print(f"  Mapped {present.sum()}/{len(DEAP_32_NAMES)} target channels.")
    if present_names:
        print(f"    Recovered: {', '.join(present_names)}")
    if missing_names:
        print(f"    Zero-filled: {', '.join(missing_names)}")

    windows = window_non_overlapping(mix_32, window_size)
    print(f"  Windows from file: {windows.shape[0]} (shape per window: {windows.shape[1:]})")
    return windows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    edf_files = find_edf_files(input_dir)
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows: List[np.ndarray] = []
    all_groups: List[np.ndarray] = []
    total = 0
    for session_idx, path in enumerate(edf_files):
        windows = process_edf_file(path, target_sfreq=args.sfreq, window_size=args.window_size)
        if windows.size == 0:
            print(f"  Warning: {path} produced 0 windows (too short for window_size={args.window_size}).")
            continue
        all_windows.append(windows)
        all_groups.append(np.full(windows.shape[0], session_idx, dtype=np.int32))
        total += windows.shape[0]

    if not all_windows:
        raise RuntimeError("No windows produced from any EDF file; aborting.")

    mix = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    window_groups = np.concatenate(all_groups, axis=0)
    if np.isnan(mix).any():
        raise RuntimeError("NaNs detected in processed clinical mix array.")

    print(f"\nFinal clinical mix shape: {mix.shape} (N, 32, {args.window_size})")
    if mix.shape[1] != len(DEAP_32_NAMES):
        raise RuntimeError(f"Expected 32 channels, got {mix.shape[1]}")

    out_path = output_dir / "mix.npy"
    groups_path = output_dir / "window_groups.npy"
    np.save(out_path, mix)
    np.save(groups_path, window_groups)
    print(f"Saved {out_path}")
    print(f"Saved {groups_path} (unique_sessions={len(np.unique(window_groups))})")


if __name__ == "__main__":
    main()

