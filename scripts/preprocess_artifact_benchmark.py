"""
Preprocess the SSVEP Artifact Benchmark dataset into MRANC-ready numpy arrays.

Dataset source:
  - Downloaded via scripts/download_artifact_benchmark.sh (MNE OSF-hosted archive)
  - Archive contains BrainVision files; this script converts them to a single FIF file,
    then reloads using mne.io.read_raw_fif() (as required by the pipeline spec).

Outputs:
  data/artifact_benchmark/processed/mix.npy      float32, shape (N, 32, T) in microvolts
  data/artifact_benchmark/processed/channel_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
from runtime import setup_src_path

setup_src_path()

from paths import (  # noqa: E402
    ARTIFACT_BENCHMARK_PROCESSED,
    ARTIFACT_BENCHMARK_RAW,
)

DEAP_32_NAMES = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess SSVEP Artifact Benchmark into mix.npy windows")
    p.add_argument("--raw-dir", type=str, default=str(ARTIFACT_BENCHMARK_RAW))
    p.add_argument("--out-dir", type=str, default=str(ARTIFACT_BENCHMARK_PROCESSED))
    p.add_argument("--t", type=int, default=1000, help="Window length in samples after resampling")
    p.add_argument("--sfreq", type=float, default=200.0, help="Target sampling rate (Hz)")
    p.add_argument("--highpass", type=float, default=0.5)
    p.add_argument("--lowpass", type=float, default=40.0)
    p.add_argument("--max-windows", type=int, default=0, help="0 = no limit")
    return p.parse_args()


def find_first_vhdr(root: Path) -> Path:
    candidates = sorted(
        p
        for p in root.rglob("*.vhdr")
        if "__MACOSX" not in {part.upper() for part in p.parts}
        and not p.name.startswith("._")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .vhdr found under {root}. Did you run scripts/download_artifact_benchmark.sh?"
        )
    return candidates[0]


def standardize_eeg_only(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy()
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
    # Use common average reference for stability across devices.
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def to_deap32_layout(raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[str]]:
    """
    Return (data_uV_32xN, channel_labels_32).

    - Uses intersection with the DEAP 32 montage names when present.
    - If some names are missing, pads remaining channels with zeros.
    """
    ch_name_map = {c.upper(): c for c in raw.ch_names}
    picks: list[int] = []
    labels: list[str] = []
    for name in DEAP_32_NAMES:
        key = name.upper()
        if key in ch_name_map:
            picks.append(raw.ch_names.index(ch_name_map[key]))
            labels.append(name)
        else:
            picks.append(-1)
            labels.append(name)

    data_v = raw.get_data()  # Volts, shape (n_ch, n_samples)
    n = data_v.shape[1]
    out_uV = np.zeros((32, n), dtype=np.float32)
    for out_i, pick in enumerate(picks):
        if pick >= 0:
            out_uV[out_i] = (data_v[pick] * 1e6).astype(np.float32, copy=False)
    return out_uV, labels


def window_contiguous(x: np.ndarray, t: int, max_windows: int = 0) -> np.ndarray:
    """
    x: (C, N)
    Returns: (W, C, t) non-overlapping windows.
    """
    c, n = x.shape
    w = n // t
    if max_windows and w > max_windows:
        w = max_windows
    n_use = w * t
    if w <= 0:
        raise ValueError(f"Not enough samples ({n}) for a single window of T={t}")
    return x[:, :n_use].reshape(c, w, t).transpose(1, 0, 2).copy()


def ensure_fif(raw_dir: Path) -> Path:
    """
    Ensure a single FIF exists in raw_dir.
    If missing, convert the first BrainVision recording to FIF.
    """
    fif = sorted(raw_dir.rglob("*.fif"))
    if fif:
        return fif[0]

    vhdr = find_first_vhdr(raw_dir)
    print(f"Converting BrainVision -> FIF: {vhdr}")
    raw_bv = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    fif_path = raw_dir / "artifact_benchmark_raw.fif"
    raw_bv.save(fif_path, overwrite=True)
    return fif_path


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fif_path = ensure_fif(raw_dir)
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    raw = standardize_eeg_only(raw)

    # Filter + resample to match our MRANC presets (SEED/Clinical-like).
    raw.filter(l_freq=float(args.highpass), h_freq=float(args.lowpass), verbose=False)
    raw.resample(float(args.sfreq), npad="auto", verbose=False)

    x_uV_32, labels = to_deap32_layout(raw)
    windows = window_contiguous(x_uV_32, t=int(args.t), max_windows=int(args.max_windows))

    mix_path = out_dir / "mix.npy"
    np.save(mix_path, windows.astype(np.float32, copy=False))
    window_groups = np.zeros(windows.shape[0], dtype=np.int32)
    np.save(out_dir / "window_groups.npy", window_groups)
    (out_dir / "channel_map.json").write_text(
        json.dumps(
            {
                "dataset": "artifact_benchmark",
                "deap_channels": labels,
                "split_policy": "contiguous",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "split_policy.json").write_text(
        json.dumps({"split_policy": "contiguous"}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {mix_path} with shape {windows.shape} (float32, microvolts)")
    print(f"Saved window_groups.npy shape={window_groups.shape} (single recording)")


if __name__ == "__main__":
    main()

