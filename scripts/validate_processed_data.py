"""
Validate processed_data/*.npy shapes for MRANC without loading full arrays into RAM.
Run: py scripts/validate_processed_data.py
"""

from __future__ import annotations

import struct
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from paths import PROCESSED_DEAP


def read_npy_header(path: Path) -> tuple[tuple[int, ...], str]:
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path}: not a .npy file")
        major, minor = struct.unpack("BB", f.read(2))
        if (major, minor) == (1, 0):
            header_len = struct.unpack("<H", f.read(2))[0]
        elif (major, minor) in ((2, 0), (3, 0)):
            header_len = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"{path}: unsupported npy version {(major, minor)}")
        header = f.read(header_len).decode("latin1").strip()
    shape_start = header.index("'shape': (") + len("'shape': (")
    shape_end = header.index(")", shape_start)
    shape_str = header[shape_start:shape_end]
    shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
    dtype = "float32" if "'<f4'" in header else "unknown"
    return shape, dtype


def main() -> None:
    root = PROCESSED_DEAP
    expected_channels = {
        "mix.npy": 32,
        "ref_eog.npy": 2,
        "ref_emg.npy": 2,
        "ref_ecg.npy": 1,
    }

    shapes: dict[str, tuple[int, ...]] = {}
    for name, exp_c in expected_channels.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        shape, dtype = read_npy_header(path)
        shapes[name] = shape
        if dtype != "float32":
            raise ValueError(f"{name}: expected float32, got {dtype}")
        if len(shape) != 3 or shape[1] != exp_c:
            raise ValueError(f"{name}: expected (N, {exp_c}, T), got {shape}")
        print(f"  {name}: shape={shape}, dtype=float32")

    n0, _, t0 = shapes["mix.npy"]
    for name, shape in shapes.items():
        if shape[0] != n0 or shape[2] != t0:
            raise ValueError(f"{name}: N/T mismatch vs mix ({n0}, *, {t0}) vs {shape}")

    print(f"OK: {n0} windows, T={t0}, ready for MRANC training.")


if __name__ == "__main__":
    print("Validating processed_data...")
    main()
