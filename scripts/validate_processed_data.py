"""
Validate processed_data/*.npy shapes for MRANC without loading full arrays into RAM.
Run: py scripts/validate_processed_data.py
"""

from __future__ import annotations

import struct
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from paths import DATASET_DIRS


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


def validate_dataset_dir(root: Path, require_groups: bool = True) -> None:
    expected_channels = {
        "mix.npy": 32,
        "ref_eog.npy": 2,
        "ref_emg.npy": 2,
        "ref_ecg.npy": 1,
    }

    shapes: dict[str, tuple[int, ...]] = {}
    mix_path = root / "mix.npy"
    if not mix_path.exists():
        raise FileNotFoundError(f"Missing {mix_path}")

    mix_shape, mix_dtype = read_npy_header(mix_path)
    if mix_dtype != "float32" or len(mix_shape) != 3 or mix_shape[1] != 32:
        raise ValueError(f"mix.npy: expected (N, 32, T) float32, got {mix_shape} {mix_dtype}")
    shapes["mix.npy"] = mix_shape
    print(f"  mix.npy: shape={mix_shape}, dtype=float32")

    for name, exp_c in expected_channels.items():
        if name == "mix.npy":
            continue
        path = root / name
        if not path.exists():
            continue
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

    groups_path = root / "window_groups.npy"
    if require_groups:
        if not groups_path.exists():
            raise FileNotFoundError(
                f"Missing {groups_path}. Re-run preprocessing to generate subject/session groups."
            )
        group_shape, _group_dtype = read_npy_header(groups_path)
        if len(group_shape) != 1 or group_shape[0] != n0:
            raise ValueError(
                f"window_groups.npy: expected ({n0},), got {group_shape}"
            )
        print(f"  window_groups.npy: shape=({n0},) unique_groups=see audit script")

    print(f"OK: {n0} windows, T={t0}, ready for MRANC training.")


def main() -> None:
    print("Validating processed datasets...")
    for name, root in DATASET_DIRS.items():
        if not root.exists():
            print(f"SKIP {name}: missing {root}")
            continue
        print(f"\n{name} ({root}):")
        validate_dataset_dir(root, require_groups=True)


if __name__ == "__main__":
    main()
