"""
Inspect the first .mat file inside .seeda_workspace/seeda.zip without full extraction.

Uses scipy.io.whosmat to list variable names, shapes, and MATLAB classes.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from runtime import setup_src_path
from scipy.io import whosmat

setup_src_path()

from paths import SEEDA_WORKSPACE

DEFAULT_ZIP = SEEDA_WORKSPACE / "seeda.zip"


def find_first_mat_member(zip_path: Path) -> tuple[str, bytes]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        mat_names = sorted(
            n for n in archive.namelist()
            if n.lower().endswith(".mat") and not n.endswith("/")
        )
        if not mat_names:
            raise FileNotFoundError(f"No .mat files in {zip_path}")

        first = mat_names[0]
        return first, archive.read(first)


def inspect_mat_bytes(mat_bytes: bytes) -> list[tuple[str, tuple[int, ...], str, bool | None]]:
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(mat_bytes)
        tmp.flush()

    try:
        raw = whosmat(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    entries: list[tuple[str, tuple[int, ...], str, bool | None]] = []
    for row in raw:
        if len(row) == 4:
            name, shape, mat_class, is_complex = row
        elif len(row) == 3:
            name, shape, mat_class = row
            is_complex = None
        else:
            raise ValueError(f"Unexpected whosmat row: {row!r}")
        entries.append((name, tuple(int(s) for s in shape), mat_class, is_complex))
    return entries


def main() -> None:
    zip_path = DEFAULT_ZIP
    print(f"Archive: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / 1e6:.2f} MB\n")

    member_name, mat_bytes = find_first_mat_member(zip_path)
    print(f"First .mat member: {member_name}")
    print(f"Compressed payload read: {len(mat_bytes) / 1e6:.2f} MB\n")

    entries = inspect_mat_bytes(mat_bytes)
    print(f"Variables ({len(entries)}):\n")
    print(f"{'Name':<32} {'Shape':<24} {'MATLAB class':<16} {'Complex'}")
    print("-" * 80)
    for name, shape, mat_class, is_complex in entries:
        shape_str = str(tuple(int(s) for s in shape))
        complex_str = str(is_complex) if is_complex is not None else "n/a"
        print(f"{name:<32} {shape_str:<24} {mat_class:<16} {complex_str}")


if __name__ == "__main__":
    main()
