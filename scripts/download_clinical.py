"""
Download CHB-MIT Scalp EEG subset (patient chb01) from PhysioNet.

Uses the system `curl` binary via subprocess (resume, retries, redirects).
Fetches chb01_01.edf, chb01_02.edf, chb01_03.edf into raw_clinical_data/.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from runtime import setup_src_path

setup_src_path()

from paths import RAW_CLINICAL

DEFAULT_OUTPUT_DIR = RAW_CLINICAL

# Patient 1 recordings live under chb01/ on PhysioNet.
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/"
PATIENT_FILES = [
    "chb01_01.edf",
    "chb01_02.edf",
    "chb01_03.edf",
]

MIN_EDF_BYTES = 1024 * 100  # reject tiny/HTML bodies

CURL = "curl.exe"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download CHB-MIT chb01 EDF files from PhysioNet (via curl)")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory for downloaded EDF files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid file already exists",
    )
    p.add_argument(
        "--sanity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate downloaded files (size, not HTML, EDF header)",
    )
    return p.parse_args()


def file_url(filename: str) -> str:
    return f"{BASE_URL}{filename}"


def _curl_head_headers(url: str) -> str:
    """Fetch response headers with curl (-I, follow redirects)."""
    result = subprocess.run(
        [CURL, "-sI", "-L", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def remote_content_length(url: str) -> int | None:
    """Parse the last Content-Length from curl -I output (after redirects)."""
    text = _curl_head_headers(url)
    lengths: list[int] = []
    for line in text.splitlines():
        m = re.match(r"^content-length:\s*(\d+)\s*$", line, re.IGNORECASE)
        if m:
            lengths.append(int(m.group(1)))
    return lengths[-1] if lengths else None


def _unlink_with_retry(path: Path, attempts: int = 5, delay_s: float = 1.0) -> None:
    last_exc: OSError | None = None
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(delay_s)
    raise RuntimeError(
        f"Cannot remove {path.name}; close any program using the file."
    ) from last_exc


def _looks_like_html(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False
    low = head.lstrip().lower()
    return low.startswith(b"<!doctype") or low.startswith(b"<html") or b"<head" in low[:300]


def cleanup_stale_partials(output_dir: Path, filename: str, url: str) -> None:
    """
    Remove corrupted / oversized / wrong-size partials from earlier runs.
    Keeps a file only if it matches the server's Content-Length (when known).
    """
    dest = output_dir / filename
    if not dest.is_file():
        return

    expected = remote_content_length(url)
    size = dest.stat().st_size

    if expected is not None:
        if size != expected:
            print(
                f"  Cleanup: removing {filename} "
                f"({size} bytes; expected {expected} bytes)"
            )
            _unlink_with_retry(dest)
        return

    if size < MIN_EDF_BYTES or _looks_like_html(dest):
        print(f"  Cleanup: removing suspicious {filename} ({size} bytes)")
        _unlink_with_retry(dest)


def run_curl_download(url: str, local_path: Path) -> None:
    """Download (or resume) one file using system curl."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            CURL,
            "-L",
            "-C",
            "-",
            "--connect-timeout",
            "15",
            "--retry",
            "5",
            "-o",
            str(local_path),
            url,
        ],
        check=True,
    )


def sanity_check_file(path: Path, expected_size: int | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing file: {path}")

    size_bytes = path.stat().st_size
    if size_bytes < MIN_EDF_BYTES:
        raise RuntimeError(
            f"{path.name}: file too small ({size_bytes} bytes). "
            "Likely an HTML error page or incomplete download."
        )

    if expected_size is not None and size_bytes != expected_size:
        raise RuntimeError(
            f"{path.name}: size mismatch ({size_bytes} / {expected_size} bytes)."
        )

    with open(path, "rb") as handle:
        head = handle.read(512)

    head_lower = head.lstrip().lower()
    if head_lower.startswith(b"<!doctype") or head_lower.startswith(b"<html"):
        raise RuntimeError(f"{path.name}: content looks like HTML, not an EDF recording.")

    if b"text/html" in head_lower[:200] or b"<head" in head_lower[:300]:
        raise RuntimeError(f"{path.name}: HTML markers detected in file header.")

    with open(path, "rb") as handle:
        version = handle.read(8)
    if not version.startswith(b"0"):
        print(f"  Warning: {path.name} EDF version field unexpected: {version!r}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"CHB-MIT download (curl) -> {output_dir.resolve()}")
    print(f"Base URL: {BASE_URL}")

    # Remove bad partials / oversized files before any new download.
    print("\nCleaning stale or corrupted partial downloads...")
    for filename in PATIENT_FILES:
        url = file_url(filename)
        cleanup_stale_partials(output_dir, filename, url)

    summary: list[tuple[str, int, str]] = []

    for filename in PATIENT_FILES:
        url = file_url(filename)
        dest = output_dir / filename
        expected = remote_content_length(url)
        print(f"\n[{filename}]")

        if dest.exists() and not args.force and expected is not None:
            if dest.stat().st_size == expected:
                try:
                    sanity_check_file(dest, expected_size=expected)
                    print(f"  Skip (exists): {filename} ({expected / 1e6:.2f} MB)")
                    summary.append((filename, dest.stat().st_size, "skipped"))
                    continue
                except RuntimeError as exc:
                    print(f"  Removing invalid cached file: {filename} ({exc})")
                    _unlink_with_retry(dest)

        if args.force and dest.exists():
            _unlink_with_retry(dest)

        print(f"  Downloading: {url}")
        run_curl_download(url, dest)

        if args.sanity:
            sanity_check_file(dest, expected_size=remote_content_length(url))
            summary.append((filename, dest.stat().st_size, "valid"))
        else:
            summary.append((filename, dest.stat().st_size, "downloaded"))

    print("\n" + "=" * 60)
    print(f"{'File':<20} {'Bytes':>12}  Status")
    print("-" * 60)
    for name, nbytes, status in summary:
        print(f"{name:<20} {nbytes:>12}  {status}")
    print("=" * 60)
    print("All requested files ready.")


if __name__ == "__main__":
    main()
