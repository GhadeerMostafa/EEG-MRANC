#!/usr/bin/env bash
set -e

# Download the MNE SSVEP example dataset archive (open-source MNE repository),
# and place it under: data/artifact_benchmark/raw/
#
# Note:
# - The upstream data is distributed as a ZIP containing BrainVision files.
# - The preprocessing step will convert the BrainVision recording into a single
#   FIF file and then load it via mne.io.read_raw_fif().

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/artifact_benchmark/raw"
mkdir -p "${OUT_DIR}"

ZIP_URL="https://osf.io/download/z8h6k?version=5"
ZIP_PATH="${OUT_DIR}/ssvep_example_data.zip"

echo "Downloading SSVEP Artifact Benchmark archive to: ${ZIP_PATH}"
curl -L --fail --retry 5 --retry-delay 2 -o "${ZIP_PATH}" "${ZIP_URL}"

# Extraction block (archive contains BrainVision EEG files).
# Use PowerShell to avoid dependency on `unzip` in minimal shells.
echo "Extracting ZIP into: ${OUT_DIR}/ssvep-example-data"
rm -rf "${OUT_DIR}/ssvep-example-data"
powershell.exe -NoProfile -Command "Expand-Archive -Force -Path '${ZIP_PATH}' -DestinationPath '${OUT_DIR}'"

echo "Done."

