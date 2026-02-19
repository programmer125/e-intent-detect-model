#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-intent-llm}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Install Miniconda/Anaconda first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[1/3] Creating conda env: ${ENV_NAME}"
conda env create -f "${ROOT_DIR}/environment.yml" -n "${ENV_NAME}" || true

echo "[2/3] Installing Python packages"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/requirements-cu118.txt"

echo "[3/3] Verifying CUDA visibility"
conda run -n "${ENV_NAME}" python "${ROOT_DIR}/scripts/verify_gpu.py"

echo "Done. Activate with: conda activate ${ENV_NAME}"
