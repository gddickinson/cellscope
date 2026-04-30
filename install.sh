#!/usr/bin/env bash
# Cellscope install script for macOS / Linux.
# Creates two conda envs (`cellpose` + `cellpose4`) and verifies imports.
#
# Usage:
#   bash install.sh
#
# Requirements: Miniconda or Anaconda installed.
#   https://docs.conda.io/en/latest/miniconda.html
set -e

echo "=== Cellscope install (macOS / Linux) ==="
echo

# Step 1: verify conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: \`conda\` not found on PATH."
  echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

echo "[1/4] conda found ($(conda --version))."
echo

# Step 2: create the main `cellpose` env (GUI + CP3 models)
echo "[2/4] Creating \`cellpose\` env (CP3 + GUI)..."
echo "This takes 5-10 minutes on the first run."
if conda env create -f environment.yml; then
  :
else
  echo
  echo "NOTE: env may already exist. To rebuild it, run:"
  echo "  conda env remove -n cellpose && conda env create -f environment.yml"
  echo "Continuing anyway..."
fi
echo

# Step 3: create the `cellpose4` env (cpsam ViT)
echo "[3/4] Creating \`cellpose4\` env (cpsam ViT)..."
if conda env create -f environment-cellpose4.yml; then
  :
else
  echo
  echo "NOTE: env may already exist. To rebuild it, run:"
  echo "  conda env remove -n cellpose4 && conda env create -f environment-cellpose4.yml"
  echo "Continuing anyway..."
fi
echo

# Step 4: verify both envs load the right cellpose version
echo "[4/4] Verifying envs..."
conda run -n cellpose  python -c "import cellpose; print('cellpose env: cellpose', cellpose.version)"
conda run -n cellpose4 python -c "import cellpose; print('cellpose4 env: cellpose', cellpose.version)"
echo

echo "=== Install complete ==="
echo
echo "Next step: download the cpsam_dic model (1.1 GB) by running:"
echo "  conda run -n cellpose python download_models.py"
echo
echo "Then launch the GUI:"
echo "  conda activate cellpose"
echo "  python main_focused.py"
echo
