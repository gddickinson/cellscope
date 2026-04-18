# CellScope — Installation Guide

## Overview

This software analyzes DIC and phase-contrast time-lapse microscopy of
migrating keratinocytes. It detects cell boundaries using Cellpose-SAM
(cpsam), tracks cells across frames, and quantifies migration, morphology,
and edge dynamics.

The suite runs on **macOS**, **Linux**, and **Windows** with optional
GPU acceleration (NVIDIA CUDA or Apple MPS).

## Requirements

- **Python 3.10** (tested; 3.9-3.11 should also work)
- **Conda** (Anaconda or Miniconda) for environment management
- **~4 GB disk** for environments + models
- **GPU** (optional): NVIDIA with CUDA 11.8+ or Apple Silicon (MPS)

## Step 1: Clone or download the project

```bash
git clone <repository-url> cellscope
cd cellscope
```

Or download and unzip the project folder.

## Step 2: Create the conda environments

The project uses **two conda environments**:

| Environment | Cellpose version | Purpose |
|---|---|---|
| `cellpose` | 3.1.1.1 | Legacy CP3 models (fallback detection) |
| `cellpose4` | 4.1.1 | Cellpose-SAM (cpsam, primary detector) |

### Create the primary environment (cellpose4)

```bash
conda create -n cellpose4 python=3.10 -y
conda activate cellpose4

# Core dependencies
pip install cellpose==4.1.1
pip install PyQt5 matplotlib scikit-image scikit-learn scipy
pip install tifffile opencv-python-headless
pip install transformers huggingface_hub peft

# Optional: for advanced tracking
pip install trackastra

# Optional: for VAMPIRE shape mode analysis
pip install vampire-analysis
```

### Create the fallback environment (cellpose)

```bash
conda create -n cellpose python=3.10 -y
conda activate cellpose

# Same dependencies but with cellpose 3.x
pip install cellpose==3.1.1.1
pip install PyQt5 matplotlib scikit-image scikit-learn scipy
pip install tifffile opencv-python-headless
pip install transformers huggingface_hub peft
```

### GPU-specific installation

**Apple Silicon (M1/M2/M3/M4):**
GPU acceleration via MPS is automatic — no extra steps. PyTorch detects
MPS when `torch.backends.mps.is_available()` returns True.

**NVIDIA CUDA (Linux/Windows):**
Replace the default PyTorch with the CUDA build:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Check with: `python -c "import torch; print(torch.cuda.is_available())"`

**CPU only:**
No extra steps. The software detects missing GPU and falls back to CPU
automatically. Detection will be slower (~10x) but functional. Uncheck
"Use GPU acceleration" in Settings menu if you get GPU errors.

## Step 3: Download required models

### DeepSea (required for refinement)

```bash
git clone --depth 1 https://github.com/abzargar/DeepSea.git /tmp/DeepSea_tmp
mkdir -p data/models/deepsea
cp /tmp/DeepSea_tmp/deepsea/trained_models/* data/models/deepsea/
cp /tmp/DeepSea_tmp/deepsea/model.py data/models/deepsea/
rm -rf /tmp/DeepSea_tmp
```

The DeepSea segmentation model (`data/models/deepsea/segmentation.pth`,
~8 MB) is loaded automatically during detection. Alternatively, use
the Setup Wizard (`python setup_wizard.py`) to install it with one click.

### MedSAM (required for fallback refinement)

Downloaded automatically on first use from HuggingFace
(`flaviagiammarino/medsam-vit-base`, ~375 MB). No manual step needed.

### Cellpose models

The default cpsam model downloads automatically on first use (~2.4 GB
for the ViT-H weights). Trained custom models are stored in
`data/models/` and detected by the GUI automatically.

## Step 4: Verify installation

```bash
conda activate cellpose4
python -c "
import cellpose; print(f'Cellpose: {cellpose.version}')
import torch; print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MPS: {torch.backends.mps.is_available()}')
from PyQt5.QtWidgets import QApplication; print('PyQt5: OK')
from core.io import load_recording; print('Core modules: OK')
print('Installation OK!')
"
```

Expected output:
```
Cellpose: 4.1.1
PyTorch: 2.x.x
CUDA: True/False (depending on your GPU)
MPS: True/False (True on Apple Silicon)
PyQt5: OK
Core modules: OK
Installation OK!
```

## Step 5: Launch the application

```bash
conda activate cellpose4
python main_suite.py
```

This opens the suite launcher with buttons for each application:

| Application | Direct launch | Description |
|---|---|---|
| Detection & Analysis | `python main_focused.py` | Single-recording cpsam pipeline |
| Batch Processing | `python main_batch.py` | Process multiple recordings |
| Tracking & Comparison | `python main_tracking.py` | Per-cell tracking + group statistics |
| Mask Editor | `python main_editor.py` | View/edit/create cell masks |
| Model Training | `python main_training.py` | Fine-tune cellpose on your data |

## Data format

### Recordings

Supported formats: `.mp4`, `.avi`, `.mov`, `.tif`, `.tiff`

Each video needs a JSON sidecar with pixel scale and time interval:
```
data/my_cells/
  recording.tif
  recording.json     ← same base name
```

**recording.json:**
```json
{
  "name": "My Recording",
  "um_per_px": 0.65,
  "time_interval_min": 5.0
}
```

For `.ome.tif` files, the sidecar is `recording.ome.json`.

If no JSON sidecar exists, the software uses defaults: `um_per_px=1.0`,
`time_interval_min=1.0`, `name=<filename>`.

### Batch processing directory structure

Organize recordings by treatment group (folder name = group name):

```
experiment/
  control/
    cell1.tif + cell1.json
    cell2.tif + cell2.json
  treated/
    cell3.tif + cell3.json
    cell4.tif + cell4.json
```

The batch GUI discovers recordings via `File > Scan` and groups them
by parent folder. Statistical comparisons use folder names as group
labels.

### Ground truth masks

For training or evaluation, save masks as cellpose-format PNGs:
```
data/manual_gt/my_recording/
  frame_0000_masks.png    ← uint16, pixel value = cell ID (0=background)
  frame_0001_masks.png
  ...
```

Or as a single NPZ file: `masks.npz` with key `"masks"` containing an
`(N, H, W)` array (bool for single-cell, int32 for multi-cell labels).

## Troubleshooting

### "not a CP4 model" error
You're running a cellpose 3.x model in the cellpose4 environment.
The hybrid pipeline handles this automatically via subprocess fallback.
If you see this error in the GUI, ensure the `cellpose` (3.x) environment
exists alongside `cellpose4`.

### Slow detection (no GPU)
Check Settings > System Info to verify GPU is detected. If not:
- **Mac**: ensure you have macOS 12.3+ and PyTorch 1.12+
- **Linux/Windows**: install the CUDA version of PyTorch (see Step 2)
- Uncheck Settings > "Use GPU acceleration" to acknowledge CPU mode

### DeepSea not found
Install DeepSea model into `data/models/deepsea/`:
```bash
python setup_wizard.py   # click "Install" next to DeepSea
```
Or manually:
```bash
git clone --depth 1 https://github.com/abzargar/DeepSea.git /tmp/DeepSea_tmp
cp /tmp/DeepSea_tmp/deepsea/trained_models/* data/models/deepsea/
cp /tmp/DeepSea_tmp/deepsea/model.py data/models/deepsea/
rm -rf /tmp/DeepSea_tmp
```

### Import errors
Ensure you're in the project root directory:
```bash
cd /path/to/cellscope
conda activate cellpose4
python main_suite.py
```

### Memory errors on large recordings
For recordings with >200 frames or >2000x2000 pixels, consider:
- Processing a subset of frames first
- Using ROI to restrict analysis to a region of interest
- Closing other GPU-heavy applications

## Project structure

```
cellscope/
├── main_suite.py          ← Unified launcher (start here)
├── main_focused.py        ← Detection & Analysis
├── main_batch.py          ← Batch Processing
├── main_tracking.py       ← Tracking & Comparison
├── main_editor.py         ← Mask Editor
├── main_training.py       ← Model Training
├── core/                  ← Analysis pipeline modules
├── gui/                   ← Shared GUI components
├── gui_focused/           ← Detection GUI
├── gui_batch/             ← Batch GUI
├── gui_tracking/          ← Tracking GUI
├── gui_editor/            ← Editor GUI
├── gui_training/          ← Training GUI
├── output/                ← Result writers
├── data/
│   ├── models/            ← Trained models (auto-populated)
│   ├── manual_gt/         ← Ground truth masks
│   └── examples/          ← Example recordings
├── results/               ← Analysis output
├── INSTALLATION.md        ← This file
├── README.md              ← Feature overview
├── INTERFACE.md           ← Module map
└── ROADMAP.md             ← Development history
```

## Updating

To update cellpose or other dependencies:
```bash
conda activate cellpose4
pip install --upgrade cellpose
```

Note: upgrading cellpose may change the default model. The hybrid
pipeline is designed to handle version differences via the dual-env
architecture.
