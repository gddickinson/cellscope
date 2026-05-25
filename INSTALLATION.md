# CellScope — Installation Guide

## Overview

CellScope analyses DIC and phase-contrast time-lapse microscopy of
migrating cells. It detects cell boundaries (Cellpose-SAM ViT for
phase-contrast, a DIC-fine-tuned ViT for DIC), tracks cells across
frames, and quantifies migration, morphology, and edge dynamics.

Runs on **macOS**, **Linux**, and **Windows** with optional GPU
acceleration (NVIDIA CUDA or Apple MPS).

## What you'll install

CellScope uses **two conda environments**:

| Env | Cellpose | Purpose |
|---|---|---|
| `cellpose4` ★ | 4.1.1 | **Default — launch all GUIs + run scripts from here.** cpsam ViT (default cpsam + the **cpsam_dic** fine-tune — current best on DIC). The auto-select pipeline uses both backbones, both of which need cellpose 4.x. |
| `cellpose` | 3.1.1.1 | CP3 fallback fine-tunes (cellpose_dic, cellpose_combined_robust, …) + MedSAM. Invoked automatically via subprocess for Phase-2 gap fill, MedSAM refinement, etc. |

The main GUI runs from the `cellpose4` env. Whenever the pipeline needs
a CP3 model (e.g. cellpose+MedSAM+DeepSea fallback in the 4-phase gap
fill cascade), it subprocess-delegates to `cellpose` automatically —
you never have to switch envs by hand.

## Requirements

- **Miniconda** or Anaconda — https://docs.conda.io/en/latest/miniconda.html
- **~6 GB disk** for envs + models (4 GB envs + ~1.3 GB models)
- **GPU** (optional but recommended): NVIDIA with CUDA 11.8+ or Apple Silicon (MPS)

## Step 1: Get the project

```bash
git clone <repository-url> cellscope
cd cellscope
```

Or unzip the project folder if you received a `cellscope-dist.zip`.

## Step 2: Create the conda environments

Run the install script for your platform.

### macOS / Linux

```bash
bash install.sh
```

### Windows

Open an **Anaconda Prompt** (Start → Anaconda → Anaconda Prompt), then:

```bat
install.bat
```

Either way, the script:

1. Creates `cellpose` env from `environment.yml` (~5 min)
2. Creates `cellpose4` env from `environment-cellpose4.yml` (~5 min)
3. Verifies both can import `cellpose` at the right version

If something goes wrong (network blip, package solver error), re-running
the script picks up where it left off. To force a clean rebuild of one
env:

```bash
conda env remove -n cellpose && bash install.sh
```

## Step 3: Download the models

Two separate downloads, both handled by one command:

| Bundle | Size | Contents | Always needed? |
|---|---:|---|---|
| **Small models** | ~120 MB | `cellpose_dic`, `cellpose_dic_v2/v3`, `cellpose_combined_robust`, `deepsea/` | Only if cloned from GitHub. The prebuilt `cellscope-dist.zip` already includes them. |
| **cpsam_dic** | 1.1 GB | `data/models/cpsam_dic` (CP4 ViT fine-tune — current best on DIC) | Always. Too large to ship in either zip. |

Run:

```bash
conda run -n cellpose4 python download_models.py
```

The script:

- Reports which models are present and which are missing.
- Pulls the small-models bundle from Drive *only if any are missing*
  (so a re-run after a dist-zip install does nothing extra).
- Pulls `cpsam_dic` from Drive.
- Both Drive URLs are hard-coded in `download_models.py` — recipients
  don't need to know them.

Useful flags:

```bash
python download_models.py --check-only     # report only, no download
python download_models.py --cpsam-only     # skip the small-models bundle
python download_models.py --bundle-only    # skip cpsam_dic
python download_models.py --force          # re-download even if present
```

## Step 4: GPU setup (optional)

### Apple Silicon (M1/M2/M3/M4)

GPU via MPS works out of the box — no extra steps. Verify:

```bash
conda run -n cellpose4 python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### NVIDIA (Linux / Windows)

The default install pulls a CPU-only PyTorch. To enable CUDA, replace
torch in **both** envs:

```bash
conda run -n cellpose  pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu121
conda run -n cellpose4 pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with `cu118` to match your CUDA version. Verify:

```bash
conda run -n cellpose4 python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### CPU only

Works but slow on cpsam ViT (~30 s/frame vs ~2 s on GPU). The pipeline
detects missing GPU and falls back automatically.

## Step 5: Verify and launch

```bash
conda activate cellpose4
python main_suite.py
```

This opens the suite launcher. From here you can launch any of the
specialised tools:

| Tool | Direct launch | Purpose |
|---|---|---|
| Detection & Analysis | `python main_focused.py` | Single-recording pipeline (load → detect → edit → analyse → export). Includes the **🔬 Test on frame** button for quick parameter-tuning previews on the currently displayed frame. |
| Batch Processing | `python main_batch.py` | Multiple recordings + group summaries |
| Tracking & Comparison | `python main_tracking.py` | Per-cell tracking + group statistics (t-test, ANOVA) |
| Mask Editor | `python main_editor.py` | View/edit/create cell masks |
| Model Training | `python main_training.py` | Fine-tune cellpose on your own GT |

You **always run from the `cellpose4` env** (cellpose 4.1.1 +
cpsam ViT). Both detection backbones (`cpsam_dic` and raw `cpsam`)
need cellpose 4.x. The pipeline subprocess-delegates to the
`cellpose` env automatically when it needs a CP3 model (e.g. for
the Phase-2 cellpose+MedSAM+DeepSea fallback in track gap fill).

## Data format

### Recordings

Supported: `.tif` / `.tiff` / `.ome.tif` / `.mp4` / `.avi` / `.mov`.

Each recording needs a JSON sidecar with pixel scale and time interval
(same base name, next to the file):

```
data/my_cells/
  recording.tif
  recording.json
```

```json
{
  "name": "My Recording",
  "um_per_px": 0.65,
  "time_interval_min": 5.0
}
```

For `.ome.tif`, the sidecar is `<base>.ome.json`. If absent, defaults
are 1.0 µm/px and 1.0 min/frame — fine for development, but pixel
sizes/intervals **must** be set for analysis to be quantitative.

### Batch directory layout

Group recordings by treatment (folder name = group name):

```
experiment/
  control/   recA.tif + recA.json   recB.tif + recB.json
  treated/   recC.tif + recC.json   recD.tif + recD.json
```

The Batch GUI's `File > Scan` finds recordings and groups by parent
folder. Statistical comparisons use folder names as labels.

### Ground truth masks (for training or evaluation)

cellpose-format PNGs (uint16, pixel value = cell ID):

```
data/manual_gt/my_rec/
  frame_0000_masks.png
  frame_0001_masks.png
```

Or a single NPZ: `masks.npz` with key `"masks"` and an `(N, H, W)` array
(bool for single-cell, int32 labels for multi-cell).

## Troubleshooting

### `conda: command not found`

Miniconda isn't on PATH. On Windows, use the **Anaconda Prompt** rather
than the regular `cmd`. On macOS / Linux, run `source ~/miniconda3/etc/profile.d/conda.sh`
or restart the terminal.

### `not a CP4 model` when using cpsam

You're trying to load a CP3 fine-tune (`cellpose_dic*`, `cellpose_combined_robust`)
in `cellpose4`, or vice versa. The pipeline handles this automatically
via subprocess delegation, but if you're calling cellpose directly in
your own scripts, match env to model:

| Model | Env |
|---|---|
| `cellpose_dic`, `cellpose_dic_v2`, `cellpose_dic_v3`, `cellpose_combined_robust` | `cellpose` |
| `cpsam_dic`, default cpsam | `cellpose4` |

### `ModuleNotFoundError: No module named 'core'` in a script

You're running from outside the project root. Either `cd cellscope/`
first, or use the helper:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports; setup_imports()
```

(Most scripts in `scripts/` already do this.)

### Slow detection (CPU)

Verify GPU detection: `Settings > System Info` in the GUI, or
```bash
conda run -n cellpose4 python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

### DeepSea / MedSAM not found

DeepSea ships with the source under `data/models/deepsea/`. MedSAM
auto-downloads from HuggingFace (`flaviagiammarino/medsam-vit-base`,
~375 MB) on first use. Check with:

```bash
conda run -n cellpose4 python download_models.py --check-only
```

### `gdown` errors during model download

Re-run the download — Drive sometimes throttles. As a fallback, open
the URL printed by `download_models.py` in a browser, save the file
manually, and place it at `data/models/cpsam_dic`.

### Memory errors on large recordings

For recordings >200 frames or >2000² pixels:

- Use **ROI selection** in the Detection GUI (Edit → Select ROI) to
  restrict to a region.
- For multi-position OME stacks, work on one position at a time.
- Close other GPU-heavy apps.

### `pydensecrf2` install fails on Windows

This is a known issue — pydensecrf2 has no Windows wheels. It's marked
optional in `environment.yml` and only used by the legacy `mode="full_stack"`
refinement path; the recommended pipelines (`hybrid_dic`, `hybrid_cpsam`,
`hybrid_dic_multi`) don't need it.

## Updating

To update a single env:

```bash
conda env update -f environment.yml --prune
conda env update -f environment-cellpose4.yml --prune
```

Or rebuild from scratch:

```bash
conda env remove -n cellpose && bash install.sh
```

## Sharing CellScope with collaborators

```bash
python make_dist.py
# → ../cellscope-dist.zip   (~118 MB, source + small models)
```

Send the zip; recipient runs `install.bat` (or `install.sh`), then
`download_models.py`. The cpsam_dic Drive URL is already baked in.

For an offline-friendly bundle (no Drive access needed):

```bash
python make_dist.py --include-cpsam
# → ~1.3 GB, fully self-contained
```

## Project structure

```
cellscope/
├── install.bat                ← Windows installer
├── install.sh                 ← macOS/Linux installer
├── environment.yml            ← cellpose env spec
├── environment-cellpose4.yml  ← cellpose4 env spec
├── download_models.py         ← Drive fetcher for cpsam_dic
├── make_dist.py               ← Build distribution zip
│
├── main_suite.py              ← Unified launcher (start here)
├── main_focused.py            ← Detection & Analysis GUI
├── main_batch.py              ← Batch Processing GUI
├── main_tracking.py           ← Tracking & Comparison GUI
├── main_editor.py             ← Mask Editor GUI
├── main_training.py           ← Model Training GUI
│
├── core/                      ← Analysis pipeline modules
│   ├── modality.py            ← Auto-detect DIC vs phase-contrast
│   ├── hybrid_cpsam.py        ← Phase-contrast pipeline
│   ├── hybrid_cpsam_multi.py  ← Multi-cell phase-contrast
│   ├── hybrid_dic.py          ← DIC pipeline (single + multi)
│   └── …
├── gui/                       ← Shared GUI components
├── gui_focused/               ← Detection GUI implementation
├── gui_batch/, gui_tracking/  ← Other GUI implementations
├── gui_editor/, gui_training/
├── output/                    ← Result writers
├── scripts/
│   ├── _paths.py              ← Path helpers (project root + benchmark data)
│   └── …
│
├── data/
│   ├── models/
│   │   ├── cpsam_dic            ← (1.1 GB, downloaded by download_models.py)
│   │   ├── cellpose_dic_v3      ← (CP3, ships with source)
│   │   ├── cellpose_dic_v2, _v1 ← (CP3, legacy fine-tunes)
│   │   ├── cellpose_combined_robust  ← (CP3, robust to noise)
│   │   └── deepsea/             ← (DeepSea refiner, ~16 MB)
│   ├── manual_gt/              ← Ground truth (optional, for training)
│   └── examples/               ← Example recordings (optional)
├── results/                    ← Analysis output
│
├── docs/
│   ├── user_manual.md          ← How to use the GUIs
│   ├── recording_recommendations.md ← Best settings per recording type
│   ├── pipeline_description.md ← Pipeline internals
│   └── IMPROVEMENTS.md         ← Research roadmap
│
├── INSTALLATION.md             ← This file
├── README.md                   ← Feature overview
├── INTERFACE.md                ← Module map
└── PROJECT_STATUS.md           ← Current results + benchmarks
```
