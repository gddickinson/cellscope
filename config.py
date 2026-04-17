"""Configuration constants for CellScope cell analysis.

Shared by both GUI versions and all core modules.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "data", "models")
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# --- Cellpose ---
CELLPOSE_MODEL_NAME = "cellpose_dic"
CELLPOSE_FLOW_THRESHOLD = 0.0  # was 0.4; 0.0 disables flow QC, helps
                               # pre-cropped single-cell frames (Jesse)
                               # without regressing our full-frame data.
                               # See SESSION_LOG Test 21 (2026-04-13).
CELLPOSE_CELLPROB_THRESHOLD = 0.0

# --- Optical flow (Farneback) ---
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2

# --- Edge snap ---
EDGE_SNAP_GRADIENT_SIGMA = 1.5
EDGE_SNAP_DEFAULT_SEARCH = 8
EDGE_SNAP_DEFAULT_MAX_STEP = 5
EDGE_SNAP_DEFAULT_SMOOTH_SIGMA = 2.0

# --- Auto-parameter selection ---
AUTO_REFINE_CONFIGS = [
    {"name": "snap5",  "search": 5,  "max_step": 3,  "smooth": 1.5, "fourier": True},
    {"name": "snap8",  "search": 8,  "max_step": 5,  "smooth": 2.0, "fourier": True},
    {"name": "snap12", "search": 12, "max_step": 7,  "smooth": 2.5, "fourier": True},
    {"name": "snap15", "search": 15, "max_step": 8,  "smooth": 3.0, "fourier": True},
    {"name": "snap20", "search": 20, "max_step": 12, "smooth": 5.0, "fourier": True},
    {"name": "snap8",  "search": 8,  "max_step": 5,  "smooth": 2.0, "fourier": False},
    {"name": "snap20", "search": 20, "max_step": 12, "smooth": 5.0, "fourier": False},
]

TEMPORAL_SMOOTH_SIGMA = 1.5
AUTO_MIN_IOU = 0.45
AUTO_IOU_DROP_OK = 0.05
AUTO_IOU_DROP_REJECT = 0.18

# --- Fourier contour smoothing ---
FOURIER_N_DESCRIPTORS = 40

# --- Edge dynamics ---
N_ANGULAR_SECTORS = 72
EDGE_AGG_METHOD = "median"
EDGE_ANGULAR_SMOOTH_WINDOW = 5
EDGE_TEMPORAL_SIGMA = 1.0

# --- Morphology ---
MIN_CELL_AREA_PX = 300

# --- Refinement: cropping defaults (core.pipeline.refine) ---
# "global" = single union bbox across all frames (fastest, temporal-safe)
# "per_cell" = each frame gets its own tight bbox (for frame-local methods)
# "none" = refine on full frame
REFINE_DEFAULT_CROP_MODE = "global"
REFINE_DEFAULT_CROP_PADDING_PX = 50  # global crop padding
REFINE_DEFAULT_PER_CELL_PADDING_PX = 30

# --- Threshold-retry detection ---
DETECT_RETRY_CELLPROB_THRESHOLDS = (-2.0, -4.0)  # applied in order on misses
DETECT_RETRY_INTERPOLATE_GAPS = True  # fill remaining frames via optical flow

# --- RF filter bank (for mode="fixed"/"iterative") ---
# "default" → data/models/rf_boundary_model.pkl (winning 2021 model)
# Other options trained by scripts/train_rf_filter_banks.py:
#   fine_scale, fine_scale_gmm, fine_scale_morph, texture_rich,
#   minimal_texture, edge_focused, gabor_bank, kitchen_sink,
#   coarse_scale, wide_original
DEFAULT_RF_FILTER_BANK = "default"

# --- Hybrid RF refinement (mode="hybrid") ---
HYBRID_RF_THRESHOLD = 0.6       # RF prob threshold inside boundary band
HYBRID_ENSEMBLE_THRESHOLD = 0.5  # threshold for averaged prob (ensemble)
HYBRID_BOUNDARY_DILATE = 10      # iterations to dilate seed for band
HYBRID_BOUNDARY_ERODE = 5        # iterations to erode seed for band

# --- SAM refinement (mode="sam") ---
# Checkpoints expected at data/models/sam/sam_vit_<b|l|h>_*.pth
# Only vit_b ships with the project; download larger ones if needed.
SAM_DEFAULT_MODEL_TYPE = "vit_b"
SAM_USE_MASK_PROMPT = True  # pass seed mask as low-res logit prompt

# --- Detection preprocessing (applied before cellpose when enabled) ---
# See core.preprocess.preprocess_sequence. Each has a disable value.
PREPROCESS_DEFAULT_TEMPORAL_METHOD = None  # "median" / "mean" / "min" / "max"
PREPROCESS_DEFAULT_SPATIAL_HIGHPASS_SIGMA = 0.0   # 0 = disabled
PREPROCESS_DEFAULT_DEBRIS_DIAMETER_UM = 0.0       # 0 = disabled
PREPROCESS_OFFSET = 128.0  # recentering offset for bg subtract & highpass

# --- ROI masking (set per-recording via GUI, not a global default) ---
# The GUI stores a bbox on the recording dict as `roi = (r0, r1, c0, c1)`.
# pipeline.detect() applies `apply_roi_mask` when that key is present.
# Pixels outside the bbox are set to ROI_FILL_VALUE in every frame.
ROI_FILL_VALUE = 0

# --- CRF ---
CRF_ITERATIONS = 5

# --- Plotting ---
FIGURE_DPI = 110
PANEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
CONTROL_COLOR = "#1f77b4"
CKO_COLOR = "#d62728"

# --- v1 GUI legacy defaults ---
SEGMENTATION_METHOD = "cellpose"
GAUSSIAN_SIGMA = 3
STD_KERNEL_SIZE = 21
MORPH_CLOSE_SIZE = 25
GRAD_SIGMA = 2
FILL_SIGMA = 20
OPEN_SIZE = 5

# --- Outputs ---
RESULT_FIGURES = ["boundary_overlay", "trajectory", "edge_kymograph",
                  "shape_timeseries"]
