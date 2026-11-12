"""Full-recording runtime estimate from one "Test on frame" detection (pure, no Qt).

Used by the focused GUI's 🔬 Test on frame button (`gui_focused/main_window.py`) and
the web planner. Detection time is linear in the number of frames; the full pipeline
adds multi-frame post-processing (tracking + gap fill + Cy5 fusion + filter) whose cost
grows with cell density, so the estimate multiplies by a density-dependent factor:

    sparse (<5 cells/frame):    full ≈ 1.5 × detection
    medium (5-10 cells/frame):  full ≈ 2.0 × detection
    dense (≥10 cells/frame):    full ≈ 2.5 × detection

Calibrated against Pos10_WT (4 cells, canonical ~1.5× extrapolated detection) and
Pos68_DMSO (17 cells, canonical ~2.3× extrapolated detection).
"""
from __future__ import annotations

DENSITY_MULTIPLIERS = ((5, 1.5, "sparse"), (10, 2.0, "medium"), (None, 2.5, "dense"))


def density_multiplier(n_cells):
    """(multiplier, density label) for the cell count found on the test frame."""
    for limit, mult, name in DENSITY_MULTIPLIERS:
        if limit is None or n_cells < limit:
            return mult, name


def estimate_full_runtime(elapsed_s, n_cells, n_frames):
    """Extrapolate one frame's detection time to the whole recording.

    Returns {"detect_only_s", "full_s", "multiplier", "density"}.
    """
    mult, density = density_multiplier(n_cells)
    detect_only = elapsed_s * n_frames
    return {"detect_only_s": detect_only, "full_s": detect_only * mult,
            "multiplier": mult, "density": density}


def format_duration(secs):
    """Short human form: '42s', '3.5 min', '2.7 h'."""
    return (f"{secs:.0f}s" if secs < 60
            else f"{secs / 60:.1f} min" if secs < 3600
            else f"{secs / 3600:.1f} h")
