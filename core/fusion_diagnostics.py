"""Diagnostic visualisations for DIC ∪ Cy5 fusion.

Two things this module produces:

1. `annotate_tracks_with_source(tracks, source_stack)` — sets
   `t["fusion_source"] = "dic_only" | "cy5_only" | "both"` based on
   the majority pixel source under the track's mask, summed across
   all frames where the track is present.

2. `render_fusion_diagnostic(dic_frames, cy5_frames, source_stack,
   labels, out_path, n_sample_frames=6)` — saves a multi-frame
   figure showing what each channel contributed.

Colour legend used throughout:
   red       — DIC-only (DIC found this cell, Cy5 didn't)
   yellow    — Cy5-only (cpsam(Cy5) added it; DIC missed it)
   green     — both     (DIC found it AND Cy5 matched it)
"""
from __future__ import annotations

import os
import numpy as np

from .dic_cy5_fusion import (
    SRC_BACKGROUND, SRC_DIC_ONLY, SRC_CY5_ONLY, SRC_BOTH)


# RGBA strings matplotlib-friendly. Lime is used so it's visible on
# both bright DIC and dark Cy5 backgrounds.
SOURCE_COLOR = {
    "dic_only": "red",
    "cy5_only": "yellow",
    "both":     "lime",
}

SOURCE_NAME = {
    SRC_DIC_ONLY: "dic_only",
    SRC_CY5_ONLY: "cy5_only",
    SRC_BOTH:     "both",
}


def annotate_tracks_with_source(tracks, source_stack):
    """For each track, decide its `fusion_source` by majority vote
    of pixel source codes across the track's full lifetime.

    Sets `t["fusion_source"]` ∈ {"dic_only", "cy5_only", "both"} on
    every track. Tracks that never overlap any non-background source
    pixel are labelled "dic_only" by default (legacy mode without
    fusion).
    """
    if source_stack is None:
        for t in tracks:
            t["fusion_source"] = "dic_only"
        return tracks
    for t in tracks:
        stack = t.get("stack")
        if stack is None:
            t["fusion_source"] = "dic_only"
            continue
        counts = {SRC_DIC_ONLY: 0, SRC_CY5_ONLY: 0, SRC_BOTH: 0}
        for fi in range(min(len(stack), len(source_stack))):
            mask = stack[fi]
            if not mask.any():
                continue
            src_under = source_stack[fi][mask]
            for code in (SRC_DIC_ONLY, SRC_CY5_ONLY, SRC_BOTH):
                counts[code] += int((src_under == code).sum())
        if sum(counts.values()) == 0:
            t["fusion_source"] = "dic_only"
        else:
            best = max(counts, key=counts.get)
            t["fusion_source"] = SOURCE_NAME[best]
        # Also keep the raw counts for transparency
        t["fusion_source_pixel_counts"] = counts
    return tracks


def pick_diagnostic_frames(n_frames, k=6):
    """Pick `k` evenly-spaced frame indices for the diagnostic grid."""
    if n_frames <= k:
        return list(range(n_frames))
    return list(np.linspace(0, n_frames - 1, k, dtype=int))


def render_fusion_diagnostic(dic_frames, cy5_frames, source_stack,
                              labels, out_path, n_sample_frames=6,
                              tracks=None):
    """Save a multi-frame diagnostic showing per-channel contributions.

    Layout: 3 columns × n_sample_frames rows
      Col 0: DIC image with all merged-cell contours, coloured by
             pixel source (red/yellow/green)
      Col 1: Cy5 image with same contours
      Col 2: DIC image with source legend / counts overlaid
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    indices = pick_diagnostic_frames(len(dic_frames), n_sample_frames)
    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Per-frame counts for the summary column
    for ri, fi in enumerate(indices):
        dic = dic_frames[fi]
        cy5 = cy5_frames[fi]
        lab_frame = labels[fi]
        src_frame = (source_stack[fi]
                     if source_stack is not None
                     else np.zeros_like(lab_frame, dtype=np.uint8))

        # Count cells by source in this frame
        counts = {"dic_only": 0, "cy5_only": 0, "both": 0}
        for lab in range(1, int(lab_frame.max()) + 1):
            cell_mask = lab_frame == lab
            if not cell_mask.any():
                continue
            src_under = src_frame[cell_mask]
            if src_under.size == 0:
                continue
            # Pick dominant source for THIS cell
            cell_counts = {
                SRC_DIC_ONLY: int((src_under == SRC_DIC_ONLY).sum()),
                SRC_CY5_ONLY: int((src_under == SRC_CY5_ONLY).sum()),
                SRC_BOTH: int((src_under == SRC_BOTH).sum()),
            }
            dominant = max(cell_counts, key=cell_counts.get)
            counts[SOURCE_NAME[dominant]] += 1

        # Plot col 0: DIC
        axes[ri, 0].imshow(dic, cmap="gray")
        axes[ri, 0].set_title(
            f"F{fi}  DIC  "
            f"dic_only:{counts['dic_only']} "
            f"both:{counts['both']} "
            f"cy5_only:{counts['cy5_only']}")
        _draw_source_contours(axes[ri, 0], lab_frame, src_frame)

        # Plot col 1: Cy5
        axes[ri, 1].imshow(cy5, cmap="inferno",
                            vmin=0, vmax=np.percentile(cy5, 99))
        axes[ri, 1].set_title(f"F{fi}  Cy5  (same cells, same colours)")
        _draw_source_contours(axes[ri, 1], lab_frame, src_frame)

        # Plot col 2: per-source overlay summary
        axes[ri, 2].imshow(dic, cmap="gray", alpha=0.6)
        axes[ri, 2].set_title(f"F{fi}  source breakdown")
        # Draw filled translucent regions per source
        for code, name in SOURCE_NAME.items():
            mask = src_frame == code
            if not mask.any():
                continue
            # Use semi-transparent filled coloured region
            overlay = np.zeros((*mask.shape, 4))
            color_rgb = _mpl_to_rgba(SOURCE_COLOR[name])
            overlay[mask] = (*color_rgb[:3], 0.55)
            axes[ri, 2].imshow(overlay)

        for ax in axes[ri]:
            ax.axis("off")

    # Big title
    fig.suptitle(
        "Fusion diagnostic — colour legend: "
        "red = DIC-only  ·  green = both channels  ·  "
        "yellow = Cy5-only",
        fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _mpl_to_rgba(name):
    """Convert matplotlib named colour to (r, g, b, 1.0) tuple."""
    import matplotlib.colors as mc
    return mc.to_rgba(name)


def _draw_source_contours(ax, lab_frame, src_frame, lw=1.4):
    """Draw per-cell contour coloured by its dominant source code."""
    from skimage import measure
    for lab in range(1, int(lab_frame.max()) + 1):
        cell_mask = lab_frame == lab
        if not cell_mask.any():
            continue
        src_under = src_frame[cell_mask]
        if src_under.size == 0:
            color = SOURCE_COLOR["dic_only"]
        else:
            counts = {
                SRC_DIC_ONLY: int((src_under == SRC_DIC_ONLY).sum()),
                SRC_CY5_ONLY: int((src_under == SRC_CY5_ONLY).sum()),
                SRC_BOTH:     int((src_under == SRC_BOTH).sum()),
            }
            dominant = max(counts, key=counts.get)
            color = SOURCE_COLOR[SOURCE_NAME[dominant]]
        for c in measure.find_contours(cell_mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, lw=lw)


def per_track_source_summary(tracks):
    """Return a compact dict summarising fusion sources across tracks.

    Useful for the run log: 'kept 18 tracks (8 dic_only, 4 both,
    6 cy5_only)'."""
    counts = {"dic_only": 0, "cy5_only": 0, "both": 0}
    for t in tracks:
        s = t.get("fusion_source", "dic_only")
        counts[s] = counts.get(s, 0) + 1
    return counts
