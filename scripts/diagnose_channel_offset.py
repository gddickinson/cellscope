"""Measure the DIC ↔ Cy5 alignment offset for a multichannel recording.

Uses phase correlation on Gaussian-filtered cell-containing pixels.
Reports:
  - Per-frame offset (dy, dx) for every frame in the recording
  - Median offset across the stack (the value that should be applied
    if alignment is stable)
  - Range / SD of the per-frame offsets (tells us if the offset is
    a single constant or drifts in time)
  - Visual: overlay of DIC + Cy5 at frame 0 BEFORE and AFTER applying
    the median offset, so you can eyeball whether the fix lands

Usage:
  conda run -n cellpose python scripts/diagnose_channel_offset.py \\
      data/ic295_gt_full/Pos7_WT
"""
import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter, shift as nd_shift

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)


def measure_per_frame_offset(dic_stack, cy5_stack, smooth_sigma=3.0,
                              upsample=4):
    """Phase-correlation alignment per frame. Returns (n_frames, 2)
    array of (dy, dx) — what you'd ADD to Cy5 coords to land on DIC.

    Gaussian-smoothed inputs so phase correlation locks on cell-scale
    structure, not pixel noise. Sub-pixel via `upsample_factor`.
    """
    n = len(dic_stack)
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        d = gaussian_filter(dic_stack[i].astype(np.float32),
                             sigma=smooth_sigma)
        c = gaussian_filter(cy5_stack[i].astype(np.float32),
                             sigma=smooth_sigma)
        # Invert DIC so its features (dark cell interiors with bright
        # halos) have similar polarity to Cy5 (bright cells, dark bg).
        d_inv = d.max() - d
        # Trim to common reference region — phase_cross_correlation
        # returns (shift_y, shift_x) such that reference = moved shifted.
        # We treat DIC as the reference; positive shift means Cy5 was
        # moved (+shift) relative to DIC, so to align Cy5 to DIC we'd
        # SUBTRACT the shift from Cy5's coordinates.
        result = phase_cross_correlation(
            d_inv, c, upsample_factor=upsample, normalization="phase")
        # New scikit-image returns (shift, error, phase_diff); old returns shift
        if isinstance(result, tuple) and len(result) >= 1:
            shift = result[0]
        else:
            shift = result
        out[i] = shift
    return out


def find_recording(folder):
    """Return the .ome.tif inside the folder."""
    for f in os.listdir(folder):
        if f.endswith((".ome.tif", ".tif")):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"no .ome.tif in {folder}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--sample-frames", type=int, default=10,
                   help="Number of frames to sample for the diagnostic")
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    tif = find_recording(folder)
    print(f"Loading {tif} …")

    from core.io import load_recording
    rec = load_recording(tif, dic_channel=1, fluo_channel=0)
    dic = rec["frames"]
    cy5 = rec["cy5_frames"]
    if cy5 is None:
        print("ERROR: no fluorescence channel in this recording.")
        sys.exit(1)
    print(f"  DIC: {dic.shape}, Cy5: {cy5.shape}")

    # Subsample for the per-frame estimate (full stack on a 2048²
    # recording is heavy)
    n_total = len(dic)
    if n_total > args.sample_frames:
        idx = np.linspace(0, n_total - 1, args.sample_frames,
                           dtype=int)
    else:
        idx = np.arange(n_total)
    print(f"\nMeasuring offset on {len(idx)} sampled frames "
          f"(every {max(1, n_total // len(idx))} frames) …")
    offsets = measure_per_frame_offset(dic[idx], cy5[idx])

    print(f"\nPer-sampled-frame offsets (dy, dx) in pixels:")
    print(f"{'frame':>5} {'dy':>8} {'dx':>8}")
    for j, fi in enumerate(idx):
        print(f"{fi:>5} {offsets[j, 0]:>8.2f} {offsets[j, 1]:>8.2f}")

    median_dy = float(np.median(offsets[:, 0]))
    median_dx = float(np.median(offsets[:, 1]))
    sd_dy = float(np.std(offsets[:, 0]))
    sd_dx = float(np.std(offsets[:, 1]))
    range_dy = (float(offsets[:, 0].min()),
                 float(offsets[:, 0].max()))
    range_dx = (float(offsets[:, 1].min()),
                 float(offsets[:, 1].max()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Median offset (dy, dx):  ({median_dy:+.2f}, "
          f"{median_dx:+.2f}) px")
    print(f"  SD across sampled frames: dy {sd_dy:.2f}, dx {sd_dx:.2f}")
    print(f"  Range dy: [{range_dy[0]:+.2f}, {range_dy[1]:+.2f}]")
    print(f"  Range dx: [{range_dx[0]:+.2f}, {range_dx[1]:+.2f}]")
    print()
    if sd_dy < 1.0 and sd_dx < 1.0:
        print(f"  → Offset is STABLE across the stack (SD < 1 px).")
        print(f"    Apply the median offset as a single global shift "
              f"to Cy5 before fusion.")
    else:
        print(f"  → Offset DRIFTS across frames (SD ≥ 1 px).")
        print(f"    Use per-frame correction.")

    # Visual: before vs after on frame 0
    print("\nRendering before/after diagnostic …")
    out_dir = os.path.join(folder, "channel_alignment")
    os.makedirs(out_dir, exist_ok=True)
    f0_dic = dic[0]
    f0_cy5 = cy5[0]
    f0_cy5_shifted = nd_shift(f0_cy5.astype(np.float32),
                                shift=(median_dy, median_dx),
                                order=1).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Overlay each on top of the other as separate RGB channels
    # DIC in grayscale base + Cy5 in inferno overlay
    for col, (ttl, cy5_overlay) in enumerate([
            (f"F0  DIC + Cy5 RAW", f0_cy5),
            (f"F0  DIC + Cy5 shifted by ({median_dy:+.1f}, "
             f"{median_dx:+.1f}) px", f0_cy5_shifted)]):
        axes[col].imshow(f0_dic, cmap="gray")
        axes[col].imshow(cy5_overlay, cmap="inferno",
                          alpha=0.45,
                          vmin=0,
                          vmax=np.percentile(cy5_overlay, 99))
        axes[col].set_title(ttl, fontsize=12)
        axes[col].axis("off")

    # Plot per-frame drift
    axes[2].plot(idx, offsets[:, 0], "o-", label="dy")
    axes[2].plot(idx, offsets[:, 1], "s-", label="dx")
    axes[2].axhline(median_dy, color="C0", ls="--", lw=0.7,
                     label=f"median dy = {median_dy:+.2f}")
    axes[2].axhline(median_dx, color="C1", ls="--", lw=0.7,
                     label=f"median dx = {median_dx:+.2f}")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Offset (px)")
    axes[2].set_title(f"Per-frame offset (SD dy={sd_dy:.2f}, "
                       f"dx={sd_dx:.2f})")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "alignment_diagnostic.png")
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    print(f"Saved {out_path}")

    # Also dump the offsets to JSON for later use
    import json
    with open(os.path.join(out_dir, "offset.json"), "w") as f:
        json.dump({
            "recording": tif,
            "median_dy_px": median_dy,
            "median_dx_px": median_dx,
            "sd_dy_px": sd_dy,
            "sd_dx_px": sd_dx,
            "per_frame_offsets": [
                {"frame": int(fi), "dy": float(offsets[j, 0]),
                 "dx": float(offsets[j, 1])}
                for j, fi in enumerate(idx)],
            "recommendation": (
                "stable_global_shift"
                if (sd_dy < 1.0 and sd_dx < 1.0)
                else "per_frame_correction"),
        }, f, indent=2)
    print(f"Saved {out_dir}/offset.json")


if __name__ == "__main__":
    main()
