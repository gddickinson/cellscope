"""Measure DIC ↔ Cy5 alignment using CELL DETECTIONS as landmarks.

Phase correlation fails on DIC/Cy5 pairs because the two channels
have very different signatures. Cell-based alignment is more
reliable: cpsam(DIC) and cpsam(Cy5) typically find the SAME cells;
the systematic offset between their matched centroids is the
alignment shift we want.

Algorithm per sampled frame:
  1. cpsam on DIC, cpsam on Cy5 (raw vit-h, no fine-tune)
  2. For every DIC cell, find the nearest Cy5 cell (max distance =
     50 px to avoid wild matches)
  3. Collect (dy, dx) of every match
  4. Report median + IQR

Output:
  channel_alignment/offset_from_detections.json
  channel_alignment/detection_offset.png  — scatter of all matched
                                            pairs, plus median arrow
"""
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

MAX_MATCH_DIST_PX = 50    # cell pairs further than this don't match
N_SAMPLE_FRAMES = 5
MIN_AREA = 200


def centroids_of(labels, min_area=MIN_AREA):
    out = []
    for cid in range(1, int(labels.max()) + 1):
        m = labels == cid
        if m.sum() < min_area:
            continue
        ys, xs = np.where(m)
        out.append((float(ys.mean()), float(xs.mean()),
                    int(m.sum())))
    return out


def match_centroids(dic_cents, cy5_cents, max_dist):
    """Hungarian assignment, filtered to dist ≤ max_dist."""
    if not dic_cents or not cy5_cents:
        return []
    a = np.array([(y, x) for y, x, _ in dic_cents])
    b = np.array([(y, x) for y, x, _ in cy5_cents])
    cost = cdist(a, b)
    rows, cols = linear_sum_assignment(cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_dist:
            out.append({
                "dic_cy": a[r, 0], "dic_cx": a[r, 1],
                "cy5_cy": b[c, 0], "cy5_cx": b[c, 1],
                "dy": float(a[r, 0] - b[c, 0]),   # DIC - Cy5
                "dx": float(a[r, 1] - b[c, 1]),
                "dist": float(cost[r, c]),
            })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    for f in os.listdir(folder):
        if f.endswith((".ome.tif", ".tif")):
            tif = os.path.join(folder, f)
            break
    else:
        sys.exit(f"no .ome.tif in {folder}")

    from core.io import load_recording
    rec = load_recording(tif, dic_channel=1, fluo_channel=0)
    dic, cy5 = rec["frames"], rec["cy5_frames"]
    if cy5 is None:
        sys.exit("no fluorescence channel")
    n = len(dic)
    idx = np.linspace(0, n - 1, min(N_SAMPLE_FRAMES, n), dtype=int)
    print(f"Sampling {len(idx)} frames out of {n} …")

    from cellpose import models
    print("Loading cpsam …")
    m = models.CellposeModel(gpu=True)

    all_matches = []
    print(f"\n{'frame':>5} {'n_dic':>6} {'n_cy5':>6} "
          f"{'matched':>8} {'med_dy':>8} {'med_dx':>8}")
    for fi in idx:
        labA = m.eval(dic[fi], augment=False)[0].astype(np.int32)
        labB = m.eval(cy5[fi], augment=False)[0].astype(np.int32)
        cA = centroids_of(labA)
        cB = centroids_of(labB)
        matches = match_centroids(cA, cB, MAX_MATCH_DIST_PX)
        for d in matches:
            d["frame"] = int(fi)
            all_matches.append(d)
        if matches:
            dys = [d["dy"] for d in matches]
            dxs = [d["dx"] for d in matches]
            print(f"{fi:>5} {len(cA):>6} {len(cB):>6} "
                  f"{len(matches):>8} {np.median(dys):>+8.2f} "
                  f"{np.median(dxs):>+8.2f}")
        else:
            print(f"{fi:>5} {len(cA):>6} {len(cB):>6} "
                  f"       0       —       —")

    if not all_matches:
        sys.exit("\nNo matched cell pairs — cannot estimate offset.")

    dys = np.array([d["dy"] for d in all_matches])
    dxs = np.array([d["dx"] for d in all_matches])
    med_dy, med_dx = float(np.median(dys)), float(np.median(dxs))
    iqr_dy = float(np.percentile(dys, 75) - np.percentile(dys, 25))
    iqr_dx = float(np.percentile(dxs, 75) - np.percentile(dxs, 25))

    # Convert to µm using metadata
    um_per_px = rec.get("um_per_px", 1.0) or 1.0

    print()
    print("=" * 60)
    print(f"SUMMARY ({len(all_matches)} matched cell pairs across "
          f"{len(idx)} frames)")
    print("=" * 60)
    print(f"  Median offset (DIC − Cy5):  dy = {med_dy:+.2f} px "
          f"({med_dy * um_per_px:+.2f} µm)")
    print(f"                                dx = {med_dx:+.2f} px "
          f"({med_dx * um_per_px:+.2f} µm)")
    print(f"  IQR:                          dy {iqr_dy:.2f}, "
          f"dx {iqr_dx:.2f}")
    print()
    if abs(med_dy) < 2 and abs(med_dx) < 2:
        print(f"  → Offset is <2 px in both axes. The 'misalignment'"
              f" you're seeing visually may be the fusion threshold "
              f"creating duplicates rather than a real shift.")
        print(f"    Fix: add a centroid-distance check to fusion's "
              f"same-cell merge logic (cheap; no re-detection needed).")
    elif iqr_dy < 2 and iqr_dx < 2:
        print(f"  → Offset is STABLE — apply ({med_dy:+.1f}, "
              f"{med_dx:+.1f}) px shift to Cy5 before fusion.")
    else:
        print(f"  → Offset DRIFTS — per-frame correction needed.")
    print()

    # Plot
    out_dir = os.path.join(folder, "channel_alignment")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(dxs, dys, s=15, alpha=0.5, color="C0",
                label=f"{len(all_matches)} matched pairs")
    ax.scatter([med_dx], [med_dy], s=200, marker="X",
                color="red", zorder=5, label=f"median "
                f"({med_dy:+.2f}, {med_dx:+.2f})")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("dx — DIC x – Cy5 x (px)")
    ax.set_ylabel("dy — DIC y – Cy5 y (px)")
    ax.set_title(f"DIC – Cy5 centroid offsets, {os.path.basename(folder)}\n"
                  f"({um_per_px:.4f} µm/px → "
                  f"{med_dy * um_per_px:+.2f}, "
                  f"{med_dx * um_per_px:+.2f} µm shift)")
    lim = max(15, abs(med_dy) * 2, abs(med_dx) * 2,
              float(np.max(np.abs(dxs))), float(np.max(np.abs(dys))))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detection_offset.png"), dpi=85)
    print(f"Saved {out_dir}/detection_offset.png")

    with open(os.path.join(out_dir,
                             "offset_from_detections.json"), "w") as f:
        json.dump({
            "recording": tif,
            "n_matched_pairs": len(all_matches),
            "median_dy_px": med_dy, "median_dx_px": med_dx,
            "median_dy_um": med_dy * um_per_px,
            "median_dx_um": med_dx * um_per_px,
            "iqr_dy_px": iqr_dy, "iqr_dx_px": iqr_dx,
            "matches": all_matches,
        }, f, indent=2, default=float)
    print(f"Saved {out_dir}/offset_from_detections.json")


if __name__ == "__main__":
    main()
