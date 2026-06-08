"""Multi-feature diagnostic for the rounded/spread boundary.

The current cut keys on circularity (a clean-sphere detector), so it
misses the beads-on-a-string and crumpled forms of the balled-up state.
This tool recomputes a RICHER shape feature set for every cell-frame and
asks: which feature actually separates the two states best?

Per cell-frame it computes:
  area            (µm² if scaled)        footprint
  rel_area        area / cell's own 90th-pctl area    ← footprint collapse
  circularity     4πA/P²                 clean-sphere detector (current)
  solidity        area / convex area     concavity (blebbing / beads)
  extent          area / bbox area
  eccentricity    elongation
  aspect_ratio    bbox aspect
  convexity       convex-perimeter / perimeter   boundary roughness

Outputs under ic295_analysis/compare/state_features/:
  feature_hists.png    per-feature histograms + bimodality coefficient
  rel_area_by_state.png  rel_area split by the CURRENT (circ-based) call —
                       reveals small-footprint frames currently mislabelled
                       spread (the beads / crumpled the cut misses)
  rel_area_vs_circ.png 2D, coloured by current call
  summary.txt          per-feature bimodality + missed-retraction estimate

Mask-only (no intensity yet — add as a follow-up if shape isn't enough).
Usage:  conda run -n cellpose4 python scripts/ic295_state_features.py
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
import numpy as np  # noqa: E402

OUT_DIR = os.path.join(COMPARE_DIR, "state_features")
FEATURES = ["area", "rel_area", "circularity", "solidity", "extent",
            "eccentricity", "aspect_ratio", "convexity"]
# shape features whose rounded threshold (high = rounded) is worth drawing
_THR_KEY = {"circularity": "rounded_circ", "solidity": "rounded_solid"}


def _frame_feats(mask, min_area):
    from skimage import measure
    # Crop to the cell's bbox first — all features below are
    # translation-invariant, and regionprops on a small crop is ~100×
    # faster than scanning the full 2048² frame.
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    from core.mask_cleanup import clean_cell_mask
    mask = clean_cell_mask(mask[r0:r1 + 1, c0:c1 + 1])  # fill holes + despeck
    props = measure.regionprops(mask.astype(np.uint8))
    if not props:
        return None
    p = props[0]
    if p.area < min_area:
        return None
    area = float(p.area)
    per = float(p.perimeter)
    circ = (4 * np.pi * area / per ** 2) if per > 0 else np.nan
    try:
        conv_per = float(measure.perimeter(p.convex_image))
        convexity = conv_per / per if per > 0 else np.nan
    except Exception:
        convexity = np.nan
    minr, minc, maxr, maxc = p.bbox
    aspect = (maxc - minc) / max(maxr - minr, 1e-6)
    return {"area": area, "circularity": float(circ),
            "solidity": float(p.solidity), "extent": float(p.extent),
            "eccentricity": float(p.eccentricity),
            "aspect_ratio": float(aspect), "convexity": float(convexity)}


def collect(um_per_px_default=0.6523):
    """Per cell-frame feature rows over all recordings."""
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    min_area = TH["min_area_px"]
    tc, ts = TH["rounded_circ"], TH["rounded_solid"]
    rows = {f: [] for f in FEATURES}
    rows["condition"] = []
    rows["current_rounded"] = []
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    for i, mp in enumerate(paths):
        label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        cond = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(mp))))
        if cond not in CONDITIONS:
            cond = parse_condition(label)
        if cond not in CONDITIONS:
            continue
        # scale from sidecar if available, else default
        um2 = um_per_px_default ** 2
        try:
            labels = np.load(mp)["labels"]
        except Exception as e:
            print(f"  WARN {mp}: {e}"); continue
        for cid in [int(v) for v in np.unique(labels) if v > 0]:
            stack = labels == cid
            present = np.where(stack.any(axis=(1, 2)))[0]
            per_frame = {}
            for fi in present:
                d = _frame_feats(stack[fi], min_area)
                if d is not None:
                    per_frame[fi] = d
            if not per_frame:
                continue
            areas = np.array([per_frame[fi]["area"] for fi in per_frame])
            baseline = float(np.percentile(areas, 90)) or 1.0
            for fi, d in per_frame.items():
                rows["area"].append(d["area"] * um2)
                rows["rel_area"].append(d["area"] / baseline)
                rows["circularity"].append(d["circularity"])
                rows["solidity"].append(d["solidity"])
                rows["extent"].append(d["extent"])
                rows["eccentricity"].append(d["eccentricity"])
                rows["aspect_ratio"].append(d["aspect_ratio"])
                rows["convexity"].append(d["convexity"])
                rows["condition"].append(cond)
                rows["current_rounded"].append(
                    d["circularity"] >= tc and d["solidity"] >= ts)
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return {k: np.asarray(v) for k, v in rows.items()}


def bimodality_coeff(x):
    """Sarle's bimodality coefficient: >5/9≈0.555 hints at bimodality."""
    from scipy import stats as sps
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan
    g = sps.skew(x)
    k = sps.kurtosis(x, fisher=False)  # Pearson (normal = 3)
    n = x.size
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g * g + 1.0) / denom) if denom else np.nan


def _feature_hists(data, thr, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, f in zip(axes.ravel(), FEATURES):
        v = data[f][np.isfinite(data[f])]
        bc = bimodality_coeff(v)
        ax.hist(v, bins=60, color="#88aacc", edgecolor="#3a5a78")
        tk = _THR_KEY.get(f)
        if tk:
            ax.axvline(thr[tk], color="#c00", lw=2, ls="--")
        flag = "  ←bimodal?" if (bc is not None and bc > 0.555) else ""
        ax.set_title(f"{f}   BC={bc:.2f}{flag}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("per-frame shape features (all cells) — "
                 "BC>0.56 suggests two modes", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def _rel_area_by_state(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ra = data["rel_area"]; cur = data["current_rounded"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1.2, 60)
    ax.hist(ra[cur], bins=bins, color="#c0392b", alpha=0.7,
            label=f"current ROUNDED (n={int(cur.sum())})", density=True)
    ax.hist(ra[~cur], bins=bins, color="#2980b9", alpha=0.6,
            label=f"current SPREAD (n={int((~cur).sum())})", density=True)
    ax.axvline(0.4, color="k", lw=1.5, ls=":",
               label="candidate rel-area cut 0.40")
    ax.set_xlabel("relative area  (frame area / cell 90th-pctl area)")
    ax.set_ylabel("density")
    ax.set_title("Footprint collapse vs the current circularity call.\n"
                 "Blue mass left of the cut = retractions the circularity "
                 "rule MISSES (beads / crumpled).")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _2d(data, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ra = data["rel_area"]; circ = data["circularity"]
    cur = data["current_rounded"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(ra[~cur], circ[~cur], s=4, c="#2980b9", alpha=0.25,
               label="current spread")
    ax.scatter(ra[cur], circ[cur], s=6, c="#c0392b", alpha=0.5,
               label="current rounded")
    ax.axvline(0.4, color="k", ls=":")
    ax.axhline(0.8, color="#c00", ls="--")
    ax.set_xlabel("relative area"); ax.set_ylabel("circularity")
    ax.set_title("relative area vs circularity (one dot per cell-frame)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _summary(data, out_path):
    lines = ["Multi-feature state diagnostic", ""]
    lines.append(f"{'feature':14s} {'bimod.coeff':>11s} {'two-mode?':>10s}")
    for f in FEATURES:
        bc = bimodality_coeff(data[f])
        lines.append(f"{f:14s} {bc:11.3f} "
                     f"{'YES' if (bc and bc > 0.555) else 'no':>10s}")
    ra = data["rel_area"]; cur = data["current_rounded"]
    miss = int(((ra < 0.4) & (~cur)).sum())
    tot_small = int((ra < 0.4).sum())
    lines += ["",
              f"frames with rel_area < 0.40 (collapsed footprint): "
              f"{tot_small} ({100*tot_small/len(ra):.1f}%)",
              f"  ...currently called ROUNDED: {int(((ra<0.4)&cur).sum())}",
              f"  ...currently called SPREAD (MISSED retractions): {miss} "
              f"({100*miss/max(tot_small,1):.0f}% of collapsed frames)",
              f"current rounded (circ-based) total: {int(cur.sum())} "
              f"({100*cur.mean():.1f}%)"]
    txt = "\n".join(lines)
    with open(out_path, "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


def main():
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    print("Collecting multi-feature per-frame shapes…", flush=True)
    data = collect()
    os.makedirs(OUT_DIR, exist_ok=True)
    _feature_hists(data, TH, os.path.join(OUT_DIR, "feature_hists.png"))
    _rel_area_by_state(data, os.path.join(OUT_DIR, "rel_area_by_state.png"))
    _2d(data, os.path.join(OUT_DIR, "rel_area_vs_circ.png"))
    _summary(data, os.path.join(OUT_DIR, "summary.txt"))
    print(f"\nWrote → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
