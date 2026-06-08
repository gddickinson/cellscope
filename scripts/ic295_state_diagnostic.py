"""Confirm the rounded-vs-spread threshold from the data.

The deployed binary cut (core.cell_state.DEFAULT_THRESHOLDS, learned from
the hand labels — see scripts/ic295_eval_state_rule.py) is: a cell-frame
is ROUNDED when `area_um2 <= rounded_area_um2` AND `eccentricity <=
rounded_eccentricity`. This tool loads EVERY recording's masks.npz,
recomputes per-frame area (µm²) + eccentricity for every classifiable
(non-edge) cell-frame, and plots their distributions with the thresholds
drawn — so you can see whether the cut sits at a sensible place (a trough
between a small/rounded mode and a large/spread mode), pooled and per
condition.

Edge-truncated frames are excluded (they are unclassifiable — their
footprint is only partly in view).

Outputs under ic295_analysis/compare/state_diagnostic/:
  area_um2_hist.png       pooled + per-condition, rounded_area_um2 line
  eccentricity_hist.png   pooled + per-condition, rounded_eccentricity line
  area_vs_ecc_2d.png      2D density, the rounded region boxed (lower-left)
  summary.txt             % frames rounded (+ per gate, per condition)

Usage:
  conda run -n cellpose4 python scripts/ic295_state_diagnostic.py
  conda run -n cellpose4 python scripts/ic295_state_diagnostic.py --um 0.4
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

OUT_DIR = os.path.join(COMPARE_DIR, "state_diagnostic")
DEFAULT_UM = 0.6523                 # IC295 scope (single magnification)
_COND_COLORS = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
                "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}


def collect_frame_shapes(um):
    """Per classifiable (non-edge) cell-frame (area_um2, eccentricity,
    condition) over all recordings. Returns dict cond -> {area:[], ecc:[]}."""
    from core.cell_state import classify_track_states
    out = {c: {"area": [], "ecc": []} for c in CONDITIONS}
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    for i, mp in enumerate(paths):
        label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        cond = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(mp))))
        if cond not in CONDITIONS:
            cond = parse_condition(label)
        if cond not in out:
            continue
        try:
            labels = np.load(mp)["labels"]
        except Exception as e:
            print(f"  WARN {mp}: {e}")
            continue
        for cid in [int(v) for v in np.unique(labels) if v > 0]:
            sd = classify_track_states((labels == cid))
            m = sd["metrics"]
            area = m["area"] * um * um
            ecc = m["eccentricity"]
            edge = sd.get("edge")
            ok = np.isfinite(area) & np.isfinite(ecc)
            if edge is not None:
                ok = ok & ~np.asarray(edge)        # drop truncated frames
            out[cond]["area"].extend(area[ok].tolist())
            out[cond]["ecc"].extend(ecc[ok].tolist())
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return out


def _hist_metric(data, key, thr, name, out_path, xmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pooled = np.concatenate(
        [np.asarray(data[c][key]) for c in CONDITIONS if data[c][key]]
        or [np.array([])])
    if pooled.size == 0:
        return
    hi = xmax or float(np.nanpercentile(pooled, 99.5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, hi, 60)
    ax1.hist(pooled, bins=bins, color="#88aacc", edgecolor="#3a5a78",
             density=True)
    ax1.axvline(thr, color="#c00", lw=2,
                label=f"rounded cut = {thr}  (rounded ≤)")
    frac = float((pooled <= thr).mean())
    ax1.set_title(f"{name} — all classifiable cell-frames (n={pooled.size})\n"
                  f"{100*frac:.1f}% ≤ cut")
    ax1.set_xlabel(name); ax1.set_ylabel("density")
    ax1.set_xlim(0, hi); ax1.legend(); ax1.grid(alpha=0.3)
    for c in CONDITIONS:
        v = np.asarray(data[c][key])
        if v.size < 5:
            continue
        ax2.hist(v, bins=np.linspace(0, hi, 40), histtype="step",
                 density=True, lw=1.6, color=_COND_COLORS.get(c),
                 label=f"{c} (n={v.size})")
    ax2.axvline(thr, color="#c00", lw=2, ls="--")
    ax2.set_title(f"{name} by condition")
    ax2.set_xlabel(name); ax2.set_ylabel("density")
    ax2.set_xlim(0, hi); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130); plt.close(fig)


def _hist2d(data, ta, te, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    area = np.concatenate(
        [np.asarray(data[c]["area"]) for c in CONDITIONS if data[c]["area"]]
        or [np.array([])])
    ecc = np.concatenate(
        [np.asarray(data[c]["ecc"]) for c in CONDITIONS if data[c]["ecc"]]
        or [np.array([])])
    if area.size == 0:
        return
    hi = float(np.nanpercentile(area, 99.5))
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    hb = ax.hexbin(np.clip(area, 0, hi), ecc, gridsize=50, cmap="magma",
                   bins="log", mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    # rounded region: area ≤ ta AND ecc ≤ te (lower-left)
    ax.add_patch(Rectangle((0, 0), ta, te, fill=False, edgecolor="cyan",
                           lw=2.5))
    ax.axvline(ta, color="cyan", lw=1, ls=":")
    ax.axhline(te, color="cyan", lw=1, ls=":")
    rounded = float(((area <= ta) & (ecc <= te)).mean())
    ax.set_xlim(0, hi); ax.set_ylim(0, 1.02)
    ax.set_title(f"area (µm²) vs eccentricity (all classifiable frames)\n"
                 f"rounded region = area≤{ta:.0f} & ecc≤{te}  "
                 f"({100*rounded:.1f}% of frames)")
    ax.set_xlabel("area  (µm²)"); ax.set_ylabel("eccentricity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close(fig)


def _summary(data, ta, te, out_path):
    lines = ["State-threshold diagnostic (deployed rule)",
             f"  rounded_area_um2 = {ta}   rounded_eccentricity = {te}", ""]
    lines.append(f"{'condition':10s} {'frames':>8s} {'%area≤':>8s} "
                 f"{'%ecc≤':>8s} {'%rounded':>9s}")
    alla, alle = [], []
    for c in CONDITIONS:
        area = np.asarray(data[c]["area"]); ecc = np.asarray(data[c]["ecc"])
        if area.size == 0:
            continue
        alla.append(area); alle.append(ecc)
        rr = float(((area <= ta) & (ecc <= te)).mean())
        lines.append(f"{c:10s} {area.size:8d} {100*(area<=ta).mean():8.1f} "
                     f"{100*(ecc<=te).mean():8.1f} {100*rr:9.1f}")
    if alla:
        area = np.concatenate(alla); ecc = np.concatenate(alle)
        rr = float(((area <= ta) & (ecc <= te)).mean())
        lines.append(f"{'ALL':10s} {area.size:8d} {100*(area<=ta).mean():8.1f} "
                     f"{100*(ecc<=te).mean():8.1f} {100*rr:9.1f}")
    txt = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


def main():
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    um = DEFAULT_UM
    if "--um" in sys.argv:
        um = float(sys.argv[sys.argv.index("--um") + 1])
    ta, te = TH["rounded_area_um2"], TH["rounded_eccentricity"]
    print(f"Collecting per-frame shapes (rounded cut: area≤{ta} µm² & "
          f"ecc≤{te}; um/px={um})…", flush=True)
    data = collect_frame_shapes(um)
    os.makedirs(OUT_DIR, exist_ok=True)
    _hist_metric(data, "area", ta, "area  (µm²)",
                 os.path.join(OUT_DIR, "area_um2_hist.png"))
    _hist_metric(data, "ecc", te, "eccentricity",
                 os.path.join(OUT_DIR, "eccentricity_hist.png"), xmax=1.0)
    _hist2d(data, ta, te, os.path.join(OUT_DIR, "area_vs_ecc_2d.png"))
    _summary(data, ta, te, os.path.join(OUT_DIR, "summary.txt"))
    print(f"\nWrote diagnostics → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
