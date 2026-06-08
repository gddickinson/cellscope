"""Confirm the rounded-vs-spread threshold from the data.

The binary state cut is: a cell-frame is ROUNDED when circularity ≥
`rounded_circ` AND solidity ≥ `rounded_solid` (core.cell_state.
DEFAULT_THRESHOLDS). This tool loads EVERY recording's masks.npz,
recomputes per-frame circularity + solidity for every cell-frame, and
plots their distributions with the thresholds drawn — so you can see
whether the cut sits at a sensible place (ideally a trough between a
broad spread mode and a tight rounded mode), pooled and split by
condition.

Outputs under ic295_analysis/compare/state_diagnostic/:
  circularity_hist.png    pooled + per-condition, rounded_circ line
  solidity_hist.png       pooled + per-condition, rounded_solid line
  circ_vs_solid_2d.png     2D density, the rounded region boxed
  summary.txt             % frames rounded (+ per gate, per condition)

Usage:
  conda run -n cellpose4 python scripts/ic295_state_diagnostic.py
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
_COND_COLORS = {"WT": "#1f77b4", "KO": "#d62728", "GOF": "#2ca02c",
                "Y1": "#9467bd", "OT": "#ff7f0e", "DMSO": "#7f7f7f"}


def collect_frame_shapes():
    """Per cell-frame (circularity, solidity, condition) over all
    recordings. Returns dict condition -> {circ:[...], solid:[...]}."""
    from core.cell_state import classify_track_states
    out = {c: {"circ": [], "solid": []} for c in CONDITIONS}
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    for i, mp in enumerate(paths):
        label = os.path.basename(
            os.path.dirname(os.path.dirname(mp)))
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
        ids = [int(v) for v in np.unique(labels) if v > 0]
        for cid in ids:
            sd = classify_track_states((labels == cid))
            m = sd["metrics"]
            circ = m["circularity"]
            solid = m["solidity"]
            ok = np.isfinite(circ) & np.isfinite(solid)
            out[cond]["circ"].extend(circ[ok].tolist())
            out[cond]["solid"].extend(solid[ok].tolist())
        print(f"  [{i+1}/{len(paths)}] {label} ({cond})", flush=True)
    return out


def _hist_metric(data, key, thr, name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pooled = np.concatenate(
        [np.asarray(data[c][key]) for c in CONDITIONS if data[c][key]]
        or [np.array([])])
    if pooled.size == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # pooled
    ax1.hist(pooled, bins=60, color="#88aacc", edgecolor="#3a5a78",
             density=True)
    ax1.axvline(thr, color="#c00", lw=2,
                label=f"rounded threshold = {thr}")
    frac = float((pooled >= thr).mean())
    ax1.set_title(f"{name} — all cell-frames (n={pooled.size})\n"
                  f"{100*frac:.1f}% ≥ threshold")
    ax1.set_xlabel(name); ax1.set_ylabel("density")
    ax1.legend(); ax1.grid(alpha=0.3)
    # per-condition overlay (step, density)
    for c in CONDITIONS:
        v = np.asarray(data[c][key])
        if v.size < 5:
            continue
        ax2.hist(v, bins=40, histtype="step", density=True, lw=1.6,
                 color=_COND_COLORS.get(c), label=f"{c} (n={v.size})")
    ax2.axvline(thr, color="#c00", lw=2, ls="--")
    ax2.set_title(f"{name} by condition")
    ax2.set_xlabel(name); ax2.set_ylabel("density")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130); plt.close(fig)


def _hist2d(data, tc, ts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    circ = np.concatenate(
        [np.asarray(data[c]["circ"]) for c in CONDITIONS if data[c]["circ"]]
        or [np.array([])])
    solid = np.concatenate(
        [np.asarray(data[c]["solid"]) for c in CONDITIONS if data[c]["solid"]]
        or [np.array([])])
    if circ.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hb = ax.hexbin(circ, solid, gridsize=50, cmap="magma", bins="log",
                   mincnt=1)
    fig.colorbar(hb, ax=ax, label="log10(count)")
    # rounded region: circ ≥ tc AND solid ≥ ts (upper-right)
    ax.add_patch(Rectangle((tc, ts), 1.0 - tc, 1.0 - ts, fill=False,
                           edgecolor="cyan", lw=2.5))
    ax.axvline(tc, color="cyan", lw=1, ls=":")
    ax.axhline(ts, color="cyan", lw=1, ls=":")
    rounded = float(((circ >= tc) & (solid >= ts)).mean())
    ax.set_title(f"circularity vs solidity (all cell-frames)\n"
                 f"rounded region = circ≥{tc} & solid≥{ts}  "
                 f"({100*rounded:.1f}% of frames)")
    ax.set_xlabel("circularity"); ax.set_ylabel("solidity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130); plt.close(fig)


def _summary(data, tc, ts, out_path):
    lines = ["State-threshold diagnostic",
             f"  rounded_circ = {tc}   rounded_solid = {ts}", ""]
    allc, alls = [], []
    lines.append(f"{'condition':10s} {'frames':>8s} {'%circ≥':>8s} "
                 f"{'%solid≥':>8s} {'%rounded':>9s}")
    for c in CONDITIONS:
        circ = np.asarray(data[c]["circ"]); solid = np.asarray(data[c]["solid"])
        if circ.size == 0:
            continue
        allc.append(circ); alls.append(solid)
        rr = float(((circ >= tc) & (solid >= ts)).mean())
        lines.append(f"{c:10s} {circ.size:8d} {100*(circ>=tc).mean():8.1f} "
                     f"{100*(solid>=ts).mean():8.1f} {100*rr:9.1f}")
    if allc:
        circ = np.concatenate(allc); solid = np.concatenate(alls)
        rr = float(((circ >= tc) & (solid >= ts)).mean())
        lines.append(f"{'ALL':10s} {circ.size:8d} {100*(circ>=tc).mean():8.1f} "
                     f"{100*(solid>=ts).mean():8.1f} {100*rr:9.1f}")
    txt = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


def main():
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    tc, ts = TH["rounded_circ"], TH["rounded_solid"]
    print(f"Collecting per-frame shapes (rounded cut: circ≥{tc} & "
          f"solid≥{ts})…", flush=True)
    data = collect_frame_shapes()
    os.makedirs(OUT_DIR, exist_ok=True)
    _hist_metric(data, "circ", tc, "circularity",
                 os.path.join(OUT_DIR, "circularity_hist.png"))
    _hist_metric(data, "solid", ts, "solidity",
                 os.path.join(OUT_DIR, "solidity_hist.png"))
    _hist2d(data, tc, ts, os.path.join(OUT_DIR, "circ_vs_solid_2d.png"))
    _summary(data, tc, ts, os.path.join(OUT_DIR, "summary.txt"))
    print(f"\nWrote diagnostics → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
