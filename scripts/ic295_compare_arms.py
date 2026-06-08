"""Arm-aware treatment comparison — respects the IC295 experimental design.

The IC295 conditions are TWO independent experiments, each with its own
control, plus a vehicle check:

  GENETIC arm — control WT      — vs GOF, KO
  DRUG    arm — vehicle DMSO    — vs YODA1 (Y1), OT
  VEHICLE     — WT vs DMSO      — effect of the vehicle alone

A flat 6-way Kruskal-Wallis + all-15-pairwise-Bonferroni
(`ic295_compare.py`) is WRONG for this design: it tests biologically
meaningless cross-arm contrasts (e.g. GOF vs OT) and over-corrects the
ones that matter. Here each arm gets its OWN Kruskal-Wallis across its 3
conditions + pairwise Mann-Whitney, **Bonferroni-corrected WITHIN the
arm**; the vehicle is a single MWU (no correction).

Reads the `per_recording.csv` / `per_cell_pooled.csv` already written by
`ic295_compare.py` / `ic295_compare_pooled.py` — run those first.

Outputs (per level):
  <compare_dir>/stats_arms.json        per metric -> {genetic, drug, vehicle}
  <compare_dir>/plots_arms/<metric>.png  genetic | drug panels, control
                                         highlighted, test-vs-control stars

Usage:
  conda run -n cellpose4 python scripts/ic295_compare_arms.py
  conda run -n cellpose4 python scripts/ic295_compare_arms.py --level recording
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import COMPARE_DIR, atomic_write_json  # noqa: E402
from scripts.ic295_compare import (  # noqa: E402
    _summary_stats, _kw_and_pairs, DEFAULT_METRICS)
from scripts.ic295_compare_pooled import (  # noqa: E402
    COMPARE_POOLED_DIR, DEFAULT_CELL_METRICS)
import numpy as np  # noqa: E402

# The experimental design. Each arm's control is listed FIRST so the
# pairwise keys read "<control>_vs_<test>".
ARMS = {
    "genetic": {"control": "WT",   "conditions": ["WT", "GOF", "KO"]},
    "drug":    {"control": "DMSO", "conditions": ["DMSO", "Y1", "OT"]},
}
VEHICLE = ["WT", "DMSO"]                       # vehicle-alone effect

_CTRL_COLOR = "#bdbdbd"                        # WT / DMSO boxes
_TEST_COLOR = "#cce4ff"

LEVELS = {
    "recording": (COMPARE_DIR, "per_recording.csv", DEFAULT_METRICS,
                  "recording"),
    "pooled":    (COMPARE_POOLED_DIR, "per_cell_pooled.csv",
                  DEFAULT_CELL_METRICS, "cell"),
}


def _group_by_condition(csv_path, metric):
    """Read one metric column from a comparison CSV → {condition: [floats]}."""
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            c = r.get("condition")
            try:
                v = float(r.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            out.setdefault(c, []).append(v)
    return out


def _arm_stats(groups_all):
    """Per-arm KW + within-arm Bonferroni pairwise, plus the vehicle MWU."""
    res = {}
    for arm, spec in ARMS.items():
        sub = {c: groups_all.get(c, []) for c in spec["conditions"]}
        res[arm] = _kw_and_pairs(sub)         # Bonferroni n = pairs in THIS arm
    res["vehicle"] = _kw_and_pairs(
        {c: groups_all.get(c, []) for c in VEHICLE})
    return res


def _stars(p):
    if p is None:
        return "ns"
    return ("***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "ns")


def _bracket(ax, x1, x2, y, label):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="#333")
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom",
            fontsize=10, color="#333")


def _plot_arms(metric, groups_all, stats, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (arm, spec) in zip(axs, ARMS.items()):
        conds = spec["conditions"]; ctrl = spec["control"]
        data = [groups_all.get(c, []) for c in conds]
        if not any(len(d) for d in data):
            ax.set_visible(False); continue
        bp = ax.boxplot(data, labels=[f"{c}\n(control)" if c == ctrl else c
                                       for c in conds],
                        widths=0.6, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], conds):
            patch.set(facecolor=_CTRL_COLOR if c == ctrl else _TEST_COLOR,
                      edgecolor="#446")
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            ax.scatter(rng.normal(i + 1, 0.06, size=len(vals)), vals,
                       s=20, color="#234", alpha=0.7, zorder=3)
        # significance brackets: each TEST vs the arm CONTROL
        allv = [v for d in data for v in d]
        if allv:
            top = max(allv); step = (top - min(allv) or 1) * 0.12
            k = 0
            for j, c in enumerate(conds):
                if c == ctrl:
                    continue
                pr = stats[arm]["pairs"].get(f"{ctrl}_vs_{c}", {})
                _bracket(ax, conds.index(ctrl) + 1, j + 1,
                         top + step * (1 + k),
                         f"{_stars(pr.get('p_bonf'))}")
                k += 1
        kw = (stats[arm].get("kw") or {}).get("p_value")
        ax.set_title(f"{arm} arm" + (f"   KW p={kw:.3f}" if kw is not None
                                     else ""))
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel(metric)
    veh = stats.get("vehicle", {}).get("pairs", {}).get("WT_vs_DMSO", {})
    vp = veh.get("p_raw")
    sup = metric + (f"      vehicle WT vs DMSO: p={vp:.3f} {_stars(vp)}"
                    if vp is not None else "")
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)


def _print_summary(level, all_stats):
    """Print the test-vs-control results that are significant (p_bonf<0.05)."""
    print(f"\n=== {level}: significant test-vs-control (within-arm "
          f"Bonferroni) ===")
    any_sig = False
    for m, st in all_stats.items():
        hits = []
        for arm, spec in ARMS.items():
            ctrl = spec["control"]
            for c in spec["conditions"]:
                if c == ctrl:
                    continue
                pr = st[arm]["pairs"].get(f"{ctrl}_vs_{c}", {})
                if pr.get("sig_bonf"):
                    hits.append(f"{c} vs {ctrl} (p={pr['p_bonf']:.3f})")
        veh = st["vehicle"]["pairs"].get("WT_vs_DMSO", {})
        if veh.get("p_raw") is not None and veh["p_raw"] < 0.05:
            hits.append(f"WT vs DMSO vehicle (p={veh['p_raw']:.3f})")
        if hits:
            any_sig = True
            print(f"  {m}: " + "; ".join(hits))
    if not any_sig:
        print("  (none — see stats_arms.json for raw/omnibus p-values)")


def _run_level(level):
    cdir, csv_name, metrics, _unit = LEVELS[level]
    csv_path = os.path.join(cdir, csv_name)
    if not os.path.exists(csv_path):
        print(f"  SKIP {level}: {csv_path} missing — run the matching "
              f"ic295_compare first.")
        return
    all_stats = {}
    for m in metrics:
        groups = _group_by_condition(csv_path, m)
        all_stats[m] = _arm_stats(groups)
        _plot_arms(m, groups, all_stats[m],
                   os.path.join(cdir, "plots_arms", f"{m}.png"))
    atomic_write_json(os.path.join(cdir, "stats_arms.json"), all_stats)
    _print_summary(level, all_stats)
    print(f"  wrote {cdir}/stats_arms.json + plots_arms/*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["recording", "pooled", "both"],
                    default="both")
    args = ap.parse_args()
    levels = (["recording", "pooled"] if args.level == "both"
              else [args.level])
    print("Arm-aware comparison — genetic (control WT) | drug (vehicle "
          "DMSO) | WT-vs-DMSO vehicle check")
    for lv in levels:
        _run_level(lv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
