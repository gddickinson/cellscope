"""Flower (origin-centred track) plots + per-cell motility + MSD for IC293.

Thin wrapper: the plotting is reused verbatim from ic295_flower_plots
(same per-condition `data` schema), fed the IC293 single-cell-crop cache
and a shorter MSD lag (crops are 30–97 frames). The rounded/spread
flower panels use the IC295 keratinocyte state rule and are EXPLORATORY
on these endothelial cells — the 'all cells' panels are the primary view.

Usage:
  conda run -n cellpose4 python scripts/ic293_flower_plots.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic293_common import COMPARE_DIR, CONDITIONS  # noqa: E402
from scripts.ic293_track_data import load_or_build, DEFAULT_UM, DEFAULT_DT  # noqa: E402
from scripts.ic295_flower_plots import (  # noqa: E402
    _plot_grid, _plot_by_cond, _plot_msd, _axis_limit,
    _MOTILITY, _GROUP_TITLE, _MSD_ARMS)

OUT_DIR = os.path.join(COMPARE_DIR, "flower_plots")
MSD_MAXLAG = 24          # frames (× 10 min = 240 min) — fits the 30-frame crops


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_or_build(DEFAULT_UM, DEFAULT_DT,
                         from_cache="--from-cache" in sys.argv)
    lim = _axis_limit(data)

    for key, title, fname in [
        ("all", "All cell tracks — full track (IC293 EC crops)",
         "flower_all.png"),
        ("rounded", "Cells ROUNDED whole track (state rule EXPLORATORY)",
         "flower_rounded_only.png"),
        ("spread", "Cells SPREAD whole track (state rule EXPLORATORY)",
         "flower_spread_only.png"),
    ]:
        n = sum(len(data[c][key]) for c in CONDITIONS)
        _plot_grid(data, key, title, os.path.join(OUT_DIR, fname), lim)
        print(f"  flower {fname}  ({n} cells, axes ±{lim:.0f} µm)")

    for key in ("all", "rounded", "spread"):
        for mkey, ylabel, stem in _MOTILITY:
            title = f"{ylabel.split('  (')[0]} — {_GROUP_TITLE[key]} (full track)"
            _plot_by_cond(data, key, mkey, ylabel, title,
                          os.path.join(OUT_DIR, f"{stem}_{key}.png"))
        print(f"  motility plots ({key}): speed / distance / netdisp")

    _plot_msd(data, "all", os.path.join(OUT_DIR, "msd_by_treatment.png"),
              DEFAULT_DT, maxlag=MSD_MAXLAG)
    _plot_msd(data, "all", os.path.join(OUT_DIR, "msd_by_treatment_median.png"),
              DEFAULT_DT, maxlag=MSD_MAXLAG, stat="median")
    for arm, label, conds in _MSD_ARMS:
        _plot_msd(data, "all", os.path.join(OUT_DIR, f"msd_{arm}.png"),
                  DEFAULT_DT, maxlag=MSD_MAXLAG, conds=conds, arm_label=label)
    print("  MSD plots (all + per-arm, maxlag=240 min)")
    print(f"\nWrote flower + motility + MSD plots → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
