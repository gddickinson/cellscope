"""Run the FULL IC293 analysis battery on the derived single-cell-crop dataset
`ic295_single-cell-crop_analysis/` (built by scripts/ic295sc_crop_cells.py).

It sets IC293_ANALYSIS_ROOT so every ic293_* script retargets to the derived
dataset (see the env-override note in scripts/ic293_common.py), then runs the
identical battery used for IC293 — NO code forks:

  1. analyze each crop      -> ic293_analyze_one.py  (per_cell.csv,
                               recording_summary.json, analysis.json,
                               RUN_METADATA) — parallel pool
  2. shape/state comparison -> ic293_compare.py       (per_position.csv,
                               stats_arms_shape.json, plots_arms/)
  3. build track cache      -> ic293_track_data.py --rebuild
  4. motility stats         -> ic293_motility_stats.py (stats_arms_motility.json,
                               REPORT.md, plots_arms/)
  5. flower / MSD plots     -> ic293_flower_plots.py

Unit of replication = the SOURCE IC295 recording (the `PosN` parsed from each
crop label — unique across conditions), the exact analogue of IC293's
"position is the unit". These are endothelial cells (IC295 = EC migration), same
cell type as IC293. The rounded/spread state rule was fitted on IC295's OWN hand
labels, so state metrics here are on the same footing as the main IC295
population analysis (a primary readout) — IC293 is a SEPARATE endothelial
experiment with no labels of its own, so its scripts flag the transferred rule
"exploratory". The auto-generated ic293 REPORT.md prints that IC293 wording
verbatim; see the dataset README + compare/INTERPRETATION.md for the framing
that applies here.

Run from cellpose4:
  conda run -n cellpose4 python scripts/ic295sc_run.py
  conda run -n cellpose4 python scripts/ic295sc_run.py --jobs 4 --force
  conda run -n cellpose4 python scripts/ic295sc_run.py --only analyze
"""
import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_ROOT = os.path.join(PROJECT_ROOT, "ic295_single-cell-crop_analysis")
os.environ["IC293_ANALYSIS_ROOT"] = NEW_ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()
from scripts.ic293_common import inventory_cache, CONDITIONS, COMPARE_DIR  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, IC293_ANALYSIS_ROOT=NEW_ROOT)


def _run(script, *script_args):
    cmd = [sys.executable, os.path.join(SCRIPTS, script), *script_args]
    print(f"\n>>> {script} {' '.join(script_args)}")
    return subprocess.run(cmd, env=ENV).returncode


def analyze_all(jobs, force):
    inv = inventory_cache()
    labels = sorted(inv)
    if not labels:
        print("No crops found — run scripts/ic295sc_crop_cells.py first")
        return 1
    percond = {c: 0 for c in CONDITIONS}
    for lab in labels:
        percond[inv[lab]["condition"]] += 1
    print(f"Analyzing {len(labels)} crops  per-condition: {percond}  "
          f"(jobs={jobs})")

    def one(lab):
        cmd = [sys.executable, os.path.join(SCRIPTS, "ic293_analyze_one.py"),
               lab] + (["--force"] if force else [])
        p = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
        return lab, p.returncode, (p.stdout or "").strip().splitlines()[-1:] \
            or [(p.stderr or "").strip().splitlines()[-1:]]

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = {ex.submit(one, lab): lab for lab in labels}
        for i, fut in enumerate(as_completed(futs), 1):
            lab, rc, tail = fut.result()
            tag = "ok " if rc == 0 else "FAIL"
            print(f"  [{i}/{len(labels)}] {tag} {lab}: "
                  f"{tail[0] if tail else ''}")
            ok += rc == 0
            fail += rc != 0
    print(f"analyze: {ok} ok, {fail} failed")
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel analyze workers (analysis is mask-only, "
                         "no cellpose — safe to parallelize)")
    ap.add_argument("--force", action="store_true",
                    help="re-analyze crops that already have analysis.json")
    ap.add_argument("--only", choices=["analyze", "compare", "track", "motility",
                                       "flowers"], default=None,
                    help="run only one stage (default: all in order)")
    args = ap.parse_args()
    t0 = time.time()
    print(f"IC293 battery retargeted to: {NEW_ROOT}")

    stages = {
        "analyze": lambda: analyze_all(args.jobs, args.force),
        "compare": lambda: _run("ic293_compare.py"),
        "track":   lambda: _run("ic293_track_data.py", "--rebuild"),
        "motility": lambda: _run("ic293_motility_stats.py"),
        "flowers": lambda: _run("ic293_flower_plots.py"),
    }
    order = ["analyze", "compare", "track", "motility", "flowers"]
    todo = [args.only] if args.only else order
    for st in todo:
        rc = stages[st]()
        if rc:
            print(f"[stage {st} returned {rc}] continuing")
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs under {COMPARE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
