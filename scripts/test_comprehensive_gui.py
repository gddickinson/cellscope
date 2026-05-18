"""Comprehensive multi-phase GUI test for CellScope.

Phases B-G run against the CellScope GUIs taking screenshots at every
stage. Phase A is covered by the existing scripts/test_focused_gui.py.

Phases:
  B  Multi-cell detection + analysis (gui_focused multi-cell)
  C  ROI + mask-editor integration
  D  Batch GUI (scan + settings + dry-run)
  E  Tracking GUI (load, track, table, plots)
  F  Training + Editor GUIs
  G  Parameter-flow verification

Usage:
  conda run -n cellpose4 python scripts/test_comprehensive_gui.py
  conda run -n cellpose4 python scripts/test_comprehensive_gui.py --phase B
"""
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Defer all heavy imports to phase modules. Make sure offscreen platform
# is set before any Qt import happens.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication  # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)

from scripts.comprehensive import _common  # noqa: E402
from scripts.comprehensive import (phase_b, phase_c, phase_d,
                                    phase_e, phase_f, phase_g)  # noqa

PHASES = {
    "B": phase_b.run,
    "C": phase_c.run,
    "D": phase_d.run,
    "E": phase_e.run,
    "F": phase_f.run,
    "G": phase_g.run,
}


def write_reports():
    pass_n = sum(1 for r in _common.RESULTS if r[2] == "PASS")
    fail_n = sum(1 for r in _common.RESULTS if r[2] == "FAIL")
    total = len(_common.RESULTS)

    out_dir = _common.OUT_DIR
    shots_dir = _common.SHOTS

    print(f"\n{'='*60}")
    print(f"Comprehensive GUI tests: {pass_n}/{total} pass")
    print(f"Screenshots: {shots_dir}/")

    report = {
        "summary": {"pass": pass_n, "fail": fail_n, "total": total},
        "tests": [
            {"phase": p, "name": n, "status": s, "detail": d}
            for p, n, s, d in _common.RESULTS],
        "screenshots": sorted(os.listdir(shots_dir)),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    by_phase = {}
    for p, n, s, d in _common.RESULTS:
        by_phase.setdefault(p, []).append((n, s, d))

    lines = [
        "# CellScope GUI - Comprehensive Test Report",
        f"\n**Summary:** {pass_n}/{total} pass\n",
        "## Per-phase results\n",
    ]
    for ph in sorted(by_phase):
        pa = sum(1 for _, s, _ in by_phase[ph] if s == "PASS")
        tot = len(by_phase[ph])
        lines.append(f"### Phase {ph} ({pa}/{tot})\n")
        for n, s, d in by_phase[ph]:
            flag = "PASS" if s == "PASS" else "FAIL"
            tail = f" -- {d}" if d else ""
            lines.append(f"- **{flag}** `{n}`{tail}")
        lines.append("")

    lines.append("## Screenshots\n")
    for sh in sorted(os.listdir(shots_dir)):
        lines.append(f"- `screenshots/{sh}`")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    help="B, C, D, E, F, G, or all (default: all)")
    args = ap.parse_args()

    phases = (list(PHASES.keys()) if args.phase == "all"
              else [args.phase.upper()])
    for ph in phases:
        if ph not in PHASES:
            print(f"Unknown phase: {ph}")
            continue
        try:
            PHASES[ph]()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n!!! Phase {ph} crashed:\n{tb}")
            _common.RESULTS.append(
                (ph, "PHASE_CRASH", "FAIL", str(e)))

    write_reports()
    fails = sum(1 for r in _common.RESULTS if r[2] == "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
