"""Check every GUI's defaults match core.pipeline_defaults.DEFAULTS.

This is a regression test for the May-2026 GUI-vs-script default-drift
class of bugs (see CLAUDE.md "Pipeline defaults live in ONE place").

Run via:
  conda run -n cellpose4 python scripts/test_defaults_consistency.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication  # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)

from core.pipeline_defaults import DEFAULTS  # noqa: E402
from core.cell_state import DEFAULT_THRESHOLDS as STH  # noqa: E402

failures = []


def expect(name, got, want):
    if got != want:
        failures.append(f"  {name}: got {got!r}, want {want!r}")
        print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}: {got!r}")


print("=== gui_focused/params_panel.py ===")
from gui_focused.params_panel import ParamsPanel
p = ParamsPanel()
expect("min_area.value", p.min_area.value(), DEFAULTS.min_area_px)
expect("use_deepsea.isChecked", p.use_deepsea.isChecked(),
       DEFAULTS.use_deepsea)
expect("use_fallback.isChecked", p.use_fallback.isChecked(),
       DEFAULTS.use_fallback)
expect("use_gap_fill.isChecked", p.use_gap_fill.isChecked(),
       DEFAULTS.use_gap_fill)
expect("use_tta.isChecked", p.use_tta.isChecked(), DEFAULTS.use_tta)
expect("search_radius.value", p.search_radius.value(),
       DEFAULTS.search_radius_px)
expect("min_track_len.value", p.min_track_len.value(),
       DEFAULTS.min_track_length)
expect("cy5_filter_threshold.value", p.cy5_filter_threshold.value(),
       DEFAULTS.cy5_filter_threshold)
expect("state_balled_circ", p.state_balled_circ.value(),
       STH["balled_circ"])
expect("state_balled_solid", p.state_balled_solid.value(),
       STH["balled_solid"])
expect("state_attached_circ", p.state_attached_circ.value(),
       STH["attached_circ"])
expect("state_attached_solid", p.state_attached_solid.value(),
       STH["attached_solid"])

print()
print("=== gui_batch/batch_window.py ===")
from gui_batch.batch_window import BatchWindow
b = BatchWindow()
expect("min_area.value", b.min_area.value(), DEFAULTS.min_area_px)
expect("use_deepsea.isChecked", b.use_deepsea.isChecked(),
       DEFAULTS.use_deepsea)
expect("use_fallback.isChecked", b.use_fallback.isChecked(),
       DEFAULTS.use_fallback)
expect("use_gap_fill.isChecked", b.use_gap_fill.isChecked(),
       DEFAULTS.use_gap_fill)
expect("cy5_filter_threshold.value", b.cy5_filter_threshold.value(),
       DEFAULTS.cy5_filter_threshold)
expect("vampire_clusters.value", b.vampire_clusters.value(),
       DEFAULTS.vampire_n_clusters)
expect("compute_states.isChecked", b.compute_states.isChecked(),
       DEFAULTS.compute_state_classification)
expect("state_balled_circ", b.state_balled_circ.value(),
       STH["balled_circ"])
expect("state_balled_solid", b.state_balled_solid.value(),
       STH["balled_solid"])
expect("state_attached_circ", b.state_attached_circ.value(),
       STH["attached_circ"])
expect("state_attached_solid", b.state_attached_solid.value(),
       STH["attached_solid"])

print()
print("=== Worker module imports (verify _PD wired) ===")
import gui_focused.workers as fw
import gui_batch.batch_worker as bw
import gui_tracking.single_view as tsv
import gui_tracking.batch_view as tbv
import gui_tracking.batch_worker as tbw
for mod_name, mod in [
        ("gui_focused.workers", fw),
        ("gui_batch.batch_worker", bw),
        ("gui_tracking.single_view", tsv),
        ("gui_tracking.batch_view", tbv),
        ("gui_tracking.batch_worker", tbw)]:
    expect(f"{mod_name}._PD.min_area_px", mod._PD.min_area_px,
           DEFAULTS.min_area_px)

print()
if failures:
    print(f"FAIL: {len(failures)} mismatches")
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print("PASS: every GUI default matches DEFAULTS / DEFAULT_THRESHOLDS")
    sys.exit(0)
