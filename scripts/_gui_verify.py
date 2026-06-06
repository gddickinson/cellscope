"""Headless test-drive of every CellScope GUI + every focused-GUI option.

Faithfully drives the real widgets / slots (not internal QTest pokes):
  A. all five GUI main windows construct
  B. focused GUI deep drive on a real multi-cell recording:
       - detect params expose gap_fill_crop + gap_fill_augment (+ all keys)
       - gap-fill child toggles gate on use_gap_fill; revert flows work
       - the detect worker actually receives the two new kwargs
       - analysis params incl. divisions toggle; mode switch
       - colour-by EVERY metric (viewer + legend) with no exception
       - run Analyze (loaded masks) → drive every registered graph
       - share-image export (PNG + JPEG) via the real render path
       - overlay toggles redraw
  C. mask-editor colour-by + legend + refresh

Run:  QT_QPA_PLATFORM=offscreen conda run -n cellpose4 python -u scripts/_gui_verify.py
"""
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REC_DIR = "ic295_analysis/by_condition/WT/Pos7-WT"
TIF = os.path.join(REC_DIR, "IC295__1_MMStack_Pos7-WT.ome.tif")

_n_pass = 0
_n_fail = 0


def check(tag, cond, detail=""):
    global _n_pass, _n_fail
    ok = bool(cond)
    _n_pass += ok
    _n_fail += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}"
          + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def section(name):
    print(f"\n=== {name} ===", flush=True)


def pump(app, ms=50):
    """Process pending Qt events for ~ms milliseconds."""
    from PyQt5.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


# ----------------------------------------------------------------------
def construct_all_guis():
    section("A. all five GUI main windows construct")
    specs = [
        ("focused", "gui_focused.main_window", "FocusedMainWindow", ()),
        ("batch", "gui_batch.batch_window", "BatchWindow", ()),
        ("tracking", "gui_tracking.tracking_window", "TrackingWindow", ()),
        ("training", "gui_training.training_window", "TrainingWindow", ()),
        ("mask-editor", "gui.mask_editor", "MaskEditor", ()),
    ]
    wins = {}
    for name, mod, cls, args in specs:
        try:
            m = __import__(mod, fromlist=[cls])
            w = getattr(m, cls)(*args)
            wins[name] = w
            check(f"construct {name} ({cls})", True)
        except Exception as e:
            check(f"construct {name} ({cls})", False, repr(e))
            traceback.print_exc()
    return wins


# ----------------------------------------------------------------------
def drive_focused(app, win):
    section("B. focused GUI — load real multi-cell recording")
    if not os.path.exists(TIF):
        check("recording present", False, TIF)
        return
    win._remote_load_recording({"path": TIF})
    pump(app)
    check("recording loaded",
          win.recording is not None and len(win.recording["frames"]) > 1,
          f'{len(win.recording["frames"])} frames')
    win._remote_load_pipeline_results({"path": REC_DIR})
    pump(app)
    check("pipeline results loaded (detect_result set)",
          win.detect_result is not None)

    # --- detect params: all keys incl. the two gap-fill toggles ---
    section("B1. detect params — gap-fill toggles present + defaulted")
    from core.pipeline_defaults import DEFAULTS as PD
    p = win.params.get_detect_params()
    for k in ("use_gap_fill", "gap_fill_crop", "gap_fill_augment"):
        check(f"detect param key '{k}'", k in p)
    check("gap_fill_crop default == DEFAULTS",
          p.get("gap_fill_crop") == PD.gap_fill_crop,
          f"{p.get('gap_fill_crop')} vs {PD.gap_fill_crop}")
    check("gap_fill_augment default == DEFAULTS",
          p.get("gap_fill_augment") == PD.gap_fill_augment,
          f"{p.get('gap_fill_augment')} vs {PD.gap_fill_augment}")

    # --- gating: child toggles enable/disable with use_gap_fill ---
    section("B2. gap-fill child toggles gate on use_gap_fill")
    pp = win.params
    pp.use_gap_fill.setChecked(True)
    pump(app)
    check("crop enabled when gap-fill ON", pp.gap_fill_crop.isEnabled())
    check("augment enabled when gap-fill ON", pp.gap_fill_augment.isEnabled())
    pp.use_gap_fill.setChecked(False)
    pump(app)
    check("crop disabled when gap-fill OFF", not pp.gap_fill_crop.isEnabled())
    check("augment disabled when gap-fill OFF",
          not pp.gap_fill_augment.isEnabled())
    pp.use_gap_fill.setChecked(True)
    pump(app)

    # --- revert flows: flip the two toggles, confirm params follow ---
    section("B3. revert-to-old-behaviour flows")
    pp.gap_fill_crop.setChecked(False)        # old: full-frame
    pp.gap_fill_augment.setChecked(True)      # old: always augment
    pump(app)
    p2 = pp.get_detect_params()
    check("crop=False reaches params (old full-frame)",
          p2["gap_fill_crop"] is False)
    check("augment=True reaches params (old always-augment)",
          p2["gap_fill_augment"] is True)
    # restore latest defaults
    pp.gap_fill_crop.setChecked(PD.gap_fill_crop)
    pp.gap_fill_augment.setChecked(PD.gap_fill_augment)
    pump(app)

    # --- the detect worker actually receives the new kwargs ---
    section("B4. detect worker receives gap-fill kwargs")
    try:
        from gui_focused.workers import FocusedDetectWorker
        params = pp.get_detect_params()
        w = FocusedDetectWorker(win.recording, "auto", params)
        wp = getattr(w, "params", None) or getattr(w, "_params", None)
        check("worker stores params with gap_fill_crop",
              isinstance(wp, dict) and "gap_fill_crop" in wp)
        check("worker stores params with gap_fill_augment",
              isinstance(wp, dict) and "gap_fill_augment" in wp)
    except Exception as e:
        check("detect worker constructs", False, repr(e))
        traceback.print_exc()

    # --- analysis params + divisions toggle + mode switch ---
    section("B5. analysis params, divisions toggle, mode switch")
    dp = pp.get_division_params()
    check("division params default enabled", dp.get("enabled") is True)
    pp.compute_divisions.setChecked(False)
    check("divisions toggle OFF reaches params",
          pp.get_division_params().get("enabled") is False)
    pp.compute_divisions.setChecked(True)
    sp = pp.get_state_params()
    check("state params returned", isinstance(sp, dict))
    try:
        win._on_mode_changed("multi")
        pump(app)
        check("mode switch → multi", win.mode == "multi")
    except Exception as e:
        check("mode switch → multi", False, repr(e))

    # --- colour-by EVERY metric in the focused viewer ---
    section("B6. colour-by every metric (focused viewer + legend)")
    from gui.metric_coloring import metric_names, ID_METRIC
    vw = win.viewer
    for name in metric_names():
        try:
            vw.color_combo.setCurrentText(name)
            pump(app, 20)
            ok = (vw.color_metric == name)
            if name != ID_METRIC:
                ok = ok and (vw._metric_colorizer is not None)
            check(f"colour-by '{name}'", ok)
        except Exception as e:
            check(f"colour-by '{name}'", False, repr(e))
            traceback.print_exc()
    vw.color_combo.setCurrentText(ID_METRIC)
    pump(app)

    # --- overlay toggles redraw without error ---
    section("B7. overlay toggles redraw")
    for attr, fn in (("chk_tracks", "_on_toggle_tracks"),
                     ("chk_source", "_on_toggle_source"),
                     ("chk_dropped", "_on_toggle_dropped")):
        try:
            getattr(vw, fn)(True)
            pump(app, 20)
            getattr(vw, fn)(False)
            check(f"overlay {attr} redraw", True)
        except Exception as e:
            check(f"overlay {attr} redraw", False, repr(e))

    # --- run Analyze on loaded masks, then drive every graph ---
    drive_analyze_and_graphs(app, win)
    # --- share-image export ---
    drive_share(app, win)


def drive_analyze_and_graphs(app, win):
    section("B8. Analyze (loaded masks) → every registered graph")
    if win.detect_result is None:
        check("detect_result available for analyze", False)
        return
    win.mode = "multi"
    win.analysis_result = None
    try:
        win._on_analyze()
    except Exception as e:
        check("Analyze started", False, repr(e))
        traceback.print_exc()
        return
    # spin until the worker finishes (analysis on masks is fast)
    from PyQt5.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    done = {"v": False}

    def poll():
        if win.analysis_result is not None:
            done["v"] = True
            loop.quit()
    t = QTimer()
    t.timeout.connect(poll)
    t.start(200)
    QTimer.singleShot(180000, loop.quit)   # 3-min ceiling
    loop.exec_()
    t.stop()
    check("Analyze completed", done["v"] and win.analysis_result is not None)
    if not done["v"]:
        return
    av = win.analysis
    names = [av.graph_combo.itemText(i)
             for i in range(av.graph_combo.count())]
    check("graph combo populated (multi)", len(names) > 0,
          f"{len(names)} graphs")
    has_div = any("Lineage" in n or "Division" in n for n in names)
    check("division graphs present in multi mode", has_div,
          ", ".join(n for n in names if "Lineage" in n or "Division" in n))
    for n in names:
        try:
            av.graph_combo.setCurrentText(n)
            pump(app, 60)
            check(f"render graph '{n}'", True)
        except Exception as e:
            check(f"render graph '{n}'", False, repr(e))
            traceback.print_exc()


def drive_share(app, win):
    section("B9. shareable-image export (dialog's real _export path)")
    try:
        from gui_focused.share_export import ShareImageDialog
        dlg = ShareImageDialog(win.viewer, win.recording,
                               win.detect_result, parent=win)
        check("ShareImageDialog constructs", dlg is not None)
        dlg.rb_current.setChecked(True)
        dlg._refresh_formats()
        pump(app, 20)
        fmts = [dlg.fmt.itemText(i) for i in range(dlg.fmt.count())]
        check("current-frame formats offered", len(fmts) > 0, ", ".join(fmts))
        td = tempfile.mkdtemp(prefix="cs_share_")
        for fmt in ("PNG", "JPEG"):
            if fmt not in fmts:
                continue
            ext = "png" if fmt == "PNG" else "jpg"
            outp = os.path.join(td, f"share.{ext}")
            dlg.fmt.setCurrentText(fmt)
            dlg.path_edit.setText(outp)
            dlg.max_px.setValue(800)
            pump(app, 20)
            dlg._export()                     # the real production save path
            pump(app, 30)
            ok = os.path.exists(outp) and os.path.getsize(outp) > 0
            check(f"_export {fmt} (current frame)", ok,
                  f"{os.path.getsize(outp)} B" if ok else "missing")
    except Exception as e:
        check("share export", False, repr(e))
        traceback.print_exc()


def drive_mask_editor(app, editor):
    section("C. mask-editor colour-by + legend + refresh")
    if editor is None:
        check("mask editor available", False)
        return
    try:
        import numpy as np
        # give it a tiny labelled stack so colour-by has something to map
        H = W = 64
        stack = np.zeros((4, H, W), np.int32)
        stack[:, 10:30, 10:30] = 1
        stack[1:, 35:55, 35:55] = 2
        if hasattr(editor, "set_masks"):
            editor.set_masks(stack)
        elif hasattr(editor, "masks"):
            editor.masks = stack
        from gui.metric_coloring import metric_names, ID_METRIC
        for name in metric_names():
            editor.color_combo.setCurrentText(name)
            pump(app, 20)
            check(f"editor colour-by '{name}'",
                  editor.color_metric == name)
        # refresh recompute
        if hasattr(editor, "_on_refresh_metric"):
            editor._on_refresh_metric()
            pump(app, 20)
            check("editor colour refresh", True)
        editor.color_combo.setCurrentText(ID_METRIC)
    except Exception as e:
        check("mask editor colour-by", False, repr(e))
        traceback.print_exc()


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    wins = construct_all_guis()
    if "focused" in wins:
        drive_focused(app, wins["focused"])
    drive_mask_editor(app, wins.get("mask-editor"))

    print("\n" + "=" * 60)
    print(f"TOTAL: {_n_pass} passed, {_n_fail} failed")
    print("=" * 60)
    return 1 if _n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
