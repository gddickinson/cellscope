"""Phase C: ROI + Mask Editor integration in the Focused GUI."""
import tempfile
import numpy as np
from PyQt5.QtWidgets import QApplication
from ._common import check, shot, trim, SINGLE_CELL


def run():
    from gui_focused.main_window import FocusedMainWindow
    from gui.mask_editor import MaskEditor
    from core.io import load_recording

    app = QApplication.instance()

    print("\n=== Phase C: ROI + Mask Editor ===")
    w = FocusedMainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    rec = load_recording(SINGLE_CELL)
    rec["frames"] = trim(rec["frames"], 5)
    w.recording = rec
    w.viewer.set_data(rec["frames"])
    w.pipeline.set_stage_status("load", "done")
    w.pipeline.enable_stage("detect", True)
    w.params.set_from_recording(rec)
    app.processEvents()

    # Programmatically draw a rectangle ROI
    H, W = rec["frames"][0].shape
    bbox = (H // 4, 3 * H // 4, W // 4, 3 * W // 4)
    mask = np.zeros((H, W), dtype=bool)
    mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = True
    w.roi.roi_mask = mask
    w.roi.bbox = bbox
    w.roi.shape = "rectangle"
    w.roi.active = True
    w.params.use_roi.setChecked(True)
    w.viewer._roi_selector = w.roi
    w.viewer._redraw()
    app.processEvents()
    shot(w, "C_01_roi_drawn")
    check("C", "roi_drawn", w.roi.has_roi())
    check("C", "roi_active", w.roi.active)

    # Persistence across frames
    w.viewer._on_frame(2)
    app.processEvents()
    check("C", "roi_persists_across_frames",
          w.roi.has_roi() and w.roi.active)

    # ROI application zeros pixels outside the bbox
    out = w.roi.apply_to_frames(rec["frames"])
    outside_zero = (out[0][:bbox[0], :].sum() == 0)
    check("C", "roi_zeros_outside", outside_zero)
    inside_kept = (out[0][bbox[0]:bbox[1], bbox[2]:bbox[3]].sum() > 0)
    check("C", "roi_preserves_inside", inside_kept)
    shot(w, "C_02_roi_applied")

    # Clear ROI
    w.roi.clear()
    w.params.use_roi.setChecked(False)
    app.processEvents()
    check("C", "roi_cleared", not w.roi.has_roi())

    # Mask editor integration
    print("  testing mask editor integration...")
    # Build fake mask stack matching the full recording length (load_video
    # inside MaskEditor returns the full 97 frames; a smaller stack would
    # trip a QMessageBox shape-mismatch and block in headless mode).
    n_full = len(rec["frames"])  # frames trimmed to 5 above, but we want full
    # Re-load the actual video frame count to match what MaskEditor will see
    from core.io import load_video
    full_frames = load_video(rec.get("video_path"))
    n_full = len(full_frames)
    fake_masks = np.zeros((n_full, H, W), dtype=np.int32)
    fake_masks[:, H // 3:2 * H // 3, W // 3:2 * W // 3] = 1
    w.detect_result = {"masks": fake_masks > 0, "labels": fake_masks}

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez_compressed(tmp.name, masks=fake_masks)
    tmp.close()
    editor = MaskEditor(video_path=rec.get("video_path"),
                        mask_path=tmp.name)
    editor.show()
    app.processEvents()
    shot(editor, "C_03_mask_editor_open")
    check("C", "mask_editor_opens", editor.isVisible())
    check("C", "mask_editor_loaded_masks",
          editor.masks.max() == 1,
          f"max id = {editor.masks.max()}")

    # Simulate "Send to GUI": modify masks → emit
    edited = fake_masks.copy()
    edited[:, :H // 4, :W // 4] = 2   # new cell #2
    editor.masks_sent.connect(w._on_masks_received)
    editor.masks_sent.emit(edited)
    app.processEvents()
    check("C", "masks_sent_received",
          int(w.detect_result["labels"].max()) == 2,
          f"max_id={w.detect_result['labels'].max()}")
    shot(w, "C_04_after_edit_received")

    editor.close()
    w.close()
    app.processEvents()
