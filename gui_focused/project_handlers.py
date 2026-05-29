"""Project save/load + scan handlers extracted from main_window.
Keeps main_window.py under 500 lines."""
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt5.QtCore import Qt


def _rebuild_tracks_from_labels(labels):
    """Reconstruct a multi-cell tracks list from an (N, H, W) int32
    label stack: one track per unique non-zero ID, with `stack` = the
    per-frame boolean mask and `first_frame` = the first frame the
    cell appears in. Matches the shape `FocusedAnalyzeWorker` expects
    (`track["stack"]` + `track["first_frame"]`) so loaded masks can be
    analyzed without re-running detection."""
    if labels is None:
        return []
    ids = np.unique(labels)
    ids = ids[ids > 0]
    tracks = []
    for cid in ids.tolist():
        stack = (labels == int(cid))
        present = stack.any(axis=(1, 2))
        if not present.any():
            continue
        tracks.append({
            "stack": stack,
            "first_frame": int(np.argmax(present)),
        })
    return tracks


def on_save_project(win):
    if win.recording is None:
        return
    path, _ = QFileDialog.getSaveFileName(
        win, "Save Project", "",
        "CellScope Project (*.cellscope)")
    if not path:
        return
    if not path.endswith(".cellscope"):
        path += ".cellscope"
    from core.project import save_project
    save_project(
        path, win.recording, win.detect_result,
        win.analysis_result, win.params.get_detect_params(),
        win.mode,
        roi_mask=win.roi.roi_mask if win.roi.has_roi() else None)
    win.logger.log("info", f"Project saved: {path}")
    win.status.showMessage(f"Project saved: {os.path.basename(path)}")
    win._dirty = False


def on_load_project(win, path=None):
    """Load a .cellscope project. If `path` is None, prompt via file
    dialog; otherwise load the given file directly (used by drag-drop)."""
    if not win._confirm_discard(
            "Loading a project will replace them."):
        return
    if path is None:
        path, _ = QFileDialog.getOpenFileName(
            win, "Open Project", "",
            "CellScope Project (*.cellscope)")
        if not path:
            return
    from core.project import load_project
    proj = load_project(path)
    info = proj["recording_info"]
    video = info.get("video_path", "")
    if video and os.path.exists(video):
        win._load_path(video)
    if proj["masks"] is not None:
        win.detect_result = {"masks": proj["masks"]}
        if proj["labels"] is not None:
            win.detect_result["labels"] = proj["labels"]
        # Restore the per-cell tracks list (rebuilt by load_project from
        # the labels stack). Without this the multi-cell analysis worker
        # iterates an empty list and reports "0 cells".
        if proj.get("tracks"):
            win.detect_result["tracks"] = proj["tracks"]
        if proj.get("fusion_source_stack") is not None:
            win.detect_result["fusion_source_stack"] = (
                proj["fusion_source_stack"])
        masks = proj["labels"] if proj["labels"] is not None \
            else proj["masks"]
        win.viewer.update_masks(masks)
        if (hasattr(win.viewer, "set_source_stack")
                and proj.get("fusion_source_stack") is not None):
            win.viewer.set_source_stack(proj["fusion_source_stack"])
        win.viewer.nav_bar.set_status(proj["masks"])
        win.pipeline.set_stage_status("detect", "done")
        win.pipeline.enable_stage("edit", True)
        win.pipeline.enable_stage("analyze", True)
    win.pipeline.set_mode(proj["mode"])
    win.logger.log("info", f"Project loaded: {path}")
    win.status.showMessage(
        f"Project loaded: {os.path.basename(path)}")
    # Loaded state IS the saved state.
    win._dirty = False


def on_load_pipeline_results(win, path=None):
    """Load a batch-pipeline output folder.

    Looks for masks.npz / masks_unfiltered.npz / filter_decisions.json
    / divisions.json / RUN_METADATA.json. The user can pick either the
    recording folder (containing a `pipeline_results/` subdir) or the
    `pipeline_results/` folder directly.

    Loads the recording from RUN_METADATA.source_recording.video_path
    if it exists; otherwise prompts the user to locate the video.
    """
    import json
    if not win._confirm_discard(
            "Loading pipeline_results will replace them."):
        return
    if path is None:
        path = QFileDialog.getExistingDirectory(
            win, "Open Pipeline Results", "")
        if not path:
            return
    pr_dir = path
    # User can pick either the recording folder OR pipeline_results/.
    if not os.path.exists(os.path.join(pr_dir, "masks.npz")):
        candidate = os.path.join(path, "pipeline_results")
        if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "masks.npz")):
            pr_dir = candidate
    masks_path = os.path.join(pr_dir, "masks.npz")
    if not os.path.exists(masks_path):
        QMessageBox.warning(
            win, "Open Pipeline Results",
            f"No masks.npz in:\n  {pr_dir}\n\nPick either the recording "
            f"folder (containing pipeline_results/) or the "
            f"pipeline_results/ folder itself.")
        return

    # Resolve the source recording. Priority order:
    #   1. RUN_METADATA.json's source_recording.video_path (if accessible).
    #   2. A sibling recording in pr_dir's parent (handles the gt_review
    #      pattern: gt_review/<rec>/<rec>.ome.tif sits next to
    #      gt_review/<rec>/pipeline_results/masks.npz).
    #   3. The currently-loaded recording, if any — the user already
    #      told us what to overlay onto by loading it first.
    #   4. Prompt the user to locate it.
    meta_path = os.path.join(pr_dir, "RUN_METADATA.json")
    video_path = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            video_path = (meta.get("source_recording") or {}).get("video_path")
        except Exception:
            video_path = None
    if not video_path or not os.path.exists(video_path):
        import glob
        parent = os.path.dirname(os.path.abspath(pr_dir))
        for pat in ("*.ome.tif", "*.tif", "*.tiff",
                    "*.mp4", "*.avi", "*.mov"):
            sibs = sorted(glob.glob(os.path.join(parent, pat)))
            if sibs:
                video_path = sibs[0]
                break
    reuse_current = False
    if not video_path or not os.path.exists(video_path):
        if win.recording is not None:
            reuse_current = True
        else:
            prompt = (f"The recording referenced by RUN_METADATA "
                       f"({video_path or 'unknown'}) is not accessible.\n\n"
                       f"Locate the recording file:")
            QMessageBox.information(win, "Locate recording", prompt)
            video_path, _ = QFileDialog.getOpenFileName(
                win, "Locate recording", "",
                "Recording (*.mp4 *.avi *.mov *.tif *.tiff)")
            if not video_path:
                return
    elif win.recording is not None:
        # Resolved video matches what's already loaded — skip the
        # redundant reload (matters for multi-GB recordings).
        cur = win.recording.get("video_path", "")
        try:
            if (cur and os.path.realpath(cur)
                    == os.path.realpath(video_path)):
                reuse_current = True
        except Exception:
            pass
    if reuse_current:
        # Replicate the stage reset _load_path would have done.
        win.analysis_result = None
        win.analysis.clear()
        win.pipeline.reset_all()
        win.pipeline.set_stage_status("load", "done")
        win.pipeline.enable_stage("detect", True)
    else:
        win._load_path(video_path)
        if win.recording is None:
            return

    # Load masks (+ optional fusion source stack from same archive).
    npz = np.load(masks_path, allow_pickle=True)
    keys = set(npz.files)
    labels = npz["labels"] if "labels" in keys else None
    masks = npz["masks"].astype(bool) if "masks" in keys else (
        labels > 0 if labels is not None else None)
    fusion_src = (npz["fusion_source_stack"].astype(np.uint8)
                   if "fusion_source_stack" in keys else None)

    win.detect_result = {"masks": masks}
    if labels is not None:
        win.detect_result["labels"] = labels
        # Multi-cell analyze (FocusedAnalyzeWorker) walks
        # detect_result["tracks"] — when loading from masks.npz there
        # were no tracks, so analyze reported "0 cells detected".
        # Rebuild per-cell tracks from the labels stack here.
        tracks = _rebuild_tracks_from_labels(labels)
        win.detect_result["tracks"] = tracks
        # Cell-division lineage: fast path uses the divisions.json
        # sidecar produced by the pipeline / export (instant). When
        # absent, annotation is deferred to FocusedAnalyzeWorker —
        # `find_candidates` takes ~16 s on a 97-frame 2048² stack,
        # too slow to do synchronously on every mask drop.
        divisions_path = os.path.join(pr_dir, "divisions.json")
        if os.path.exists(divisions_path):
            try:
                with open(divisions_path) as f:
                    dj = json.load(f)
                for entry in dj.get("track_lineage", []) or []:
                    d = entry.get("track_index")
                    if d is not None and 0 <= d < len(tracks):
                        tracks[d]["parent_id"] = entry.get(
                            "parent_track_index")
                        tracks[d]["division_frame"] = entry.get(
                            "division_frame")
                        tracks[d]["division_score"] = entry.get(
                            "division_score")
                win.detect_result["divisions"] = (
                    dj.get("candidates", []) or [])
                n_div = sum(1 for t in tracks
                            if t.get("parent_id") is not None)
                win.logger.log(
                    "info",
                    f"Loaded {n_div} cell division(s) from "
                    f"divisions.json")
            except Exception as e:
                win.logger.log(
                    "warn",
                    f"Couldn't parse divisions.json: {e}")
    if fusion_src is not None:
        win.detect_result["fusion_source_stack"] = fusion_src

    overlay = labels if labels is not None else masks
    win.viewer.update_masks(overlay)
    if hasattr(win.viewer, "set_source_stack"):
        win.viewer.set_source_stack(fusion_src)

    # Load pre-filter dropped overlay (optional, only present when
    # persistence_guard ran on this recording).
    dropped_path = os.path.join(pr_dir, "masks_unfiltered.npz")
    dec_path = os.path.join(pr_dir, "filter_decisions.json")
    if (os.path.exists(dropped_path) and labels is not None
            and hasattr(win.viewer, "set_dropped_labels")):
        unf = np.load(dropped_path, allow_pickle=True)["labels"]
        # Show only cells the filter REJECTED. Prefer the explicit
        # raw_id → kept mapping from filter_decisions.json; without
        # it, fall back to "raw cells not covered by any kept label".
        if os.path.exists(dec_path):
            with open(dec_path) as f:
                dec = json.load(f)
            dropped_ids = {d["raw_id"] for d in dec.get("decisions", [])
                           if not d.get("kept")}
            mask = np.isin(unf, list(dropped_ids))
            dropped_only = np.where(mask, unf, 0).astype(np.int32)
            win.viewer.set_dropped_labels(dropped_only)
            n_drop = len(dropped_ids)
            n_keep = dec.get("n_kept", 0)
            win.logger.log(
                "info",
                f"Loaded filter decisions: {n_keep} kept, "
                f"{n_drop} dropped (toggle 'Dropped' in viewer to view)")
        else:
            # No decisions sidecar — fall back to showing all raw IDs
            # whose pixels don't already belong to a kept cell.
            kept_per_frame = (labels > 0)
            dropped_only = np.where(kept_per_frame, 0, unf).astype(np.int32)
            win.viewer.set_dropped_labels(dropped_only)
            win.logger.log(
                "info",
                "Loaded masks_unfiltered.npz (no filter_decisions.json — "
                "showing all unmatched raw cells)")

    win.viewer.nav_bar.set_status(masks)
    win.pipeline.set_stage_status("detect", "done")
    win.pipeline.enable_stage("edit", True)
    win.pipeline.enable_stage("analyze", True)
    label = os.path.basename(os.path.dirname(pr_dir)) or os.path.basename(pr_dir)
    win.status.showMessage(f"Loaded pipeline_results: {label}")
    win.logger.log("info", f"Loaded pipeline_results from {pr_dir}")
    win._dirty = False


def on_scan_cells(win):
    if win.recording is None:
        QMessageBox.information(win, "Scan", "Load a recording first.")
        return
    try:
        from core.hybrid_cpsam_multi import scan_cell_count
        QApplication.setOverrideCursor(Qt.WaitCursor)
        win.status.showMessage("Scanning for cell count...")
        QApplication.processEvents()
        count = scan_cell_count(
            win.recording["frames"], n_sample=5,
            min_area_px=win.params.min_area.value())
        win.params.expected_cells.setValue(max(1, count))
        win.logger.log("info", f"Scan: {count} cells detected")
        win.status.showMessage(f"Scan complete: {count} cells/frame")
    except Exception as e:
        QMessageBox.warning(win, "Scan Error", str(e))
    finally:
        QApplication.restoreOverrideCursor()
