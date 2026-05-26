"""Run the full pipeline on the DMSO_busy multichannel recording.

End-to-end:
  1. Load DIC + Cy5 channels via core.io
  2. Run detect_hybrid_dic_multi with Cy5 fusion ON
  3. Apply Cy5 multi-metric filter (false-positive rejection)
  4. Annotate tracks with cy5 + fusion-source
  5. Analyse each tracked cell (morphology + edge dynamics + VAMPIRE
     + cell-state classification)
  6. Write masks.npz, metrics.json, per-cell metrics, project file,
     fusion_diagnostic.png, overlay.tif, run_log.md AND
     RUN_METADATA.{md,json} to the results dir

After this runs the user can open test.cellscope in the GUI and see
everything restored: masks, tracks, source toggle, analysis metrics.

All pipeline defaults come from core.pipeline_defaults — single
source of truth shared with the GUI. Don't pass parameter overrides
unless you have a reason; they get logged into RUN_METADATA as
deviations.

Run from the cellpose env (default) — cpsam(Cy5) will be delegated
to cellpose4 via subprocess.
"""
import os
import sys
import time
import json
import logging
import numpy as np
import tifffile

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

REC_DIR = os.path.join(
    CELLSCOPE_ROOT,
    "data/examples/multichannel_DIC_Cy5_DMSO_busy")
TIFF = os.path.join(REC_DIR, "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
JSON = os.path.join(REC_DIR, "multichannel_DIC_Cy5_DMSO_busy.ome.json")
OUT_DIR = os.path.join(REC_DIR, "results")
PROJECT_FILE = os.path.join(REC_DIR, "test.cellscope")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)s · %(message)s")
log = logging.getLogger("busy_demo")


def load_channels():
    """Return DIC, Cy5 stacks (both N,H,W uint8) + metadata."""
    with open(JSON) as f:
        meta = json.load(f)
    with tifffile.TiffFile(TIFF) as tf:
        n = len(tf.pages) // 2
        h, w = tf.pages[0].shape
        cy5 = np.empty((n, h, w), dtype=np.uint8)
        dic = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            cy5[i] = tf.pages[2 * i].asarray()      # ch0 per metadata
            dic[i] = tf.pages[2 * i + 1].asarray()  # ch1 per metadata
    return dic, cy5, meta


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    log.info("Loading %s", TIFF)
    dic, cy5, meta = load_channels()
    n = len(dic)
    log.info("Loaded %d frames; DIC %s, Cy5 %s", n, dic.shape, cy5.shape)

    recording = {
        "frames": dic,
        "cy5_frames": cy5,
        "name": meta.get("name", "busy_demo"),
        "video_path": TIFF,
        "um_per_px": meta.get("um_per_px", 1.0),
        "time_interval_min": meta.get("time_interval_min", 1.0),
        "n_channels": meta.get("n_channels", 2),
    }

    # ---- 1. Detection + fusion + multi-metric filter ----
    # Use canonical pipeline defaults — model auto-resolves to
    # cpsam_dic (best available) and min_area_px = 500. Do NOT
    # hardcode overrides here; that's what caused the May-2026
    # GUI-vs-script mismatch.
    from core.hybrid_dic import detect_hybrid_dic_multi
    from core.pipeline_defaults import resolve_dic_model_path
    model_path = resolve_dic_model_path()
    log.info("Running detect_hybrid_dic_multi with Cy5 fusion ON "
             "(model = %s)…", model_path)
    t0 = time.time()
    detect = detect_hybrid_dic_multi(
        dic, progress_fn=lambda m, p: log.info("  detect[%d%%]: %s",
                                                p, m),
        # All other args left at defaults from pipeline_defaults
        cy5_frames=cy5,
        use_cy5_fusion=True,
    )
    log.info("Detection done in %.0fs (tracks=%d, "
             "cy5_fusion_added=%d)",
             time.time() - t0,
             len(detect.get("tracks", [])),
             detect.get("n_cy5_fusion_added", 0))

    # ---- 2. Annotate tracks with Cy5 + apply multi-metric filter ----
    from core.multichannel import (
        per_cell_cy5_features, cy5_presence_score,
        cy5_inside_outside_ratio, cy5_fraction_positive)
    log.info("Annotating tracks with Cy5 metrics …")
    tracks = detect["tracks"]
    for t in tracks:
        stack = t.get("stack")
        if stack is None:
            continue
        mean = np.full(n, np.nan, dtype=np.float32)
        score = np.full(n, np.nan, dtype=np.float32)
        ior = np.full(n, np.nan, dtype=np.float32)
        fp = np.full(n, np.nan, dtype=np.float32)
        for i in range(n):
            m = stack[i].astype(bool)
            if not m.any():
                continue
            mean[i] = per_cell_cy5_features(m, cy5[i])["mean"]
            score[i] = cy5_presence_score(m, cy5[i])
            ior[i] = cy5_inside_outside_ratio(m, cy5[i])
            fp[i] = cy5_fraction_positive(m, cy5[i])
        t["cy5_mean"] = mean
        t["cy5_score"] = score
        t["cy5_io_ratio"] = ior
        t["cy5_fraction_positive"] = fp
        valid = ~np.isnan(score)
        t["cy5_mean_score"] = (float(np.nanmean(score))
                                if valid.any() else 0.0)

    from core.cy5_filter import apply_cy5_filter, rebuild_label_stack
    log.info("Applying Cy5 multi-metric filter …")
    kept, dropped, info = apply_cy5_filter(tracks, mode="multi_metric")
    log.info("Cy5 filter kept %d / %d (dropped %d)",
             len(kept), len(tracks), len(dropped))
    detect["tracks_raw"] = tracks
    detect["tracks"] = kept
    detect["tracks_dropped"] = dropped
    detect["labels"] = rebuild_label_stack(kept, dic.shape)
    detect["masks"] = detect["labels"] > 0

    # ---- 3. Per-cell analysis (analytics + VAMPIRE + state) ----
    from core.pipeline import analyze_recording
    from core.vampire_analysis import run_vampire_analysis
    from core.cell_state import (
        classify_track_states, state_fraction,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)
    from core.tracking import extract_centroids

    um_per_px = float(recording["um_per_px"])
    dt_min = float(recording["time_interval_min"])

    log.info("Analysing %d tracked cells …", len(kept))
    per_cell = []
    for tid, track in enumerate(kept):
        cell_masks = track["stack"]
        result = analyze_recording(recording, cell_masks)
        # VAMPIRE
        try:
            vamp = run_vampire_analysis(cell_masks, n_clusters=5)
            if vamp:
                result["vampire"] = vamp
        except Exception as e:
            log.warning("VAMPIRE failed for cell %d: %s", tid + 1, e)
        # State classification + per-state motility
        try:
            sd = classify_track_states(cell_masks.astype(bool))
            states = sd["states"]
            cents = extract_centroids(cell_masks.astype(bool))
            result["state_per_frame"] = states
            result["state_frac_balled"] = state_fraction(
                states, STATE_BALLED)
            result["state_frac_attached"] = state_fraction(
                states, STATE_ATTACHED)
            result["state_frac_transitional"] = state_fraction(
                states, STATE_TRANSITIONAL)
            for state, prefix in ((STATE_BALLED, "balled"),
                                    (STATE_ATTACHED, "attached")):
                sp = state_speeds(cents, states, state, um_per_px,
                                   dt_min)
                msd = state_msd(cents, states, state, um_per_px,
                                 max_lag=20)
                per = state_persistence(cents, states, state,
                                         um_per_px)
                td = state_total_displacement(cents, states, state,
                                                um_per_px)
                result[f"{prefix}_n_speed_samples"] = int(len(sp))
                result[f"{prefix}_mean_speed_um_per_min"] = (
                    float(np.mean(sp)) if len(sp) else float("nan"))
                result[f"{prefix}_median_speed_um_per_min"] = (
                    float(np.median(sp)) if len(sp) else float("nan"))
                result[f"{prefix}_persistence_lag1"] = per["lag1"]
                result[f"{prefix}_msd_lag5_um2"] = (
                    float(msd["msd"][4])
                    if len(msd["msd"]) > 4
                    and not np.isnan(msd["msd"][4])
                    else float("nan"))
                result[f"{prefix}_total_displacement_um"] = (
                    td["total_displacement_um"])
                result[f"{prefix}_straightness"] = td["straightness"]
        except Exception as e:
            log.warning("State classification failed for cell %d: %s",
                        tid + 1, e)
        result["cell_id"] = tid + 1
        result["track_info"] = {
            "first_frame": track["first_frame"],
            "frames_tracked": int(cell_masks.any(axis=(1, 2)).sum()),
            "parent_id": track.get("parent_id"),
        }
        result["fusion_source"] = track.get("fusion_source",
                                              "dic_only")
        per_cell.append(result)

    # ---- 4. Save masks.npz ----
    from gui_focused.export_dialog import _to_jsonable
    log.info("Writing results to %s", OUT_DIR)
    save_dict = {
        "masks": detect["masks"],
        "labels": detect["labels"],
    }
    if detect.get("fusion_source_stack") is not None:
        save_dict["fusion_source_stack"] = detect[
            "fusion_source_stack"].astype(np.uint8)
    np.savez_compressed(os.path.join(OUT_DIR, "masks.npz"),
                         **save_dict)

    # Pre-Cy5-filter snapshot (no-op for single-channel runs).
    try:
        from output.results import save_unfiltered_detections
        unf = save_unfiltered_detections(OUT_DIR, detect)
        if unf is not None:
            n_raw, n_kept, n_dropped = unf
            log.info("Wrote masks_unfiltered.npz + "
                     "filter_decisions.json (%d raw, %d kept, "
                     "%d dropped)", n_raw, n_kept, n_dropped)
    except Exception as e:
        log.warning("Unfiltered save failed: %s", e)

    # ---- 5. metrics.json + per-cell metrics ----
    def _build_metrics(r):
        out = {}
        for key in ("name", "n_frames", "um_per_px", "time_interval_min",
                    "mean_speed", "total_distance", "net_displacement",
                    "persistence", "mean_boundary_confidence",
                    "cell_id", "fusion_source"):
            if key in r:
                out[key] = _to_jsonable(r[key])
        for key in ("shape_summary", "edge_summary", "area_stability",
                    "track_info", "vampire",
                    "state_frac_balled", "state_frac_attached",
                    "state_frac_transitional"):
            if key in r:
                out[key] = _to_jsonable(r[key])
        for key in r:
            if (key.startswith("balled_") or key.startswith("attached_")):
                out[key] = _to_jsonable(r[key])
        return out

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump([_build_metrics(r) for r in per_cell], f, indent=2)
    for r in per_cell:
        cid = r["cell_id"]
        with open(os.path.join(OUT_DIR, f"metrics_cell{cid}.json"),
                   "w") as f:
            json.dump(_build_metrics(r), f, indent=2)

    # ---- 6. fusion diagnostic figure ----
    if detect.get("fusion_source_stack") is not None:
        from core.fusion_diagnostics import render_fusion_diagnostic
        render_fusion_diagnostic(
            dic, cy5,
            detect["fusion_source_stack"],
            detect["labels"],
            os.path.join(OUT_DIR, "fusion_diagnostic.png"),
            n_sample_frames=6,
            tracks=kept)
        log.info("Wrote fusion_diagnostic.png")

    # ---- 7. project file (re-loadable in the GUI) ----
    from core.project import save_project
    from core.pipeline_defaults import DEFAULTS
    # Record the params we actually used. Where we deferred to a
    # default, we record the default value too so RUN_METADATA can
    # report "all defaults used" instead of "unknown".
    params_used = {
        **DEFAULTS.as_dict(),     # canonical defaults
        "use_cy5_fusion": True,
        "cy5_filter_mode": "persistence_guard",
        "model_path": model_path,
    }
    save_project(
        PROJECT_FILE, recording, detect, per_cell, params_used,
        mode="multi")
    log.info("Wrote project file: %s", PROJECT_FILE)

    # ---- 7b. RUN_METADATA — every analysis path must write this ----
    from core.run_metadata import write_run_metadata
    write_run_metadata(
        OUT_DIR, recording, params_used, detect,
        analysis_results=per_cell,
        pipeline_function="hybrid_dic_multi",
        mode="multi",
        runtime_seconds=time.time() - t_total,
        rerun_command=(
            "cd /Users/george/claude_test/cellscope && "
            "conda run -n cellpose python "
            "scripts/run_busy_pipeline_full.py"),
        extra={
            "model_path": model_path,
            "n_cy5_fusion_added": detect.get("n_cy5_fusion_added"),
            "n_tracks_raw": len(detect.get("tracks_raw", [])),
            "cy5_filter_result": {
                "kept": len(kept), "dropped": len(dropped)},
        },
        project_root=CELLSCOPE_ROOT)

    # ---- 8. Quick text summary ----
    src_counts = {"dic_only": 0, "both": 0, "cy5_only": 0}
    for t in kept:
        s = t.get("fusion_source", "dic_only")
        src_counts[s] = src_counts.get(s, 0) + 1
    summary = f"""DMSO_busy full pipeline run
  Detection:         {len(detect['tracks_raw'])} tracks from cellpose_dic ∪ cpsam(Cy5)
  Cy5 fusion added:  {detect.get('n_cy5_fusion_added', 0)} cells (pre-tracking)
  Cy5 filter kept:   {len(kept)} / {len(detect['tracks_raw'])} tracks
  Track sources:     {src_counts['dic_only']} dic_only,
                     {src_counts['both']} both,
                     {src_counts['cy5_only']} cy5_only
  Files written:     {os.listdir(OUT_DIR)}
  Project file:      {PROJECT_FILE}
  Total runtime:     {time.time() - t_total:.0f}s
"""
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)
    with open(os.path.join(OUT_DIR, "run_summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
