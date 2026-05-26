"""Run the canonical pipeline on a GT-labelled recording so its
predictions can be compared to manual ground truth.

Usage:
  conda run -n cellpose4 python scripts/run_pipeline_on_gt_recording.py \\
      data/ic295_gt_full/Pos7_WT

Run from the **cellpose4** env. The auto-selector
(`select_pipeline_for_recording`) may pick either:
  - hybrid_dic_multi (uses cpsam_dic, CP4 ViT — loads natively)
  - hybrid_cpsam_multi (uses raw cpsam, CP4 ViT — loads natively)

Both need cellpose 4.x. Running from cellpose env (3.x) errors out
the moment a CP4 model loads. Falls through to cellpose env via
subprocess for any CP3 fallback steps.

Saves into the recording's folder:
  pipeline_results/
    masks.npz              labels + fusion source stack
    fusion_diagnostic.png
    RUN_METADATA.{md,json}
    run_summary.txt

This script is a THIN WRAPPER around
`core.unified_detection.detect_recording()` — the same function the
focused GUI calls. By construction, running this script and running
the GUI on the same input produce identical labels.
"""
import os
import sys
import time
import json
import logging
import argparse
import numpy as np

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("run_pipeline_gt")


def find_recording(folder):
    """Return (.ome.tif path, .json path) inside the folder."""
    for f in os.listdir(folder):
        if f.endswith(".ome.tif") or f.endswith(".tif"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No .ome.tif in {folder}")


def load_channels(tif_path):
    """Load (dic, cy5) uint8 stacks. Returns cy5=None for
    single-channel recordings.

    For 2-channel TIFFs we assume the IC295 layout (DIC=ch1, Cy5=ch0)
    — that's the only multichannel arrangement we ship at the moment.
    """
    from core.io import detect_channels, load_recording
    n_ch = detect_channels(tif_path)
    if n_ch >= 2:
        rec = load_recording(tif_path, dic_channel=1, fluo_channel=0)
        return rec["frames"], rec.get("cy5_frames")
    from core.io import load_video
    return load_video(tif_path), None


def load_metadata(folder, tif_path):
    """Find the .ome.json sidecar produced by setup_ic295_gt_labeling."""
    for f in os.listdir(folder):
        if f.endswith(".ome.json"):
            with open(os.path.join(folder, f)) as fp:
                return json.load(fp)
    return {"um_per_px": 1.0, "time_interval_min": 1.0,
            "name": os.path.basename(tif_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Recording folder (e.g. "
                   "data/ic295_gt_full/Pos7_WT)")
    p.add_argument("--downsample", default="auto",
                   help="Downsample factor before detection. "
                   "'auto' (default) picks 1/2 based on image size; "
                   "'off' or '1' disables; pass an integer (e.g. 2, "
                   "3) for an explicit factor. Labels are upscaled "
                   "back to original resolution at the end.")
    p.add_argument("--downsample-small-px", type=int, default=None,
                   help="Auto-downsample: max(H,W) below this stays "
                   "at factor 1 (default 900).")
    p.add_argument("--downsample-large-px", type=int, default=None,
                   help="Auto-downsample: max(H,W) above this gets "
                   "factor 2 (default 1500).")
    p.add_argument("--no-alignment", action="store_true",
                   help="Skip automatic DIC↔Cy5 alignment "
                   "correction. Default is to measure + apply.")
    p.add_argument("--no-save-unfiltered", action="store_true",
                   help="Skip writing masks_unfiltered.npz + "
                   "filter_decisions.json. By default, when the Cy5 "
                   "filter (persistence_guard) runs we save the "
                   "pre-filter label stack and a per-track decision "
                   "JSON so users can inspect which cells were "
                   "dropped and why.")
    args = p.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)

    tif = find_recording(folder)
    out_dir = os.path.join(folder, "pipeline_results")
    os.makedirs(out_dir, exist_ok=True)

    t_total = time.time()
    log.info("Loading %s …", tif)
    dic, cy5 = load_channels(tif)
    n = len(dic)
    has_cy5 = cy5 is not None
    log.info("  %d frames, DIC %s, Cy5 %s",
             n, dic.shape, cy5.shape if has_cy5 else "(none)")

    meta = load_metadata(folder, tif)
    recording = {
        "frames": dic,
        "cy5_frames": cy5,
        "name": meta.get("name", os.path.basename(folder)),
        "video_path": tif,
        "um_per_px": meta.get("um_per_px", 1.0),
        "time_interval_min": meta.get("time_interval_min", 1.0),
        "n_channels": 2 if has_cy5 else 1,
    }

    # ===== The whole detection pipeline runs through ONE function =====
    # detect_recording() handles alignment, downsample, threshold
    # conversion, auto-select, detection, Cy5 filter, and upscale.
    # The GUI worker calls the same function — guaranteeing the runner
    # and GUI produce identical labels for the same inputs.
    # Optional per-recording use_mirror_pad override from sidecar
    # JSON. Default "auto" works for most recordings; specific ones
    # (Pos39_OT: padding merges its dividing cell pair) can set
    # `"use_mirror_pad": "off"` in <recording>.ome.json to skip
    # padding for that recording only.
    use_mirror_pad_override = meta.get("use_mirror_pad")
    if use_mirror_pad_override is not None:
        log.info("Per-recording use_mirror_pad override: %s",
                 use_mirror_pad_override)

    from core.unified_detection import detect_recording
    detect = detect_recording(
        dic, cy5_frames=cy5,
        um_per_px=recording["um_per_px"],
        time_interval_min=recording["time_interval_min"],
        downsample=args.downsample,
        downsample_small_px=args.downsample_small_px,
        downsample_large_px=args.downsample_large_px,
        align_channels=not args.no_alignment,
        pipeline_kind="auto",
        run_cy5_filter=True,
        cy5_filter_mode="persistence_guard",
        cy5_filter_threshold=0.15,
        use_mirror_pad=use_mirror_pad_override,
        progress_fn=lambda m, p: log.info("  detect[%d%%]: %s", p, m))

    auto = detect["auto"]
    pipeline_kind = auto["pipeline_kind"]
    model_path = auto.get("model_path")
    downsample = auto["downsample_factor"]
    ds_reason = auto["downsample_reason"]
    alignment_info = auto.get("alignment")
    sel_info = {"reason": auto.get("pipeline_reason"),
                "cell_counts_at_sample":
                    auto.get("pipeline_sample_counts")}
    log.info("Detection done in %.0fs (tracks=%d, "
             "cy5_fusion_added=%d, downsample=%d×, pipeline=%s)",
             time.time() - t_total, len(detect.get("tracks", [])),
             detect.get("n_cy5_fusion_added", 0),
             downsample, pipeline_kind)

    # Fusion diagnostic (multichannel only). Render at full
    # resolution using the original DIC + aligned cy5 + final labels.
    if has_cy5 and detect.get("fusion_source_stack") is not None:
        from core.fusion_diagnostics import render_fusion_diagnostic
        cy5_for_diag = detect.get("aligned_cy5_orig_scale", cy5)
        render_fusion_diagnostic(
            dic, cy5_for_diag, detect["fusion_source_stack"],
            detect["labels"],
            os.path.join(out_dir, "fusion_diagnostic.png"),
            n_sample_frames=6, tracks=detect.get("tracks"))

    # Save labels + source (always full res after detect_recording)
    save_dict = {
        "masks": detect["masks"],
        "labels": detect["labels"],
    }
    if detect.get("fusion_source_stack") is not None:
        save_dict["fusion_source_stack"] = detect[
            "fusion_source_stack"].astype(np.uint8)
    np.savez_compressed(os.path.join(out_dir, "masks.npz"), **save_dict)
    log.info("Wrote %s/masks.npz", out_dir)

    # Pre-Cy5-filter snapshot — only written when persistence_guard
    # ran AND populated tracks_raw (single-channel recordings skip
    # both). Lets the GUI surface which cells were dropped.
    if not args.no_save_unfiltered:
        from output.results import save_unfiltered_detections
        counts = save_unfiltered_detections(out_dir, detect)
        if counts is not None:
            n_raw, n_kept, n_dropped = counts
            log.info(
                "Wrote %s/masks_unfiltered.npz + "
                "filter_decisions.json (%d raw → %d kept, "
                "%d dropped)", out_dir, n_raw, n_kept, n_dropped)

    # Division-lineage sidecar — sets parent_id on tracks +
    # exports divisions.json listing detected events
    divisions = detect.get("divisions", []) or []
    if divisions:
        log.info("Detected %d cell division(s)", len(divisions))
    div_payload = {
        "recording": recording["name"],
        "n_candidates": len(divisions),
        "candidates": divisions,
        "track_lineage": [
            {"track_index": i,
             "parent_track_index": t.get("parent_id"),
             "division_frame": t.get("division_frame"),
             "division_score": t.get("division_score")}
            for i, t in enumerate(detect.get("tracks", []) or [])
            if t.get("parent_id") is not None],
    }
    import json as _json
    with open(os.path.join(out_dir, "divisions.json"), "w") as jf:
        _json.dump(div_payload, jf, indent=2)
    log.info("Wrote %s/divisions.json (%d division(s), %d lineage "
             "link(s))",
             out_dir, len(divisions),
             len(div_payload["track_lineage"]))

    # RUN_METADATA
    from core.pipeline_defaults import DEFAULTS
    from core.run_metadata import write_run_metadata
    params_used = {**DEFAULTS.as_dict(),
                   "use_cy5_fusion": has_cy5,
                   "cy5_filter_mode": ("persistence_guard" if has_cy5
                                        else None),
                   "model_path": model_path,
                   "auto_selected_pipeline": pipeline_kind,
                   "downsample": downsample,
                   "downsample_reason": ds_reason,
                   "downsample_spec": str(args.downsample)}
    pipeline_fn = ("hybrid_dic_multi" if pipeline_kind == "cpsam_dic"
                   else "hybrid_cpsam_multi")
    extra = {"model_path": model_path,
             "n_tracks": len(detect.get("tracks", [])),
             "auto_select": sel_info,
             "channel_alignment": alignment_info,
             "px_thresholds": detect.get("px_thresholds"),
             "unified_detection_path": True}
    if has_cy5:
        extra.update({
            "n_cy5_fusion_added": detect.get("n_cy5_fusion_added"),
            "n_tracks_raw": len(detect.get("tracks_raw", [])),
            "cy5_filter_kept": len(detect.get("tracks", [])),
        })
    write_run_metadata(
        out_dir, recording, params_used, detect,
        analysis_results=None,
        pipeline_function=pipeline_fn,
        mode="multi",
        runtime_seconds=time.time() - t_total,
        rerun_command=(f"conda run -n cellpose4 python "
                       f"scripts/run_pipeline_on_gt_recording.py "
                       f"{args.folder}"),
        extra=extra,
        project_root=CELLSCOPE_ROOT)

    cells_per_frame = [int(detect["labels"][i].max())
                        for i in range(n)]
    if has_cy5:
        n_raw = len(detect.get("tracks_raw", []))
        summary = (
            f"Pipeline run on {os.path.basename(folder)}\n\n"
            f"Detection:        {n_raw} raw tracks\n"
            f"Cy5 fusion added: {detect.get('n_cy5_fusion_added', 0)}"
            f" cells\n"
            f"Cy5 filter kept:  {len(detect['tracks'])} / {n_raw}\n"
            f"Cells per frame:  min={min(cells_per_frame)}"
            f" max={max(cells_per_frame)}"
            f" mean={np.mean(cells_per_frame):.1f}\n"
            f"Downsample:       {downsample}× ({ds_reason})\n"
            f"Pipeline:         {pipeline_kind}\n"
            f"Runtime:          {time.time() - t_total:.0f}s\n"
            f"Output dir:       {out_dir}\n")
    else:
        summary = (
            f"Pipeline run on {os.path.basename(folder)} "
            f"(single-channel)\n"
            f"  tracks:           {len(detect['tracks'])}\n"
            f"  cells/frame:      min={min(cells_per_frame)}"
            f" max={max(cells_per_frame)}"
            f" mean={np.mean(cells_per_frame):.1f}\n"
            f"  downsample:       {downsample}× ({ds_reason})\n"
            f"  pipeline:         {pipeline_kind}\n"
            f"  runtime:          {time.time() - t_total:.0f}s\n"
            f"  output:           {out_dir}\n")
    print("\n" + "=" * 60 + "\n" + summary + "=" * 60)
    with open(os.path.join(out_dir, "run_summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
