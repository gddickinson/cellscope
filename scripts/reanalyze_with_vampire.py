"""Quick re-analyse the saved DMSO_busy results now that vampire is
installed in the cellpose env. Skips detection — uses masks.npz from
the previous run."""
import os
import sys
import time
import json
import logging
import numpy as np
import tifffile

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

REC_DIR = os.path.join(
    CELLSCOPE_ROOT,
    "data/examples/multichannel_DIC_Cy5_DMSO_busy")
TIFF = os.path.join(REC_DIR, "multichannel_DIC_Cy5_DMSO_busy.ome.tif")
JSON = os.path.join(REC_DIR, "multichannel_DIC_Cy5_DMSO_busy.ome.json")
OUT_DIR = os.path.join(REC_DIR, "results")
PROJECT_FILE = os.path.join(REC_DIR, "test.cellscope")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("reanalyze")


def load_channels():
    with open(JSON) as f:
        meta = json.load(f)
    with tifffile.TiffFile(TIFF) as tf:
        n = len(tf.pages) // 2
        h, w = tf.pages[0].shape
        cy5 = np.empty((n, h, w), dtype=np.uint8)
        dic = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            cy5[i] = tf.pages[2 * i].asarray()
            dic[i] = tf.pages[2 * i + 1].asarray()
    return dic, cy5, meta


def tracks_from_labels(labels, source_stack=None):
    """Same logic as core.project._tracks_from_labels but annotates
    each track with its fusion_source when source_stack is given."""
    if labels is None:
        return []
    out = []
    cell_ids = sorted(int(c) for c in np.unique(labels) if c != 0)
    for cid in cell_ids:
        stack = (labels == cid)
        per_frame = stack.any(axis=(1, 2))
        if not per_frame.any():
            continue
        first = int(np.argmax(per_frame))
        t = {"stack": stack, "first_frame": first, "parent_id": None}
        if source_stack is not None:
            n_dic, n_cy5, n_both = 0, 0, 0
            for fi in range(len(stack)):
                m = stack[fi]
                if not m.any():
                    continue
                src = source_stack[fi][m]
                n_dic += int((src == 1).sum())
                n_cy5 += int((src == 2).sum())
                n_both += int((src == 3).sum())
            if n_both >= max(n_dic, n_cy5):
                t["fusion_source"] = "both"
            elif n_cy5 > n_dic:
                t["fusion_source"] = "cy5_only"
            else:
                t["fusion_source"] = "dic_only"
        out.append(t)
    return out


def main():
    t0 = time.time()
    log.info("Loading recording + saved masks …")
    dic, cy5, meta = load_channels()
    data = np.load(os.path.join(OUT_DIR, "masks.npz"))
    labels = data["labels"]
    masks_bool = data["masks"]
    source_stack = (data["fusion_source_stack"]
                    if "fusion_source_stack" in data.files else None)
    log.info("  loaded labels %s, source_stack=%s",
             labels.shape, source_stack is not None)

    tracks = tracks_from_labels(labels, source_stack)
    log.info("Reconstructed %d tracks", len(tracks))

    recording = {
        "frames": dic,
        "cy5_frames": cy5,
        "name": meta.get("name", "busy_demo"),
        "video_path": TIFF,
        "um_per_px": meta.get("um_per_px", 1.0),
        "time_interval_min": meta.get("time_interval_min", 1.0),
        "n_channels": meta.get("n_channels", 2),
    }
    um_per_px = float(recording["um_per_px"])
    dt_min = float(recording["time_interval_min"])

    # Verify vampire is importable
    import vampire
    log.info("vampire OK: %s", vampire.__file__)

    from core.pipeline import analyze_recording
    from core.vampire_analysis import run_vampire_analysis
    from core.cell_state import (
        classify_track_states, state_fraction,
        STATE_BALLED, STATE_ATTACHED, STATE_TRANSITIONAL)
    from core.motility_state import (
        state_speeds, state_msd, state_persistence,
        state_total_displacement)
    from core.tracking import extract_centroids

    per_cell = []
    for tid, track in enumerate(tracks):
        log.info("  cell %d/%d …", tid + 1, len(tracks))
        cell_masks = track["stack"]
        result = analyze_recording(recording, cell_masks)
        try:
            vamp = run_vampire_analysis(cell_masks, n_clusters=5)
            if vamp:
                result["vampire"] = vamp
                log.info("    VAMPIRE OK: %d contours, H=%.2f",
                         vamp["n_contours"],
                         vamp["heterogeneity"]["entropy"])
        except Exception as e:
            log.warning("    VAMPIRE failed: %s", e)
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
            log.warning("    State classification failed: %s", e)
        result["cell_id"] = tid + 1
        result["track_info"] = {
            "first_frame": track["first_frame"],
            "frames_tracked": int(cell_masks.any(axis=(1, 2)).sum()),
            "parent_id": None,
        }
        result["fusion_source"] = track.get("fusion_source",
                                              "dic_only")
        per_cell.append(result)

    # Write metrics.json + per-cell metrics
    from gui_focused.export_dialog import _to_jsonable

    def _build_metrics(r):
        out = {}
        for k in ("name", "n_frames", "um_per_px", "time_interval_min",
                  "mean_speed", "total_distance", "net_displacement",
                  "persistence", "mean_boundary_confidence",
                  "cell_id", "fusion_source"):
            if k in r:
                out[k] = _to_jsonable(r[k])
        for k in ("shape_summary", "edge_summary", "area_stability",
                  "track_info", "vampire",
                  "state_frac_balled", "state_frac_attached",
                  "state_frac_transitional"):
            if k in r:
                out[k] = _to_jsonable(r[k])
        for k in r:
            if k.startswith("balled_") or k.startswith("attached_"):
                out[k] = _to_jsonable(r[k])
        return out

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump([_build_metrics(r) for r in per_cell], f, indent=2)
    for r in per_cell:
        with open(os.path.join(OUT_DIR,
                                f"metrics_cell{r['cell_id']}.json"),
                   "w") as f:
            json.dump(_build_metrics(r), f, indent=2)

    # Rewrite project file (carries the same masks, but with new
    # analysis scalars)
    from core.project import save_project
    detect_result = {
        "masks": masks_bool,
        "labels": labels,
        "fusion_source_stack": source_stack,
        "tracks": tracks,
    }
    params = {
        "mode": "multi",
        "use_cy5_fusion": True,
        "cy5_filter_mode": "persistence_guard",
        "min_area_px": 200,
    }
    save_project(PROJECT_FILE, recording, detect_result, per_cell,
                 params, mode="multi")
    log.info("Wrote project file: %s", PROJECT_FILE)

    n_vamp = sum(1 for r in per_cell if r.get("vampire"))
    print(f"\nDone in {time.time()-t0:.0f}s — VAMPIRE OK on "
          f"{n_vamp}/{len(per_cell)} cells")


if __name__ == "__main__":
    main()
