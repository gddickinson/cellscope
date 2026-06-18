"""Crop individual, non-dividing single cells out of the multi-cell IC295
recordings into a DERIVED single-cell dataset, mirroring the IC293
single-cell-crop on-disk format so the SAME analysis battery runs on it.

Why: IC295's finding was "phenotype is in shape/state, not migration; state +
crowding dominate motility". This dataset isolates individual non-dividing
cells (removing the crowding/contact confound and division events) and asks
whether the result holds for clean single cells — the single-cell analogue of
IC293. These are endothelial cells (IC295 = "EC migration"). The rounded/spread
state rule was fitted on IC295's OWN hand labels, so state metrics here are on
the same footing as the main IC295 population analysis (where state is a primary
readout) — unlike IC293, a SEPARATE endothelial experiment with no hand labels
of its own, where that transferred rule is flagged exploratory.

What it does (READ-ONLY of ic295_analysis):
  * for every IC295 recording with `pipeline_results/masks.npz`, take each
    tracked cell that is (a) NOT involved in a division (parent or daughter in
    divisions.json), (b) present for >= --min-frames contiguous frames, and
    (c) not heavily edge-truncated;
  * crop the DIC video + that one cell's mask to the cell's trajectory bbox +
    a margin, over the cell's present-frame span (so the crop is "one cell,
    present every frame" — the IC293 prior);
  * write the crop in the IC293 layout under ic295_single-cell-crop_analysis/:
      _cache/<label>.ome.tif (uint8 DIC, TYX) + <label>.ome.json sidecar
      by_condition/<cond>/<label>/  (symlink + pipeline_results/masks.npz +
                                     <label>.cellscope + RUN_METADATA)
    where <label> = `<SrcPos>_cell<ID>-<COND>` (IC293 label grammar, so the
    condition/position parsers and the whole battery work unchanged).

NEVER writes into ic295_analysis. The masks are REUSED from the validated +
reviewed IC295 detection (not re-detected): we already have the tracked cell.

Run from cellpose4:
  conda run -n cellpose4 python scripts/ic295sc_crop_cells.py            # all
  conda run -n cellpose4 python scripts/ic295sc_crop_cells.py --dry-run  # plan
  conda run -n cellpose4 python scripts/ic295sc_crop_cells.py --label Pos7-WT
"""
import os
import sys
import csv
import json
import glob
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_ROOT = os.path.join(PROJECT_ROOT, "ic295_single-cell-crop_analysis")
# Retarget ic293_common's path constants at its import so the reused
# folder-setup helpers (setup_recording_folder / write_cellscope_project) write
# into the DERIVED dataset, never into ic293_analysis.
os.environ["IC293_ANALYSIS_ROOT"] = NEW_ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from scripts.ic293_common import (  # noqa: E402  (now rooted at NEW_ROOT)
    CONDITIONS, CACHE_DIR, ANALYSIS_ROOT, recording_dir,
    setup_recording_folder, write_cellscope_project, atomic_write_json)
from core.edge_filter import mask_touches_edge  # noqa: E402

IC295_BYC = os.path.join(PROJECT_ROOT, "ic295_analysis", "by_condition")

# Defaults. min_frames is the floor "no shorter than the shortest track in
# IC293" (the IC293 analysis set is all >= 30 frames; Pos36_div-GOF, the one
# 13-frame dividing crop, was excluded). Overridable via --min-frames.
DEF_MIN_FRAMES = 30
DEF_MARGIN_PX = 60
DEF_MIN_CONTIG = 0.90      # present / span — guards a genuinely-tracked cell
DEF_MAX_EDGE_FRAC = 0.50   # skip cells truncated by the FoV border most frames
DEF_MAX_CELLS = 0          # 0 = no cap; else keep the N longest-tracked per rec


# ───────────────────────── source enumeration ─────────────────────────
def source_recordings(only_label=None):
    out = []
    for cond in CONDITIONS:
        cdir = os.path.join(IC295_BYC, cond)
        if not os.path.isdir(cdir):
            continue
        for lab in sorted(os.listdir(cdir)):
            if only_label and lab != only_label:
                continue
            rdir = os.path.join(cdir, lab)
            mp = os.path.join(rdir, "pipeline_results", "masks.npz")
            if not os.path.exists(mp):
                continue
            tifs = sorted(glob.glob(os.path.join(rdir, "*.ome.tif")))
            tifs = [t for t in tifs if os.path.exists(os.path.realpath(t))]
            if not tifs:
                continue
            out.append({"label": lab, "cond": cond, "rdir": rdir,
                        "masks": mp, "tif": tifs[0]})
    return out


def dividing_labels(rdir):
    """Cell label IDs to EXCLUDE because they divide. label ID = track_index+1
    (verified against the IC295 lineage); a daughter row contributes both its
    own and its parent's label."""
    dj = os.path.join(rdir, "pipeline_results", "divisions.json")
    out = set()
    if os.path.exists(dj):
        try:
            dd = json.load(open(dj))
            for r in (dd.get("track_lineage") or []):
                p = r.get("parent_track_index")
                if p is not None:
                    out.add(int(r["track_index"]) + 1)
                    out.add(int(p) + 1)
        except Exception:
            pass
    return out


def qualifying_cells(labels, div, min_frames, min_contig, max_edge_frac,
                     max_cells):
    T, H, W = labels.shape
    rows = []
    for cid in [int(v) for v in np.unique(labels) if v > 0]:
        if cid in div:
            continue
        m = labels == cid
        present = m.any(axis=(1, 2))
        fr = np.where(present)[0]
        if len(fr) < min_frames:
            continue
        f0, f1 = int(fr[0]), int(fr[-1])
        span = f1 - f0 + 1
        contig = len(fr) / span
        if contig < min_contig:
            continue
        edge = sum(1 for t in fr if mask_touches_edge(m[t]))
        edge_frac = edge / len(fr)
        if edge_frac > max_edge_frac:
            continue
        ys, xs = np.where(m[f0:f1 + 1].any(axis=0))
        rows.append({
            "cid": cid, "f0": f0, "f1": f1, "present": int(len(fr)),
            "span": span, "contig": round(contig, 3),
            "edge_frac": round(edge_frac, 3),
            "bbox": (int(ys.min()), int(ys.max()) + 1,
                     int(xs.min()), int(xs.max()) + 1)})
    rows.sort(key=lambda r: -r["present"])          # longest-tracked first
    if max_cells and len(rows) > max_cells:
        rows = rows[:max_cells]
    return rows


# ───────────────────────── crop writers ─────────────────────────
def write_ome_tif_u8(dst, arr, um_per_px, dt_min):
    """Single-channel DIC crop, (T,H,W) uint8, axes=TYX. Mirrors
    ic293_stage_crops.write_ome_tif (ome=True → SizeC=1 so detect_channels
    reports 1 and load_recording reads it single-channel)."""
    tmp = dst + ".tmp"
    tifffile.imwrite(
        tmp, arr, ome=True, photometric="minisblack",
        metadata={"axes": "TYX",
                  "PhysicalSizeX": float(um_per_px), "PhysicalSizeXUnit": "µm",
                  "PhysicalSizeY": float(um_per_px), "PhysicalSizeYUnit": "µm",
                  "TimeIncrement": float(dt_min) * 60.0,
                  "TimeIncrementUnit": "s"})
    os.replace(tmp, dst)


def write_sidecar(path, label, um, dt, n_frames):
    atomic_write_json(path, {
        "name": label, "um_per_px": float(um), "time_interval_min": float(dt),
        "n_channels": 1, "dic_channel": 0, "n_frames": int(n_frames),
        "source": "derived single-cell crop from IC295"})


def _provenance(pr, src, cell, label, um, dt, n_frames, bbox, dic_crop):
    """RUN_METADATA for the crop (detect-phase analog, rule 2)."""
    try:
        from core.run_metadata import write_run_metadata
        rec_for_meta = {"video_path": os.path.join(
                            CACHE_DIR, f"{label}.ome.tif"),
                        "name": label, "frames": dic_crop, "cy5_frames": None,
                        "um_per_px": um, "time_interval_min": dt}
        write_run_metadata(
            pr, recording=rec_for_meta,
            params={"min_frames_present": cell["present"],
                    "crop_bbox_yxhw": list(bbox),
                    "frame_span": [cell["f0"], cell["f1"]],
                    "contig": cell["contig"], "edge_frac": cell["edge_frac"]},
            detect_result={"tracks": [{"stack": None}]},
            pipeline_function="ic295sc_crop_cells (REUSED IC295 masks; NOT re-detected)",
            mode="single",
            rerun_command=("conda run -n cellpose4 python "
                           f"scripts/ic295sc_crop_cells.py --label {src['label']}"),
            extra={"dataset": "IC295_single_cell_crop",
                   "source_recording": src["label"],
                   "source_condition": src["cond"],
                   "source_cell_id": cell["cid"], "n_frames": n_frames,
                   "cell_type": "endothelial (IC295 EC-migration)",
                   "state_rule": "IC295-fit rounded/spread rule on IC295's own cells "
                                 "(primary, as in the IC295 population analysis)",
                   "note": "masks reused from validated/reviewed IC295 detection"},
            project_root=PROJECT_ROOT)
    except Exception as e:
        print(f"    WARN: RUN_METADATA failed for {label}: {e}", file=sys.stderr)


def write_crop(src, dic, labels, cell, um, dt, margin, dry):
    src_pos = src["label"].rsplit("-", 1)[0]            # Pos7-WT -> Pos7
    label = f"{src_pos}_cell{cell['cid']}-{src['cond']}"
    f0, f1 = cell["f0"], cell["f1"]
    r0, r1, c0, c1 = cell["bbox"]
    H, W = labels.shape[1], labels.shape[2]
    r0, c0 = max(0, r0 - margin), max(0, c0 - margin)
    r1, c1 = min(H, r1 + margin), min(W, c1 + margin)
    n_frames = f1 - f0 + 1
    rec = {"label": label, "cond": src["cond"], "source": src["label"],
           "cell_id": cell["cid"], "present": cell["present"],
           "span": cell["span"], "contig": cell["contig"],
           "edge_frac": cell["edge_frac"], "n_frames": int(n_frames),
           "shape": [int(n_frames), int(r1 - r0), int(c1 - c0)],
           "bbox": [int(r0), int(r1), int(c0), int(c1)]}
    if dry:
        return rec
    dic_crop = np.ascontiguousarray(dic[f0:f1 + 1, r0:r1, c0:c1])
    lab_crop = (labels[f0:f1 + 1, r0:r1, c0:c1] == cell["cid"]).astype(np.int32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_tif = os.path.join(CACHE_DIR, f"{label}.ome.tif")
    write_ome_tif_u8(cache_tif, dic_crop, um, dt)
    sidecar = cache_tif.replace(".ome.tif", ".ome.json")
    write_sidecar(sidecar, label, um, dt, n_frames)
    info = {"label": label, "condition": src["cond"], "position": src_pos,
            "video_path": cache_tif, "json_sidecar": sidecar,
            "metadata_txt": cache_tif.replace(".ome.tif", "_metadata.txt")}
    rec_dir = setup_recording_folder(info)
    pr = os.path.join(rec_dir, "pipeline_results")
    np.savez_compressed(os.path.join(pr, "masks.npz"),
                        labels=lab_crop, masks=(lab_crop > 0))
    write_cellscope_project(rec_dir, info, {
        "um_per_px": um, "time_interval_min": dt, "n_frames": n_frames})
    _provenance(pr, src, cell, label, um, dt, n_frames,
                (r0, r1, c0, c1), dic_crop)
    return rec


# ───────────────────────── driver ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-frames", type=int, default=DEF_MIN_FRAMES)
    ap.add_argument("--margin-px", type=int, default=DEF_MARGIN_PX)
    ap.add_argument("--min-contig", type=float, default=DEF_MIN_CONTIG)
    ap.add_argument("--max-edge-frac", type=float, default=DEF_MAX_EDGE_FRAC)
    ap.add_argument("--max-cells-per-rec", type=int, default=DEF_MAX_CELLS)
    ap.add_argument("--label", default=None, help="only this IC295 recording")
    ap.add_argument("--limit", type=int, default=0, help="first N recordings")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from core.io import load_recording
    srcs = source_recordings(args.label)
    if args.limit:
        srcs = srcs[:args.limit]
    print(f"{'[dry-run] ' if args.dry_run else ''}IC295 source recordings: "
          f"{len(srcs)}  ->  {NEW_ROOT}")
    print(f"  min_frames>={args.min_frames}  margin={args.margin_px}px  "
          f"contig>={args.min_contig}  edge_frac<={args.max_edge_frac}  "
          f"max_cells/rec={args.max_cells_per_rec or 'all'}")

    manifest, per_cond = [], {c: 0 for c in CONDITIONS}
    t0 = time.time()
    for i, src in enumerate(srcs, 1):
        labels = np.load(src["masks"])["labels"].astype(np.int32)
        div = dividing_labels(src["rdir"])
        cells = qualifying_cells(labels, div, args.min_frames,
                                 args.min_contig, args.max_edge_frac,
                                 args.max_cells_per_rec)
        dic = None
        if cells and not args.dry_run:
            rec = load_recording(os.path.realpath(src["tif"]),
                                 dic_channel=1, fluo_channel=0)
            dic = rec["frames"]
            um = float(rec.get("um_per_px") or 0.6523)
            dt = float(rec.get("time_interval_min") or 10.0)
            if dic.shape[0] != labels.shape[0]:
                print(f"  ! {src['label']}: frame count mismatch "
                      f"dic={dic.shape[0]} labels={labels.shape[0]} — skip",
                      file=sys.stderr)
                continue
        else:
            um, dt = 0.6523, 10.0
        for cell in cells:
            r = write_crop(src, dic, labels, cell, um, dt,
                           args.margin_px, args.dry_run)
            manifest.append(r)
            per_cond[src["cond"]] += 1
        print(f"  [{i}/{len(srcs)}] {src['label']}: {len(cells)} cell(s) "
              f"(divs excluded: {sorted(div) or 'none'})")

    print(f"\n{'[dry-run] ' if args.dry_run else ''}TOTAL crops: "
          f"{len(manifest)}   per-condition: {per_cond}")
    print(f"  recordings contributing: "
          f"{len({m['source'] for m in manifest})}/{len(srcs)}")
    if not args.dry_run and manifest:
        os.makedirs(ANALYSIS_ROOT, exist_ok=True)
        mpath = os.path.join(ANALYSIS_ROOT, "crops_manifest.csv")
        cols = ["label", "cond", "source", "cell_id", "present", "span",
                "contig", "edge_frac", "n_frames", "shape", "bbox"]
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for m in manifest:
                w.writerow({k: m.get(k) for k in cols})
        print(f"  wrote {mpath}")
        print(f"  next: conda run -n cellpose4 python scripts/ic295sc_run.py")
    print(f"  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
