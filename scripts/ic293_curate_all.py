"""Run single-cell curation across ALL IC293 recordings (non-destructive).

For every crop: writes pipeline_results/masks_curated.npz + curation_log.json
(+ curation_montage.png when frames changed/flagged), leaving masks.npz
untouched. Aggregates everything into ic293_analysis/CURATION_REPORT.md —
the method-reporting record + manual-review worklist. NEVER excludes a
recording; prior-violating ones are flagged needs_manual_review.

Uses Ignasi's priors (isolated, flattened, not dividing, present every
frame, centred); expected cell area inferred per recording unless given.
Builds one SAM2 predictor and reuses it across recordings.

Usage:
  conda run -n cellpose4 python scripts/ic293_curate_all.py
  conda run -n cellpose4 python scripts/ic293_curate_all.py --expected-area 3500
"""
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, read_sidecar_meta, ANALYSIS_ROOT, CONDITIONS)
from scripts.ic293_curate_one import _montage  # noqa: E402
from core.single_cell_curation import (  # noqa: E402
    SingleCellCurationParams, curate_single_cell)


def _report(rows, params, path):
    L = ["# IC293 single-cell curation report", "",
         f"_Generated {datetime.now().isoformat(timespec='seconds')}_  ",
         f"Priors: isolated={params.cell_isolated}, "
         f"flattened/centred={params.roughly_centered}, "
         f"no_dividing={params.no_dividing}, "
         f"present_every_frame={params.cell_present_every_frame}; "
         f"expected_area={'per-recording (inferred)' if not params.expected_cell_area_um2 else params.expected_cell_area_um2}.",
         "",
         "Non-destructive: each recording has a `masks_curated.npz` + "
         "`curation_log.json` in its `pipeline_results/`; `masks.npz` is "
         "untouched. No recording is excluded.", ""]
    n = len(rows)
    n_art = sum(1 for r in rows if r["artifacts_removed"])
    n_rec = sum(1 for r in rows if r["frames_recovered"])
    n_rep = sum(1 for r in rows if r["frames_repaired"])
    n_rev = sum(1 for r in rows if r["needs_manual_review"])
    n_clean = sum(1 for r in rows if not r["artifacts_removed"]
                  and not r["frames_recovered"] and not r["frames_repaired"]
                  and not r["needs_manual_review"])
    tot_art = sum(len(r["artifacts_removed"]) for r in rows)
    tot_rec = sum(len(r["frames_recovered"]) for r in rows)
    tot_rep = sum(len(r["frames_repaired"]) for r in rows)
    L += ["## Summary", "",
          f"- recordings: **{n}**  (clean / no action: **{n_clean}**)",
          f"- artifacts removed: {tot_art} objects across {n_art} recordings",
          f"- frames recovered (were missing): {tot_rec} across {n_rec} recordings",
          f"- frames repaired (under-segmented): {tot_rep} across {n_rep} recordings",
          f"- **needs manual review: {n_rev}**", ""]

    rev = [r for r in rows if r["needs_manual_review"]]
    if rev:
        L += ["## ⚑ Needs manual review", "",
              "| recording | reason | unresolved frames |", "|---|---|---|"]
        for r in sorted(rev, key=lambda x: x["label"]):
            reasons = "; ".join(e["type"] for e in r["exceptions"]) or "—"
            uf = ",".join(str(f["frame"]) for f in r["frames_unresolved"]) or "—"
            L.append(f"| {r['label']} | {reasons} | {uf} |")
        L.append("")

    L += ["## All recordings", "",
          "| recording | objs | artifacts | recovered | repaired | unresolved | review |",
          "|---|--:|--:|--:|--:|--:|:--:|"]
    for r in sorted(rows, key=lambda x: (x["cond"], x["label"])):
        L.append(f"| {r['label']} | {r['n_objects_initial']} | "
                 f"{len(r['artifacts_removed'])} | {len(r['frames_recovered'])} | "
                 f"{len(r['frames_repaired'])} | {len(r['frames_unresolved'])} | "
                 f"{'⚑' if r['needs_manual_review'] else ''} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-area", type=float, default=None)
    args = ap.parse_args()
    inv = inventory_cache()
    params = SingleCellCurationParams(
        enabled=True, cell_present_every_frame=True, no_dividing=True,
        cell_isolated=True, roughly_centered=True,
        expected_cell_area_um2=args.expected_area)

    from core.io import load_recording
    predictor = None
    rows = []
    labels_sorted = sorted(inv, key=lambda l: (inv[l]["condition"], l))
    for i, label in enumerate(labels_sorted, 1):
        info = inv[label]
        rec_dir = recording_dir(label, info["condition"])
        pr = os.path.join(rec_dir, "pipeline_results")
        mp = os.path.join(pr, "masks.npz")
        if not os.path.exists(mp):
            continue
        before = np.load(mp)["labels"].astype(np.int32)
        rec = load_recording(info["video_path"])
        meta = read_sidecar_meta(info)
        um = float(rec.get("um_per_px") or meta["um_per_px"])
        # build SAM2 predictor once, only when first needed
        if predictor is None:
            try:
                from core.sam2_video import _build_predictor
                predictor = _build_predictor()
            except Exception:
                predictor = None
        after, log = curate_single_cell(before, rec["frames"], um, params,
                                        predictor=predictor)
        log["label"] = label
        log["cond"] = info["condition"]
        np.savez_compressed(os.path.join(pr, "masks_curated.npz"),
                            labels=after, masks=(after > 0))
        with open(os.path.join(pr, "curation_log.json"), "w") as f:
            json.dump(log, f, indent=2, default=str)
        _montage(rec["frames"], before, after, log,
                 os.path.join(pr, "curation_montage.png"))
        rows.append(log)
        flag = " ⚑review" if log["needs_manual_review"] else ""
        print(f"[{i}/{len(labels_sorted)}] {label}: "
              f"art={len(log['artifacts_removed'])} rec={len(log['frames_recovered'])} "
              f"rep={len(log['frames_repaired'])} unres={len(log['frames_unresolved'])}"
              f"{flag}", flush=True)

    _report(rows, params, os.path.join(ANALYSIS_ROOT, "CURATION_REPORT.md"))
    nrev = sum(1 for r in rows if r["needs_manual_review"])
    print(f"\nCurated {len(rows)} recordings → masks_curated.npz + logs. "
          f"{nrev} need manual review. Report: ic293_analysis/CURATION_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
