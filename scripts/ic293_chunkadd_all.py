"""Chunk-add outline refinement across the IC293 single-cell crops.

Applies the validated chunk-add method (core.boundary_chunk_add) to recover
large solid missing cell lobes the detector under-segmented, WITHOUT
disturbing already-correct boundaries (clean cells change +0%; low-SNR crops
+0–1% — the method is self-limiting). RF P>0.65 by default.

Per recording (NON-DESTRUCTIVE of the pre-refinement state):
  * the first run backs up the current masks.npz -> masks_pre_chunkadd.npz;
  * chunk-add is always applied to that backup (so re-runs are idempotent and
    never compound), writing masks_chunkadd.npz AND overwriting masks.npz so
    the focused-GUI review + ic293_analyze_one see the refined masks;
  * pipeline_results/chunkadd_log.json records params + per-frame stats.

Top level: ic293_analysis/review/chunkadd_summary.csv (per-recording area
delta) + chunkadd_changed.png (montage of the most-changed cells).

Usage (run from cellpose4 so the RF + skimage stack are present):
  conda run -n cellpose4 python scripts/ic293_chunkadd_all.py            # all
  conda run -n cellpose4 python scripts/ic293_chunkadd_all.py --label Pos0-WT
  conda run -n cellpose4 python scripts/ic293_chunkadd_all.py --limit 3
  conda run -n cellpose4 python scripts/ic293_chunkadd_all.py --revert    # restore
"""
import os
import sys
import csv
import json
import time
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, detection_order, recording_dir, ANALYSIS_ROOT)

REVIEW_DIR = os.path.join(ANALYSIS_ROOT, "review")
BACKUP_NAME = "masks_pre_chunkadd.npz"


def _flagged_frames():
    """label -> reviewer-flagged frame (for montage panel selection)."""
    out = {}
    p = os.path.join(REVIEW_DIR, "review_flags.csv")
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if str(r.get("flagged", "")).lower() in ("1", "true", "yes"):
                try:
                    out[r["label"]] = int(r.get("frame") or 0)
                except (TypeError, ValueError):
                    out[r["label"]] = 0
    return out


def _revert(inv, queue):
    n = 0
    for label in queue:
        pr = os.path.join(recording_dir(label, inv[label]["condition"]),
                          "pipeline_results")
        bk = os.path.join(pr, BACKUP_NAME)
        if os.path.exists(bk):
            shutil.copy2(bk, os.path.join(pr, "masks.npz"))
            n += 1
    print(f"reverted {n} recordings: masks.npz <- {BACKUP_NAME}")
    return 0


def _montage(panels, out_path, ncols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    n = len(panels)
    if not n:
        return
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.7 * nrows),
                            squeeze=False)
    for i, (label, f, img, before, after, pct) in enumerate(panels):
        ax = axs[i // ncols][i % ncols]
        ys, xs = np.where(before | after)
        if not len(ys):                          # empty frame — show whole crop
            ys, xs = np.array([0, img.shape[0] - 1]), np.array([0, img.shape[1] - 1])
        sl = (slice(max(0, ys.min() - 16), ys.max() + 16),
              slice(max(0, xs.min() - 16), xs.max() + 16))
        ax.imshow(img[sl], cmap="gray")
        for ct in measure.find_contours(before[sl].astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="red", lw=1.0)
        for ct in measure.find_contours(after[sl].astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="lime", lw=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label} F{f}  +{pct:.0f}%", fontsize=8)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle("Chunk-add (RF P>0.65): red = before, green = after — each "
                 "cell at its most-grown frame (+% = that frame's growth)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _write_summary(rows, panels):
    """Authoritative summary CSV + most-changed montage (post-run / collect).

    rows: (label, condition, agg_delta_pct, max_frame_delta_pct,
           n_frames_changed, n_frames)."""
    os.makedirs(REVIEW_DIR, exist_ok=True)
    with open(os.path.join(REVIEW_DIR, "chunkadd_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "condition", "agg_delta_pct",
                    "max_frame_delta_pct", "n_frames_changed", "n_frames"])
        for r in sorted(rows, key=lambda x: -x[3]):
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", r[4], r[5]])
    changed = sorted([p for p in panels if p[5] > 0.5], key=lambda x: -x[5])[:24]
    _montage(changed, os.path.join(REVIEW_DIR, "chunkadd_changed.png"))
    maxp = [r[3] for r in rows]
    if maxp:
        n_changed = sum(1 for p in maxp if p > 0.5)
        print(f"  {n_changed}/{len(rows)} cells gained a lobe on >=1 frame  "
              f"(max-frame growth: median +{float(np.median([p for p in maxp if p>0.5] or [0])):.1f}%, "
              f"peak +{max(maxp):.1f}%, {sum(1 for p in maxp if p > 20)} cells >+20%)")
    print("  wrote review/chunkadd_summary.csv + review/chunkadd_changed.png")


def _run_parallel(queue, args):
    """Fan the queue across N worker processes (round-robin split keeps each
    shard condition-balanced), then collect one authoritative summary. RF is
    pure-CPU so workers scale near-linearly on a multi-core host."""
    import subprocess
    shards = [queue[i::args.jobs] for i in range(args.jobs)]
    procs = []
    for sh in shards:
        if not sh:
            continue
        cmd = [sys.executable, os.path.abspath(__file__),
               "--labels", ",".join(sh), "--shard",
               "--threshold", str(args.threshold)]
        if args.no_overwrite:
            cmd.append("--no-overwrite")
        procs.append(subprocess.Popen(cmd))
    print(f"[parallel] launched {len(procs)} workers over {len(queue)} crops",
          flush=True)
    rc = 0
    for p in procs:
        rc |= p.wait()
    inv = inventory_cache()
    _collect(inv, queue, _flagged_frames())
    print("  next: conda run -n cellpose4 python scripts/ic293_batch.py "
          "--phase analyze --force")
    return rc


def _collect(inv, queue, flagged):
    """Rebuild summary + montage from per-recording chunkadd_log.json (after
    parallel shards). Reloads before/after masks only for the changed cells."""
    from core.io import load_recording
    rows, panels = [], []
    for label in queue:
        info = inv.get(label)
        if not info:
            continue
        pr = os.path.join(recording_dir(label, info["condition"]),
                          "pipeline_results")
        logp = os.path.join(pr, "chunkadd_log.json")
        if not os.path.exists(logp):
            continue
        with open(logp) as fh:
            log = json.load(fh)
        agg = float(log["area_delta_pct"])
        maxp = float(log.get("max_frame_delta_pct", 0.0))
        rows.append((label, info["condition"], agg, maxp,
                     log["n_frames_changed"], log["n_frames"]))
        ca = os.path.join(pr, "masks_chunkadd.npz")
        if maxp > 0.5 and os.path.exists(ca) and os.path.exists(
                os.path.join(pr, BACKUP_NAME)):
            before = np.load(os.path.join(pr, BACKUP_NAME))["labels"]
            after = np.load(ca)["labels"]
            frames = load_recording(info["video_path"])["frames"]
            f = int(log.get("max_frame", -1))
            if f < 0 or f >= len(frames) or not (after[f] > 0).any():
                present = np.where((after > 0).any(axis=(1, 2)))[0]
                f = int(present[0]) if len(present) else 0
            panels.append((label, f, frames[f], before[f] > 0,
                           after[f] > 0, maxp))
    print(f"=== collect: {len(rows)} recordings ===")
    _write_summary(rows, panels)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--labels", default=None,
                    help="comma-separated subset (for parallel shards)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--revert", action="store_true",
                    help="restore masks.npz from masks_pre_chunkadd.npz")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="write masks_chunkadd.npz but DON'T touch masks.npz")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (RF is pure-CPU; ~N× speedup)")
    ap.add_argument("--shard", action="store_true",
                    help="suppress shared summary/montage writes (parallel run)")
    ap.add_argument("--collect-only", action="store_true",
                    help="rebuild summary CSV + montage from chunkadd_log.json")
    args = ap.parse_args()

    inv = inventory_cache()
    if not inv:
        sys.exit("No IC293 crops found — run scripts/ic293_stage_crops.py first.")
    queue = detection_order(inv)
    if args.label:
        queue = [args.label] if args.label in inv else []
        if not queue:
            sys.exit(f"{args.label} not in IC293 cache")
    if args.labels:
        want = [s for s in args.labels.split(",") if s]
        queue = [l for l in queue if l in want]
    if args.limit:
        queue = queue[:args.limit]

    if args.revert:
        return _revert(inv, queue)
    if args.collect_only:
        return _collect(inv, queue, _flagged_frames())
    if args.jobs > 1 and not args.shard:
        return _run_parallel(queue, args)

    from core.boundary_chunk_add import refine_label_stack, ChunkAddParams
    from core.boundary_rf import load_rf_model
    from core.io import load_recording
    rf, config = load_rf_model()
    if rf is None:
        sys.exit("RF model not found (data/models/rf_boundary_model.pkl) — "
                 "train with scripts/ic293_train_boundary_rf.py")
    params = ChunkAddParams(threshold=args.threshold)
    flagged = _flagged_frames()

    rows, panels = [], []
    t_start = time.time()
    for i, label in enumerate(queue, 1):
        info = inv[label]
        pr = os.path.join(recording_dir(label, info["condition"]),
                          "pipeline_results")
        masks_path = os.path.join(pr, "masks.npz")
        if not os.path.exists(masks_path):
            print(f"[{i}/{len(queue)}] {label}: no masks.npz — skip", flush=True)
            continue
        # idempotent backup: always refine FROM the pre-chunkadd state
        backup = os.path.join(pr, BACKUP_NAME)
        if not os.path.exists(backup):
            shutil.copy2(masks_path, backup)
        labels = np.load(backup)["labels"].astype(np.int32)
        frames = load_recording(info["video_path"])["frames"]
        if labels.shape != frames.shape:
            print(f"[{i}/{len(queue)}] {label}: shape mismatch "
                  f"labels{labels.shape} vs frames{frames.shape} — skip",
                  flush=True)
            continue

        t0 = time.time()
        new_labels, stats = refine_label_stack(labels, frames, rf, config, params)
        dt = time.time() - t0

        np.savez_compressed(os.path.join(pr, "masks_chunkadd.npz"),
                            labels=new_labels, masks=(new_labels > 0))
        if not args.no_overwrite:
            np.savez_compressed(masks_path,
                                labels=new_labels, masks=(new_labels > 0))
        with open(os.path.join(pr, "chunkadd_log.json"), "w") as fh:
            json.dump({"label": label, "condition": info["condition"],
                       "method": "chunk_add", "source": BACKUP_NAME,
                       "runtime_seconds": dt, **stats}, fh, indent=2)

        rows.append((label, info["condition"], stats["area_delta_pct"],
                     stats["max_frame_delta_pct"], stats["n_frames_changed"],
                     stats["n_frames"]))
        # montage panel: the most-grown frame (max per-frame %), else the
        # reviewer-flagged frame, else the first frame the cell is present.
        present = np.where((new_labels > 0).any(axis=(1, 2)))[0]
        f = stats["max_frame"]
        if f < 0 or f >= len(frames) or not (new_labels[f] > 0).any():
            f = flagged.get(label, int(present[0]) if len(present) else 0)
        if f >= len(frames) or not (new_labels[f] > 0).any():
            f = int(present[0]) if len(present) else 0
        panels.append((label, f, frames[f], labels[f] > 0,
                       new_labels[f] > 0, stats["max_frame_delta_pct"]))
        print(f"[{i}/{len(queue)}] {label:18s} {info['condition']:5s} "
              f"agg +{stats['area_delta_pct']:4.1f}%  "
              f"max-frame +{stats['max_frame_delta_pct']:5.1f}%  "
              f"{stats['n_frames_changed']:>2}/{stats['n_frames']} frames  "
              f"({dt:.0f}s)", flush=True)

    print(f"\n=== shard done: {len(rows)} recordings in "
          f"{time.time()-t_start:.0f}s ===", flush=True)
    if not args.shard:
        _write_summary(rows, panels)
        print(f"  next: conda run -n cellpose4 python scripts/ic293_batch.py "
              f"--phase analyze --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
