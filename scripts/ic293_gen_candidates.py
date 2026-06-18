"""Generate per-cell outline-refinement CANDIDATES for human selection.

These are slightly-difficult cells: no single automatic rule wins on all of
them (chunk-add rescues real lobes on some, over-grafts shadows/expansions on
others). So instead of committing to one method we generate a few candidates
per cell and let the reviewer pick the best (or keep the original) in the
gui_review app. NON-DESTRUCTIVE — writes masks_cand_*.npz alongside the
curated masks; masks.npz is untouched until ic293_apply_choices runs.

Candidates (① Original = the curated masks_pre_chunkadd.npz, no file needed):
  ② moderate      chunk-add P>0.65 + locality gate (recovers real lobes,
                  over-expansion/debris rinds rejected)
  ③ conservative  chunk-add P>0.90 + locality gate (only confident additions)
  ④ sam2          SAM2 re-segmentation seeded by the curated mask — the only
                  candidate that can TIGHTEN as well as grow (for the noisy
                  over-extension cells). Bounded 0.5–2.0× + must overlap seed.

Phases (run from cellpose4):
  conda run -n cellpose4 python scripts/ic293_gen_candidates.py --phase chunkadd --jobs 8
  conda run -n cellpose4 python scripts/ic293_gen_candidates.py --phase sam2 --labels <review-set>
  conda run -n cellpose4 python scripts/ic293_gen_candidates.py --collect
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, detection_order, recording_dir, ANALYSIS_ROOT)

REVIEW_DIR = os.path.join(ANALYSIS_ROOT, "review")
ORIGINAL = "masks_pre_chunkadd.npz"          # the curated baseline (① Original)
CAND_FILES = {"moderate": "masks_cand_moderate.npz",
              "conservative": "masks_cand_conservative.npz",
              "sam2": "masks_cand_sam2.npz"}


def _to_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, np.uint8)
    return ((img - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _stack_delta(orig, cand):
    """Area change of cand vs orig over a (T,H,W) label stack. Tracks both %
    and ABSOLUTE px (the % alone is inflated on small/rounded frames)."""
    ob = (orig > 0); cb = (cand > 0)
    ab = int(ob.sum()); aa = int(cb.sum())
    maxp, maxf, nch, maxpx, nchunk = 0.0, -1, 0, 0, 0
    for t in range(orig.shape[0]):
        o = int(ob[t].sum())
        if not o:
            continue
        d = int(cb[t].sum()) - o
        if (cb[t] != ob[t]).any():
            nch += 1
        if abs(d) >= 250:                       # a real chunk-sized change
            nchunk += 1
        if abs(d) > abs(maxpx):
            maxpx = d
        p = d / o * 100.0
        if abs(p) > abs(maxp):
            maxp, maxf = p, t
    return {"agg_delta_pct": (aa - ab) / max(ab, 1) * 100.0,
            "max_frame_delta_pct": float(maxp), "max_frame": int(maxf),
            "max_frame_delta_px": int(maxpx), "n_frames_changed": int(nch),
            "n_chunk_frames": int(nchunk)}


def _gen_chunkadd(label, info):
    from core.boundary_chunk_add import refine_label_stack, ChunkAddParams
    from core.boundary_rf import load_rf_model, predict_cell_probability
    from core.io import load_recording
    rf, config = load_rf_model()
    pr = os.path.join(recording_dir(label, info["condition"]), "pipeline_results")
    curated = np.load(os.path.join(pr, ORIGINAL))["labels"].astype(np.int32)
    frames = load_recording(info["video_path"])["frames"]
    if curated.shape != frames.shape:
        return f"shape mismatch {curated.shape} vs {frames.shape}"
    # RF prob ONCE per frame, indexed by t (None where the cell is absent).
    # Must index by t — id(frames[t]) recycles across ephemeral views.
    probs = [None] * curated.shape[0]
    for t in range(curated.shape[0]):
        if (curated[t] > 0).any():
            probs[t] = predict_cell_probability(frames[t], rf, config)

    pmod = ChunkAddParams(threshold=0.65, max_contact_frac=0.35)
    pcons = ChunkAddParams(threshold=0.90, max_contact_frac=0.35)
    mod, _ = refine_label_stack(curated, frames, rf, config, params=pmod, probs=probs)
    cons, _ = refine_label_stack(curated, frames, rf, config, params=pcons, probs=probs)
    np.savez_compressed(os.path.join(pr, CAND_FILES["moderate"]), labels=mod, masks=(mod > 0))
    np.savez_compressed(os.path.join(pr, CAND_FILES["conservative"]), labels=cons, masks=(cons > 0))
    return None


def _keep_seed_cc(refined, seed):
    from scipy.ndimage import label as cc_label
    lab, n = cc_label(refined)
    if n == 0:
        return np.zeros_like(seed)
    best = max(range(1, n + 1), key=lambda c: ((lab == c) & seed).sum())
    return lab == best


def _gen_sam2(label, info, predictor):
    from core.sam_refine import refine_with_sam2
    from core.io import load_recording
    pr = os.path.join(recording_dir(label, info["condition"]), "pipeline_results")
    curated = np.load(os.path.join(pr, ORIGINAL))["labels"].astype(np.int32)
    frames = load_recording(info["video_path"])["frames"]
    out = np.zeros_like(curated)
    for t in range(curated.shape[0]):
        ids = [int(v) for v in np.unique(curated[t]) if v > 0]
        if not ids:
            continue
        img8 = _to_uint8(frames[t])
        for v in ids:
            M = curated[t] == v
            try:
                r = _keep_seed_cc(refine_with_sam2(img8, M, predictor=predictor).astype(bool), M)
            except Exception:
                r = M
            a0, a1 = int(M.sum()), int(r.sum())
            ok = a0 and a1 and 0.5 <= a1 / a0 <= 2.0 and (r & M).sum() / a0 >= 0.5
            keep = r if ok else M                # bound blowups/collapses
            out[t][keep & ~(out[t] > 0)] = v
    np.savez_compressed(os.path.join(pr, CAND_FILES["sam2"]), labels=out, masks=(out > 0))
    return None


# ─────────────────────────── orchestration ───────────────────────────
def _run_parallel(queue, args):
    import subprocess
    shards = [queue[i::args.jobs] for i in range(args.jobs)]
    procs = []
    for sh in shards:
        if not sh:
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", args.phase,
               "--labels", ",".join(sh), "--shard"]
        procs.append(subprocess.Popen(cmd))
    print(f"[parallel] {len(procs)} workers over {len(queue)} crops", flush=True)
    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


def _collect(inv, queue):
    """Build review_candidates.json: per recording, per candidate delta vs
    Original — the review app reads it to surface only the cells that differ."""
    man = {}
    for label in queue:
        info = inv.get(label)
        if not info:
            continue
        pr = os.path.join(recording_dir(label, info["condition"]), "pipeline_results")
        op = os.path.join(pr, ORIGINAL)
        if not os.path.exists(op):
            continue
        orig = np.load(op)["labels"]
        ent = {"condition": info["condition"], "candidates": {}}
        for name, fn in CAND_FILES.items():
            fp = os.path.join(pr, fn)
            if os.path.exists(fp):
                ent["candidates"][name] = _stack_delta(orig, np.load(fp)["labels"])
        cs = list(ent["candidates"].values())
        # review priority = biggest absolute single-frame change across cands
        ent["max_abs_px"] = max((abs(c.get("max_frame_delta_px", 0)) for c in cs),
                                default=0)
        ent["max_abs_change"] = max((abs(c["max_frame_delta_pct"]) for c in cs),
                                    default=0.0)
        # a cell is worth reviewing if any candidate grafts a real chunk on
        # >=2 frames OR shifts overall area >=2% (px-based, not %-inflated)
        ent["review"] = any(c.get("n_chunk_frames", 0) >= 2
                            or abs(c.get("agg_delta_pct", 0)) >= 2.0 for c in cs)
        man[label] = ent
    os.makedirs(REVIEW_DIR, exist_ok=True)
    with open(os.path.join(REVIEW_DIR, "review_candidates.json"), "w") as f:
        json.dump(man, f, indent=2)
    n_rev = sum(1 for e in man.values() if e.get("review"))
    print(f"=== collect: {len(man)} recordings, {n_rev} in the review set ===")
    print("  wrote review/review_candidates.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["chunkadd", "sam2"], default="chunkadd")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    inv = inventory_cache()
    queue = detection_order(inv)
    if args.labels:
        want = set(s for s in args.labels.split(",") if s)
        queue = [l for l in queue if l in want]

    if args.collect:
        return _collect(inv, queue)
    if args.jobs > 1 and not args.shard:
        rc = _run_parallel(queue, args)
        _collect(inv, queue)
        return rc

    predictor = None
    if args.phase == "sam2":
        from core.sam_refine import load_sam2
        predictor = load_sam2("hiera_t")
    for i, label in enumerate(queue, 1):
        info = inv[label]
        try:
            err = (_gen_chunkadd(label, info) if args.phase == "chunkadd"
                   else _gen_sam2(label, info, predictor))
        except Exception as e:
            err = repr(e)
        print(f"[{i}/{len(queue)}] {args.phase} {label}: {err or 'ok'}", flush=True)
    if not args.shard:
        _collect(inv, queue)
    return 0


if __name__ == "__main__":
    sys.exit(main())
