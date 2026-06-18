"""Train the boundary-refinement RandomForest on the clean IC293 masks.

Confident-band sampling: cell = eroded interior (definitely cell),
background = dilated-complement near band + far (definitely background),
the boundary band left UNLABELED. So the RF learns "cell texture vs
background", not the (slightly clipped) mask edge — and its probability
isoline can therefore extend PAST the training masks' edges into genuine
cell-textured protrusions. Reuses core.boundary_rf's feature bank; saves
data/models/rf_boundary_model.pkl + filter_config.json (both gitignored).

Trains on clean (unflagged, non-excluded) recordings; holds out every 3rd
clean recording for ic293_rf_validate.py.

Usage: conda run -n cellpose4 python scripts/ic293_train_boundary_rf.py
"""
import os
import sys
import csv
import json
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

import numpy as np  # noqa: E402
from scipy.ndimage import binary_erosion, binary_dilation  # noqa: E402
from core.io import load_recording  # noqa: E402
from core.boundary_rf import (  # noqa: E402
    compute_filter_bank, compute_one_filter, MODEL_PATH, CONFIG_PATH)
from scripts.ic293_common import (  # noqa: E402
    inventory_cache, recording_dir, ANALYSIS_EXCLUDE, ANALYSIS_ROOT)

FILTER_CONFIG = {"filters": [
    {"name": "intensity"},
    {"name": "gradient", "sigmas": [1, 2, 4]},
    {"name": "local_std", "sigmas": [1, 2, 4, 8]},
    {"name": "dog", "sigmas": [2, 4]},
    {"name": "gabor", "freqs": [0.15, 0.3], "thetas": [0.0, 1.5708]},
    {"name": "lbp", "radii": [1, 2]},
    {"name": "frangi", "sigmas": [1, 2, 4]},
    {"name": "gmm_prob", "n_components": 3},
]}


def _flagged_labels():
    p = os.path.join(ANALYSIS_ROOT, "review", "review_flags.csv")
    out = set()
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                if str(r.get("flagged", "")).lower() in ("1", "true", "yes"):
                    out.add(r["label"])
    return out


def _feature_names(sample_img):
    names = []
    for fd in FILTER_CONFIG["filters"]:
        for _arr, nm in compute_one_filter(sample_img, fd):
            names.append(nm)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=18)
    ap.add_argument("--frames-per", type=int, default=3)
    ap.add_argument("--px-per-class", type=int, default=1500)
    ap.add_argument("--holdout-out", default="/tmp/rf_holdout.json")
    args = ap.parse_args()

    inv = inventory_cache()
    flagged = _flagged_labels()
    clean = sorted(l for l in inv if l not in flagged and l not in ANALYSIS_EXCLUDE)
    train_labels = [l for i, l in enumerate(clean) if i % 3 != 0][:args.n_recordings]
    holdout = [l for i, l in enumerate(clean) if i % 3 == 0]
    print(f"clean={len(clean)}  train={len(train_labels)}  holdout={len(holdout)}",
          flush=True)

    X, y = [], []
    names = None
    for li, label in enumerate(train_labels):
        info = inv[label]
        frames = load_recording(info["video_path"])["frames"]
        labels = np.load(os.path.join(recording_dir(label, info["condition"]),
                                      "pipeline_results", "masks.npz"))["labels"]
        present = np.where((labels > 0).any(axis=(1, 2)))[0]
        if len(present) == 0:
            continue
        sel = present[np.linspace(0, len(present) - 1,
                                  min(args.frames_per, len(present))).round().astype(int)]
        for f in sel:
            M = labels[f] > 0
            if M.sum() < 200:
                continue
            cell = binary_erosion(M, iterations=3)
            d4, d10 = binary_dilation(M, iterations=4), binary_dilation(M, iterations=10)
            near_bg, far_bg = (d10 & ~d4), ~d10
            bank = compute_filter_bank(frames[f], FILTER_CONFIG)
            if names is None:
                names = _feature_names(frames[f])
            flat = bank.reshape(-1, bank.shape[-1])
            rng = np.random.default_rng(int(f))

            def samp(mask, n):
                idx = np.flatnonzero(mask.ravel())
                if len(idx) == 0:
                    return np.empty((0, flat.shape[1]), np.float32)
                return flat[rng.choice(idx, size=min(n, len(idx)), replace=False)]
            Xc = samp(cell, args.px_per_class)
            Xn = samp(near_bg, int(args.px_per_class * 0.6))
            Xf = samp(far_bg, int(args.px_per_class * 0.4))
            X += [Xc, Xn, Xf]
            y += [np.ones(len(Xc)), np.zeros(len(Xn)), np.zeros(len(Xf))]
        print(f"  [{li+1}/{len(train_labels)}] {label}", flush=True)

    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y)
    print(f"pixels: {len(y)} ({int(y.sum())} cell / {int((y==0).sum())} bg), "
          f"feats={X.shape[1]}", flush=True)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=8, n_jobs=-1,
        class_weight="balanced", oob_score=True, bootstrap=True, random_state=0)
    rf.fit(X, y)
    print(f"OOB score: {rf.oob_score_:.3f}", flush=True)
    if names:
        imp = sorted(zip(names, rf.feature_importances_), key=lambda t: -t[1])
        print("top features: " + ", ".join(f"{n}={i:.3f}" for n, i in imp[:8]),
              flush=True)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as fo:
        pickle.dump({"model": rf, "config": FILTER_CONFIG}, fo)
    with open(CONFIG_PATH, "w") as fo:
        json.dump(FILTER_CONFIG, fo, indent=2)
    json.dump(holdout, open(args.holdout_out, "w"))
    print(f"saved RF -> {MODEL_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
