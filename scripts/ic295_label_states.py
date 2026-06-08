"""Label a few cell-frames → learn a rounded/spread classifier.

No single shape feature is bimodal (see ic295_state_features.py), so a
hand-picked threshold is always arbitrary. Instead we let you label a
small, MORPHOLOGICALLY DIVERSE sample of cell-frames (you can recognise
the string-of-balls / crumpled / sphere / spread forms by eye), then fit
a small interpretable classifier on the shape features — a validated
boundary that encodes your judgment and tells us which features matter.

Subcommands:
  sample [--n 80]   pick a diverse sample, render DIC crops + a numbered
                    montage + labels.csv (with an empty `label` column).
  label             interactive labeller (one crop at a time; keys
                    r=rounded s=spread u=unsure/skip). Falls back to "edit
                    the CSV" if no interactive display.
  train             fit logistic-regression + decision-tree on the
                    labelled rows; report cross-val accuracy + which
                    features matter + a suggested rule.

Everything lives under ic295_analysis/state_labels/.
"""
import os
import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from scripts.ic295_common import (  # noqa: E402
    RECORDINGS_ROOT, COMPARE_DIR, CONDITIONS, parse_condition)
from scripts.ic295_state_features import _frame_feats, FEATURES  # noqa: E402
import numpy as np  # noqa: E402

LABELS_DIR = os.path.join(os.path.dirname(COMPARE_DIR), "state_labels")
CROPS_DIR = os.path.join(LABELS_DIR, "crops")
CSV_PATH = os.path.join(LABELS_DIR, "labels.csv")
MARGIN = 18


# ---------- feature collection with identity ----------
def _collect():
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    min_area = TH["min_area_px"]
    rows = []
    paths = sorted(glob.glob(os.path.join(
        RECORDINGS_ROOT, "*", "*", "pipeline_results", "masks.npz")))
    for i, mp in enumerate(paths):
        label = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        cond = os.path.basename(os.path.dirname(os.path.dirname(
            os.path.dirname(mp))))
        if cond not in CONDITIONS:
            cond = parse_condition(label)
        try:
            labels = np.load(mp)["labels"]
        except Exception:
            continue
        for cid in [int(v) for v in np.unique(labels) if v > 0]:
            stack = labels == cid
            present = np.where(stack.any(axis=(1, 2)))[0]
            ff = {}
            for fi in present:
                d = _frame_feats(stack[fi], min_area)
                if d is not None:
                    ff[int(fi)] = d
            if not ff:
                continue
            base = float(np.percentile(
                [ff[fi]["area"] for fi in ff], 90)) or 1.0
            for fi, d in ff.items():
                rows.append({"label_dir": label, "condition": cond,
                             "mp": mp, "cid": cid, "frame": fi,
                             "rel_area": d["area"] / base, **d})
        print(f"  [{i+1}/{len(paths)}] {label}", flush=True)
    return rows


def _feature_matrix(rows):
    cols = ["rel_area"] + [f for f in FEATURES if f != "rel_area"]
    X = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    return X, cols


# ---------- sample ----------
def cmd_sample(args):
    rows = _collect()
    if not rows:
        print("no cell-frames found."); return 1
    X, cols = _feature_matrix(rows)
    ok = np.all(np.isfinite(X), axis=1)
    rows = [r for r, o in zip(rows, ok) if o]
    X = X[ok]
    n = min(args.n, len(rows))
    # diverse pick: k-means in standardised feature space, frame nearest
    # each centroid (falls back to rel_area quantiles if sklearn absent).
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n, n_init=4, random_state=0).fit(Xs)
        idx = []
        for c in range(n):
            members = np.where(km.labels_ == c)[0]
            if len(members) == 0:
                continue
            d = np.linalg.norm(Xs[members] - km.cluster_centers_[c], axis=1)
            idx.append(int(members[np.argmin(d)]))
    except Exception:
        order = np.argsort(X[:, 0])  # rel_area
        idx = [int(order[int(k)]) for k in
               np.linspace(0, len(order) - 1, n)]
    picks = [rows[i] for i in idx]
    os.makedirs(CROPS_DIR, exist_ok=True)
    _render_crops(picks)
    _write_csv(picks)
    print(f"\nSampled {len(picks)} diverse cell-frames.")
    print(f"  montage : {LABELS_DIR}/montage.png")
    print(f"  crops   : {CROPS_DIR}/NNN.png")
    print(f"  labels  : {CSV_PATH}  (fill the `label` column: r/s/u)")
    print("Then:  ic295_label_states.py label   (or edit the CSV)  → train")
    return 0


def _render_crops(picks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    from core.io import load_recording
    rec_cache = {}
    thumbs = []
    for k, r in enumerate(picks):
        mp = r["mp"]
        if mp not in rec_cache:
            tif = next((os.path.join(os.path.dirname(os.path.dirname(mp)), f)
                        for f in os.listdir(os.path.dirname(
                            os.path.dirname(mp))) if f.endswith(".ome.tif")),
                       None)
            frames = (load_recording(tif, dic_channel=1, fluo_channel=0)
                      ["frames"] if tif else None)
            rec_cache[mp] = (frames, np.load(mp)["labels"])
        frames, labels = rec_cache[mp]
        m = labels[r["frame"]] == r["cid"]
        rr = np.any(m, axis=1); cc = np.any(m, axis=0)
        r0, r1 = np.where(rr)[0][[0, -1]]; c0, c1 = np.where(cc)[0][[0, -1]]
        r0, c0 = max(0, r0 - MARGIN), max(0, c0 - MARGIN)
        r1, c1 = r1 + MARGIN, c1 + MARGIN
        img = (frames[r["frame"]][r0:r1, c0:c1].astype(float)
               if frames is not None else m[r0:r1, c0:c1].astype(float))
        mc = m[r0:r1, c0:c1]
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        if frames is not None:
            lo, hi = np.percentile(img, [2, 98])
            ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        else:
            ax.imshow(img, cmap="gray")
        for ct in measure.find_contours(mc.astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="#ff3030", lw=1.2)
        ax.set_title(f"{k}", fontsize=9); ax.axis("off")
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(CROPS_DIR, f"{k:03d}.png"), dpi=90)
        plt.close(fig)
        thumbs.append(os.path.join(CROPS_DIR, f"{k:03d}.png"))
    _montage(thumbs)


def _montage(thumbs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n = len(thumbs); ncol = 10; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.5))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < n:
            ax.imshow(mpimg.imread(thumbs[i]));
        ax.axis("off")
    fig.suptitle("Label each by index in labels.csv  (r=rounded s=spread "
                 "u=unsure)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LABELS_DIR, "montage.png"), dpi=120)
    plt.close(fig)


def _write_csv(picks):
    cols = (["idx", "label", "condition", "label_dir", "cid", "frame",
             "rel_area"] + [f for f in FEATURES if f != "rel_area"])
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k, r in enumerate(picks):
            row = {"idx": k, "label": "", **r}
            w.writerow(row)


# ---------- label (interactive) ----------
def cmd_label(args):
    if not os.path.exists(CSV_PATH):
        print("Run `sample` first."); return 1
    rows = list(csv.DictReader(open(CSV_PATH)))
    fieldnames = list(rows[0].keys())
    try:
        import matplotlib
        if sys.platform == "darwin":
            matplotlib.use("MacOSX")
        import matplotlib as mpl
        # Disable matplotlib's built-in key shortcuts (e.g. 's' = save
        # figure → the save dialog you hit) so our r/s/u keys just advance.
        for _k in list(mpl.rcParams):
            if _k.startswith("keymap."):
                mpl.rcParams[_k] = []
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except Exception:
        print("No interactive display — open montage.png and fill the "
              "`label` column (r/s/u) in labels.csv, then `train`.")
        return 0
    state = {"i": 0}
    todo = [r for r in rows if not r["label"]]
    if not todo:
        print("all rows already labelled."); return 0

    def _persist():
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(4, 4))

    def show():
        ax.clear()
        r = todo[state["i"]]
        p = os.path.join(CROPS_DIR, f"{int(r['idx']):03d}.png")
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_title(f"{state['i']+1}/{len(todo)}  rel_area="
                     f"{float(r['rel_area']):.2f}\n"
                     f"r=rounded  s=spread  u=unsure  q=quit", fontsize=10)
        ax.axis("off"); fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == "q":
            plt.close(fig); return
        if ev.key in ("r", "s", "u"):
            todo[state["i"]]["label"] = ev.key
            _persist()                 # auto-save after every keypress
            state["i"] += 1
        if state["i"] >= len(todo):
            plt.close(fig); return
        show()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(); plt.show()
    _persist()
    done = sum(1 for r in rows if r["label"] in ("r", "s"))
    print(f"saved — {done} labelled. Run `train`.")
    return 0


# ---------- train ----------
def _recompute_clean(rows):
    """Re-derive each labelled row's features from its mask using the
    current (cleaned) _frame_feats — so a fit reflects the hole-fill /
    despeck fix. Groups by cell to load each masks.npz once."""
    import collections
    from core.cell_state import DEFAULT_THRESHOLDS as TH
    min_area = TH["min_area_px"]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["label_dir"], int(r["cid"]))].append(r)
    for (cond, ld, cid), grp in groups.items():
        mp = os.path.join(RECORDINGS_ROOT, cond, ld,
                          "pipeline_results", "masks.npz")
        if not os.path.exists(mp):
            continue
        stack = np.load(mp)["labels"] == cid
        ff = {}
        for fi in np.where(stack.any(axis=(1, 2)))[0]:
            d = _frame_feats(stack[fi], min_area)
            if d is not None:
                ff[int(fi)] = d
        if not ff:
            continue
        base = float(np.percentile([ff[fi]["area"] for fi in ff], 90)) or 1.0
        for r in grp:
            fi = int(r["frame"])
            if fi in ff:
                d = ff[fi]
                r["rel_area"] = d["area"] / base
                for f in FEATURES:
                    if f != "rel_area" and f in d:
                        r[f] = d[f]
    return rows


def cmd_train(args):
    if not os.path.exists(CSV_PATH):
        print("Run `sample` + `label` first."); return 1
    rows = [r for r in csv.DictReader(open(CSV_PATH))
            if r["label"] in ("r", "s")]
    if len(rows) < 12:
        print(f"only {len(rows)} labelled — label ~30+ for a useful fit.")
        return 1
    print("recomputing shape features from masks (hole-fill + despeck)…")
    _recompute_clean(rows)
    cols = ["rel_area"] + [f for f in FEATURES if f != "rel_area"]
    X = np.array([[float(r[c]) for c in cols] for r in rows])
    y = np.array([1 if r["label"] == "r" else 0 for r in rows])
    print(f"Labelled: {int(y.sum())} rounded / {int((1-y).sum())} spread "
          f"(n={len(y)})\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000))
    acc = cross_val_score(lr, X, y, cv=min(5, int(y.sum()), int((1-y).sum())))
    lr.fit(X, y)
    coef = lr.named_steps["logisticregression"].coef_[0]
    print(f"Logistic regression  CV accuracy = {acc.mean():.2f} ± "
          f"{acc.std():.2f}")
    print("  feature weights (standardised — bigger |w| = more important):")
    for c, w in sorted(zip(cols, coef), key=lambda t: -abs(t[1])):
        print(f"    {c:14s} {w:+.2f}")
    dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3,
                                random_state=0).fit(X, y)
    print("\n  Interpretable rule (depth-2 tree):")
    print(export_text(dt, feature_names=cols).rstrip())
    print("\nTo deploy: add a classifier path to core.cell_state using these "
          "features (rel_area needs a per-track baseline at classify time).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=80)
    s.set_defaults(func=cmd_sample)
    sub.add_parser("label").set_defaults(func=cmd_label)
    sub.add_parser("train").set_defaults(func=cmd_train)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help(); return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
