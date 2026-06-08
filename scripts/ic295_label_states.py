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
    picks = [rows[i] for i in _kmeans_diverse(X, n)]
    os.makedirs(CROPS_DIR, exist_ok=True)
    _render_crops(picks, 0, "montage.png")
    _write_csv(picks, 0, append=False)
    print(f"\nSampled {len(picks)} diverse cell-frames.")
    print(f"  montage : {LABELS_DIR}/montage.png")
    print(f"  crops   : {CROPS_DIR}/NNN.png")
    print(f"  labels  : {CSV_PATH}  (fill the `label` column: r/s/u)")
    print("Then:  ic295_label_states.py label   (or edit the CSV)  → train")
    return 0


def _kmeans_diverse(X, n):
    """Indices of n rows spanning the feature space (nearest each
    k-means centroid; falls back to rel_area quantiles)."""
    n = min(n, len(X))
    if n <= 0:
        return []
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
        return idx
    except Exception:
        order = np.argsort(X[:, 0])
        return [int(order[int(k)]) for k in np.linspace(0, len(order) - 1, n)]


def cmd_more(args):
    """Append a larger, rounded-ENRICHED batch (active-learning style):
    score every un-sampled frame with a classifier fit on the existing
    labels, then sample across P(rounded) bands (rounded is rare, so we
    over-weight the high-P + uncertain bands) for a better class mix."""
    n_new = getattr(args, "count", 150)
    if not os.path.exists(CSV_PATH):
        print("Run `sample` first."); return 1
    existing = list(csv.DictReader(open(CSV_PATH)))
    seen = {(r["condition"], r["label_dir"], int(r["cid"]), int(r["frame"]))
            for r in existing}
    labeled = [r for r in existing if r["label"] in ("r", "s")]
    print(f"existing: {len(existing)} sampled, {len(labeled)} labelled")
    rows = _collect()
    rows = [r for r in rows
            if (r["condition"], r["label_dir"], int(r["cid"]),
                int(r["frame"])) not in seen]
    if not rows:
        print("no new frames to sample."); return 1
    X, cols = _feature_matrix(rows)
    ok = np.all(np.isfinite(X), axis=1)
    rows = [r for r, o in zip(rows, ok) if o]; X = X[ok]
    P = _round_proba(X, cols, labeled)
    # bands: (lo, hi, fraction of the new batch) — enrich rounded/boundary
    bands = [(0.55, 1.01, 0.45), (0.30, 0.55, 0.30),
             (0.12, 0.30, 0.15), (0.0, 0.12, 0.10)]
    picks, taken = [], set()
    for lo, hi, frac in bands:
        idx = np.where((P >= lo) & (P < hi))[0]
        idx = np.array([i for i in idx if i not in taken])
        k = int(round(n_new * frac))
        if len(idx) == 0 or k == 0:
            continue
        for j in _kmeans_diverse(X[idx], min(k, len(idx))):
            taken.add(int(idx[j])); picks.append(rows[int(idx[j])])
    # top up from highest-P remaining if short
    if len(picks) < n_new:
        rest = [i for i in np.argsort(-P) if i not in taken]
        for i in rest[:n_new - len(picks)]:
            picks.append(rows[int(i)])
    start = len(existing)
    os.makedirs(CROPS_DIR, exist_ok=True)
    _render_crops(picks, start, "montage_more.png")
    _write_csv(picks, start, append=True)
    print(f"\nAppended {len(picks)} enriched cell-frames "
          f"(idx {start}..{start+len(picks)-1}); P(rounded) bands enrich "
          f"the rare rounded class.")
    print(f"  montage : {LABELS_DIR}/montage_more.png")
    print(f"  total now: {len(existing)+len(picks)} crops "
          f"({len(labeled)} already labelled)")
    print("Run:  ic295_label_states.py label   (shows only the new "
          "unlabelled ones)")
    return 0


def _round_proba(X, cols, labeled):
    """P(rounded) per row from a classifier fit on `labeled` (cleaned
    features). Falls back to a rel_area proxy if too few labels."""
    if len(labeled) >= 12 and len({r["label"] for r in labeled}) == 2:
        _recompute_clean(labeled)
        Xl = np.array([[float(r[c]) for c in cols] for r in labeled])
        yl = np.array([1 if r["label"] == "r" else 0 for r in labeled])
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000)).fit(Xl, yl)
        return clf.predict_proba(X)[:, 1]
    ra = X[:, 0]                       # rel_area: smaller = more rounded
    return 1.0 - (ra - ra.min()) / (np.ptp(ra) + 1e-9)


def _fixed_window(img, cy, cx, size):
    """Centre a `size`×`size` window on (cy,cx); zero-pad past edges.
    Returns (window, valid_mask) — valid is False over the padding."""
    half = size // 2
    H, W = img.shape[:2]
    out = np.zeros((size, size), dtype=img.dtype)
    valid = np.zeros((size, size), dtype=bool)
    y0, x0 = cy - half, cx - half
    sy0, sy1 = max(0, y0), min(H, y0 + size)
    sx0, sx1 = max(0, x0), min(W, x0 + size)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    valid[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
    return out, valid


def _load_frames_labels(mp, cache):
    if mp not in cache:
        from core.io import load_recording
        d = os.path.dirname(os.path.dirname(mp))
        tif = next((os.path.join(d, f) for f in os.listdir(d)
                    if f.endswith(".ome.tif")), None)
        frames = (load_recording(tif, dic_channel=1, fluo_channel=0)["frames"]
                  if tif else None)
        cache[mp] = (frames, np.load(mp)["labels"])
    return cache[mp]


def _render_crops(picks, start_idx=0, montage_name="montage.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure
    rec_cache = {}
    # Pass 1: load recordings + find each cell's centre + extent, so all
    # crops share ONE window size (large enough for the biggest cell) →
    # the cell's true scale is comparable across crops.
    meta = []
    for r in picks:
        frames, labels = _load_frames_labels(r["mp"], rec_cache)
        m = labels[r["frame"]] == r["cid"]
        rr = np.any(m, axis=1); cc = np.any(m, axis=0)
        if not rr.any():
            continue
        r0, r1 = np.where(rr)[0][[0, -1]]; c0, c1 = np.where(cc)[0][[0, -1]]
        meta.append((r, (r0 + r1) // 2, (c0 + c1) // 2,
                     max(r1 - r0, c1 - c0)))
    if not meta:
        return
    size = int(np.clip(max(e for *_, e in meta) + 2 * MARGIN, 64, 420))
    # Pass 2: render fixed-size, cell-centred crops
    thumbs = []
    for k, (r, cy, cx, _) in enumerate(meta):
        gidx = start_idx + k
        frames, labels = rec_cache[r["mp"]]
        m = labels[r["frame"]] == r["cid"]
        base = frames[r["frame"]] if frames is not None else m.astype(np.uint16)
        imgw, valid = _fixed_window(base, cy, cx, size)
        mw, _ = _fixed_window(m.astype(np.uint8), cy, cx, size)
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        vals = imgw[valid]
        lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))
        ax.imshow(imgw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
        for ct in measure.find_contours(mw.astype(float), 0.5):
            ax.plot(ct[:, 1], ct[:, 0], "-", color="#ff3030", lw=1.2)
        ax.set_title(f"{gidx}", fontsize=9); ax.axis("off")
        ax.set_xlim(0, size); ax.set_ylim(size, 0)
        fig.tight_layout(pad=0.1)
        fig.savefig(os.path.join(CROPS_DIR, f"{gidx:03d}.png"), dpi=90)
        plt.close(fig)
        thumbs.append((gidx, os.path.join(CROPS_DIR, f"{gidx:03d}.png")))
    _montage(thumbs, montage_name)


def _montage(thumbs, name="montage.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n = len(thumbs); ncol = 10; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.5))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < n:
            ax.imshow(mpimg.imread(thumbs[i][1]))
        ax.axis("off")
    fig.suptitle("Label each by index in labels.csv  (r=rounded s=spread "
                 "u=unsure)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LABELS_DIR, name), dpi=120)
    plt.close(fig)


def _write_csv(picks, start_idx=0, append=False):
    cols = (["idx", "label", "condition", "label_dir", "cid", "frame",
             "rel_area"] + [f for f in FEATURES if f != "rel_area"])
    mode = "a" if append else "w"
    with open(CSV_PATH, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if not append:
            w.writeheader()
        for k, r in enumerate(picks):
            w.writerow({"idx": start_idx + k, "label": "", **r})


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
        from skimage import measure
    except Exception:
        print("No interactive display — open montage.png and fill the "
              "`label` column (r/s/u) in labels.csv, then `train`.")
        return 0
    state = {"i": 0}
    cache = {}
    todo = [r for r in rows if not r["label"]]
    if not todo:
        print("all rows already labelled."); return 0

    def _persist():
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    def show():
        ax.clear()
        r = todo[state["i"]]
        mp = os.path.join(RECORDINGS_ROOT, r["condition"], r["label_dir"],
                          "pipeline_results", "masks.npz")
        try:
            frames, labels = _load_frames_labels(mp, cache)
            m = labels[int(r["frame"])] == int(r["cid"])
            rr = np.any(m, axis=1); cc = np.any(m, axis=0)
            r0, r1 = np.where(rr)[0][[0, -1]]
            c0, c1 = np.where(cc)[0][[0, -1]]
            cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
            size = int(max(r1 - r0, c1 - c0) + 4 * MARGIN)
            base = (frames[int(r["frame"])] if frames is not None
                    else m.astype(np.uint16))
            imgw, valid = _fixed_window(base, cy, cx, size)
            mw, _ = _fixed_window(m.astype(np.uint8), cy, cx, size)
            vals = imgw[valid]
            lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))
            ax.imshow(imgw, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            for ct in measure.find_contours(mw.astype(float), 0.5):
                ax.plot(ct[:, 1], ct[:, 0], "-", color="#ff3030", lw=1.4)
            ax.set_xlim(0, size); ax.set_ylim(size, 0)
        except Exception as e:
            ax.text(0.5, 0.5, f"render error:\n{e}", ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_title(f"{state['i']+1}/{len(todo)}   "
                     f"rel_area={float(r['rel_area']):.2f}   "
                     f"({r['label_dir']})\n"
                     f"r=rounded  s=spread  u=unsure   scroll=zoom   q=quit",
                     fontsize=10)
        ax.axis("off"); fig.canvas.draw_idle()

    def on_scroll(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        scale = (1 / 1.25) if ev.button == "up" else 1.25
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(ev.xdata - (ev.xdata - x0) * scale,
                    ev.xdata + (x1 - ev.xdata) * scale)
        ax.set_ylim(ev.ydata - (ev.ydata - y0) * scale,
                    ev.ydata + (y1 - ev.ydata) * scale)
        fig.canvas.draw_idle()

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
    fig.canvas.mpl_connect("scroll_event", on_scroll)
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
    m = sub.add_parser("more")   # positional count avoids conda's --flag quirk
    m.add_argument("count", nargs="?", type=int, default=150)
    m.set_defaults(func=cmd_more)
    sub.add_parser("label").set_defaults(func=cmd_label)
    sub.add_parser("train").set_defaults(func=cmd_train)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help(); return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
