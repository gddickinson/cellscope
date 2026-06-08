"""How well does the rounded/spread rule match the hand labels?

Evaluates `core.cell_state.classify_state` (rounded iff circ ≥ rounded_circ
AND solid ≥ rounded_solid) against the human labels in
`ic295_analysis/state_labels/labels.csv`, and asks whether a different
feature or threshold would agree with the labeller better. Read-only; no
analysis state is touched.

    conda run -n cellpose4 python scripts/ic295_eval_state_rule.py
    conda run -n cellpose4 python scripts/ic295_eval_state_rule.py path/to/labels.csv
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa: E402
setup_imports()

from core.cell_state import DEFAULT_THRESHOLDS  # noqa: E402

FEATURES = ["circularity", "solidity", "rel_area", "area", "extent",
            "eccentricity", "aspect_ratio", "convexity"]
DEFAULT_CSV = os.path.join("ic295_analysis", "state_labels", "labels.csv")
DEFAULT_UM = 0.6523                  # IC295 scope (single magnification)


def _load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        lab = (r.get("label") or "").strip()
        if lab not in ("r", "s"):           # drop unsure / unlabelled
            continue
        feats = {}
        ok = True
        for f in FEATURES:
            try:
                v = float(r.get(f, ""))
                feats[f] = v if np.isfinite(v) else np.nan
            except (TypeError, ValueError):
                feats[f] = np.nan
        rows.append((1 if lab == "r" else 0, feats))
    return rows


def _confusion(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and (prec + rec) else float("nan"))
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec, f1=f1)


def _best_threshold(x, y):
    """Best single-threshold accuracy for predicting rounded(=1).
    Tries both polarities (rounded = high feature, or = low feature)."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    if len(x) < 5 or len(set(y.tolist())) < 2:
        return None
    best = (0.0, None, None)
    for thr in np.unique(x):
        for hi in (True, False):
            pred = (x >= thr) if hi else (x <= thr)
            acc = np.mean(pred == y)
            if acc > best[0]:
                best = (acc, float(thr), "high" if hi else "low")
    return best  # (acc, threshold, polarity)


def _auc(x, y):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(x)
    if len(set(y[m].tolist())) < 2:
        return float("nan")
    a = roc_auc_score(y[m], x[m])
    return max(a, 1 - a)            # orientation-free separability


def main():
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    path = pos[0] if pos else DEFAULT_CSV
    um = DEFAULT_UM
    if "--um" in argv:
        um = float(argv[argv.index("--um") + 1])
    rows = _load(path)
    if not rows:
        print(f"No r/s labels in {path}"); return 1
    y = np.array([r[0] for r in rows])
    X = {f: np.array([r[1][f] for r in rows]) for f in FEATURES}
    n_r, n_s = int(y.sum()), int((1 - y).sum())
    print(f"Labelled set: {len(rows)} cell-frames  "
          f"({n_r} rounded, {n_s} spread)   source: {path}\n")

    # 1) the SHIPPED rule -------------------------------------------------
    tc, ts = DEFAULT_THRESHOLDS["rounded_circ"], DEFAULT_THRESHOLDS["rounded_solid"]
    pred = ((X["circularity"] >= tc) & (X["solidity"] >= ts)).astype(int)
    c = _confusion(y, pred)
    print(f"== SHIPPED RULE: rounded iff circ>={tc} AND solid>={ts} ==")
    print(f"  accuracy {c['acc']:.3f}   precision {c['prec']:.3f}   "
          f"recall {c['rec']:.3f}   F1 {c['f1']:.3f}")
    print(f"  confusion: TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")
    print(f"  -> {c['fn']} rounded MISSED, {c['fp']} spread WRONGLY rounded\n")

    # 2) single-feature separability vs the human ------------------------
    print("== single-feature agreement with the labeller ==")
    print(f"  {'feature':>13}  {'AUC':>5}  {'best-1thr acc':>13}  rule")
    ranked = []
    for f in FEATURES:
        auc = _auc(X[f], y)
        bt = _best_threshold(X[f], y)
        ranked.append((auc, f, bt))
    for auc, f, bt in sorted(ranked, reverse=True,
                             key=lambda t: (t[0] if t[0] == t[0] else 0)):
        if bt:
            acc, thr, pol = bt
            rule = f"rounded if {f} {'≥' if pol=='high' else '≤'} {thr:.3f}"
            print(f"  {f:>13}  {auc:.3f}  {acc:>11.3f}    {rule}")
        else:
            print(f"  {f:>13}  {auc:.3f}  {'n/a':>13}")
    print()

    # 3) combined models (5-fold CV) -------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    def cv(feats, est):
        M = np.column_stack([X[f] for f in feats])
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), est)
        s = cross_val_score(pipe, M, y, cv=5)
        return s.mean(), s.std()

    print("== combined models (5-fold CV accuracy) ==")
    for name, feats, est in [
        ("circ+solid  logistic", ["circularity", "solidity"],
         LogisticRegression(max_iter=1000)),
        ("ALL feats   logistic", FEATURES, LogisticRegression(max_iter=1000)),
        ("ALL feats   tree(d=2)", FEATURES,
         DecisionTreeClassifier(max_depth=2, random_state=0)),
        ("ALL feats   tree(d=3)", FEATURES,
         DecisionTreeClassifier(max_depth=3, random_state=0)),
    ]:
        m, sd = cv(feats, est)
        print(f"  {name}:  {m:.3f} ± {sd:.3f}")
    print()

    # 4) what an interpretable tree actually splits on -------------------
    M = np.column_stack([X[f] for f in FEATURES])
    from sklearn.impute import SimpleImputer
    Mi = SimpleImputer(strategy="median").fit_transform(M)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Mi, y)
    print("== depth-2 tree (interpretable) ==")
    _print_tree(tree, FEATURES)

    # 5) emit the DEPLOYED thresholds (area_um2 + eccentricity) ----------
    _emit_thresholds(X, y, um)

    # 6) validation figures (unless --no-plots) --------------------------
    if "--no-plots" not in argv:
        try:
            from scripts.ic295_common import COMPARE_DIR
            out_dir = os.path.join(COMPARE_DIR, "state_rule_validation")
        except Exception:
            out_dir = os.path.join("ic295_analysis", "compare",
                                   "state_rule_validation")
        _plots(X, y, um, out_dir)
    return 0


def _emit_thresholds(X, y, um=None):
    """Fit the deployed depth-2 rule (area_um2 + eccentricity) and print the
    DEFAULT_THRESHOLDS values — so the rule stays reproducible from labels.

    `area` in the CSV is px; converted to µm² with `um` (µm/px). The IC295
    corpus is a single scope, so a constant um is correct here; pass --um to
    override.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.impute import SimpleImputer
    if um is None:
        um = 0.6523
        for i, a in enumerate(sys.argv):
            if a == "--um" and i + 1 < len(sys.argv):
                um = float(sys.argv[i + 1])
    area_um2 = X["area"] * um * um
    M = np.column_stack([area_um2, X["eccentricity"]])
    Mi = SimpleImputer(strategy="median").fit_transform(M)
    t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Mi, y).tree_
    # root splits on area_um2 (feature 0); follow its SMALL-area child (left,
    # area<=thr) and read that branch's eccentricity split — the meaningful
    # one (the large-area branch is ~all spread regardless of ecc).
    area_thr = ecc_thr = None
    if t.feature[0] == 0:                      # root is the area split
        area_thr = t.threshold[0]
        left = t.children_left[0]              # small-area branch
        if t.children_left[left] != t.children_right[left] \
                and t.feature[left] == 1:
            ecc_thr = t.threshold[left]
    print(f"\n== suggested DEFAULT_THRESHOLDS  (um/px={um}) ==")
    print(f"  rounded_area_um2     = {area_thr:.1f}"
          if area_thr is not None else "  (no area split)")
    print(f"  rounded_eccentricity = {ecc_thr:.3f}"
          if ecc_thr is not None else "  (no ecc split)")


def _plots(X, y, um, out_dir):
    """Label-grounded validation figures for the deployed rule
    (area_um2 ≤ rounded_area_um2 AND eccentricity ≤ rounded_eccentricity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    os.makedirs(out_dir, exist_ok=True)
    Ta = DEFAULT_THRESHOLDS["rounded_area_um2"]
    Te = DEFAULT_THRESHOLDS["rounded_eccentricity"]
    area_um2 = X["area"] * um * um
    ecc = X["eccentricity"]
    pred = ((area_um2 <= Ta) & (ecc <= Te)).astype(int)
    rcol = "#c0392b"; scol = "#2980b9"      # rounded / spread (hand label)
    rnd, spr = (y == 1), (y == 0)

    # ---- Figure 1: decision boundary + per-feature dists + confusion ----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    ax = axs[0, 0]
    xmax = float(np.nanpercentile(area_um2, 99.5))
    ax.add_patch(Rectangle((0, 0), Ta, Te, facecolor="#c0392b", alpha=0.08,
                           zorder=0))
    ax.scatter(area_um2[spr], ecc[spr], s=12, c=scol, alpha=0.5,
               label="hand: spread")
    ax.scatter(area_um2[rnd], ecc[rnd], s=12, c=rcol, alpha=0.6,
               label="hand: rounded")
    wrong = pred != y
    ax.scatter(area_um2[wrong], ecc[wrong], s=46, facecolors="none",
               edgecolors="k", lw=1.1, label=f"misclassified (n={int(wrong.sum())})")
    ax.axvline(Ta, color="k", ls="--", lw=1.3)
    ax.axhline(Te, color="k", ls="--", lw=1.3)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
    ax.set_xlabel("area  (µm²)"); ax.set_ylabel("eccentricity")
    ax.set_title(f"Decision boundary — shaded = rounded\n"
                 f"(area ≤ {Ta:.0f} µm²  AND  ecc ≤ {Te})")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.25)

    ax = axs[0, 1]
    bins = np.linspace(0, xmax, 45)
    ax.hist(area_um2[spr], bins=bins, color=scol, alpha=0.6, density=True,
            label="spread")
    ax.hist(area_um2[rnd], bins=bins, color=rcol, alpha=0.6, density=True,
            label="rounded")
    ax.axvline(Ta, color="k", ls="--", lw=1.5, label=f"cut = {Ta:.0f} µm²")
    ax.set_xlabel("area  (µm²)"); ax.set_ylabel("density")
    ax.set_title("Footprint: hand-rounded vs hand-spread")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axs[1, 0]
    eb = np.linspace(0, 1, 40)
    ax.hist(ecc[spr], bins=eb, color=scol, alpha=0.6, density=True,
            label="spread")
    ax.hist(ecc[rnd], bins=eb, color=rcol, alpha=0.6, density=True,
            label="rounded")
    ax.axvline(Te, color="k", ls="--", lw=1.5, label=f"cut = {Te}")
    ax.set_xlabel("eccentricity"); ax.set_ylabel("density")
    ax.set_title("Elongation: hand-rounded vs hand-spread")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axs[1, 1]
    cm = np.array([[int((spr & (pred == 0)).sum()), int((spr & (pred == 1)).sum())],
                   [int((rnd & (pred == 0)).sum()), int((rnd & (pred == 1)).sum())]])
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred spread", "pred rounded"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["hand spread", "hand rounded"])
    acc = (pred == y).mean()
    rec = cm[1, 1] / max(cm[1].sum(), 1)
    ax.set_title(f"Confusion (acc {acc:.2f}, rounded-recall {rec:.2f})")
    fig.suptitle(f"Rounded/spread rule vs {len(y)} hand labels "
                 f"({int(rnd.sum())} rounded, {int(spr.sum())} spread)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "decision_boundary.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 2: per-feature separability (AUC) ----
    aucs = sorted(((_auc(X[f], y), f) for f in FEATURES), reverse=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [f for _, f in aucs]; vals = [a for a, _ in aucs]
    bars = ax.barh(range(len(names)), vals, color="#4c72b0")
    for i, f in enumerate(names):
        if f in ("area", "eccentricity"):
            bars[i].set_color("#c0392b")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.invert_yaxis(); ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="grey", lw=1)
    ax.set_xlabel("AUC vs hand label  (0.5 = chance)")
    ax.set_title("Single-feature agreement with the labeller\n"
                 "(red = features used by the deployed rule)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_auc.png"), dpi=130)
    plt.close(fig)
    print(f"\nWrote validation plots → {out_dir}/"
          f"  (decision_boundary.png, feature_auc.png)")


def _print_tree(tree, names, node=0, depth=0):
    t = tree.tree_
    pad = "  " + "    " * depth
    if t.children_left[node] == t.children_right[node]:   # leaf
        n = int(t.n_node_samples[node])
        vals = t.value[node][0]
        counts = vals * n if vals.sum() <= 1.0 + 1e-6 else vals  # sklearn≥1.3
        cls = "rounded" if counts[1] >= counts[0] else "spread"
        print(f"{pad}-> {cls}  (n={n}, "
              f"{counts[1]:.0f} rounded / {counts[0]:.0f} spread)")
        return
    f = names[t.feature[node]]; thr = t.threshold[node]
    print(f"{pad}if {f} <= {thr:.3f}:")
    _print_tree(tree, names, t.children_left[node], depth + 1)
    print(f"{pad}else:")
    _print_tree(tree, names, t.children_right[node], depth + 1)


if __name__ == "__main__":
    sys.exit(main())
