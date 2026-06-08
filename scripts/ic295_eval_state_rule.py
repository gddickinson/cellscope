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
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
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
    return 0


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
