"""Bench a cellpose model across the brightness-perturbed test set.

Wraps `bench_cpsam_dic.py` so we don't reimplement the model loop.
For each subdirectory of ``test_brightness/``, calls bench, then
collects the per-perturbation JSON outputs into one summary file
plus a markdown report.

The headline metric is **IoU retention** = perturbation_iou /
clean_iou. ≥0.8 retention is the ship criterion for the brightness-
augmented retrain (3.5).

Usage:

  # Baseline (current cpsam_dic v2)
  conda run -n cellpose4 python scripts/bench_brightness.py \\
      --model data/models/cpsam_dic --label cpsam_dic_v2

  # After retrain (cpsam_dic v4)
  conda run -n cellpose4 python scripts/bench_brightness.py \\
      --model data/models/cpsam_dic_v4 --label cpsam_dic_v4

  # Compare
  python scripts/bench_brightness.py --compare cpsam_dic_v2 cpsam_dic_v4
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

TEST_ROOT_DEFAULT = "data/training/dic_splits_v3/test_brightness"
OUT_ROOT = "results/brightness_eval"


def discover_perturbations(test_root):
    return sorted(
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d))
    )


def bench_one(model, label, perturbation, test_root, env, per_genotype):
    test_dir = os.path.join(test_root, perturbation)
    out_dir = os.path.join(OUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{perturbation}.json")
    if os.path.exists(out_json):
        print(f"  [cache] {out_json}")
        return out_json
    cmd = [
        "conda", "run", "-n", env,
        "python", "scripts/bench_cpsam_dic.py",
        "--model", model,
        "--test-dir", test_dir,
        "--out", out_json,
        "--per-genotype", str(per_genotype),
    ]
    print(f"  running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [error] bench failed for {perturbation}:\n"
              f"    stderr: {res.stderr[-400:]}")
        return None
    return out_json


def collect_summary(label):
    out_dir = os.path.join(OUT_ROOT, label)
    rows = {}
    for fn in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        pert = os.path.basename(fn).replace(".json", "")
        with open(fn) as f:
            data = json.load(f)
        rows[pert] = {
            "mean_iou": float(data.get("mean_iou", 0.0)),
            "n": int(data.get("n", 0)),
            "by_genotype": {
                g: float(d.get("mean_iou", 0.0)) if isinstance(d, dict)
                else float(d)
                for g, d in (data.get("by_genotype", {}) or {}).items()
            },
        }
    if not rows:
        return None
    clean = rows.get("clean", {}).get("mean_iou", None)
    for pert, r in rows.items():
        r["retention"] = (r["mean_iou"] / clean
                          if clean and clean > 0 else None)
    summary = {"label": label, "rows": rows}
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_markdown(label, summary):
    out_path = os.path.join(OUT_ROOT, label, "summary.md")
    lines = [f"# Brightness robustness — {label}", ""]
    lines.append("| Perturbation | Mean IoU | Retention vs clean | n |")
    lines.append("|---|---:|---:|---:|")
    rows = summary["rows"]
    order = ["clean", "b_plus_30", "b_plus_60", "b_minus_30",
             "b_minus_60", "gamma_05", "gamma_18", "vignette"]
    for pert in [p for p in order if p in rows] + [
            p for p in rows if p not in order]:
        r = rows[pert]
        ret = (f"{r['retention']:.2f}"
               if r["retention"] is not None else "—")
        lines.append(f"| {pert} | {r['mean_iou']:.3f} | {ret} | "
                     f"{r['n']} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


def cmd_compare(labels):
    summaries = {}
    for lab in labels:
        path = os.path.join(OUT_ROOT, lab, "summary.json")
        if not os.path.exists(path):
            print(f"  [skip] no summary at {path} — run bench first")
            continue
        with open(path) as f:
            summaries[lab] = json.load(f)
    if len(summaries) < 2:
        print("Need at least 2 summaries to compare.")
        return
    perts = sorted({p for s in summaries.values() for p in s["rows"]})
    out_path = os.path.join(OUT_ROOT, "comparison.md")
    lines = ["# Brightness robustness — comparison", "",
             "| Perturbation | "
             + " | ".join(f"IoU {l}" for l in summaries)
             + " | "
             + " | ".join(f"Ret. {l}" for l in summaries)
             + " |"]
    sep = "|---" * (1 + 2 * len(summaries)) + "|"
    lines.append(sep)
    order = ["clean", "b_plus_30", "b_plus_60", "b_minus_30",
             "b_minus_60", "gamma_05", "gamma_18", "vignette"]
    for pert in [p for p in order if p in perts] + [
            p for p in perts if p not in order]:
        ious = []
        rets = []
        for lab, s in summaries.items():
            r = s["rows"].get(pert, {})
            ious.append(f"{r.get('mean_iou', 0.0):.3f}")
            rets.append(f"{r['retention']:.2f}"
                        if r.get("retention") is not None else "—")
        lines.append(f"| {pert} | " + " | ".join(ious) + " | "
                     + " | ".join(rets) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Comparison written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="cellpose model path")
    ap.add_argument("--label", help="label for this run "
                    "(used as subdirectory name)")
    ap.add_argument("--env", default="cellpose4",
                    help="conda env (cellpose4 for cpsam, cellpose for CP3)")
    ap.add_argument("--test-root", default=TEST_ROOT_DEFAULT,
                    help=f"brightness test root (default {TEST_ROOT_DEFAULT})")
    ap.add_argument("--per-genotype", type=int, default=20,
                    help="frames per genotype (default 20)")
    ap.add_argument("--compare", nargs="+",
                    help="compare two or more existing labels")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(args.compare)
        return

    if not (args.model and args.label):
        ap.error("--model and --label required (or use --compare)")

    if not os.path.isdir(args.test_root):
        ap.error(f"missing test root: {args.test_root}\n"
                 f"  Run scripts/build_brightness_test.py first.")

    perts = discover_perturbations(args.test_root)
    print(f"[bench_brightness] perturbations: {perts}")
    print(f"[bench_brightness] model: {args.model}")
    print(f"[bench_brightness] label: {args.label}\n")

    for p in perts:
        print(f"-- {p} --")
        bench_one(args.model, args.label, p,
                  args.test_root, args.env, args.per_genotype)

    summary = collect_summary(args.label)
    if summary:
        write_markdown(args.label, summary)


if __name__ == "__main__":
    main()
