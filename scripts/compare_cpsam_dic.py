"""Compare two bench JSONs side by side.

Usage:
  python scripts/compare_cpsam_dic.py \
      results/dic_model_eval/cellpose_dic_v3.json \
      results/dic_model_eval/cpsam_dic.json
"""
import argparse
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt_iou(s):
    return f"{s['mean_iou']:.3f} ± {s['std_iou']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_json", help="first JSON (baseline)")
    ap.add_argument("candidate_json", help="second JSON (candidate)")
    args = ap.parse_args()

    a = load(args.baseline_json)
    b = load(args.candidate_json)

    print(f"\n  {'metric':18s}  {'baseline':>22s}  "
          f"{'candidate':>22s}  {'Δ':>8s}")
    print(f"  {'':18s}  {a['model']:>22s}  {b['model']:>22s}")
    print("  " + "-" * 76)

    print(f"  {'n frames':18s}  {a['n']:>22d}  {b['n']:>22d}")
    print(f"  {'mean IoU':18s}  {fmt_iou(a):>22s}  {fmt_iou(b):>22s}  "
          f"{b['mean_iou'] - a['mean_iou']:+.3f}")
    print(f"  {'det rate':18s}  {a['det_rate']:>22.0%}  "
          f"{b['det_rate']:>22.0%}  "
          f"{b['det_rate'] - a['det_rate']:+.0%}")

    geno_keys = sorted(set(a['per_genotype']) | set(b['per_genotype']))
    for g in geno_keys:
        sa = a['per_genotype'].get(g)
        sb = b['per_genotype'].get(g)
        if not sa or not sb:
            continue
        delta = sb['mean_iou'] - sa['mean_iou']
        print(f"  {('IoU ' + g):18s}  {fmt_iou(sa):>22s}  "
              f"{fmt_iou(sb):>22s}  {delta:+.3f}")

    # Recommendation
    delta = b['mean_iou'] - a['mean_iou']
    print()
    if delta >= 0.02:
        print(f"  → ship candidate (Δ {delta:+.3f} mean IoU)")
    elif delta >= -0.02:
        print(f"  → roughly tied (Δ {delta:+.3f}); consider resuming "
              f"training on cpsam_dic for more epochs")
    else:
        print(f"  → keep baseline (candidate worse by {-delta:.3f}); "
              f"resume training or discard cpsam_dic")


if __name__ == "__main__":
    main()
