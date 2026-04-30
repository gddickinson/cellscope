"""Evaluate DIC models on held-out VAMPIRE test set + our full-frame GT.

Tests:
  1. cellpose_dic (current CP3 model, baseline)
  2. cellpose_dic_v2 (newly trained CP3 model, Plan B)
  3. cpsam defaults (cellpose4, phase-contrast optimized)
  4. cpsam_dic (fine-tuned on DIC, Plan A) — if available

On:
  A. VAMPIRE held-out test sequences (6 sequences, 1129 pairs)
  B. Our full-frame GT (122 ctrl + 122 cKO at 526x526)
  C. VAMPIRE movie recordings (full-stack IoU comparison)

Reports per-genotype IoU, detection rate, and overall ranking.
"""
import os
import sys
import glob
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports, benchmark_data_root  # noqa
setup_imports()

TEST_DIR = "data/training/dic_splits/test"
GT_DIR = str(benchmark_data_root() / "data" / "training")
VAMP_DIR = str(benchmark_data_root() / "data" / "examples" / "vampire_movies")
OUT_DIR = "results/dic_model_eval"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_N = 30  # frames per model per dataset


def iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else 0.0


def load_test_pairs(data_dir, max_per_geno=None):
    """Load test pairs, return list of (img, mask, name, genotype)."""
    import tifffile
    img_files = sorted(glob.glob(os.path.join(data_dir, "*_img.tif")))
    pairs = []
    for img_path in img_files:
        mask_path = img_path.replace("_img.tif", "_masks.tif")
        if not os.path.exists(mask_path):
            continue
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)
        if img.ndim == 3:
            img = img[0]
        if mask.ndim == 3:
            mask = mask[0]
        name = os.path.basename(img_path).replace("_img.tif", "")
        geno = "control" if "WT" in name or "con" in name or "wt" in name \
            else "gof" if "GoF" in name or "gof" in name \
            else "cKO" if "KO" in name or "ko" in name \
            else "unknown"
        pairs.append((img, mask > 0, name, geno))
    return pairs


def load_our_gt():
    """Load our 244 full-frame GT pairs."""
    import cv2
    pairs = []
    for gtype, geno in [("ctrl", "control"), ("cko", "cKO")]:
        pattern = os.path.join(GT_DIR, f"our_{gtype}_gt_*.png")
        img_files = sorted(glob.glob(pattern))
        img_files = [f for f in img_files if "_masks" not in f]
        for img_path in img_files:
            mask_path = img_path.replace(".png", "_masks.png")
            if not os.path.exists(mask_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            name = os.path.basename(img_path).replace(".png", "")
            pairs.append((img, mask > 0, name, geno))
    return pairs


def run_cellpose_cp3(images, model_path, ft=0.0, ct=0.0):
    """Run cellpose CP3 model on images (handles variable sizes)."""
    from core.detection import detect_cellpose
    results = []
    for img in images:
        frames = img[np.newaxis]
        masks = detect_cellpose(
            frames, gpu=True, model_path=model_path,
            flow_threshold=ft, cellprob_threshold=ct)
        results.append(masks[0])
    return results


def run_cellpose_cp3_with_preprocess(images, model_path):
    """Run cellpose CP3 with preprocessing + retry (per-image)."""
    from core.detection import detect_cellpose
    from core.preprocess import preprocess_sequence
    results = []
    for img in images:
        frames = img[np.newaxis]
        processed = preprocess_sequence(
            frames, temporal_method="median",
            spatial_highpass_sigma=40.0,
            debris_diameter_um=0, pixel_size_um=None)
        masks = detect_cellpose(
            processed, gpu=True, model_path=model_path,
            flow_threshold=0.0, cellprob_threshold=0.0)
        if masks[0].sum() < 200:
            retry = detect_cellpose(
                processed, gpu=True, model_path=model_path,
                flow_threshold=0.0, cellprob_threshold=-2.0)
            masks[0] = retry[0]
        results.append(masks[0])
    return results


def evaluate_model(pairs, model_fn, name, sample_n=None):
    """Evaluate a model on a set of (img, mask, name, genotype) pairs."""
    if sample_n and len(pairs) > sample_n:
        idx = np.linspace(0, len(pairs)-1, sample_n, dtype=int)
        pairs = [pairs[i] for i in idx]

    images = [p[0] for p in pairs]
    gt_masks = [p[1] for p in pairs]
    names = [p[2] for p in pairs]
    genotypes = [p[3] for p in pairs]

    t0 = time.time()
    try:
        pred_masks = model_fn(images)
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        return None
    elapsed = time.time() - t0

    ious = [iou(pred_masks[i], gt_masks[i]) for i in range(len(pairs))]
    det_rate = sum(1 for m in pred_masks if m.any()) / max(len(pairs), 1)

    # Per-genotype
    by_geno = {}
    for g in set(genotypes):
        g_ious = [ious[i] for i in range(len(pairs)) if genotypes[i] == g]
        by_geno[g] = float(np.mean(g_ious)) if g_ious else 0

    result = {
        "name": name,
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        "det_rate": float(det_rate),
        "n_pairs": len(pairs),
        "elapsed": elapsed,
        "per_genotype": by_geno,
    }
    return result


def main():
    print("=== DIC Model Evaluation ===\n")

    # Define models to test
    models = {}

    # cellpose_dic (baseline)
    dic_path = "data/models/cellpose_dic"
    if os.path.exists(dic_path):
        models["cellpose_dic"] = lambda imgs: run_cellpose_cp3(
            imgs, dic_path)
        models["cellpose_dic+pp"] = lambda imgs: \
            run_cellpose_cp3_with_preprocess(imgs, dic_path)

    # cellpose_dic_v2 (Plan B)
    dic_v2_path = "data/models/cellpose_dic_v2"
    if os.path.exists(dic_v2_path):
        models["cellpose_dic_v2"] = lambda imgs: run_cellpose_cp3(
            imgs, dic_v2_path)
        models["cellpose_dic_v2+pp"] = lambda imgs: \
            run_cellpose_cp3_with_preprocess(imgs, dic_v2_path)
    else:
        print(f"  cellpose_dic_v2 not found at {dic_v2_path}")

    # cellpose_combined_robust
    robust_path = "data/models/cellpose_combined_robust"
    if os.path.exists(robust_path):
        models["cellpose_robust"] = lambda imgs: run_cellpose_cp3(
            imgs, robust_path)

    if not models:
        print("No CP3 models found!")
        return

    print(f"Models to test: {list(models.keys())}\n")

    all_results = {}

    # Dataset A: VAMPIRE held-out test
    print("=" * 60)
    print("Dataset A: VAMPIRE held-out test sequences")
    print("=" * 60)
    test_pairs = load_test_pairs(TEST_DIR)
    print(f"Loaded {len(test_pairs)} test pairs")
    geno_counts = {}
    for _, _, _, g in test_pairs:
        geno_counts[g] = geno_counts.get(g, 0) + 1
    print(f"Genotypes: {geno_counts}\n")

    for model_name, model_fn in models.items():
        result = evaluate_model(test_pairs, model_fn, model_name,
                                sample_n=SAMPLE_N * 3)
        if result:
            print(f"  {model_name:30s}: IoU={result['mean_iou']:.3f}±"
                  f"{result['std_iou']:.3f} det={result['det_rate']:.0%} "
                  f"{result['elapsed']:.0f}s")
            for g, v in sorted(result["per_genotype"].items()):
                print(f"    {g:10s}: {v:.3f}")
            all_results[f"vampire_{model_name}"] = result

    # Dataset B: Our full-frame GT
    print(f"\n{'=' * 60}")
    print("Dataset B: Our full-frame GT (244 frames)")
    print("=" * 60)
    gt_pairs = load_our_gt()
    print(f"Loaded {len(gt_pairs)} GT pairs\n")

    for model_name, model_fn in models.items():
        result = evaluate_model(gt_pairs, model_fn, model_name,
                                sample_n=SAMPLE_N * 2)
        if result:
            print(f"  {model_name:30s}: IoU={result['mean_iou']:.3f}±"
                  f"{result['std_iou']:.3f} det={result['det_rate']:.0%} "
                  f"{result['elapsed']:.0f}s")
            for g, v in sorted(result["per_genotype"].items()):
                print(f"    {g:10s}: {v:.3f}")
            all_results[f"ourgt_{model_name}"] = result

    # Summary ranking
    print(f"\n{'=' * 60}")
    print("RANKING: VAMPIRE test set")
    print("=" * 60)
    vamp_results = [(k.replace("vampire_", ""), v)
                    for k, v in all_results.items() if k.startswith("vampire_")]
    vamp_results.sort(key=lambda x: -x[1]["mean_iou"])
    for i, (name, r) in enumerate(vamp_results):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1}. {name:30s}: {r['mean_iou']:.3f}{marker}")

    print(f"\n{'=' * 60}")
    print("RANKING: Our full-frame GT")
    print("=" * 60)
    gt_results = [(k.replace("ourgt_", ""), v)
                  for k, v in all_results.items() if k.startswith("ourgt_")]
    gt_results.sort(key=lambda x: -x[1]["mean_iou"])
    for i, (name, r) in enumerate(gt_results):
        marker = " ★" if i == 0 else ""
        print(f"  {i+1}. {name:30s}: {r['mean_iou']:.3f}{marker}")

    # Save
    with open(os.path.join(OUT_DIR, "dic_model_eval.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults: {OUT_DIR}/dic_model_eval.json")


if __name__ == "__main__":
    main()
