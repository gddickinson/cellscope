"""Side-by-side model comparison on the new Ignasi recordings.

Runs each (model, options) config on a sampled set of frames and
tabulates cell counts so we can pick the best detector before
committing to the 8-hour full pipeline run.

Pipeline per (config × recording × frame):
  1. Load frame (channel 0)
  2. Flat-field correct (subtract Gaussian-blurred background, σ=80)
  3. Run model in the appropriate env via subprocess
  4. Record cells detected, mask px, runtime
  5. Save overlay PNG (frame + colored instance contours)

Default config set covers cpsam_dic ± TTA, cpsam base ± TTA, and the
two strongest CP3 models. Override with --configs.

Output:
  results/ignasi_model_comparison/
    comparison.csv         long-format rows
    by_config.csv          aggregated (mean cells / detection rate)
    by_config_condition.csv
    overlays/<cfg>_<pos>.png
    overlay_grid.png       config × condition grid
    report.md
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

OUT_DIR = "results/ignasi_model_comparison"
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")

DEFAULT_CONFIGS = [
    # (label, env, model_path-or-None for cpsam base, tta, extra_kwargs)
    ("cpsam_dic_tta",  "cellpose4", "data/models/cpsam_dic", True,  {}),
    ("cpsam_dic",      "cellpose4", "data/models/cpsam_dic", False, {}),
    ("cpsam_base_tta", "cellpose4", None,                    True,  {}),
    ("cpsam_base",     "cellpose4", None,                    False, {}),
    ("cp3_robust",     "cellpose",  "data/models/cellpose_combined_robust",
                                                              False,
     {"flow_threshold": 0.0, "cellprob_threshold": 0.0}),
    ("cp3_dic_v3",     "cellpose",  "data/models/cellpose_dic_v3", False,
     {"flow_threshold": 0.0, "cellprob_threshold": 0.0}),
]

# One representative recording per condition for the comparison.
REPRESENTATIVE_RECORDINGS = {
    "WT":   "Pos5-WT",
    "KO":   "Pos19-KO",
    "GOF":  "Pos28-GOF",
    "Y1":   "Pos54-Y1",
    "DMSO": "Pos60-DMSO",
}


def parse_metadata(tif_path):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    out = {"n_channels": 1, "n_frames": None}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            txt = f.read()
        if (m := re.search(r'"Channels"\s*:\s*(\d+)', txt)):
            out["n_channels"] = int(m.group(1))
        if (m := re.search(r'"Frames"\s*:\s*(\d+)', txt)):
            out["n_frames"] = int(m.group(1))
    return out


def flat_field(frame_uint16, sigma=80):
    from scipy.ndimage import gaussian_filter
    f = frame_uint16.astype(np.float32)
    bg = gaussian_filter(f, sigma=sigma)
    bg = np.where(bg < 1, 1, bg)
    return f / bg - 1.0


def to_uint8(frame_uint16):
    flat = flat_field(frame_uint16)
    p1, p99 = np.percentile(flat, [1, 99])
    return np.clip((flat - p1) / max(p99 - p1, 1e-6) * 255,
                   0, 255).astype(np.uint8)


def read_frame(tif_path, frame_idx, channel, n_channels):
    page = frame_idx * n_channels + channel
    with tifffile.TiffFile(tif_path) as tf:
        return tf.pages[page].asarray()


def run_config_on_frames(label, env, model_path, tta, extra_kwargs,
                         frames_list):
    """Run a model config on a list of (uint8) frames via subprocess.

    Returns list of (n_cells, mask_px, mask, elapsed_s) per frame.
    """
    with tempfile.TemporaryDirectory() as td:
        in_npz = os.path.join(td, "frames.npz")
        out_npz = os.path.join(td, "results.npz")
        np.savez_compressed(in_npz,
                             **{f"f{i}": f for i, f in enumerate(frames_list)})
        kw_json = json.dumps(extra_kwargs)
        py = f"""
import json, time, numpy as np
from cellpose import models
data = np.load({in_npz!r}, allow_pickle=False)
keys = sorted(data.files, key=lambda k: int(k[1:]))
extra = json.loads({kw_json!r})
mp = {model_path!r}
if mp is None:
    m = models.CellposeModel(gpu=True)
else:
    m = models.CellposeModel(gpu=True, pretrained_model=mp)

results = {{}}
for k in keys:
    img = data[k]
    t0 = time.time()
    out = m.eval(img, augment={tta}, **extra)
    elapsed = time.time() - t0
    masks = out[0]
    results[k + "_mask"] = masks.astype(np.int32)
    results[k + "_t"] = np.array(elapsed)
    print(f"OK {{k}} cells={{int(masks.max())}} px={{int((masks>0).sum())}} t={{elapsed:.2f}}", flush=True)
np.savez_compressed({out_npz!r}, **results)
print("ALL_DONE")
"""
        cmd = ["conda", "run", "-n", env, "python", "-c", py]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=3600)
        if "ALL_DONE" not in res.stdout:
            return None, (res.stdout, res.stderr)
        z = np.load(out_npz, allow_pickle=False)
        results = []
        for i in range(len(frames_list)):
            mask = z[f"f{i}_mask"]
            elapsed = float(z[f"f{i}_t"])
            results.append({
                "n_cells": int(mask.max()),
                "mask_px": int((mask > 0).sum()),
                "mask": mask,
                "elapsed_s": elapsed,
            })
        return results, None


def overlay_image(frame_uint8, mask_int32):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(frame_uint8, cmap="gray", vmin=0, vmax=255)
    if mask_int32.max() > 0:
        ncol = max(int(mask_int32.max()), 1) + 1
        cmap = plt.cm.tab20(np.linspace(0, 1, ncol))
        cmap[0] = (0, 0, 0, 0)
        ax.imshow(mask_int32, cmap=ListedColormap(cmap), alpha=0.45)
    ax.axis("off")
    return fig


def build_overlay_grid(rows, configs):
    """Grid: rows = configs, cols = conditions, cell = overlay PNG."""
    cond_order = ["WT", "KO", "GOF", "Y1", "DMSO"]
    fig, axes = plt.subplots(len(configs), len(cond_order),
                             figsize=(len(cond_order) * 3.5,
                                       len(configs) * 3.5))
    if len(configs) == 1:
        axes = axes.reshape(1, -1)
    for ci, (label, *_) in enumerate(configs):
        for ji, cond in enumerate(cond_order):
            ax = axes[ci, ji]
            png = os.path.join(OVERLAY_DIR, f"{label}_{cond}.png")
            if os.path.exists(png):
                img = plt.imread(png)
                ax.imshow(img)
            n_cells = next(
                (r["n_cells"] for r in rows
                 if r["config"] == label and r["condition"] == cond), "?")
            ax.set_title(f"{label}\n{cond}  n={n_cells}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "overlay_grid.png")
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


def write_csvs(rows):
    long_csv = os.path.join(OUT_DIR, "comparison.csv")
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["config", "position", "condition",
                            "frame_index", "n_cells", "mask_px",
                            "elapsed_s"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "mask"})
    print(f"Wrote {long_csv}")

    by_cfg_csv = os.path.join(OUT_DIR, "by_config.csv")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    with open(by_cfg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_frames", "mean_cells", "detection_rate",
                    "mean_mask_px", "mean_elapsed_s"])
        for cfg, rs in by_cfg.items():
            ncells = [r["n_cells"] for r in rs]
            w.writerow([cfg, len(rs),
                        round(float(np.mean(ncells)), 2),
                        round(sum(1 for c in ncells if c > 0) / len(rs), 2),
                        round(float(np.mean([r["mask_px"] for r in rs])), 0),
                        round(float(np.mean([r["elapsed_s"] for r in rs])), 1)])
    print(f"Wrote {by_cfg_csv}")

    by_cc_csv = os.path.join(OUT_DIR, "by_config_condition.csv")
    by_cc = {}
    for r in rows:
        by_cc.setdefault((r["config"], r["condition"]), []).append(r)
    with open(by_cc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "condition", "n_frames", "mean_cells",
                    "detection_rate"])
        for (cfg, cond), rs in sorted(by_cc.items()):
            ncells = [r["n_cells"] for r in rs]
            w.writerow([cfg, cond, len(rs),
                        round(float(np.mean(ncells)), 2),
                        round(sum(1 for c in ncells if c > 0) / len(rs), 2)])
    print(f"Wrote {by_cc_csv}")


def write_markdown(rows, configs):
    md = os.path.join(OUT_DIR, "report.md")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    with open(md, "w") as f:
        f.write("# Ignasi recordings — model comparison\n\n")
        f.write("## Per-config aggregate\n\n")
        f.write("| Config | N frames | Mean cells | Detection rate | "
                "Mean elapsed (s) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for cfg, rs in by_cfg.items():
            ncells = [r["n_cells"] for r in rs]
            f.write(f"| {cfg} | {len(rs)} | "
                    f"{np.mean(ncells):.2f} | "
                    f"{sum(1 for c in ncells if c > 0) / len(rs):.2f} | "
                    f"{np.mean([r['elapsed_s'] for r in rs]):.1f} |\n")
        f.write("\n## Per-config × condition\n\n")
        cond_order = ["WT", "KO", "GOF", "Y1", "DMSO"]
        f.write("| Config | " + " | ".join(cond_order) + " |\n")
        f.write("|---" * (len(cond_order) + 1) + "|\n")
        for cfg, rs in by_cfg.items():
            line = [cfg]
            for c in cond_order:
                vals = [r["n_cells"] for r in rs if r["condition"] == c]
                line.append(f"{np.mean(vals):.1f}" if vals else "—")
            f.write("| " + " | ".join(line) + " |\n")
        f.write("\n## Overlay grid\n\n![](overlay_grid.png)\n")
    print(f"Wrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--frames-per-recording", type=int, default=3,
                    help="frames per recording sampled (default 3 = "
                         "early/middle/late)")
    ap.add_argument("--configs", nargs="+",
                    help="restrict to specific config labels")
    args = ap.parse_args()

    os.makedirs(OVERLAY_DIR, exist_ok=True)

    configs = DEFAULT_CONFIGS
    if args.configs:
        keep = set(args.configs)
        configs = [c for c in configs if c[0] in keep]
    print(f"Configs: {[c[0] for c in configs]}\n")

    # For each representative recording: load + flat-field-correct
    # the sampled frames once, then iterate configs.
    sampled = {}  # cond -> {position, frames: [(idx, uint8)]}
    for cond, pos in REPRESENTATIVE_RECORDINGS.items():
        tif = next((p for p in glob.glob(os.path.join(args.src, "*.ome.tif"))
                    if pos in os.path.basename(p)), None)
        if tif is None:
            print(f"  [skip {cond}] no recording matching {pos}")
            continue
        meta = parse_metadata(tif)
        n = meta["n_frames"]
        nch = meta["n_channels"]
        idxs = [int(round(q * (n - 1)))
                for q in np.linspace(0.05, 0.95,
                                      args.frames_per_recording)]
        frames = [(i, to_uint8(read_frame(tif, i, 0, nch))) for i in idxs]
        sampled[cond] = {"position": pos, "frames": frames}
        print(f"  loaded {pos} ({cond}): frames {idxs}")

    rows = []
    print(f"\nRunning {len(configs)} configs × "
          f"{sum(len(s['frames']) for s in sampled.values())} frames...\n")
    for label, env, model_path, tta, extra in configs:
        print(f"\n=== config: {label} (env={env}, "
              f"model={os.path.basename(model_path) if model_path else 'cpsam-base'}, "
              f"tta={tta}) ===")
        # Flatten to ordered list, run, then unpack
        flat_frames, flat_meta = [], []
        for cond, info in sampled.items():
            for fi, img in info["frames"]:
                flat_frames.append(img)
                flat_meta.append((cond, info["position"], fi))
        results, err = run_config_on_frames(
            label, env, model_path, tta, extra, flat_frames)
        if results is None:
            print(f"  [FAIL] {err[1][-400:] if err else 'unknown'}")
            continue
        for (cond, pos, fi), res in zip(flat_meta, results):
            rows.append({
                "config": label,
                "position": pos,
                "condition": cond,
                "frame_index": fi,
                "n_cells": res["n_cells"],
                "mask_px": res["mask_px"],
                "elapsed_s": round(res["elapsed_s"], 2),
                "mask": res["mask"],
            })
            print(f"  {cond:<5} {pos:<12} f{fi}: "
                  f"cells={res['n_cells']} "
                  f"px={res['mask_px']} "
                  f"t={res['elapsed_s']:.1f}s")
        # Save 1 overlay per (config, condition) — middle frame
        for cond, info in sampled.items():
            mid_idx = info["frames"][len(info["frames"]) // 2][0]
            mid_img = info["frames"][len(info["frames"]) // 2][1]
            mask = next(
                (r["mask"] for r in rows
                 if r["config"] == label and r["condition"] == cond
                 and r["frame_index"] == mid_idx), None)
            if mask is None:
                continue
            fig = overlay_image(mid_img, mask)
            fig.savefig(os.path.join(OVERLAY_DIR, f"{label}_{cond}.png"),
                        dpi=70, bbox_inches="tight")
            plt.close(fig)

    if not rows:
        print("\nNo results — nothing to write.")
        return
    write_csvs(rows)
    build_overlay_grid(rows, configs)
    write_markdown(rows, configs)


if __name__ == "__main__":
    main()
