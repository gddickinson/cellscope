"""Smoke-test new Ignasi recordings — verify load + cpsam detection.

For every .ome.tif under --src:
  * read metadata (channels, frames, pixel size, interval, condition)
  * read the first/middle/last frame as uint8
  * run cpsam_dic on the middle frame (in cellpose4 env via subprocess)
  * report cells detected + total mask area
  * save a 1×3 preview PNG (raw frames) + 1×3 overlay PNG (with cpsam
    contours) for visual sanity

Outputs:
  results/ignasi_new_test/
    summary.csv            per-recording stats
    report.md              human-readable summary
    previews/<pos>.png     raw frame triptych (no inference needed)
    previews/<pos>_pred.png  cpsam-overlay triptych

Why subprocess: cpsam needs cellpose 4.1.1 (cellpose4 env). This
script runs in the default cellpose env so it can invoke other
cellscope utilities, then shells out per recording for inference.
"""
import argparse
import csv
import glob
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

OUT_DIR = "results/ignasi_new_test"
PREVIEW_DIR = os.path.join(OUT_DIR, "previews")


_COND_RE = re.compile(r"_(Pos\d+)-(WT|KO|GOF|Y1|DMSO)", re.I)


def parse_condition(filename):
    m = _COND_RE.search(filename)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def to_uint8(frame):
    if frame.dtype == np.uint8:
        return frame
    p1, p99 = np.percentile(frame, [1, 99])
    return np.clip((frame.astype(np.float32) - p1)
                   / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)


def parse_metadata(tif_path):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    out = {"n_channels": 1, "n_frames": None,
           "pixel_size_um": None, "interval_min": None}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            txt = f.read()
        if (m := re.search(r'"Channels"\s*:\s*(\d+)', txt)):
            out["n_channels"] = int(m.group(1))
        if (m := re.search(r'"Frames"\s*:\s*(\d+)', txt)):
            out["n_frames"] = int(m.group(1))
        if (m := re.search(r'"PixelSize_um"\s*:\s*([\d.]+)', txt)):
            out["pixel_size_um"] = float(m.group(1))
        if (m := re.search(r'"Interval_ms"\s*:\s*(\d+)', txt)):
            out["interval_min"] = float(m.group(1)) / 60000
    # Fallback: divide tif page count by channel count
    if out["n_frames"] is None:
        with tifffile.TiffFile(tif_path) as tf:
            out["n_frames"] = max(1, len(tf.pages) // out["n_channels"])
    return out


def read_frame(tif_path, frame_idx, channel, n_channels):
    page = frame_idx * n_channels + channel
    with tifffile.TiffFile(tif_path) as tf:
        return to_uint8(tf.pages[page].asarray())


def run_cpsam_on_frame(frame_uint8, cpsam_model_path):
    """Invoke cpsam in cellpose4 env on a single frame.

    Writes the frame to a temp PNG, calls a tiny inline python that
    runs cellpose, parses (n_cells, total_mask_px, mask_path) back.
    """
    with tempfile.TemporaryDirectory() as td:
        in_png = os.path.join(td, "frame.tif")
        out_npy = os.path.join(td, "mask.npy")
        tifffile.imwrite(in_png, frame_uint8)
        py = (
            "import numpy as np, tifffile\n"
            "from cellpose import models\n"
            f"img = tifffile.imread({in_png!r})\n"
            f"m = models.CellposeModel(gpu=True, "
            f"pretrained_model={cpsam_model_path!r})\n"
            "out = m.eval(img)\n"
            "masks = out[0]\n"
            f"np.save({out_npy!r}, masks)\n"
            "n = int(masks.max())\n"
            "px = int((masks > 0).sum())\n"
            "print(f'CPSAM_RESULT n={n} px={px}')\n"
        )
        cmd = ["conda", "run", "-n", "cellpose4", "python", "-c", py]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=240)
        elapsed = time.time() - t0
        if res.returncode != 0:
            return {
                "ok": False,
                "n_cells": None, "mask_px": None,
                "elapsed_s": elapsed,
                "error": res.stderr.strip().splitlines()[-1] if res.stderr
                else "unknown",
                "mask": None,
            }
        m = re.search(r"CPSAM_RESULT n=(\d+) px=(\d+)", res.stdout)
        if not m:
            return {"ok": False, "n_cells": None, "mask_px": None,
                    "elapsed_s": elapsed,
                    "error": "no result line",
                    "mask": None}
        mask = np.load(out_npy)
        return {
            "ok": True,
            "n_cells": int(m.group(1)),
            "mask_px": int(m.group(2)),
            "elapsed_s": elapsed,
            "error": "",
            "mask": mask,
        }


def save_triptych(frames, titles, out_path, masks=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, t in zip(axes, frames, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    if masks is not None:
        for ax, m in zip(axes, masks):
            if m is None:
                continue
            from matplotlib.colors import ListedColormap
            colors = plt.cm.tab20(np.linspace(0, 1, max(int(m.max()), 1) + 1))
            colors[0] = (0, 0, 0, 0)
            ax.imshow(m, cmap=ListedColormap(colors), alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cpsam-model",
                    default="data/models/cpsam_dic",
                    help="cpsam model path (default cpsam_dic)")
    ap.add_argument("--channel", type=int, default=0,
                    help="channel index for phase/DIC (default 0)")
    ap.add_argument("--no-detect", action="store_true",
                    help="skip cpsam inference (just preview the data)")
    args = ap.parse_args()

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    tif_files = sorted(glob.glob(os.path.join(args.src, "*.ome.tif")))
    if not tif_files:
        print(f"No .ome.tif under {args.src}")
        sys.exit(1)
    print(f"Testing {len(tif_files)} recordings\n")

    rows = []
    for tif_path in tif_files:
        base = os.path.basename(tif_path).replace(".ome.tif", "")
        pos, cond = parse_condition(base)
        meta = parse_metadata(tif_path)
        n = meta["n_frames"]
        ch = meta["n_channels"]
        print(f"-- {pos:<10} {cond:<6} ch={ch} n_frames={n}")
        try:
            frames = [
                read_frame(tif_path, 0, args.channel, ch),
                read_frame(tif_path, max(0, n // 2), args.channel, ch),
                read_frame(tif_path, max(0, n - 1), args.channel, ch),
            ]
        except Exception as e:
            print(f"  [FAIL] could not read frames: {e}")
            rows.append({
                "position": pos, "condition": cond,
                "n_frames": n, "n_channels": ch,
                "shape": "?", "ok_load": False,
                "ok_detect": False,
                "n_cells_mid": None, "mask_px_mid": None,
                "elapsed_s": None, "error": str(e),
            })
            continue
        h, w = frames[0].shape
        save_triptych(frames,
                      [f"frame 0 ({h}×{w})",
                       f"frame {n // 2}",
                       f"frame {n - 1}"],
                      os.path.join(PREVIEW_DIR, f"{pos}_raw.png"))

        det = {"ok": True, "n_cells": None, "mask_px": None,
               "elapsed_s": None, "error": "skipped"}
        if not args.no_detect:
            det = run_cpsam_on_frame(frames[1], args.cpsam_model)
            if det["ok"]:
                save_triptych(frames,
                              [f"frame 0", f"frame {n // 2}",
                               f"frame {n - 1}"],
                              os.path.join(PREVIEW_DIR,
                                           f"{pos}_pred.png"),
                              masks=[None, det["mask"], None])
                print(f"  cpsam: {det['n_cells']} cells, "
                      f"{det['mask_px']} px, {det['elapsed_s']:.1f}s")
            else:
                print(f"  [FAIL] cpsam: {det['error']}")
        rows.append({
            "position": pos, "condition": cond,
            "n_frames": n, "n_channels": ch,
            "shape": f"{h}x{w}",
            "ok_load": True,
            "ok_detect": det["ok"],
            "n_cells_mid": det["n_cells"],
            "mask_px_mid": det["mask_px"],
            "elapsed_s": (f"{det['elapsed_s']:.1f}"
                          if det["elapsed_s"] else ""),
            "error": det.get("error", ""),
            "pixel_size_um": meta["pixel_size_um"],
            "interval_min": meta["interval_min"],
        })

    # CSV
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    # Markdown report
    md_path = os.path.join(OUT_DIR, "report.md")
    with open(md_path, "w") as f:
        f.write("# Ignasi recordings — smoke test\n\n")
        f.write(f"Source: `{args.src}`\n")
        f.write(f"Model: `{args.cpsam_model}` (skipped if "
                f"`--no-detect`)\n\n")
        f.write("## Per-recording\n\n")
        f.write("| Position | Condition | Frames | Shape | "
                "Cells (mid) | Mask px | Time (s) | Status |\n")
        f.write("|---|---|---:|---|---:|---:|---:|---|\n")
        for r in rows:
            status = "OK" if r["ok_detect"] else "FAIL: " + (
                r.get("error", "") or "")[:60]
            f.write(f"| {r['position']} | {r['condition']} | "
                    f"{r['n_frames']} | {r['shape']} | "
                    f"{r['n_cells_mid'] if r['n_cells_mid'] is not None else '-'} | "
                    f"{r['mask_px_mid'] if r['mask_px_mid'] is not None else '-'} | "
                    f"{r['elapsed_s'] or '-'} | {status} |\n")
        f.write("\n## By condition (mean cells/frame on the mid frame)\n\n")
        by = {}
        for r in rows:
            if r["n_cells_mid"] is None:
                continue
            by.setdefault(r["condition"], []).append(r["n_cells_mid"])
        f.write("| Condition | n recordings | Mean cells | Range |\n")
        f.write("|---|---:|---:|---|\n")
        for cond, vals in sorted(by.items()):
            f.write(f"| {cond} | {len(vals)} | "
                    f"{np.mean(vals):.1f} | {min(vals)}–{max(vals)} |\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
