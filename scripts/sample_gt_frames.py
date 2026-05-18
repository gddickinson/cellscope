"""Sample frames from new Ignasi recordings for GT labelling.

For each .ome.tif under --src, picks N frames spread across the
timecourse and writes them as uint8 PNGs at --out/candidates/.
A contact sheet + manifest CSV + LABELLING.md are also written so the
user can scan the sample, then label each candidate by hand.

Default pick: frames at the 5/35/65/95 % time points → 4 per recording.
At 16 recordings × 4 frames = 64 candidates, this is realistic to
hand-label in a single session.

Multi-channel mode (--multichannel): for IC295-style data with both
DIC and a fluorescence channel, saves three PNGs per candidate:
  <name>_dic.png         labelled here (canonical boundary)
  <name>_cy5.png         reference (is this a viable cell?)
  <name>_composite.png   overlay reference

Usage (single channel — IC293):
  python scripts/sample_gt_frames.py \\
      --src /Users/george/Desktop/ignasi_cellscope_test_data \\
      --out data/ignasi_new_gt \\
      --n-per-recording 4

Usage (multi-channel — IC295):
  python scripts/sample_gt_frames.py \\
      --src /Volumes/GeorgeDrive/ignasi/IC295 \\
      --out data/ic295_gt \\
      --n-per-recording 2 \\
      --multichannel
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_COND_RE = re.compile(r"_(Pos\d+)-(WT|KO|GOF|Y1|DMSO|OT)", re.I)


def parse_condition(filename):
    m = _COND_RE.search(filename)
    if not m:
        return ("unknown", "unknown")
    return (m.group(1), m.group(2).upper())


def flat_field_correct(frame_uint16, sigma=80):
    """Subtract Gaussian-blurred background to remove illumination
    vignette / shading. Returns float32 in roughly [-1, +1] range
    (later rescaled to uint8 by to_uint8)."""
    from scipy.ndimage import gaussian_filter
    f = frame_uint16.astype(np.float32)
    bg = gaussian_filter(f, sigma=sigma)
    # Avoid divide-by-zero
    bg = np.where(bg < 1, 1, bg)
    return f / bg - 1.0


def to_uint8(frame, flatten=True):
    """Per-frame uint8 conversion. If flatten=True, applies
    flat-field correction first to remove illumination shading."""
    if flatten and frame.dtype != np.uint8:
        flat = flat_field_correct(frame)
        p1, p99 = np.percentile(flat, [1, 99])
        return np.clip((flat - p1) / max(p99 - p1, 1e-6) * 255,
                       0, 255).astype(np.uint8)
    if frame.dtype == np.uint8:
        return frame
    p1, p99 = np.percentile(frame, [1, 99])
    return np.clip((frame.astype(np.float32) - p1)
                   / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)


def pick_frames(n_frames, n_pick):
    """Return n_pick frame indices spread across the timecourse."""
    if n_frames <= n_pick:
        return list(range(n_frames))
    # Quantile spacing (skip very-first and very-last frames)
    qs = np.linspace(0.05, 0.95, n_pick)
    return [int(round(q * (n_frames - 1))) for q in qs]


def read_frame(tif_path, frame_idx, channel=0, n_channels=1):
    """Read a single (T, ch, H, W) → 2D uint8 frame."""
    page_idx = frame_idx * n_channels + channel
    with tifffile.TiffFile(tif_path) as tf:
        page = tf.pages[page_idx]
        arr = page.asarray()
    return to_uint8(arr)


def detect_n_channels(tif_path):
    """Heuristic: many micromanager OMEs interleave channels per frame.

    We use the metadata sidecar if present, else default to 1.
    """
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            txt = f.read()
        m = re.search(r'"Channels"\s*:\s*(\d+)', txt)
        if m:
            return int(m.group(1))
    return 1


def detect_n_frames(tif_path, n_channels):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            txt = f.read()
        m = re.search(r'"Frames"\s*:\s*(\d+)', txt)
        if m:
            return int(m.group(1))
    # Fallback: pages / channels
    with tifffile.TiffFile(tif_path) as tf:
        return len(tf.pages) // max(n_channels, 1)


def detect_pixel_size_um(tif_path):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        txt = f.read()
    m = re.search(r'"PixelSize_um"\s*:\s*([\d.]+)', txt)
    return float(m.group(1)) if m else None


def detect_interval_min(tif_path):
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        txt = f.read()
    m = re.search(r'"Interval_ms"\s*:\s*(\d+)', txt)
    return float(m.group(1)) / 60000 if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder of .ome.tif recordings")
    ap.add_argument("--out", default="data/ignasi_new_gt",
                    help="output root (default data/ignasi_new_gt)")
    ap.add_argument("--n-per-recording", type=int, default=4,
                    help="frames to sample per recording (default 4)")
    ap.add_argument("--channel", type=int, default=0,
                    help="channel index for phase/DIC (default 0)")
    ap.add_argument("--multichannel", action="store_true",
                    help="multi-channel mode (e.g. IC295 with SiR-actin "
                         "Cy5 + DIC). Writes <name>_dic.png + _cy5.png "
                         "+ _composite.png per candidate.")
    ap.add_argument("--dic-channel", type=int, default=1,
                    help="(multichannel) DIC channel index (default 1)")
    ap.add_argument("--fluo-channel", type=int, default=0,
                    help="(multichannel) fluorescence channel index "
                         "(default 0)")
    args = ap.parse_args()

    tif_files = sorted(glob.glob(os.path.join(args.src, "*.ome.tif")))
    if not tif_files:
        print(f"No .ome.tif files in {args.src}")
        sys.exit(1)

    candidates_dir = os.path.join(args.out, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    manifest_path = os.path.join(args.out, "manifest.csv")
    rows = []
    print(f"Sampling {args.n_per_recording} frames × "
          f"{len(tif_files)} recordings → {candidates_dir}/\n")
    for tif_path in tif_files:
        base = os.path.basename(tif_path).replace(".ome.tif", "")
        pos, cond = parse_condition(base)
        n_channels = detect_n_channels(tif_path)
        n_frames = detect_n_frames(tif_path, n_channels)
        pixel_size = detect_pixel_size_um(tif_path)
        interval_min = detect_interval_min(tif_path)
        idxs = pick_frames(n_frames, args.n_per_recording)
        print(f"  {pos:<10} {cond:<6} ch={n_channels} "
              f"frames={n_frames} → picks {idxs}")
        for fi in idxs:
            if args.multichannel:
                from core.multichannel import (
                    to_uint8_dic, to_uint8_fluorescence)
                # Read both channels uint16 then preprocess
                with tifffile.TiffFile(tif_path) as tf:
                    dic_raw = tf.pages[
                        fi * n_channels + args.dic_channel].asarray()
                    fluo_raw = tf.pages[
                        fi * n_channels + args.fluo_channel].asarray()
                dic_u8 = to_uint8_dic(dic_raw)
                cy5_u8 = to_uint8_fluorescence(fluo_raw)
                # Composite: gray DIC + red Cy5 overlay
                comp = np.stack([dic_u8] * 3, axis=-1).astype(np.float32)
                cy5n = cy5_u8.astype(np.float32) / 255.0
                comp[..., 0] = comp[..., 0] * (1 - 0.65 * cy5n) + 255 * 0.65 * cy5n
                comp[..., 1] = comp[..., 1] * (1 - 0.65 * cy5n * 0.3)
                comp[..., 2] = comp[..., 2] * (1 - 0.65 * cy5n * 0.3)
                comp = np.clip(comp, 0, 255).astype(np.uint8)

                stem = f"{pos}_{cond}_f{fi:03d}"
                dic_name = f"{stem}_dic.png"
                cy5_name = f"{stem}_cy5.png"
                comp_name = f"{stem}_composite.png"
                plt.imsave(os.path.join(candidates_dir, dic_name),
                            dic_u8, cmap="gray", vmin=0, vmax=255)
                plt.imsave(os.path.join(candidates_dir, cy5_name),
                            cy5_u8, cmap="gray", vmin=0, vmax=255)
                plt.imsave(os.path.join(candidates_dir, comp_name),
                            comp)
                rows.append({
                    "candidate": dic_name,
                    "candidate_cy5": cy5_name,
                    "candidate_composite": comp_name,
                    "source": os.path.basename(tif_path),
                    "position": pos,
                    "condition": cond,
                    "frame_index": fi,
                    "n_frames": n_frames,
                    "shape": "x".join(str(s) for s in dic_u8.shape),
                    "pixel_size_um": (f"{pixel_size:.4f}"
                                      if pixel_size else ""),
                    "interval_min": (f"{interval_min:.2f}"
                                     if interval_min else ""),
                })
            else:
                frame = read_frame(tif_path, fi, args.channel, n_channels)
                out_name = f"{pos}_{cond}_f{fi:03d}.png"
                out_path = os.path.join(candidates_dir, out_name)
                plt.imsave(out_path, frame, cmap="gray", vmin=0, vmax=255)
                rows.append({
                    "candidate": out_name,
                    "source": os.path.basename(tif_path),
                    "position": pos,
                    "condition": cond,
                    "frame_index": fi,
                    "n_frames": n_frames,
                    "shape": "x".join(str(s) for s in frame.shape),
                    "pixel_size_um": (f"{pixel_size:.4f}"
                                      if pixel_size else ""),
                    "interval_min": (f"{interval_min:.2f}"
                                     if interval_min else ""),
                })

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} candidates + {manifest_path}")

    # Build contact-sheet preview with CLAHE so cells are visible
    # despite the strong vignette / illumination shading. The
    # stored candidate PNGs are NOT CLAHE-enhanced — they're what
    # the pipeline sees. Adjust contrast in cellpose GUI to see
    # cells when labelling.
    try:
        import cv2
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    except ImportError:
        clahe = None
    contact_path = os.path.join(args.out, "contact_sheet.png")
    n = len(rows)
    cols = min(args.n_per_recording, 4)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols,
                             figsize=(cols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)
    for ax, r in zip(axes, rows):
        # Multichannel: prefer the composite for the contact sheet
        # since it shows cells + actin signal at once. Single-channel:
        # use the candidate.
        preview_name = r.get("candidate_composite", r["candidate"])
        img = plt.imread(os.path.join(candidates_dir, preview_name))
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 \
                else img.astype(np.uint8)
        if img.ndim == 3 and "composite" not in preview_name:
            img = img[..., 0]
            if clahe is not None:
                img = clahe.apply(img)
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)  # composite in colour
        ax.set_title(f"{r['position']} {r['condition']} f{r['frame_index']}",
                     fontsize=8)
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(contact_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote contact sheet (CLAHE-enhanced for visibility): "
          f"{contact_path}")

    # Labelling instructions — pick template based on mode
    instructions = os.path.join(args.out, "LABELLING.md")
    template = (_LABELLING_INSTRUCTIONS_MULTICHANNEL
                if args.multichannel else _LABELLING_INSTRUCTIONS)
    with open(instructions, "w") as f:
        f.write(template.format(
            n=len(rows), candidates_dir=candidates_dir,
            manifest=manifest_path,
            contact_sheet=contact_path,
            labels_dir=os.path.join(args.out, "labels")))
    print(f"Wrote labelling instructions: {instructions}")


_LABELLING_INSTRUCTIONS = """\
# GT labelling instructions

You have **{n} candidate frames** under `{candidates_dir}/` ready to
be hand-labelled as ground truth. The contact sheet at
`{contact_sheet}` shows all candidates at a glance.

## A note on preprocessing

The new Ignasi recordings have severe illumination shading (a
strong vignette gradient from corner to corner). The candidates are
**flat-field corrected** (background subtracted via Gaussian blur,
σ=80 px) so cells are visible. This means:

* Your masks are valid for the *flat-field-corrected* image.
* When evaluating the pipeline, the same correction must be applied
  before passing the frame to cpsam — otherwise IoU numbers will be
  meaningless. The benchmark wrapper handles this automatically;
  don't bypass it by feeding raw `.ome.tif` data through.
* If you want to look at the raw image to compare, the source TIFFs
  are at `/Users/george/Desktop/ignasi_cellscope_test_data/`.

## Pick your tool

### Option 1 — cellpose GUI (recommended)

```bash
conda activate cellpose4
python -m cellpose
```

* File → Open the candidate PNG.
* Use the brush / outline tools to draw cell boundaries.
  - Use the `+` cell button to start a new instance
  - Each cell becomes a different colour
* File → Save → "Save outlines as PNG" (`<name>_masks.png` in the
  same folder as the source).

### Option 2 — cellscope mask editor

```bash
python main_editor.py
```

Open each candidate PNG and use the brush/eraser. Saves as int32
multi-cell `*_masks.png`.

## Where to save labels

Save each `<name>_masks.png` next to its source PNG inside
`{candidates_dir}/`. The `bench_cpsam_dic.py` evaluator already
recognises this PNG-pair convention.

(If you'd rather keep them separate, mirror the candidate filenames
into `{labels_dir}/`. The evaluator can be pointed at either folder.)

## After labelling

Quick benchmark of the current model:

```bash
conda run -n cellpose4 python scripts/bench_cpsam_dic.py \\
    --model data/models/cpsam_dic \\
    --test-dir {candidates_dir} \\
    --out results/ignasi_new_eval/cpsam_dic.json \\
    --per-genotype 50
```

Then write a side-by-side report by condition (WT / KO / GOF / Y1 /
DMSO) — see the existing `scripts/piezo1_comparison.py` for an
example of grouping by condition.
"""


_LABELLING_INSTRUCTIONS_MULTICHANNEL = """\
# GT labelling instructions — multi-channel (DIC + SiR-actin Cy5)

You have **{n} candidate frames** under `{candidates_dir}/` ready to
be hand-labelled. The contact sheet at `{contact_sheet}` shows all
candidates at a glance (composite view: Cy5 in red on DIC).

Per candidate frame there are three PNGs:

| File | Use |
|---|---|
| `<name>_dic.png` | **Label this** — the canonical mask boundary |
| `<name>_cy5.png` | Reference — SiR-actin signal, "is this a cell?" |
| `<name>_composite.png` | Reference — Cy5 red overlaid on DIC gray |

## Workflow

1. Open `<name>_composite.png` for context (which features are real
   cells vs debris) — anything WITHOUT red overlay is likely debris
   and should NOT be labelled.
2. Open `<name>_dic.png` in the cellpose GUI:
   ```bash
   conda activate cellpose4
   python -m cellpose
   ```
   File → Open → pick the `_dic.png` candidate.
3. Trace each REAL cell (one with Cy5 signal in the composite). Use
   the brush / outline tools; press `+` to start a new cell.
4. File → Save → "Save outlines as PNG" — produces
   `<name>_dic_masks.png` next to the source.

## Don't label

* DIC features without any Cy5 signal in the composite — these are
  dust / debris.
* Faint Cy5 patches without a clear DIC outline — these are
  fluorescence artefacts (rare).

## After labelling

```bash
conda run -n cellpose4 python scripts/bench_multichannel.py \\
    --candidates {candidates_dir} \\
    --out results/ic295_eval/cpsam_base.json
```

(`bench_multichannel.py` to be written; will use the same
`<name>_dic.png` + `<name>_dic_masks.png` PNG-pair convention as the
single-channel evaluator, plus runs the multi-channel pipeline so we
can quantify the AND-fusion benefit vs DIC-only.)
"""


if __name__ == "__main__":
    main()
