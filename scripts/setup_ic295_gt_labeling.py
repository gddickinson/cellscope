"""Set up an IC295 ground-truth labeling sandbox.

Picks 2 random recordings per condition (DMSO/GOF/KO/OT/WT/Y1) from
the combined IC295 + IC295_batch2 pool (66 recordings total, balanced
11 per condition). Uses seed=0 so the selection is reproducible.

For each chosen recording:
  - Symlinks the .ome.tif + _metadata.txt into a per-recording folder
  - Writes a JSON sidecar with um_per_px, time_interval_min, etc.
  - Writes GT_FRAMES.txt listing the frames the user should annotate
    (every 10th: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
  - Creates an empty subfolder `gt_masks/` for the user's labels

Output root: data/ic295_gt_full/
Top-level files:
  manifest.csv           — list of 12 recordings + condition + path
  contact_sheet.png      — frame 0 thumbnail of each recording
  LABELLING.md           — workflow doc

Usage:
  conda run -n cellpose python scripts/setup_ic295_gt_labeling.py
"""
import os
import csv
import sys
import json
import random
import logging
import numpy as np
import tifffile
import matplotlib.pyplot as plt

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

SOURCE_DIRS = [
    "/Volumes/GeorgeDrive/ignasi/IC295",
    "/Volumes/GeorgeDrive/ignasi/IC295_batch2",
]

OUT_ROOT = os.path.join(CELLSCOPE_ROOT, "data/ic295_gt_full")
SEED = 0
RECORDINGS_PER_CONDITION = 2
ANNOTATE_EVERY_N_FRAMES = 10
N_FRAMES_PER_RECORDING = 97  # known from inspection

# Pixel size + frame interval for IC295 (from existing manifests)
UM_PER_PX = 0.6523
TIME_INTERVAL_MIN = 10.0

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("setup_gt")


def parse_filename(path):
    """Extract Pos id and condition from an IC295 .ome.tif filename.

    Returns (pos_id, condition) where pos_id is "Pos14" and condition
    is "KO" / "WT" / etc. Returns (None, None) if pattern doesn't match.
    """
    base = os.path.basename(path)
    # Pattern: IC295__1_MMStack_Pos14-KO.ome.tif
    if "MMStack_" not in base:
        return None, None
    tail = base.split("MMStack_")[1]
    if "-" not in tail:
        return None, None
    pos_part, rest = tail.split("-", 1)
    cond_part = rest.split(".")[0]
    return pos_part, cond_part


def find_all_recordings():
    """Return list of (path, pos_id, condition) tuples."""
    out = []
    for src in SOURCE_DIRS:
        if not os.path.isdir(src):
            log.warning("Source missing: %s", src)
            continue
        for f in sorted(os.listdir(src)):
            if not f.endswith(".ome.tif"):
                continue
            pos, cond = parse_filename(f)
            if not pos or not cond:
                continue
            out.append((os.path.join(src, f), pos, cond))
    return out


def stratified_sample(records, k_per_cond, seed):
    """Pick k random recordings per condition. Returns sorted list."""
    rng = random.Random(seed)
    by_cond = {}
    for rec in records:
        by_cond.setdefault(rec[2], []).append(rec)
    chosen = []
    for cond in sorted(by_cond):
        pool = by_cond[cond]
        rng.shuffle(pool)
        chosen.extend(pool[:k_per_cond])
    return chosen


def thumb_from_tif(path, frame_idx=0, downscale=4):
    """Return a small uint8 preview of the DIC channel at frame_idx."""
    with tifffile.TiffFile(path) as tf:
        # 2 channels per frame (Cy5 + DIC), DIC = ch1
        page_idx = frame_idx * 2 + 1
        if page_idx >= len(tf.pages):
            page_idx = len(tf.pages) - 1
        raw = tf.pages[page_idx].asarray()
    # Downsample
    h, w = raw.shape
    h2, w2 = h // downscale, w // downscale
    small = raw[:h2 * downscale, :w2 * downscale].reshape(
        h2, downscale, w2, downscale).mean(axis=(1, 3))
    p1, p99 = np.percentile(small, [1, 99])
    norm = np.clip((small - p1) / max(p99 - p1, 1e-6), 0, 1)
    return (norm * 255).astype(np.uint8)


def setup_recording_folder(rec_path, out_dir):
    """Symlink the .ome.tif + metadata into out_dir, write JSON sidecar
    + GT_FRAMES.txt + empty gt_masks subfolder."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(rec_path)
    target_tif = os.path.join(out_dir, base)
    if os.path.lexists(target_tif):
        os.remove(target_tif)
    os.symlink(rec_path, target_tif)
    # Metadata sidecar
    meta_src = rec_path.replace(".ome.tif", "_metadata.txt")
    if os.path.exists(meta_src):
        meta_dst = os.path.join(out_dir, os.path.basename(meta_src))
        if os.path.lexists(meta_dst):
            os.remove(meta_dst)
        os.symlink(meta_src, meta_dst)
    # JSON sidecar (CellScope reads this)
    pos, cond = parse_filename(rec_path)
    json_path = os.path.join(
        out_dir, base.replace(".ome.tif", ".ome.json"))
    json_data = {
        "name": f"{pos}_{cond}",
        "um_per_px": UM_PER_PX,
        "time_interval_min": TIME_INTERVAL_MIN,
        "condition": cond,
        "cell_type": "keratinocyte (SiR-actin labelled)",
        "n_channels": 2,
        "channels": {"0": "Cy5 (F-actin)", "1": "DIC"},
        "source": rec_path,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    # GT_FRAMES.txt
    frames = list(range(0, N_FRAMES_PER_RECORDING,
                        ANNOTATE_EVERY_N_FRAMES))
    with open(os.path.join(out_dir, "GT_FRAMES.txt"), "w") as f:
        f.write("# Frames to annotate for ground truth.\n")
        f.write(f"# Every {ANNOTATE_EVERY_N_FRAMES}th frame from a "
                f"{N_FRAMES_PER_RECORDING}-frame recording.\n")
        f.write("# When you save masks, name them mask_F<frame>.png\n")
        f.write("# Save to: gt_masks/\n\n")
        for fi in frames:
            f.write(f"{fi}\n")
    # Empty gt_masks folder
    os.makedirs(os.path.join(out_dir, "gt_masks"), exist_ok=True)
    return frames


def render_contact_sheet(chosen, out_path):
    """3 cols × 4 rows = 12 tiles, one frame-0 DIC thumbnail per
    recording, labelled with Pos + condition."""
    fig, axes = plt.subplots(4, 3, figsize=(13, 17))
    for ax, (path, pos, cond) in zip(axes.flat, chosen):
        try:
            thumb = thumb_from_tif(path, frame_idx=0)
            ax.imshow(thumb, cmap="gray")
        except Exception as e:
            ax.text(0.5, 0.5, f"(failed: {e})",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{pos}-{cond}", fontsize=11)
        ax.axis("off")
    plt.suptitle(
        f"IC295 ground-truth sample (n={len(chosen)} recordings, "
        f"seed={SEED})", fontsize=14, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(out_path, dpi=85, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote contact sheet: %s", out_path)


def write_manifest(chosen, frames_per_rec, out_path):
    fields = ["pos", "condition", "source_path", "rec_dir",
              "n_frames_total", "frames_to_annotate"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for path, pos, cond in chosen:
            w.writerow({
                "pos": pos, "condition": cond, "source_path": path,
                "rec_dir": os.path.join(
                    OUT_ROOT, f"{pos}_{cond}"),
                "n_frames_total": N_FRAMES_PER_RECORDING,
                "frames_to_annotate": ",".join(
                    str(x) for x in frames_per_rec),
            })
    log.info("Wrote manifest: %s", out_path)


def write_labelling_doc(chosen, frames_per_rec, out_path):
    n_frames_each = len(frames_per_rec)
    n_total = n_frames_each * len(chosen)
    cond_summary = {}
    for _, _, c in chosen:
        cond_summary[c] = cond_summary.get(c, 0) + 1
    cond_str = ", ".join(f"{c}: {n}"
                          for c, n in sorted(cond_summary.items()))
    md = f"""# IC295 ground-truth labeling

## What you have

- **{len(chosen)} recordings**, sampled randomly with `seed={SEED}` so
  the selection is reproducible.
- **{n_frames_each} frames per recording** to annotate (every
  {ANNOTATE_EVERY_N_FRAMES}th frame): {frames_per_rec}.
- **Total: {n_total} annotated frames** across all 6 conditions.

## Conditions covered
{cond_str}

## Folder layout

```
data/ic295_gt_full/
├── manifest.csv            ← list of all 12 recordings
├── contact_sheet.png       ← thumbnail grid of frame 0 of each
├── LABELLING.md            ← this file
└── <Pos>_<Cond>/           ← one subfolder per recording
    ├── IC295__1_MMStack_<Pos>-<Cond>.ome.tif   (symlink)
    ├── IC295__1_MMStack_<Pos>-<Cond>_metadata.txt (symlink)
    ├── IC295__1_MMStack_<Pos>-<Cond>.ome.json     (sidecar with
    │                                               um_per_px etc.)
    ├── GT_FRAMES.txt       ← which frames to annotate
    └── gt_masks/           ← put your labelled masks here
```

## Workflow

1. Open the GUI:
   ```
   conda run -n cellpose python main_focused.py
   ```
2. **File → Open Recording**, pick one of the .ome.tif files (or drag
   it into the window).
3. When the channel chooser appears, pick **DIC = ch1, Fluo = ch0**.
4. Use the Frame slider to scrub through the recording and watch the
   cells move. This is the whole point of using full recordings — the
   temporal context makes it MUCH easier to identify which blobs are
   real cells vs debris.
5. For each frame listed in `GT_FRAMES.txt`, open the **Mask Editor**
   (Edit menu → Open Mask Editor) and label every visible cell.
6. **Save** masks as `mask_F<frame_idx>.png` (e.g. `mask_F30.png`)
   into the `gt_masks/` subfolder.

## Mask format

- Use **int32 PNGs** with one unique integer per cell (1, 2, 3, ...,
  background = 0). The mask editor saves this format by default.
- **Track identity matters.** Cell 1 in `mask_F0.png` should be the
  same physical cell as Cell 1 in `mask_F10.png` if it's still
  visible — that lets us evaluate tracking accuracy too, not just
  per-frame detection.
- If a cell first appears at F30, give it a fresh ID (e.g. cell 5)
  and use the same ID in F40, F50, etc. while it remains visible.

## Tips

- **Use the channel toggle** (DIC ⇄ Fluo) to confirm a faint DIC
  blob is actually a real cell with actin signal.
- **Use the playback** (Play button or arrow keys) to verify you're
  not labelling debris that doesn't move.
- **Don't worry about perfect boundaries** — what matters most for
  accuracy estimation is "did we detect this cell at all" and "did
  we keep the same ID across frames". Boundary IoU >0.5 is plenty.
- If a cell is undergoing mitosis / dying / detaching across the gap
  between two annotated frames (e.g. F30 → F40), that's fine — give
  the daughter cells fresh IDs starting at F40 and note it in
  `gt_masks/notes.txt`.

## When done

Once you've labelled at least one recording, ping me and I'll set up
an evaluation script that:
  - Compares the saved pipeline output against your masks
  - Computes per-frame detection precision/recall
  - Computes track-level identity preservation (TRA score)
  - Reports per-condition accuracy
"""
    with open(out_path, "w") as f:
        f.write(md)
    log.info("Wrote labelling doc: %s", out_path)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    log.info("Scanning %s …", SOURCE_DIRS)
    all_recs = find_all_recordings()
    log.info("Found %d recordings", len(all_recs))
    by_cond = {}
    for rec in all_recs:
        by_cond.setdefault(rec[2], []).append(rec)
    log.info("By condition: %s",
             {c: len(v) for c, v in sorted(by_cond.items())})

    chosen = stratified_sample(all_recs, RECORDINGS_PER_CONDITION,
                                SEED)
    log.info("Sampled %d recordings:", len(chosen))
    for path, pos, cond in chosen:
        log.info("  %s  %s  %s", cond, pos, os.path.basename(path))

    log.info("\nSetting up per-recording folders …")
    frames = None
    for path, pos, cond in chosen:
        rec_dir = os.path.join(OUT_ROOT, f"{pos}_{cond}")
        frames = setup_recording_folder(path, rec_dir)
        log.info("  %s", rec_dir)

    log.info("\nWriting top-level files …")
    write_manifest(chosen, frames,
                    os.path.join(OUT_ROOT, "manifest.csv"))
    write_labelling_doc(chosen, frames,
                         os.path.join(OUT_ROOT, "LABELLING.md"))
    log.info("\nRendering contact sheet …")
    render_contact_sheet(
        chosen, os.path.join(OUT_ROOT, "contact_sheet.png"))

    print()
    print("=" * 60)
    print(f"GT labeling sandbox ready at: {OUT_ROOT}")
    print(f"  {len(chosen)} recordings × {len(frames)} frames each "
          f"= {len(chosen) * len(frames)} annotated frames total")
    print()
    print("Next: open data/ic295_gt_full/LABELLING.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
