"""Generate side-by-side DIC + Cy5 panels for IC295 frame inspection.

Lets the user judge:
  1. Cy5 probe — what does the actin distribution look like?
     (SiR-actin: cortex + stress fibres bright; phalloidin-equivalent
     pattern; LifeAct: similar but uses a different fluorophore)
  2. Are there cells visible in DIC but NOT in Cy5?
     (un-stained cells vs debris)
  3. Photobleaching across the timecourse?

Picks 1 recording per condition (WT/KO/GOF/Y1/DMSO/OT) × 3 frames
per recording (early/middle/late) and writes a multi-page PDF + a
PNG contact sheet.

No cpsam calls — pure I/O + matplotlib so we don't compete with the
GPU pipeline running in another env.
"""
import os
import sys
import re
import glob

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import setup_imports  # noqa
setup_imports()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

SRC = "/Volumes/GeorgeDrive/ignasi/IC295"
OUT_DIR = "data/ic295_inspection"
N_CHANNELS = 3
CY5_CH = 0
DIC_CH = 1
FRAMES = [5, 48, 91]


def parse_condition(name):
    m = re.search(r"_(Pos\d+)-(WT|KO|GOF|Y1|DMSO|OT)",
                   name, re.I)
    return (m.group(1), m.group(2).upper()) if m else ("?", "?")


def to_u8_pct(arr, low=1, high=99):
    p1, p99 = np.percentile(arr, [low, high])
    return np.clip((arr.astype(np.float32) - p1)
                   / max(p99 - p1, 1e-6) * 255,
                   0, 255).astype(np.uint8)


def to_u8_dic(raw_uint16, sigma=80):
    from scipy.ndimage import gaussian_filter
    f = raw_uint16.astype(np.float32)
    bg = gaussian_filter(f, sigma=sigma)
    bg = np.where(bg < 1, 1, bg)
    flat = f / bg - 1.0
    p1, p99 = np.percentile(flat, [1, 99])
    return np.clip((flat - p1) / max(p99 - p1, 1e-6) * 255,
                   0, 255).astype(np.uint8)


def to_u8_cy5(raw_uint16):
    """Cy5 is bright on near-zero background — use p1/p99.5
    to keep cells visible without saturating."""
    return to_u8_pct(raw_uint16, low=1, high=99.5)


def composite(dic_u8, cy5_u8, cy5_alpha=0.65):
    """Gray DIC with red Cy5 overlay — makes Cy5+ vs Cy5- cells
    immediately obvious."""
    rgb = np.stack([dic_u8, dic_u8, dic_u8], axis=-1).astype(np.float32)
    cy5_norm = cy5_u8.astype(np.float32) / 255.0
    rgb[..., 0] = rgb[..., 0] * (1 - cy5_alpha * cy5_norm) + 255 * cy5_alpha * cy5_norm
    rgb[..., 1] = rgb[..., 1] * (1 - cy5_alpha * cy5_norm * 0.3)
    rgb[..., 2] = rgb[..., 2] * (1 - cy5_alpha * cy5_norm * 0.3)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_recording_page(pdf, tif_path):
    base = os.path.basename(tif_path).replace(".ome.tif", "")
    pos, cond = parse_condition(base)
    print(f"  rendering {pos} ({cond})")
    with tifffile.TiffFile(tif_path) as tf:
        n_pages = len(tf.pages)
        n_frames = n_pages // N_CHANNELS
        for fi in FRAMES:
            if fi >= n_frames:
                continue
            cy5_raw = tf.pages[fi * N_CHANNELS + CY5_CH].asarray()
            dic_raw = tf.pages[fi * N_CHANNELS + DIC_CH].asarray()
            dic_u8 = to_u8_dic(dic_raw)
            cy5_u8 = to_u8_cy5(cy5_raw)
            comp = composite(dic_u8, cy5_u8)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
            axes[0].imshow(dic_u8, cmap="gray")
            axes[0].set_title(f"DIC (flat-field) — frame {fi}")
            axes[1].imshow(cy5_u8, cmap="gray")
            axes[1].set_title(f"Cy5 (actin) — p1/p99.5 — "
                              f"max={cy5_raw.max()} p99.9="
                              f"{int(np.percentile(cy5_raw, 99.9))}")
            axes[2].imshow(comp)
            axes[2].set_title("Composite — Cy5 on DIC")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{pos} ({cond})  —  frame {fi}/97  —  "
                         f"{tif_path.split('/')[-1]}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, dpi=70, bbox_inches="tight")
            plt.close(fig)


def render_zoom_page(pdf, tif_path, frame_idx=48,
                     crops=((400, 400), (1200, 1200))):
    """Per recording, also include a zoomed 600×600 view at two
    locations — easier to inspect cell-level detail than the full
    2048×2048."""
    base = os.path.basename(tif_path).replace(".ome.tif", "")
    pos, cond = parse_condition(base)
    with tifffile.TiffFile(tif_path) as tf:
        cy5_raw = tf.pages[frame_idx * N_CHANNELS + CY5_CH].asarray()
        dic_raw = tf.pages[frame_idx * N_CHANNELS + DIC_CH].asarray()
        dic_u8 = to_u8_dic(dic_raw)
        cy5_u8 = to_u8_cy5(cy5_raw)
        comp = composite(dic_u8, cy5_u8)

    fig, axes = plt.subplots(len(crops), 3, figsize=(15, 5 * len(crops)))
    if len(crops) == 1:
        axes = axes.reshape(1, -1)
    for r, (cy, cx) in enumerate(crops):
        h = 300
        crop_dic = dic_u8[cy - h:cy + h, cx - h:cx + h]
        crop_cy5 = cy5_u8[cy - h:cy + h, cx - h:cx + h]
        crop_comp = comp[cy - h:cy + h, cx - h:cx + h]
        axes[r, 0].imshow(crop_dic, cmap="gray")
        axes[r, 0].set_title(f"DIC zoom @ ({cy},{cx})")
        axes[r, 1].imshow(crop_cy5, cmap="gray")
        axes[r, 1].set_title("Cy5 zoom")
        axes[r, 2].imshow(crop_comp)
        axes[r, 2].set_title("Composite zoom")
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle(f"{pos} ({cond}) — zoom views, frame {frame_idx}",
                 fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # One representative recording per condition
    by_cond = {}
    for tif in sorted(glob.glob(os.path.join(SRC, "*.ome.tif"))):
        _, cond = parse_condition(os.path.basename(tif))
        by_cond.setdefault(cond, []).append(tif)
    chosen = [v[0] for v in by_cond.values()]

    print(f"Rendering {len(chosen)} recordings × "
          f"{len(FRAMES)} frames + zooms → {OUT_DIR}/")

    pdf_path = os.path.join(OUT_DIR, "channel_inspection.pdf")
    with PdfPages(pdf_path) as pdf:
        for tif in chosen:
            render_recording_page(pdf, tif)
            render_zoom_page(pdf, tif, frame_idx=48)
    print(f"\nWrote {pdf_path}")

    # Also write a single-page contact sheet summary
    fig, axes = plt.subplots(len(chosen), 3,
                              figsize=(15, 5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)
    for ri, tif in enumerate(chosen):
        base = os.path.basename(tif).replace(".ome.tif", "")
        pos, cond = parse_condition(base)
        with tifffile.TiffFile(tif) as tf:
            cy5_raw = tf.pages[48 * N_CHANNELS + CY5_CH].asarray()
            dic_raw = tf.pages[48 * N_CHANNELS + DIC_CH].asarray()
            dic_u8 = to_u8_dic(dic_raw)
            cy5_u8 = to_u8_cy5(cy5_raw)
            comp = composite(dic_u8, cy5_u8)
        axes[ri, 0].imshow(dic_u8, cmap="gray")
        axes[ri, 1].imshow(cy5_u8, cmap="gray")
        axes[ri, 2].imshow(comp)
        axes[ri, 0].set_ylabel(f"{pos}\n{cond}", rotation=0,
                                labelpad=40, fontsize=11, va="center")
        for c in range(3):
            axes[ri, c].set_xticks([])
            axes[ri, c].set_yticks([])
        axes[ri, 0].set_title("DIC" if ri == 0 else "")
        axes[ri, 1].set_title("Cy5" if ri == 0 else "")
        axes[ri, 2].set_title("Composite (Cy5 red on DIC)" if ri == 0 else "")
    fig.suptitle("IC295 — DIC vs Cy5 per condition  (frame 48/97)",
                 fontsize=13)
    fig.tight_layout()
    contact_png = os.path.join(OUT_DIR, "contact_sheet.png")
    fig.savefig(contact_png, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {contact_png}")


if __name__ == "__main__":
    main()
