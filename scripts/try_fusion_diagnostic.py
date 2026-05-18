"""End-to-end smoke test of the diagnostic system.

Runs detect_hybrid_dic_multi with use_cy5_fusion=True on the DMSO_busy
demo, then writes the diagnostic figure showing per-channel contributions.

Output: /tmp/fluo_investigation/06_diagnostic_busy.png
        /tmp/fluo_investigation/06_diagnostic_summary.txt
"""
import os
import sys
import time
import numpy as np
import tifffile

CELLSCOPE_ROOT = "/Users/george/claude_test/cellscope"
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

OUT_DIR = "/tmp/fluo_investigation"
os.makedirs(OUT_DIR, exist_ok=True)

TIFF = ("data/examples/multichannel_DIC_Cy5_DMSO_busy/"
        "multichannel_DIC_Cy5_DMSO_busy.ome.tif")


def load_channels(tif_path):
    with tifffile.TiffFile(tif_path) as tf:
        n = len(tf.pages) // 2
        h, w = tf.pages[0].shape
        cy5 = np.empty((n, h, w), dtype=np.uint8)
        dic = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            cy5[i] = tf.pages[2 * i].asarray()
            dic[i] = tf.pages[2 * i + 1].asarray()
    return dic, cy5


def main():
    from core.hybrid_dic import detect_hybrid_dic_multi
    from core.fusion_diagnostics import (
        render_fusion_diagnostic, per_track_source_summary)

    print(f"Loading {TIFF} ...")
    dic, cy5 = load_channels(TIFF)
    n = len(dic)
    print(f"  loaded {n} frames")

    print("\nRunning detect_hybrid_dic_multi with Cy5 fusion ON ...")
    t0 = time.time()
    result = detect_hybrid_dic_multi(
        dic, progress_fn=None,
        min_area_px=200,
        use_preprocess=True,
        use_deepsea=True,
        use_retry=True,
        use_gap_fill=True,
        model_path="data/models/cellpose_dic",
        cy5_frames=cy5,
        use_cy5_fusion=True,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")
    print(f"  n_cy5_fusion_added = {result.get('n_cy5_fusion_added')}")
    print(f"  tracks = {len(result.get('tracks', []))}")

    tracks = result.get("tracks", [])
    summary = per_track_source_summary(tracks)
    print(f"  Track sources: {summary}")
    for i, t in enumerate(tracks[:10]):
        counts = t.get("fusion_source_pixel_counts", {})
        print(f"    Track {i + 1}: source={t.get('fusion_source')} "
              f"counts={counts}")

    if result.get("fusion_source_stack") is None:
        print("  no fusion_source_stack — diagnostic skipped")
        return

    out = f"{OUT_DIR}/06_diagnostic_busy.png"
    render_fusion_diagnostic(
        dic, cy5,
        result["fusion_source_stack"],
        result["labels"],
        out,
        n_sample_frames=6,
        tracks=tracks)
    print(f"  wrote {out} ({os.path.getsize(out) // 1024} KB)")

    with open(f"{OUT_DIR}/06_diagnostic_summary.txt", "w") as f:
        f.write(f"DMSO_busy fusion diagnostic ({n} frames, "
                f"{elapsed:.0f}s)\n\n")
        f.write(f"Total fusion-added cells (pre-tracking): "
                f"{result.get('n_cy5_fusion_added', 0)}\n")
        f.write(f"Final tracks: {len(tracks)}\n")
        f.write(f"Track source breakdown:\n")
        f.write(f"  dic_only (cellpose_dic only): "
                f"{summary['dic_only']}\n")
        f.write(f"  both     (DIC + Cy5 agreed):  "
                f"{summary['both']}\n")
        f.write(f"  cy5_only (cpsam(Cy5) only):   "
                f"{summary['cy5_only']}\n")


if __name__ == "__main__":
    main()
