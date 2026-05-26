"""Re-run Pos3 detection using hybrid_cpsam_multi (raw cpsam ViT-H,
no DIC fine-tune) and save into pipeline_results_cpsam/ for
side-by-side evaluation.
"""
import os
import sys
import time
import logging
import numpy as np

CELLSCOPE_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
os.chdir(CELLSCOPE_ROOT)
sys.path.insert(0, CELLSCOPE_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rerun_cpsam")

FOLDER = "data/legacy_gt/ignasi_3_cells_control_IC293_Pos3"
TIF = (FOLDER + "/IC293__1_MMStack_Pos3-WT.ome-cropped.tif")
OUT_DIR = FOLDER + "/pipeline_results_cpsam"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from core.io import load_video
    log.info("Loading %s …", TIF)
    frames = load_video(TIF)
    log.info("  %d frames, %s", len(frames), frames.shape)

    from core.hybrid_cpsam_multi import detect_hybrid_cpsam_multi
    from core.pipeline_defaults import DEFAULTS
    log.info("\nDetecting with hybrid_cpsam_multi (raw cpsam ViT-H) …")
    t0 = time.time()
    detect = detect_hybrid_cpsam_multi(
        frames,
        progress_fn=lambda m, p: log.info("  detect[%d%%]: %s", p, m),
        min_area_px=DEFAULTS.min_area_px,
        use_fallback=False,    # no DIC fallback (we ARE cpsam already)
        use_deepsea=DEFAULTS.use_deepsea,
        use_gap_fill=DEFAULTS.use_gap_fill,
        use_tta=False,
        cy5_frames=None,
        recover_with_cy5=False,
        use_cy5_fusion=False)
    elapsed = time.time() - t0
    log.info("Detection done in %.0fs (tracks=%d)",
             elapsed, len(detect.get("tracks", [])))

    save_dict = {
        "masks": detect["masks"],
        "labels": detect["labels"],
    }
    np.savez_compressed(os.path.join(OUT_DIR, "masks.npz"),
                         **save_dict)
    log.info("Wrote %s/masks.npz", OUT_DIR)

    from core.run_metadata import write_run_metadata
    recording = {
        "frames": frames, "cy5_frames": None,
        "name": "ignasi_3_cells_control_IC293_Pos3 (cpsam variant)",
        "video_path": TIF,
        "um_per_px": 0.6523, "time_interval_min": 10.0,
        "n_channels": 1,
    }
    params = {**DEFAULTS.as_dict(),
              "model_used": "raw cpsam (vit_h)",
              "use_cy5_fusion": False}
    write_run_metadata(
        OUT_DIR, recording, params, detect,
        analysis_results=None,
        pipeline_function="hybrid_cpsam_multi",
        mode="multi",
        runtime_seconds=elapsed,
        rerun_command="python scripts/rerun_pos3_with_cpsam.py",
        extra={"reason": "Test fix for under-detection — cpsam_dic "
                          "merges touching cells; raw cpsam separates"},
        project_root=CELLSCOPE_ROOT)
    print(f"\n✓ Pipeline output saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
