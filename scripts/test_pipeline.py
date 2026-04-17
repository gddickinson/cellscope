"""End-to-end smoke test on the bundled example recordings."""
import os
import sys
import shutil
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.io import load_recording, find_recordings
from core.pipeline import detect, refine, analyze_recording
from output.results import write_recording_results, _build_metrics_dict
from output.summary import write_all_summaries

EXAMPLES_DIR = "data/examples"
TEST_RESULTS_DIR = "results/test_run"


def progress(msg):
    print(f"  {msg}", flush=True)


def main():
    if os.path.isdir(TEST_RESULTS_DIR):
        shutil.rmtree(TEST_RESULTS_DIR)

    groups = find_recordings(EXAMPLES_DIR)
    print(f"Discovered groups: {list(groups.keys())}")
    for grp, paths in groups.items():
        print(f"  {grp}: {paths}")

    all_metrics = []
    for grp, paths in sorted(groups.items()):
        for video_path in paths:
            print(f"\n=== {grp} :: {os.path.basename(video_path)} ===")
            rec = load_recording(video_path)
            print(f"  Loaded {rec['frames'].shape}")

            print("  Detection...")
            det = detect(rec["frames"], progress_fn=lambda i, m=None: None)
            print(f"  Detection done in {det['elapsed']:.1f}s")

            print("  Auto-refinement...")
            ref = refine(rec["frames"], det["masks"], progress_fn=progress)
            print(f"  Refinement done in {ref['elapsed']:.1f}s, "
                  f"chosen={ref['config']['name']}")

            print("  Analyzing...")
            result = analyze_recording(rec, ref["masks"], progress_fn=progress)
            result["detection_elapsed_s"] = det["elapsed"]
            result["refinement_elapsed_s"] = ref["elapsed"]
            result["refinement_config"] = ref["config"]
            result["refinement_score_log"] = ref["score_log"]
            result["mean_flow_quality"] = float(det["flow_quality"].mean())

            print(f"  Mean speed: {result['mean_speed']:.3f} μm/min")
            print(f"  Mean boundary conf: {result['mean_boundary_confidence']:.1f}")
            print(f"  Area CV: {result['area_stability']['area_cv']:.3f}")

            base = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(TEST_RESULTS_DIR, grp, base)
            write_recording_results(result, out_dir)
            print(f"  Saved to {out_dir}")

            metrics = _build_metrics_dict(result)
            all_metrics.append((grp, metrics))

    if all_metrics:
        paths = write_all_summaries(all_metrics, TEST_RESULTS_DIR)
        print(f"\nSummaries:")
        for k, v in paths.items():
            print(f"  {k}: {v}")

    print("\nDONE")


if __name__ == "__main__":
    main()
