"""Example: run CellScope analysis from Python code.

Shows how to detect cells, analyze, and export results
without using the GUI. Useful for scripting and batch automation.

Usage:
    conda run -n cellpose4 python scripts/example_analysis.py path/to/recording.tif
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="CellScope example analysis")
    parser.add_argument("recording", help="Path to recording (.tif/.mp4)")
    parser.add_argument("--mode", default="hybrid_cpsam",
                        choices=["hybrid_cpsam", "hybrid_cpsam_multi"],
                        help="Detection mode (default: hybrid_cpsam)")
    parser.add_argument("--output", default="results/example",
                        help="Output directory")
    args = parser.parse_args()

    from core.io import load_recording
    from core.pipeline import detect, analyze_recording
    from output.results import write_recording_results

    # 1. Load recording
    print(f"Loading {args.recording}...")
    rec = load_recording(args.recording)
    print(f"  {len(rec['frames'])} frames @ {rec['frames'].shape[1:]}")
    print(f"  Scale: {rec['um_per_px']} um/px, "
          f"{rec['time_interval_min']} min/frame")

    # 2. Detect cells
    print(f"\nDetecting cells (mode={args.mode})...")
    det = detect(rec["frames"], mode=args.mode)
    n_detected = int(det["masks"].any(axis=(1, 2)).sum())
    print(f"  Detected cells in {n_detected}/{len(rec['frames'])} frames")

    # 3. Analyze
    print("\nAnalyzing...")
    result = analyze_recording(rec, det["masks"])
    print(f"  Mean speed: {result.get('mean_speed', 0):.3f} um/min")
    print(f"  Persistence: {result.get('persistence', 0):.3f}")
    ss = result.get("shape_summary", {}).get("area_um2", {})
    print(f"  Mean area: {ss.get('mean', 0):.0f} um^2")

    # 4. Export
    os.makedirs(args.output, exist_ok=True)
    write_recording_results(result, args.output)
    print(f"\nResults saved to {args.output}/")
    print("  masks.npz, metrics.json, trajectory.png, "
          "edge_kymograph.png, shape_timeseries.png")


if __name__ == "__main__":
    main()
