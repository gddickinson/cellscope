"""Poll an incoming IC295 directory and process new recordings as
they finish downloading.

Designed to run during a long network transfer:
  * Lists *.ome.tif in the source directory
  * Skips files where the matching `_metadata.txt` sidecar is missing
    (the sidecar is the LAST file written by Micro-Manager during a
    save — its presence means the .ome.tif is fully written).
  * Skips files whose mtime changed in the last `--quiet-seconds`
    (defensive against rsync / Finder copy mid-write).
  * Skips files already processed (NPZ exists in --out-dir).
  * Processes each remaining file via run_ignasi_ic295_full pipeline
    (cpsam_base + Cy5 recovery + DeepSea + tracking).
  * Sleeps `--poll-seconds` and repeats.

Stop with Ctrl-C or by killing the process. Each completed recording's
NPZ + HTML + overlay TIFF is written immediately so partial results
are usable mid-run.

Usage:
  conda run -n cellpose4 python scripts/process_incoming_ic295.py \\
      --src /Volumes/GeorgeDrive/ignasi/IC295_batch2 \\
      --out-dir results/ic295_batch2_full
"""
import argparse
import glob
import os
import subprocess
import sys
import time

DEFAULT_QUIET = 120        # seconds since last mtime change
DEFAULT_POLL = 300         # seconds between scans
EXPECTED_TIF_BYTES = 2_400_000_000  # ~2.45 GB; complete IC295 stack


def is_stable(tif_path, quiet_seconds):
    """File is stable if ALL of these hold:
      1. The `_metadata.txt` sidecar exists (Micro-Manager writes
         this LAST when saving — its presence implies the .tif is
         fully written).
      2. Neither the .tif nor sidecar has been modified in
         `quiet_seconds`.
      3. The .tif is at least EXPECTED_TIF_BYTES (2.4 GB) — defends
         against partial copies that finish writing the sidecar
         before the data file.
    """
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if not os.path.exists(sidecar):
        return False
    try:
        size = os.path.getsize(tif_path)
    except OSError:
        return False
    if size < EXPECTED_TIF_BYTES:
        return False
    now = time.time()
    for p in (tif_path, sidecar):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            return False
        if now - mt < quiet_seconds:
            return False
    return True


def already_processed(tif_path, out_dir):
    """Check if the NPZ for this recording exists already."""
    import re
    name = os.path.basename(tif_path).replace(".ome.tif", "")
    m = re.search(r"(Pos\d+)-(WT|KO|GOF|OT|Y1|DMSO)", name, re.I)
    if not m:
        return False
    pos, cond = m.group(1), m.group(2).upper()
    return os.path.exists(os.path.join(out_dir, f"{pos}_{cond}.npz"))


def list_ready(src, out_dir, quiet_seconds):
    """Return (ready, queued_unstable, processed) lists of full paths."""
    tifs = sorted(glob.glob(os.path.join(src, "*.ome.tif")))
    ready, unstable, done = [], [], []
    for t in tifs:
        if already_processed(t, out_dir):
            done.append(t)
        elif is_stable(t, quiet_seconds):
            ready.append(t)
        else:
            unstable.append(t)
    return ready, unstable, done


def process_one(tif_path, src, out_dir, conda_env="cellpose4"):
    """Run the full pipeline on a single recording via the existing
    run_ignasi_ic295_full.py with --positions filter.

    Uses position+condition tag (e.g. 'Pos4-WT') to filter so
    'Pos4' doesn't accidentally also match 'Pos40', 'Pos41', etc.
    """
    import re
    name = os.path.basename(tif_path).replace(".ome.tif", "")
    m = re.search(r"(Pos\d+-(?:WT|KO|GOF|OT|Y1|DMSO))", name, re.I)
    if not m:
        print(f"  [skip] cannot parse position-condition "
              f"from {name}", flush=True)
        return False
    tag = m.group(1)
    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "scripts/run_ignasi_ic295_full.py",
        "--src", src,
        "--positions", tag,
        "--out-dir", out_dir,
    ]
    print(f"  → processing {tag}…", flush=True)
    t0 = time.time()
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, env=env)
    dt = time.time() - t0
    if result.returncode == 0:
        print(f"  ✓ {tag} done in {dt/60:.1f} min", flush=True)
        return True
    else:
        print(f"  ✗ {tag} failed (exit {result.returncode}) "
              f"after {dt/60:.1f} min", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet-seconds", type=int, default=DEFAULT_QUIET,
                    help="file mtime must be older than this to be"
                         " considered stable (default: 120)")
    ap.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL,
                    help="seconds between scans (default: 300)")
    ap.add_argument("--conda-env", default="cellpose4",
                    help="conda env for cpsam (default: cellpose4)")
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="stop after this many poll cycles "
                         "(0 = run until killed, default)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Polling {args.src} → {args.out_dir}", flush=True)
    print(f"  quiet threshold: {args.quiet_seconds}s, "
          f"poll interval: {args.poll_seconds}s", flush=True)

    iteration = 0
    while True:
        iteration += 1
        ready, unstable, done = list_ready(
            args.src, args.out_dir, args.quiet_seconds)
        print(f"\n[iter {iteration}] {time.strftime('%H:%M:%S')} — "
              f"{len(ready)} ready, {len(unstable)} downloading, "
              f"{len(done)} already done", flush=True)
        if unstable:
            for u in unstable[:5]:
                print(f"  [waiting] {os.path.basename(u)}", flush=True)
            if len(unstable) > 5:
                print(f"  …and {len(unstable) - 5} more", flush=True)

        for tif in ready:
            process_one(tif, args.src, args.out_dir,
                          conda_env=args.conda_env)

        if args.max_iterations and iteration >= args.max_iterations:
            print("Max iterations reached — exiting.", flush=True)
            break

        # If nothing ready and nothing unstable, we're either at the
        # start (no files yet) or done. Keep polling.
        print(f"  sleep {args.poll_seconds}s…", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
