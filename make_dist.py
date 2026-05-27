"""Package CellScope into a zip ready to send to collaborators.

Default: produces a ~150 MB zip containing source + small CP3 models +
DeepSea, but NOT the 1.1 GB cpsam_dic ViT model. Recipients fetch
cpsam_dic separately via download_models.py (Google Drive).

Usage:
  python make_dist.py
      → ../cellscope-dist.zip   (~150 MB, source + small models only)

  python make_dist.py --include-cpsam
      → ../cellscope-dist.zip   (~1.3 GB, fully self-contained)

  python make_dist.py --out /tmp/cellscope.zip
      → /tmp/cellscope.zip

  python make_dist.py --dry-run
      → list what would be archived without writing
"""
import argparse
import fnmatch
import os
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TOP = PROJECT_ROOT.name   # "cellscope" — prefix inside the archive

# Always-excluded subtrees (relative to PROJECT_ROOT). Too big or
# not needed at runtime by recipients. NOTE: data/examples is
# intentionally INCLUDED — the JSON sidecars + small fixture .tif
# ship with the repo and are needed for the lab-install validation +
# remote-control smoke test. Large example .tif files are excluded
# per-file by EXCLUDE_FILE_PATTERNS below.
EXCLUDE_DIRS = {
    "data/training",         # ~6 GB of crops + masks (dev only)
    "data/VAMPIRE",          # raw VAMPIRE dataset (dev only)
    "data/manual_gt",        # manual annotations (dev only)
    "data/splits",           # train/test splits (dev only)
    "data/splits_eval",      # eval splits (dev only)
    "data/ic295_gt_full",    # multichannel GT corpus (research provenance, 324 MB)
    "data/legacy_gt",        # legacy single-channel GT (research provenance)
    "data/ic295_gt",         # GT scaffold dir (placeholder only on disk)
    "data/ic295_inspection", # channel inspection PDFs (dev only)
    "data/ignasi_new_gt",    # IC293 GT scaffold (placeholder only on disk)
    "data/gt_backups",       # tarball backups of GT (redundant)
    "results",               # generated figures + caches
    "notebooks",             # Colab notebooks (maintainer only)
    ".git",
    ".github",
    ".vscode",
    ".idea",
    ".claude",
    ".pytest_cache",
    ".mypy_cache",
}

# Always-excluded file globs (matched against the leaf filename).
EXCLUDE_FILE_PATTERNS = [
    "*.pyc",
    "*.pyo",
    "*.log",
    ".DS_Store",
    "Thumbs.db",
    "*~",
    "*.swp",
    "*.zip",          # don't pack other dist zips
    "cellscope*.zip",
]

# Per-path exclusions: large example recordings (>20 MB) that don't
# fit the "small fixture" criterion. The test_small_multichannel
# fixture (~7.4 MB) is intentionally kept.
EXCLUDE_FILE_GLOBS_BY_PATH = [
    "data/examples/multichannel_DIC_Cy5_DMSO_busy/*.tif",
    "data/examples/multichannel_DIC_Cy5_DMSO_busy/*.tiff",
    "data/examples/multichannel_DIC_Cy5_WT/*.tif",
    "data/examples/multichannel_DIC_Cy5_WT/*.tiff",
    "data/examples/single_cell_phase_WT/*.tif",
    "data/examples/single_cell_phase_WT/*.tiff",
    "data/examples/multi_cell_DIC_*/*.tif",
    "data/examples/multi_cell_DIC_*/*.tiff",
]


def is_excluded(rel_path: str, include_cpsam: bool) -> bool:
    """True if the path (relative to PROJECT_ROOT, forward slashes)
    should be skipped."""
    rp = rel_path.replace(os.sep, "/")

    for d in EXCLUDE_DIRS:
        if rp == d or rp.startswith(d + "/"):
            return True

    parts = rp.split("/")
    if "__pycache__" in parts:
        return True

    # cpsam_dic (~1.1 GB) is downloaded post-install via download_models.py.
    # Only bundle when --include-cpsam is set. Match the bare name and
    # all versioned variants (cpsam_dic_v2, cpsam_dic_gt_ft_v1, etc.).
    if not include_cpsam:
        if rp == "data/models/cpsam_dic" or rp.startswith("data/models/cpsam_dic_"):
            return True

    # SAM2 (~150 MB) and SAM (~360 MB) checkpoints are downloaded
    # post-install. Never bundle.
    if rp == "data/models/sam2" or rp.startswith("data/models/sam2/"):
        return True
    if rp == "data/models/sam" or rp.startswith("data/models/sam/"):
        return True

    # Generated analysis output bundled with example recordings -
    # recipients regenerate via the GUI. Mirrors .gitignore.
    if "/results/" in f"/{rp}/" or rp.endswith("/results"):
        # Only skip when results/ is nested under data/examples/.
        if rp.startswith("data/examples/") and "/results" in rp:
            return True

    name = parts[-1]
    for pat in EXCLUDE_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True

    for pat in EXCLUDE_FILE_GLOBS_BY_PATH:
        if fnmatch.fnmatch(rp, pat):
            return True

    return False


def collect_files(include_cpsam: bool):
    """Walk PROJECT_ROOT, yield (abs_path, archive_path) pairs.

    Silently skips broken symlinks (top-level fiji_export_test +
    results point at /Volumes/GeorgeDrive on the maintainer's
    machine; missing on other clones)."""
    for root, dirs, files in os.walk(PROJECT_ROOT, followlinks=False):
        rel_root = os.path.relpath(root, PROJECT_ROOT).replace(os.sep, "/")
        if rel_root == ".":
            rel_root = ""

        # Prune dirs in-place so os.walk skips excluded subtrees entirely.
        keep = []
        for d in dirs:
            full_rel = f"{rel_root}/{d}" if rel_root else d
            if is_excluded(full_rel, include_cpsam):
                continue
            # Skip broken-symlink "dirs" (os.walk yields them as dirs).
            full_abs = os.path.join(root, d)
            if os.path.islink(full_abs) and not os.path.exists(full_abs):
                continue
            keep.append(d)
        dirs[:] = keep

        for f in files:
            full_rel = f"{rel_root}/{f}" if rel_root else f
            if is_excluded(full_rel, include_cpsam):
                continue
            abs_path = os.path.join(root, f)
            # Skip broken symlinks at file level too.
            if os.path.islink(abs_path) and not os.path.exists(abs_path):
                continue
            archive_path = f"{TOP}/{full_rel}"
            yield abs_path, archive_path


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT.parent /
                                          "cellscope-dist.zip"),
                    help="output zip path (default: ../cellscope-dist.zip)")
    ap.add_argument("--include-cpsam", action="store_true",
                    help="bundle the 1.1 GB cpsam_dic model too "
                         "(produces a ~1.3 GB self-contained zip)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be archived without writing")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(collect_files(args.include_cpsam))
    total_bytes = sum(os.path.getsize(p) for p, _ in files)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Files to pack: {len(files)}")
    print(f"Uncompressed : {human_size(total_bytes)}")
    print(f"Include cpsam: {args.include_cpsam}")
    print(f"Output       : {out_path}")
    print()

    if args.dry_run:
        sized = sorted(((os.path.getsize(p), ap) for p, ap in files),
                       reverse=True)[:20]
        print("Largest files (top 20):")
        for sz, ap_ in sized:
            print(f"  {human_size(sz):>10s}  {ap_}")
        print("\n(dry-run; no zip written)")
        return

    if out_path.exists():
        print(f"Removing existing {out_path}")
        out_path.unlink()

    print("Writing zip…")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for i, (abs_path, archive_path) in enumerate(files):
            zf.write(abs_path, archive_path)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}…")

    out_size = out_path.stat().st_size
    ratio = (1 - out_size / total_bytes) * 100 if total_bytes else 0
    print(f"\n✓ Wrote {out_path}")
    print(f"  Compressed: {human_size(out_size)}  "
          f"(saved {ratio:.0f}%)")
    print()
    if args.include_cpsam:
        print("Self-contained zip — recipient doesn't need the Drive URL.")
        print()
        print("Recipient steps:")
        print("  1. Unzip cellscope-dist.zip")
        print("  2. cd cellscope")
        print("  3. install.bat (Windows)   |   bash install.sh (Mac/Linux)")
        print("  4. conda activate cellpose && python main_suite.py")
    else:
        print("Send the zip + the Drive URL for cpsam_dic to your collaborator.")
        print()
        print("Recipient steps:")
        print("  1. Unzip cellscope-dist.zip")
        print("  2. cd cellscope")
        print("  3. install.bat (Windows)   |   bash install.sh (Mac/Linux)")
        print("  4. conda run -n cellpose python download_models.py")
        print("  5. conda activate cellpose && python main_suite.py")


if __name__ == "__main__":
    main()
