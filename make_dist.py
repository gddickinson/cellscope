"""Create a distribution zip of CellScope.

Creates cellscope_full.zip containing everything needed to run
(code + models + examples). For sharing via Google Drive, Dropbox, etc.

Usage:
    python make_dist.py              # full zip (~13 GB)
    python make_dist.py --code-only  # code only (~5 MB)
"""
import os
import sys
import zipfile
import argparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".DS_Store",
    ".git/",
    ".claude/",
    "Thumbs.db",
    ".pyc",
]


def should_include(path, code_only=False):
    """Check if a file should be included in the zip."""
    rel = os.path.relpath(path, PROJECT_DIR)
    for pat in EXCLUDE_PATTERNS:
        if pat in rel:
            return False
    if code_only:
        if rel.startswith("data/models/") and not rel.endswith(".json"):
            return False
        if rel.startswith("data/examples/") and (
                rel.endswith(".tif") or rel.endswith(".tiff")
                or rel.endswith(".mp4")):
            return False
        if rel.startswith("results/"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create CellScope distribution zip")
    parser.add_argument("--code-only", action="store_true",
                        help="Exclude models and example data")
    parser.add_argument("--output", default=None,
                        help="Output zip path")
    args = parser.parse_args()

    suffix = "code" if args.code_only else "full"
    out_path = args.output or f"cellscope_{suffix}.zip"

    print(f"Creating {out_path}...")
    n_files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(PROJECT_DIR):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", ".claude")]
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                if should_include(fpath, args.code_only):
                    arcname = os.path.join(
                        "cellscope",
                        os.path.relpath(fpath, PROJECT_DIR))
                    zf.write(fpath, arcname)
                    n_files += 1
                    if n_files % 100 == 0:
                        print(f"  {n_files} files...")

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nCreated {out_path}")
    print(f"  {n_files} files, {size_mb:.1f} MB")
    if args.code_only:
        print("\nNote: models and example data excluded.")
        print("Run setup_wizard.py after extracting to download them.")


if __name__ == "__main__":
    main()
