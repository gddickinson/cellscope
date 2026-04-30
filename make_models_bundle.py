"""Bundle the small models (CP3 fine-tunes + DeepSea) into one zip
ready to upload to Google Drive.

This is needed for the GitHub-clone install path. GitHub doesn't host
binary models (and Git LFS is awkward for end users), so the small
models live on Drive too — `download_models.py` fetches them.

Usage:
  python make_models_bundle.py
      → ../cellscope-models-bundle.zip   (~120 MB)

Then:
  1. Upload that zip to Google Drive
  2. Get a share link
  3. Paste the link into MODELS_BUNDLE_URL near the top of
     download_models.py

The 1.1 GB cpsam_dic model is hosted separately (different Drive file)
because it's much larger and harder to re-upload — see CPSAM_DIC_URL
in download_models.py.
"""
import argparse
import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Files / dirs to include in the bundle. Keys are paths relative to
# data/models/ and become entries in the zip.
INCLUDE = [
    "cellpose_dic",
    "cellpose_dic_v2",
    "cellpose_dic_v3",
    "cellpose_combined_robust",
    "deepsea",
]
EXCLUDE_NAMES = {"__pycache__"}


def human_size(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def collect():
    """Yield (abs_path, archive_path) for everything to include."""
    for entry in INCLUDE:
        src = MODELS_DIR / entry
        if not src.exists():
            print(f"  WARN: {src} not found — skipping")
            continue
        if src.is_file():
            yield src, f"models/{entry}"
        else:
            for root, dirs, files in os.walk(src):
                # prune caches
                dirs[:] = [d for d in dirs if d not in EXCLUDE_NAMES]
                for f in files:
                    if f.startswith("."):
                        continue
                    abs_p = Path(root) / f
                    rel = abs_p.relative_to(MODELS_DIR)
                    yield abs_p, f"models/{rel.as_posix()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PROJECT_ROOT.parent /
                                          "cellscope-models-bundle.zip"),
                    help="output zip path "
                         "(default: ../cellscope-models-bundle.zip)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = list(collect())
    total_bytes = sum(p.stat().st_size for p, _ in files)
    out = Path(args.out).resolve()

    print(f"Files to bundle: {len(files)}")
    print(f"Uncompressed   : {human_size(total_bytes)}")
    print(f"Output         : {out}")

    if args.dry_run:
        sized = sorted(((p.stat().st_size, ap_) for p, ap_ in files),
                       reverse=True)[:15]
        print("\nLargest entries:")
        for sz, ap_ in sized:
            print(f"  {human_size(sz):>10s}  {ap_}")
        return

    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    print("\nWriting zip…")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, (abs_p, archive_p) in enumerate(files):
            zf.write(abs_p, archive_p)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}…")

    sz = out.stat().st_size
    saved = (1 - sz / total_bytes) * 100 if total_bytes else 0
    print(f"\n✓ Wrote {out}")
    print(f"  Compressed: {human_size(sz)}  (saved {saved:.0f}%)")
    print()
    print("Next steps:")
    print("  1. Upload the zip to Google Drive.")
    print("  2. Right-click → Get link → 'Anyone with the link'.")
    print("  3. Copy the URL.")
    print("  4. Paste it into MODELS_BUNDLE_URL near the top of "
          "download_models.py.")
    print("  5. Commit + push download_models.py.")


if __name__ == "__main__":
    main()
