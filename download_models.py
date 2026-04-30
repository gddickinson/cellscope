"""Download large model weights that can't ship in the source bundle.

Models bundled with the source (small):
  data/models/cellpose_dic       (25 MB, CP3 fine-tune)
  data/models/cellpose_dic_v2    (25 MB)
  data/models/cellpose_dic_v3    (25 MB)
  data/models/cellpose_combined_robust  (25 MB)
  data/models/deepsea/           (~16 MB)

Model fetched by this script (large):
  data/models/cpsam_dic          (1.1 GB, CP4 ViT fine-tune — best DIC)

Source: Google Drive shared link (set CPSAM_DIC_URL below or pass --url).

Usage:
  conda run -n cellpose python download_models.py
  python download_models.py --url 'https://drive.google.com/...'
  python download_models.py --check-only      # just verify what's present
"""
import argparse
import hashlib
import os
import sys

# ---------------------------------------------------------------
# Paste the Google Drive share link here once you have one.
# Acceptable forms:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#   https://drive.google.com/uc?id=<FILE_ID>
# Leave as None to require the user to pass --url on first run.
# ---------------------------------------------------------------
CPSAM_DIC_URL = (
    "https://drive.google.com/open?id=15M1BqDc1dt_Jj8L-BJ2RBpfPGmihaIOW"
    "&usp=drive_fs"
)
EXPECTED_BYTES = 1_218_640_971  # size of the file you trained
EXPECTED_MIN_BYTES = 800_000_000  # accept partial-trained variants too

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "models", "cpsam_dic")


def file_ok(path):
    """Return True if file exists and is at least roughly the right size."""
    return os.path.exists(path) and os.path.getsize(path) >= EXPECTED_MIN_BYTES


def download_via_gdown(url, dest):
    """Pull from Drive via gdown. Auto-handles big-file confirmation."""
    try:
        import gdown
    except ImportError:
        print("ERROR: `gdown` not installed. Run:")
        print("  conda run -n cellpose pip install gdown")
        sys.exit(2)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading cpsam_dic (~1.1 GB) from Drive…")
    print(f"Destination: {dest}")
    # gdown.download with fuzzy=True accepts the share-link form too.
    gdown.download(url, dest, quiet=False, fuzzy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CPSAM_DIC_URL,
                    help="Google Drive share link for cpsam_dic. "
                         "Defaults to the constant in this file.")
    ap.add_argument("--check-only", action="store_true",
                    help="Just report which models are present.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if a valid file is present.")
    args = ap.parse_args()

    print("=== Cellscope model check ===")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "models")

    bundled = [
        ("cellpose_dic",            "CP3 DIC v1 (legacy)"),
        ("cellpose_dic_v2",         "CP3 DIC v2"),
        ("cellpose_dic_v3",         "CP3 DIC v3"),
        ("cellpose_combined_robust", "CP3 robust (noise-tolerant)"),
        ("deepsea",                 "DeepSea refiner (model dir)"),
    ]
    for name, desc in bundled:
        p = os.path.join(base, name)
        if os.path.exists(p):
            sz = (os.path.getsize(p) if os.path.isfile(p)
                  else sum(os.path.getsize(os.path.join(p, f))
                           for f in os.listdir(p) if os.path.isfile(
                               os.path.join(p, f))))
            print(f"  ✓ {name:30s} ({sz/1e6:.0f} MB)  — {desc}")
        else:
            print(f"  ✗ {name:30s} MISSING        — {desc}")

    cpsam_present = file_ok(DEST)
    if cpsam_present and not args.force:
        sz = os.path.getsize(DEST) / 1e9
        print(f"  ✓ cpsam_dic                       ({sz:.2f} GB)  "
              f"— CP4 ViT (best DIC). Already downloaded.")
    else:
        print(f"  ✗ cpsam_dic                       MISSING        "
              f"— CP4 ViT (best DIC). Will download (~1.1 GB).")

    if args.check_only:
        return

    if cpsam_present and not args.force:
        print("\nNothing to download. Pass --force to re-download.")
        return

    if not args.url:
        print("\nERROR: no Drive URL provided.")
        print("  • Either: open download_models.py and set CPSAM_DIC_URL")
        print("  • Or:     pass --url 'https://drive.google.com/...'")
        print("\nAsk the project maintainer for the link.")
        sys.exit(1)

    download_via_gdown(args.url, DEST)

    if not file_ok(DEST):
        print(f"\nERROR: download produced an unexpectedly small file at "
              f"{DEST}. Re-try, or check your Drive link.")
        sys.exit(3)

    sz = os.path.getsize(DEST) / 1e9
    print(f"\n✓ cpsam_dic downloaded ({sz:.2f} GB)")
    print(f"  Path: {DEST}")
    print("\nReady. Launch with:")
    print("  conda activate cellpose")
    print("  python main_focused.py")


if __name__ == "__main__":
    main()
