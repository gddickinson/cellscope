"""Download model weights from Google Drive.

Two bundles are hosted on Drive:

  1. Small models bundle  (~120 MB)
       Contains: cellpose_dic, cellpose_dic_v2, cellpose_dic_v3,
                 cellpose_combined_robust, deepsea/
       Needed by: GitHub-clone installs (the repo is code-only).
       Skipped if all those models are already on disk
       (e.g. you installed via the prebuilt cellscope-dist.zip).

  2. cpsam_dic ViT fine-tune  (1.1 GB)
       Contains: data/models/cpsam_dic
       Needed by: every install (this is the best DIC detector and
                  is too large to ship in either zip).

Usage:
  conda run -n cellpose python download_models.py
  python download_models.py --check-only       # just report what's present
  python download_models.py --bundle-only      # skip cpsam_dic
  python download_models.py --cpsam-only       # skip small-models bundle
  python download_models.py --force            # re-download everything
  python download_models.py --bundle-url <url> --cpsam-url <url>

Maintainer: edit the URL constants near the top after uploading the
zips to Drive. To rebuild the small-models bundle from scratch:
  python make_models_bundle.py
"""
import argparse
import os
import sys
import zipfile

# ─────────────────────────────────────────────────────────────────
# Drive URLs — set these once after uploading. Both must be
# "Anyone with the link" share links.
# ─────────────────────────────────────────────────────────────────
CPSAM_DIC_URL = (
    "https://drive.google.com/file/d/15M1BqDc1dt_Jj8L-BJ2RBpfPGmihaIOW"
    "/view?usp=sharing"
)

# Set after uploading cellscope-models-bundle.zip from make_models_bundle.py.
# Until set, the script gracefully skips the bundle and only fetches
# cpsam_dic — fine for installs from a dist zip that already includes
# the small models.
MODELS_BUNDLE_URL = None

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
CPSAM_DEST = os.path.join(MODELS_DIR, "cpsam_dic")

CPSAM_EXPECTED_MIN = 800_000_000   # accept partial-trained variants too
BUNDLE_EXPECTED_MIN = 50_000_000   # roughly 120 MB; allow some headroom

SMALL_MODEL_NAMES = [
    ("cellpose_dic",            "CP3 DIC v1 (legacy)"),
    ("cellpose_dic_v2",         "CP3 DIC v2"),
    ("cellpose_dic_v3",         "CP3 DIC v3 (current best CP3)"),
    ("cellpose_combined_robust", "CP3 robust (noise-tolerant)"),
    ("deepsea",                 "DeepSea refiner (model dir)"),
]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def cpsam_ok():
    return (os.path.exists(CPSAM_DEST)
            and os.path.getsize(CPSAM_DEST) >= CPSAM_EXPECTED_MIN)


def small_models_ok():
    """All five small-model entries are present."""
    return all(os.path.exists(os.path.join(MODELS_DIR, n))
               for n, _ in SMALL_MODEL_NAMES)


def gdown_or_die():
    try:
        import gdown
        return gdown
    except ImportError:
        print("ERROR: `gdown` not installed. Run:")
        print("  conda run -n cellpose pip install gdown")
        sys.exit(2)


def _drive_file_id(url):
    """Extract the file ID from a Drive share URL.

    Handles all common share-link forms:
      https://drive.google.com/file/d/<ID>/view?usp=sharing
      https://drive.google.com/open?id=<ID>
      https://drive.google.com/uc?id=<ID>
    """
    import re
    import urllib.parse
    m = re.search(r"/file/d/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    qs = urllib.parse.urlparse(url).query
    fid = urllib.parse.parse_qs(qs).get("id", [None])[0]
    if fid:
        return fid
    raise ValueError(f"Could not extract Drive file ID from URL: {url}")


def download_via_gdown(url, dest, label):
    """Download from a Drive share URL via gdown 6.x.

    gdown 6.x dropped `fuzzy=` and requires the bare file ID via `id=`.
    """
    gdown = gdown_or_die()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"\nDownloading {label} from Drive…")
    print(f"Destination: {dest}")
    file_id = _drive_file_id(url)
    gdown.download(id=file_id, output=dest, quiet=False)


def fetch_small_bundle(url, force=False):
    """Pull the small-models zip from Drive and extract into data/models/."""
    if small_models_ok() and not force:
        print("Small models already present — skipping bundle.")
        return True

    if not url:
        print(
            "\n⚠  Small-models bundle URL not configured "
            "(MODELS_BUNDLE_URL is None).\n"
            "   This is fine if you installed from a prebuilt dist zip "
            "(it already includes the\n   small models). If you cloned "
            "from GitHub instead, ask the project maintainer\n"
            "   for the bundle URL, or pass --bundle-url <url>.")
        return False

    tmp = os.path.join(PROJECT_ROOT, "_models_bundle_tmp.zip")
    download_via_gdown(url, tmp, label="small-models bundle (~120 MB)")
    if not (os.path.exists(tmp)
            and os.path.getsize(tmp) >= BUNDLE_EXPECTED_MIN):
        print(f"\nERROR: bundle download produced an unexpectedly small file "
              f"at {tmp}. Aborting.")
        sys.exit(3)

    print(f"\nExtracting bundle into {MODELS_DIR}/…")
    os.makedirs(MODELS_DIR, exist_ok=True)
    target = os.path.dirname(MODELS_DIR)  # data/
    with zipfile.ZipFile(tmp, "r") as zf:
        zf.extractall(target)

    os.remove(tmp)
    if not small_models_ok():
        print("ERROR: bundle extracted but some small models still missing.")
        sys.exit(4)
    print("✓ small-models bundle extracted.")
    return True


def fetch_cpsam(url, force=False):
    if cpsam_ok() and not force:
        sz = os.path.getsize(CPSAM_DEST) / 1e9
        print(f"cpsam_dic already present ({sz:.2f} GB) — skipping.")
        return True

    if not url:
        print("\nERROR: no cpsam_dic URL provided.")
        print("  Either: edit CPSAM_DIC_URL near the top of "
              "download_models.py, or")
        print("  Pass:   --cpsam-url 'https://drive.google.com/...'")
        sys.exit(1)

    download_via_gdown(url, CPSAM_DEST, label="cpsam_dic (~1.1 GB)")
    if not cpsam_ok():
        print(f"\nERROR: cpsam_dic download produced an unexpectedly small "
              f"file at {CPSAM_DEST}. Re-try, or check the Drive link.")
        sys.exit(3)
    sz = os.path.getsize(CPSAM_DEST) / 1e9
    print(f"✓ cpsam_dic downloaded ({sz:.2f} GB)")
    return True


def report_status():
    print("=== Cellscope model check ===")
    print(f"Models dir: {MODELS_DIR}\n")

    for name, desc in SMALL_MODEL_NAMES:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            sz = (os.path.getsize(p) if os.path.isfile(p)
                  else sum(
                      os.path.getsize(os.path.join(root, f))
                      for root, _, files in os.walk(p)
                      for f in files))
            print(f"  ✓ {name:28s} ({sz/1e6:.0f} MB)  — {desc}")
        else:
            print(f"  ✗ {name:28s} MISSING        — {desc}")

    if cpsam_ok():
        sz = os.path.getsize(CPSAM_DEST) / 1e9
        print(f"  ✓ cpsam_dic                  ({sz:.2f} GB)  "
              f"— CP4 ViT (current best DIC, downloaded)")
    else:
        print(f"  ✗ cpsam_dic                  MISSING        "
              f"— CP4 ViT (current best DIC). Will download ~1.1 GB.")
    print()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-url", default=MODELS_BUNDLE_URL,
                    help="Override MODELS_BUNDLE_URL constant")
    ap.add_argument("--cpsam-url", default=CPSAM_DIC_URL,
                    help="Override CPSAM_DIC_URL constant")
    ap.add_argument("--check-only", action="store_true",
                    help="Just print model status; do not download.")
    ap.add_argument("--bundle-only", action="store_true",
                    help="Only fetch the small-models bundle.")
    ap.add_argument("--cpsam-only", action="store_true",
                    help="Only fetch cpsam_dic.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if files are already present.")
    args = ap.parse_args()

    report_status()
    if args.check_only:
        return

    do_bundle = not args.cpsam_only
    do_cpsam = not args.bundle_only

    if do_bundle:
        fetch_small_bundle(args.bundle_url, force=args.force)
    if do_cpsam:
        fetch_cpsam(args.cpsam_url, force=args.force)

    print()
    report_status()
    print("Ready. Launch with:")
    print("  conda activate cellpose")
    print("  python main_suite.py")


if __name__ == "__main__":
    main()
