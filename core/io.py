"""Load video files and per-recording metadata."""
import json
import os
import numpy as np
import cv2


def _to_uint8(arr, pct_lo=1.0, pct_hi=99.0):
    """Rescale an integer / float image (or stack) to uint8.

    Uses percentile-based normalization computed over the WHOLE stack so
    the intensity range is consistent across frames (important for
    cellpose and temporal analyses). The saved sidecar JSON's
    `um_per_px` / `time_interval_min` are unaffected.

    - If the array is already uint8, returned as-is.
    - If pct_hi == pct_lo (constant image), returns zeros.
    """
    if arr.dtype == np.uint8:
        return arr
    lo = float(np.percentile(arr, pct_lo))
    hi = float(np.percentile(arr, pct_hi))
    if hi <= lo:
        # Flat image — fall back to min/max, or zeros
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _read_tiff_rawpages(path):
    """Read a multi-page TIFF via raw page bytes (numpy 2.0 workaround).

    Bypasses tifffile's `result.newbyteorder()` call that was removed
    in NumPy 2.0. Handles uint8/16/32 and float32 by reading the
    DataOffsets / DataByteCounts of each page directly.
    """
    import tifffile
    fmt_to_kind = {1: "u", 2: "i", 3: "f"}
    with tifffile.TiffFile(path) as tf:
        pages = []
        for pg in tf.pages:
            h, w = pg.imagelength, pg.imagewidth
            bps = pg.bitspersample
            sf = pg.sampleformat
            byteorder = pg.parent.byteorder
            kind = fmt_to_kind.get(sf, "u")
            np_dtype = np.dtype(f"{byteorder}{kind}{bps // 8}")
            with open(path, "rb") as f:
                buf = []
                for off, n in zip(pg.dataoffsets, pg.databytecounts):
                    f.seek(off)
                    buf.append(f.read(n))
            arr = np.frombuffer(b"".join(buf), dtype=np_dtype).reshape(h, w)
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("=")).copy()
            pages.append(arr)
    return np.stack(pages) if len(pages) > 1 else pages[0]


def load_video(path):
    """Load a video file as (N, H, W) uint8 grayscale.

    Handles .mp4/.avi/.mov/.tif/.tiff/.ome.tif. For multi-bit TIFFs
    (common: uint16 from sCMOS cameras), intensity is rescaled using
    per-stack 1/99 percentiles so the full dynamic range maps to
    0-255 linearly. The previous naive `astype(uint8)` truncated the
    top byte and produced visible staircase artifacts.

    Args:
        path: path to .mp4/.avi/.mov/.tif/.tiff file

    Returns:
        frames: (N, H, W) uint8 array
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        try:
            arr = tifffile.imread(path)
        except AttributeError as e:
            # NumPy 2.0 + non-native-byteorder tifffile workaround:
            # `ndarray.newbyteorder` was removed in NumPy 2.0. Read
            # raw page bytes via the file handle and assemble manually.
            if "newbyteorder" not in str(e):
                raise
            arr = _read_tiff_rawpages(path)
        if arr.ndim == 4:  # (N, H, W, C) — collapse channels
            arr = arr.mean(axis=-1)
        return _to_uint8(arr)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames in video: {path}")
    return np.stack(frames).astype(np.uint8)


def load_metadata(video_path):
    """Load JSON sidecar metadata for a video.

    Looks for {video_basename}.json next to the video. If not found,
    returns sensible defaults.

    Returns:
        dict with keys: name, um_per_px, time_interval_min
    """
    base, _ = os.path.splitext(video_path)
    json_path = base + ".json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    meta.setdefault("name", os.path.basename(base))
    meta.setdefault("um_per_px", 1.0)
    meta.setdefault("time_interval_min", 1.0)
    return meta


def load_recording(video_path):
    """Load a video and its metadata as a dict.

    Returns:
        {
            "name": str,
            "frames": (N, H, W) uint8,
            "um_per_px": float,
            "time_interval_min": float,
            "video_path": str,
        }
    """
    frames = load_video(video_path)
    meta = load_metadata(video_path)
    meta["frames"] = frames
    meta["video_path"] = video_path
    return meta


def find_recordings(root_dir):
    """Recursively find video files grouped by parent folder.

    Returns:
        dict mapping group_name (parent folder) → list of video paths
    """
    extensions = (".mp4", ".avi", ".mov", ".tif", ".tiff")
    groups = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        videos = sorted(
            os.path.join(dirpath, f)
            for f in filenames
            if f.lower().endswith(extensions)
        )
        if videos:
            group_name = os.path.basename(dirpath) or "root"
            groups.setdefault(group_name, []).extend(videos)
    return groups
