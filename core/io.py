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


def detect_channels(tif_path):
    """Inspect a TIFF/OME-TIFF and return n_channels (1 if not multichannel).

    Reads from the metadata sidecar first ({base}_metadata.txt for OME),
    falls back to the first IFD's SamplesPerPixel.
    """
    import re
    # Look for a Micro-Manager-style sidecar. Only valid when the path
    # actually ends in `.ome.tif` — otherwise .replace returns the
    # original .tif path and we'd try to decode binary TIFF as UTF-8.
    sidecar = tif_path.replace(".ome.tif", "_metadata.txt")
    if sidecar != tif_path and os.path.exists(sidecar):
        with open(sidecar) as f:
            m = re.search(r'"Channels"\s*:\s*(\d+)', f.read())
        if m:
            return int(m.group(1))
    try:
        import tifffile
        with tifffile.TiffFile(tif_path) as tf:
            ome = getattr(tf, "ome_metadata", None)
            if ome:
                m = re.search(r'SizeC="(\d+)"', ome)
                if m:
                    return int(m.group(1))
            n_pages = len(tf.pages)
            if n_pages == 1:
                return 1
            # Heuristic: if N_pages divisible by likely channel counts {2,3,4}
            # AND first page has 1 sample, treat first dim as channels.
            # Without more info we default to 1 here; multichannel callers
            # should pass n_channels explicitly when known.
            return 1
    except Exception:
        return 1


def load_video_multichannel(tif_path, dic_channel=1, fluo_channel=0,
                             n_channels=None):
    """Load a multi-channel OME-TIFF as (dic_uint8, fluo_uint8) stacks.

    Channels are interleaved frame-by-frame: page i*nch+ch is frame i,
    channel ch. DIC gets flat-field correction (σ=80); fluorescence
    gets flat-field σ=200 (broader because fluorescent clusters span
    larger spatial scales) — both via core.multichannel utilities.

    Args:
        tif_path: path to multichannel .ome.tif
        dic_channel: which channel index is DIC (default 1)
        fluo_channel: which channel index is fluorescence (default 0)
        n_channels: total channels; auto-detected from metadata if None

    Returns:
        (dic_uint8, fluo_uint8) — both (N, H, W) uint8
    """
    if n_channels is None:
        n_channels = detect_channels(tif_path)
    from core.multichannel import load_recording_multi
    return load_recording_multi(tif_path, dic_ch=dic_channel,
                                 fluo_ch=fluo_channel)


def load_recording(video_path, dic_channel=None, fluo_channel=None):
    """Load a video and its metadata as a dict.

    For single-channel files the result has the existing shape. For
    multichannel TIFFs, pass `dic_channel` (and optionally
    `fluo_channel`) to load DIC + fluorescence stacks; otherwise the
    file is loaded with channels collapsed (legacy behaviour).

    Returns:
        {
            "name": str,
            "frames": (N, H, W) uint8 — DIC if multichannel else raw,
            "cy5_frames": (N, H, W) uint8 or None — present when
                fluo_channel is given,
            "n_channels": int,
            "dic_channel": int (if multichannel),
            "fluo_channel": int or None,
            "um_per_px": float,
            "time_interval_min": float,
            "video_path": str,
        }
    """
    n_ch = detect_channels(video_path) if video_path.lower().endswith(
        (".tif", ".tiff")) else 1
    meta = load_metadata(video_path)
    meta["video_path"] = video_path
    meta["n_channels"] = n_ch
    meta["cy5_frames"] = None
    meta["dic_channel"] = None
    meta["fluo_channel"] = None
    if n_ch > 1 and dic_channel is not None:
        dic_frames, fluo_frames = load_video_multichannel(
            video_path, dic_channel=dic_channel,
            fluo_channel=fluo_channel if fluo_channel is not None else 0,
            n_channels=n_ch)
        meta["frames"] = dic_frames
        meta["cy5_frames"] = (
            fluo_frames if fluo_channel is not None else None)
        meta["dic_channel"] = dic_channel
        meta["fluo_channel"] = fluo_channel
    else:
        meta["frames"] = load_video(video_path)
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
