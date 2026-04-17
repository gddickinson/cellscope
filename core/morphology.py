"""Cell shape descriptors."""
import numpy as np
from skimage import measure


def shape_descriptors(mask, um_per_px):
    """Per-frame shape metrics."""
    if not mask.any():
        return {k: float("nan") for k in [
            "area_um2", "perimeter_um", "circularity", "solidity",
            "aspect_ratio", "eccentricity",
        ]}
    props = measure.regionprops(mask.astype(np.uint8))[0]
    area_px = float(props.area)
    per_px = float(props.perimeter)
    area_um = area_px * um_per_px ** 2
    per_um = per_px * um_per_px
    circ = (4 * np.pi * area_px / (per_px ** 2)) if per_px > 0 else 0.0
    minr, minc, maxr, maxc = props.bbox
    aspect = (maxc - minc) / max(maxr - minr, 1e-6)
    return {
        "area_um2": area_um,
        "perimeter_um": per_um,
        "circularity": float(circ),
        "solidity": float(props.solidity),
        "aspect_ratio": float(aspect),
        "eccentricity": float(props.eccentricity),
    }


def shape_timeseries(masks, um_per_px):
    """Stack of shape descriptors over frames."""
    keys = ["area_um2", "perimeter_um", "circularity",
            "solidity", "aspect_ratio", "eccentricity"]
    out = {k: np.full(len(masks), np.nan) for k in keys}
    for i, m in enumerate(masks):
        d = shape_descriptors(m, um_per_px)
        for k in keys:
            out[k][i] = d[k]
    return out


def shape_summary(ts):
    """Mean/std/min/max of each descriptor."""
    out = {}
    for k, v in ts.items():
        valid = v[~np.isnan(v)]
        out[k] = {
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "std": float(np.std(valid)) if len(valid) else float("nan"),
            "min": float(np.min(valid)) if len(valid) else float("nan"),
            "max": float(np.max(valid)) if len(valid) else float("nan"),
        }
    return out
