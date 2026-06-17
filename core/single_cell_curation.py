"""Parameterised single-cell curation for curated single-cell recordings.

When an experimenter hand-crops recordings to ONE target cell that is
isolated, flattened (not rounded), not dividing, present in every frame,
and roughly centred, those constraints are strong priors. Because exactly
one cell is present every frame, curation is posed as **assembling that
single cell's track frame-by-frame, label-agnostically**:

  * per (frame, object) features: area, centroid, and DIC TEXTURE (the
    in-mask intensity std). A real DIC cell is full of relief/organelle
    contrast → high texture; a uniform optical shadow is flat → low
    texture. Texture is what separates a large persistent SHADOW from the
    cell when size/persistence cannot;
  * a single-target track is assembled from a confident anchor (largest ×
    most-textured instance) by linking forward/backward, each frame
    picking argmax(size_score · texture · motion-continuity). This is
    LABEL-AGNOSTIC, so it ID-stitches across tracker label switches (the
    cell crossing onto a shadow and taking its label) universally;
  * the cell is tracked through ALL states — flattened AND rounded (a cell
    can crumple to a ball and return), so a small present mask is kept as a
    valid rounded state, not forced back to flattened. Only frames with NO
    acceptable pick (cell truly absent) are filled by SAM2 from a good
    neighbour. (Under-segmentation repair is an opt-in param, off by
    default, so genuine rounding isn't overwritten.);
  * EXCEPTIONS are flagged, never excluded: a second persistent, textured,
    cell-sized track (two cells / division) or unrecoverable frames mark
    the recording needs_manual_review.

Every action is recorded in a curation log (method reporting + the spec
for the built-in pipeline parameters).
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

CURATION_SCHEMA_VERSION = 2


@dataclass
class SingleCellCurationParams:
    """Experimenter priors. Defaults match Ignasi's IC293 curation."""
    enabled: bool = False
    cell_present_every_frame: bool = True
    no_dividing: bool = True
    cell_isolated: bool = True
    roughly_centered: bool = True
    expected_cell_area_um2: float = None    # None → inferred from the anchor
    area_low_frac: float = 0.4              # accept band (× expected)
    area_high_frac: float = 2.5
    recover_missing: bool = True
    # OFF by default: a cell can legitimately crumple to a ball and return
    # to flattened, so a small present mask is a real ROUNDED state to track
    # through (and keep outlined), NOT a detection error to enlarge. Enable
    # only when small masks are known to be under-segmentations, not rounding.
    repair_undersized: bool = False
    max_hop_px: float = 150.0               # per-frame cell motion gate
    min_pick_score: float = 0.04            # floor to accept a per-frame pick


# ───────────────────── per-object/per-frame features ─────────────────────
def _size_score(area_um2, expected):
    if expected <= 0 or area_um2 <= 0:
        return 0.0
    return math.exp(-(math.log(area_um2 / expected) / math.log(2.5)) ** 2)


def _texture(frame2d, mask2d):
    """In-mask DIC intensity std — high for a textured cell, low for a flat
    optical shadow."""
    px = frame2d[mask2d]
    return float(px.std()) if px.size else 0.0


def _object_table(labels, frames):
    """{cid: {frame: (area_px, (cy,cx), texture)}} over present frames."""
    tab = {}
    for cid in [int(v) for v in np.unique(labels) if v > 0]:
        m = labels == cid
        d = {}
        for f in np.where(m.any(axis=(1, 2)))[0]:
            mm = m[f]
            ys, xs = np.where(mm)
            d[int(f)] = (int(mm.sum()), (float(ys.mean()), float(xs.mean())),
                         _texture(frames[f], mm))
        if d:
            tab[cid] = d
    return tab


def _aggregate(tab, T):
    agg = {}
    for cid, d in tab.items():
        areas = np.array([v[0] for v in d.values()], float)
        texs = np.array([v[2] for v in d.values()], float)
        agg[cid] = {"present_frac": len(d) / T,
                    "med_area_px": float(np.median(areas)),
                    "med_tex": float(np.median(texs))}
    return agg


# ───────────────────────── track assembly ─────────────────────────
def _assemble(tab, T, um, params):
    """Assemble the single cell track. Returns
    (picks{frame:cid}, gaps[frames], E_um2, TB_texture, agg, anchor_cid)."""
    a2 = um * um
    agg = _aggregate(tab, T)
    # Anchor = the single clearest cell instance: max (area × texture).
    best = None
    for cid, d in tab.items():
        for f, (area, cen, tex) in d.items():
            s = area * tex
            if best is None or s > best[0]:
                best = (s, cid, f, cen)
    _, acid, af, acen = best
    E = (float(params.expected_cell_area_um2) if params.expected_cell_area_um2
         else float(agg[acid]["med_area_px"]) * a2)
    TB = float(agg[acid]["med_tex"]) or 1.0
    hop = params.max_hop_px

    byframe = {f: [] for f in range(T)}
    for cid, d in tab.items():
        for f, (area, cen, tex) in d.items():
            byframe[f].append((cid, area, cen, tex))

    def tex_norm(t):
        return t / (t + TB) if (t + TB) > 0 else 0.0

    def cont(c, prev):
        dist = ((c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2) ** 0.5
        return math.exp(-(dist / hop) ** 2)

    def size_cont(area_um2, prev_area):
        # size CONTINUITY vs the cell's previous size, NOT the flattened
        # expected: a cell rounding up shrinks smoothly (tracked); debris
        # has an abrupt size jump (rejected). Lets the cell be followed
        # through its rounded state.
        if prev_area <= 0 or area_um2 <= 0:
            return 0.0
        return math.exp(-(math.log(area_um2 / prev_area) / math.log(3.0)) ** 2)

    anchor_area = float(agg[acid]["med_area_px"]) * a2
    picks = {af: acid}

    def link(order, prev_pos, prev_area):
        for f in order:
            scored = sorted(
                ((tex_norm(tex) * cont(cen, prev_pos) * size_cont(area * a2, prev_area),
                  cid, cen, area * a2) for cid, area, cen, tex in byframe[f]),
                reverse=True)
            if scored and scored[0][0] >= params.min_pick_score:
                picks[f], prev_pos, prev_area = scored[0][1], scored[0][2], scored[0][3]

    link(range(af + 1, T), acen, anchor_area)
    link(range(af - 1, -1, -1), acen, anchor_area)
    gaps = [f for f in range(T) if f not in picks]
    return picks, gaps, E, TB, agg, acid


# ───────────────────────── SAM2 recovery ─────────────────────────
def _runs(sorted_frames):
    runs, s, p = [], None, None
    for f in sorted_frames:
        if s is None:
            s = p = f
        elif f == p + 1:
            p = f
        else:
            runs.append((s, p)); s = p = f
    if s is not None:
        runs.append((s, p))
    return runs


def _sam2_recover(frames, anchor_idx, anchor_mask, targets, predictor,
                  min_area, max_hop):
    from core.sam2_video import propagate_through_gap
    out = {}
    fwd = sorted(f for f in targets if f > anchor_idx)
    bwd = sorted((f for f in targets if f < anchor_idx), reverse=True)
    if fwd:
        out.update(propagate_through_gap(
            frames, anchor_idx, anchor_mask, fwd, predictor=predictor,
            min_area=min_area, max_hop_px=max_hop))
    if bwd:
        lo = min(bwd)
        sub = frames[lo:anchor_idx + 1][::-1]
        res = propagate_through_gap(
            sub, 0, anchor_mask, [anchor_idx - g for g in bwd],
            predictor=predictor, min_area=min_area, max_hop_px=max_hop)
        for li, m in res.items():
            out[anchor_idx - li] = m
    return out


# ───────────────────────── orchestration ─────────────────────────
def curate_single_cell(labels, frames, um_per_px, params, predictor=None):
    """Curate one single-cell recording → (new_labels, log)."""
    labels = np.asarray(labels)
    T = labels.shape[0]
    a2 = um_per_px * um_per_px
    log = {"schema": CURATION_SCHEMA_VERSION, "params": asdict(params),
           "expected_cell_area_um2": None, "primary_id": None,
           "primary_cellness": None, "n_objects_initial": 0,
           "artifacts_removed": [], "frames_recovered": [],
           "frames_repaired": [], "frames_unresolved": [], "id_stitch": [],
           "exceptions": [], "needs_manual_review": False,
           "final_presence": None}

    tab = _object_table(labels, frames)
    log["n_objects_initial"] = len(tab)
    if not tab:
        log["needs_manual_review"] = True
        log["exceptions"].append({"type": "no_objects", "detail": "no cells detected"})
        return labels, log

    picks, gaps, E, TB, agg, acid = _assemble(tab, T, um_per_px, params)
    log["expected_cell_area_um2"] = round(E, 1)
    picked_cids = set(picks.values())
    # dominant label = "primary"; multiple → an ID stitch
    from collections import Counter
    cnt = Counter(picks.values())
    log["primary_id"] = int(cnt.most_common(1)[0][0])
    if len(picked_cids) > 1:
        switches = sorted(f for f in range(1, T)
                          if f in picks and (f - 1) in picks
                          and picks[f] != picks[f - 1])
        log["id_stitch"] = [{"frame": int(f), "from": int(picks[f - 1]),
                             "to": int(picks[f])} for f in switches]
        # One clean switch (e.g. cell crossing a shadow) is fine; many
        # switches signal a division / unstable scene → flag for review.
        if len(log["id_stitch"]) >= 3:
            log["needs_manual_review"] = True
            log["exceptions"].append({
                "type": "frequent_label_switches",
                "detail": f"{len(log['id_stitch'])} label switches in the "
                          "assembled track — possible division / unstable "
                          "detection; review"})

    # Build the assembled cell mask stack (relabel to 1).
    new_labels = np.zeros_like(labels, dtype=np.int32)
    for f, cid in picks.items():
        new_labels[f][labels[f] == cid] = 1

    # SAM2 recovery: missing frames (gaps) + under-segmented picked frames.
    good = {f for f, cid in picks.items()
            if a2 * (labels[f] == cid).sum() >= params.area_low_frac * E
            and a2 * (labels[f] == cid).sum() <= params.area_high_frac * E}
    targets = set(gaps if (params.cell_present_every_frame
                           and params.recover_missing) else [])
    if params.repair_undersized:
        targets |= {f for f, cid in picks.items()
                    if a2 * (labels[f] == cid).sum() < params.area_low_frac * E}
    targets = sorted(targets)
    lo_a, hi_a = params.area_low_frac * E, params.area_high_frac * E
    if targets and good:
        try:
            if predictor is None:
                from core.sam2_video import _build_predictor
                predictor = _build_predictor()
        except Exception as e:
            predictor = None
            log["exceptions"].append({"type": "sam2_unavailable", "detail": str(e)})
        for a, b in _runs(targets):
            run = list(range(a, b + 1))
            rec = {}
            for anchor in (max((g for g in good if g < a), default=None),
                           min((g for g in good if g > b), default=None)):
                if anchor is None or predictor is None:
                    continue
                need = [f for f in run if f not in rec]
                if not need:
                    break
                for f, m in _sam2_recover(frames, anchor, new_labels[anchor] == 1,
                                          need, predictor, int(lo_a / a2),
                                          params.max_hop_px).items():
                    if lo_a <= float(m.sum()) * a2 <= hi_a:
                        rec[f] = m
            for f in run:
                was_missing = f in gaps
                if f in rec:
                    new_labels[f][:] = 0
                    new_labels[f][rec[f]] = 1
                    e = {"frame": int(f), "method": "sam2",
                         "area_um2": round(float(rec[f].sum()) * a2, 0)}
                    (log["frames_recovered"] if was_missing
                     else log["frames_repaired"]).append(e)
                elif was_missing:
                    log["frames_unresolved"].append({"frame": int(f), "was_missing": True})
    elif targets:
        log["frames_unresolved"] += [{"frame": int(f), "was_missing": f in gaps}
                                     for f in targets if f in gaps]

    # Artifacts = objects never used by the cell track.
    for cid, d in tab.items():
        if cid in picked_cids:
            continue
        log["artifacts_removed"].append({
            "id": cid, "n_frames": len(d),
            "med_area_um2": round(agg[cid]["med_area_px"] * a2, 0),
            "texture": round(agg[cid]["med_tex"], 1),
            "reason": ("low-texture optical artifact"
                       if agg[cid]["med_tex"] < 0.6 * TB else "debris/non-target")})

    # Exception: a second persistent, textured, cell-sized track (2 cells /
    # division). Flag, never exclude.
    for cid, d in tab.items():
        if cid in picked_cids:
            continue
        if (agg[cid]["present_frac"] >= 0.5
                and _size_score(agg[cid]["med_area_px"] * a2, E) >= 0.5
                and agg[cid]["med_tex"] >= 0.6 * TB):
            log["needs_manual_review"] = True
            log["exceptions"].append({
                "type": "possible_division_or_multiple_cells",
                "detail": f"object {cid} is a second persistent textured "
                          f"cell-sized track (present {agg[cid]['present_frac']:.0%}, "
                          f"{agg[cid]['med_area_px']*a2:.0f} µm², tex "
                          f"{agg[cid]['med_tex']:.0f}) — review"})

    if log["frames_unresolved"]:
        log["needs_manual_review"] = True
        log["exceptions"].append({
            "type": "unresolved_frames",
            "detail": f"{len(log['frames_unresolved'])} frame(s) could not be "
                      "conformed to the flattened profile — manual review"})

    # Confidence = mean per-frame (size·texture) over the assembled track.
    sc = []
    for f, cid in picks.items():
        area = a2 * (labels[f] == cid).sum()
        tex = tab[cid][f][2]
        sc.append(_size_score(area, E) * (tex / (tex + TB) if tex + TB else 0))
    log["primary_cellness"] = round(float(np.mean(sc)), 3) if sc else None
    log["final_presence"] = round(int((new_labels > 0).any(axis=(1, 2)).sum()) / T, 3)
    return new_labels, log
