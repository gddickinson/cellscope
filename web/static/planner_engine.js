// JS ports of the small pure planning functions in core/pipeline_defaults.py (resolve_downsample,
// should_use_mirror_pad, decide_backbone, PipelineDefaults.pixel_thresholds) and core/runtime_estimate.py.
// Constants come from the API (/api/planner, read from the Python module), so the page and Python never disagree.
// Checked against the Python code by planner_engine.test.mjs.

// Python's round(): halves go to the even neighbour.
export function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// resolve_downsample(spec, frame_shape) -> [factor, reason]
export function resolveDownsample(spec, [h, w], C) {
  if (spec == null || ['off', 'Off', 'OFF'].includes(spec)) return [1, 'downsample disabled'];
  if (typeof spec === 'string' && spec.toLowerCase() === 'auto') {
    const maxDim = Math.max(h, w), small = C.DOWNSAMPLE_SMALL_PX, large = C.DOWNSAMPLE_LARGE_PX;
    if (maxDim < small) return [1, `auto: max dim ${maxDim} < ${small}; cells would be too small at ds≥2`];
    if (maxDim < large) return [2, `auto: max dim ${maxDim} in [${small}, ${large}); ds=2 keeps cells resolvable while giving a ~3× speedup`];
    return [2, `auto: max dim ${maxDim} ≥ ${large}; ds=2 gives ~5× speedup with minimal accuracy loss`];
  }
  const factor = Number.parseInt(spec, 10);
  if (!Number.isFinite(factor)) throw new Error(`downsample spec not understood: ${spec}`);
  if (factor < 1) throw new Error(`downsample factor must be ≥ 1, got ${factor}`);
  return [factor, `explicit factor ${factor}`];
}

// should_use_mirror_pad(use_mirror_pad, det_frame_shape)
export function shouldUseMirrorPad(spec, shape, C) {
  if (spec == null || ['off', 'Off', 'OFF', false].includes(spec)) return false;
  if (['on', 'On', 'ON', true].includes(spec)) return true;
  if (!shape) return false;
  return Math.min(shape[0], shape[1]) >= C.MIRROR_PAD_MIN_DIM_PX;
}

// numpy.percentile, linear interpolation
export function percentile(a, q) {
  const s = [...a].sort((x, y) => x - y), pos = (s.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// decide_backbone(counts) -> {choice, p75_count, median_count, threshold, reason}
export function decideBackbone(counts, C) {
  const p = C.PROBE_AGGREGATOR_PERCENTILE, thr = C.MULTICELL_THRESHOLD, d = percentile(counts, p);
  const out = { p75_count: d, decision_count: d, median_count: percentile(counts, 50), threshold: thr, aggregator: `p${p}` };
  if (d >= thr) Object.assign(out, { choice: 'cpsam', reason: `p${p} ${d.toFixed(1)} cells/frame ≥ ${thr}: multi-cell scene, switching to raw cpsam (cpsam_dic merges touching cells)` });
  else Object.assign(out, { choice: 'cpsam_dic', reason: `p${p} ${d.toFixed(1)} cells/frame < ${thr}: single-cell scene, using cpsam_dic (tighter boundaries on isolated cells)` });
  return out;
}

// PipelineDefaults.pixel_thresholds(um_per_px, time_interval_min); D = DEFAULTS as JSON
export function pixelThresholds(D, um, dt) {
  if (!um || um <= 0) {
    return { min_area_px: D.min_area_px, max_hop_px: dt ? pyRound(D.max_hop_um_per_min * (dt || 1) / 1) : 150,
      search_radius_px: D.search_radius_px, cell_diameter_px: null };
  }
  const minArea = Math.max(1, pyRound(D.min_area_um2 / (um * um)));
  const diam = pyRound(D.expected_cell_diameter_um / um);
  const maxHop = dt && dt > 0 ? Math.max(10, pyRound(D.max_hop_um_per_min * dt / um)) : D.search_radius_px;
  return { min_area_px: minArea, max_hop_px: maxHop, search_radius_px: Math.max(10, pyRound(D.search_radius_um / um)),
    cell_diameter_px: diam };
}

// core/runtime_estimate.py
export function estimateFullRuntime(elapsed, nCells, nFrames, mults) {
  const [mult, density] = (mults.find(([lim]) => lim === null || nCells < lim) || []).slice(1);
  const detect = elapsed * nFrames;
  return { detect_only_s: detect, full_s: detect * mult, multiplier: mult, density };
}

export const formatDuration = s => (s < 60 ? `${s.toFixed(0)}s` : s < 3600 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`);

// numpy.linspace(0, n - 1, k, dtype=int): the frames the auto-select probe samples
export function probeFrames(n, k) {
  k = Math.min(k, n);
  if (k <= 1) return [0];
  const step = (n - 1) / (k - 1);
  return Array.from({ length: k }, (_, i) => Math.trunc(i === k - 1 ? n - 1 : i * step));
}
