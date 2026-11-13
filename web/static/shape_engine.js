// JS port of CellScope's per-cell shape + state + colour-by metrics: core/mask_cleanup.py::clean_cell_mask,
// core/cell_state.py (shape_metrics_for_mask, classify_state, state_fraction), core/edge_filter.py
// (mask_touches_edge) and core/mask_metrics.py::compute_label_metrics. scikit-image's regionprops quantities are
// reproduced: area (pixel count), perimeter (Benkrid-Crookes, neighbourhood 4), solidity (area / pixel count of
// convex_hull_image, the hull of the pixel-edge midpoints, borders included), eccentricity (inertia tensor of the
// central moments) and the bbox aspect ratio (width / height). Checked against the Python code by
// shape_engine.test.mjs. Coordinates: pixel indices (row, col); centroids = mean pixel position.

export const STATE_CODE = { unknown: 0, spread: 1, rounded: 2 };
export const STATE_CODE_EDGE = 3;
export const SHAPE_KEYS = ['area', 'perimeter', 'circularity', 'solidity', 'eccentricity', 'aspect_ratio'];
const F = Math.fround;

// Perimeter weights indexed by the 3x3 convolution code (skimage measure.perimeter, neighborhood=4).
const PW = new Float64Array(50);
[5, 7, 15, 17, 25, 27].forEach(i => { PW[i] = 1; });
[21, 33].forEach(i => { PW[i] = Math.SQRT2; });
[13, 23].forEach(i => { PW[i] = (1 + Math.SQRT2) / 2; });

// 4-connected components of a binary crop (Uint8Array h*w) in raster order of first pixel; returns Int32Array.
export function components(b, h, w) {
  const lab = new Int32Array(h * w), stack = [];
  let n = 0;
  for (let s = 0; s < b.length; s++) {
    if (!b[s] || lab[s]) continue;
    lab[s] = ++n; stack.push(s);
    while (stack.length) {
      const i = stack.pop(), y = (i / w) | 0, x = i - y * w;
      if (y > 0 && b[i - w] && !lab[i - w]) { lab[i - w] = n; stack.push(i - w); }
      if (y < h - 1 && b[i + w] && !lab[i + w]) { lab[i + w] = n; stack.push(i + w); }
      if (x > 0 && b[i - 1] && !lab[i - 1]) { lab[i - 1] = n; stack.push(i - 1); }
      if (x < w - 1 && b[i + 1] && !lab[i + 1]) { lab[i + 1] = n; stack.push(i + 1); }
    }
  }
  return { lab, n };
}

// clean_cell_mask on a crop with a one-pixel empty border: keep the largest 4-connected component (first on ties,
// like argmax over ndimage.label sizes), then fill holes (background not 4-connected to the border).
function clean(b, h, w) {
  const { lab, n } = components(b, h, w);
  if (n > 1) {
    const size = new Int32Array(n + 1);
    for (const l of lab) size[l]++;
    let best = 1;
    for (let k = 2; k <= n; k++) if (size[k] > size[best]) best = k;
    for (let i = 0; i < b.length; i++) b[i] = lab[i] === best ? 1 : 0;
  }
  const out = new Uint8Array(b.length), stack = [0];
  out[0] = 1;                                         // flood the outside background from the padded corner
  while (stack.length) {
    const i = stack.pop(), y = (i / w) | 0, x = i - y * w;
    for (const j of [y > 0 ? i - w : -1, y < h - 1 ? i + w : -1, x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1]) {
      if (j >= 0 && !out[j] && !b[j]) { out[j] = 1; stack.push(j); }
    }
  }
  for (let i = 0; i < b.length; i++) b[i] = out[i] ? 0 : 1;
  return b;
}

const nanMetrics = (edge = false) => ({ area: NaN, perimeter: NaN, circularity: NaN, solidity: NaN, eccentricity: NaN,
  aspect_ratio: NaN, edge_touch: !!edge });

// shape_metrics_for_mask for one cell-frame given its pixel indices in a (H, W) frame. minArea = DEFAULT_THRESHOLDS
// min_area_px, margin = edge_margin_px.
export function shapeMetrics(px, H, W, { minArea = 200, margin = 0 } = {}) {
  if (!px.length) return nanMetrics();
  let y0 = H, y1 = -1, x0 = W, x1 = -1;
  for (const i of px) { const y = (i / W) | 0, x = i - y * W; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x; }
  const h = y1 - y0 + 3, w = x1 - x0 + 3, b = new Uint8Array(h * w);
  for (const i of px) { const y = (i / W) | 0, x = i - y * W; b[(y - y0 + 1) * w + (x - x0 + 1)] = 1; }
  clean(b, h, w);
  // regionprops of the single cleaned region (crop coordinates; +y0-1 / +x0-1 back to the frame)
  let n = 0, sy = 0, sx = 0, ry0 = h, ry1 = -1, rx0 = w, rx1 = -1;
  const rowL = new Int32Array(h).fill(-1), rowR = new Int32Array(h).fill(-1);
  for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
    if (!b[i]) continue;
    n++; sy += y; sx += x;
    if (y < ry0) ry0 = y; if (y > ry1) ry1 = y; if (x < rx0) rx0 = x; if (x > rx1) rx1 = x;
    if (rowL[y] < 0) rowL[y] = x;
    rowR[y] = x;
  }
  const m = margin, fy0 = ry0 + y0 - 1, fy1 = ry1 + y0 - 1, fx0 = rx0 + x0 - 1, fx1 = rx1 + x0 - 1;
  const edge = fy0 <= m || fx0 <= m || fy1 >= H - 1 - m || fx1 >= W - 1 - m;
  if (n < minArea) return nanMetrics(edge);
  const cy = sy / n, cx = sx / n;
  let mrr = 0, mcc = 0, mrc = 0;
  for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
    if (!b[i]) continue;
    const dr = y - cy, dc = x - cx;
    mrr += dr * dr; mcc += dc * dc; mrc += dr * dc;
  }
  // inertia tensor [[mcc, -mrc], [-mrc, mrr]] / n -> eigenvalues
  const a = mcc / n, d = mrr / n, c = -mrc / n, half = (a + d) / 2, disc = Math.sqrt(((a - d) / 2) ** 2 + c * c);
  const l1 = Math.max(half + disc, 0), l2 = Math.max(half - disc, 0);
  const ecc = l1 === 0 ? 0 : Math.sqrt(1 - l2 / l1);
  // perimeter (border pixels: in the mask with a 4-neighbour outside)
  const border = new Uint8Array(h * w);
  for (let y = 1, i = w + 1; y < h - 1; y++, i = y * w + 1) for (let x = 1; x < w - 1; x++, i++) {
    if (b[i] && !(b[i - 1] && b[i + 1] && b[i - w] && b[i + w])) border[i] = 1;
  }
  let per = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    if (!border[i]) continue;
    const code = 1 + 2 * (border[i - 1] + border[i + 1] + border[i - w] + border[i + w])
      + 10 * (border[i - w - 1] + border[i - w + 1] + border[i + w - 1] + border[i + w + 1]);
    per += code < 50 ? PW[code] : 0;
  }
  const circ = per > 0 ? 4 * Math.PI * n / (per * per) : NaN;
  return { area: n, perimeter: per, circularity: circ, solidity: n / convexPixels(rowL, rowR, h, w),
    eccentricity: ecc, aspect_ratio: (rx1 - rx0 + 1) / Math.max(ry1 - ry0 + 1, 1e-6), edge_touch: edge };
}

// Pixel count of skimage convex_hull_image: hull of the pixel-edge midpoints (the extreme pixels of each row
// suffice), pixel centres inside or on the hull. Doubled coordinates keep every test exact.
function convexPixels(rowL, rowR, h, w) {
  const pts = [];
  for (let y = 0; y < h; y++) {
    if (rowL[y] < 0) continue;
    for (const x of rowL[y] === rowR[y] ? [rowL[y]] : [rowL[y], rowR[y]]) {
      pts.push([2 * y - 1, 2 * x], [2 * y + 1, 2 * x], [2 * y, 2 * x - 1], [2 * y, 2 * x + 1]);
    }
  }
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of pts) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = pts.length - 1; i >= 0; i--) { const q = pts[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  const hull = lo.slice(0, -1).concat(up.slice(0, -1)), k = hull.length;    // counter-clockwise in (row, col)
  let count = 0;
  for (let y = 0; y < h; y++) {                       // each row: the hull's x-interval at Y = 2y
    const Y = 2 * y;
    let lo = Infinity, hi = -Infinity;
    for (let e = 0; e < k; e++) {
      const [ya, xa] = hull[e], [yb, xb] = hull[(e + 1) % k];
      if (Y < Math.min(ya, yb) || Y > Math.max(ya, yb)) continue;
      const xs = ya === yb ? [xa, xb] : [xa + (Y - ya) * (xb - xa) / (yb - ya)];
      for (const x of xs) { if (x < lo) lo = x; if (x > hi) hi = x; }
    }
    if (lo > hi) continue;
    const a = Math.max(0, Math.ceil(lo / 2)), b = Math.min(w - 1, Math.floor(hi / 2));
    if (b >= a) count += b - a + 1;
  }
  return count;
}

// classify_state(metrics, thresholds, um_per_px) -> 'rounded' | 'spread' | 'unknown'
export function classifyState(mt, th, um) {
  if (mt.edge_touch) return 'unknown';
  const area = mt.area, ecc = mt.eccentricity;
  if (um && Number.isFinite(area) && Number.isFinite(ecc)) {
    return area * um ** 2 <= th.rounded_area_um2 && ecc <= th.rounded_eccentricity ? 'rounded' : 'spread';
  }
  const circ = mt.circularity, sol = mt.solidity;
  if (Number.isNaN(circ) || Number.isNaN(sol)) return 'unknown';
  return circ >= th.rounded_circ && sol >= th.rounded_solid ? 'rounded' : 'spread';
}

// Pixel lists of every label in a stack: Map(id -> Array(T) of Int32Array|null).
export function pixelsById(m) {
  const HW = m.H * m.W, out = new Map();
  for (let t = 0; t < m.T; t++) {
    const acc = new Map(), base = t * HW;
    for (let i = 0; i < HW; i++) { const l = m.labels[base + i]; if (l > 0) { let a = acc.get(l); if (!a) acc.set(l, a = []); a.push(i); } }
    for (const [l, a] of acc) { if (!out.has(l)) out.set(l, new Array(m.T).fill(null)); out.get(l)[t] = Int32Array.from(a); }
  }
  return out;
}

const mean = a => { const v = a.filter(Number.isFinite); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN; };

// compute_label_metrics(labels, um_per_px, dt_min, thresholds) -> {per_track, per_frame, states, metrics, n_frames,
// cell_ids}. metrics[id] holds the per-frame shape arrays (float32 values, as in Python) so callers can reclassify.
export function computeLabelMetrics(m, um, dt, th, { maxCells = 200, onCell } = {}) {
  const out = { per_track: {}, per_frame: {}, states: {}, metrics: {}, n_frames: m.T, cell_ids: [] };
  const byId = pixelsById(m), ids = [...byId.keys()].sort((a, b) => a - b).slice(0, maxCells);
  out.cell_ids = ids;
  const hasScale = !!(um && dt), U = um ? +um : 1, D = dt ? +dt : 1, n = m.T;
  ids.forEach((cid, ci) => {
    const pxs = byId.get(cid), mt = {}, states = [], edge = [];
    SHAPE_KEYS.forEach(k => { mt[k] = new Array(n).fill(NaN); });
    const cents = [];
    for (let t = 0; t < n; t++) {
      const px = pxs[t];
      const d = px ? shapeMetrics(px, m.H, m.W, { minArea: th.min_area_px, margin: th.edge_margin_px }) : nanMetrics();
      SHAPE_KEYS.forEach(k => { mt[k][t] = F(d[k]); });
      edge.push(!!d.edge_touch);
      states.push(classifyState(d, th, um));
      if (px) { let sy = 0, sx = 0; for (const i of px) { sy += (i / m.W) | 0; sx += i % m.W; } cents.push([sy / px.length, sx / px.length]); }
      else cents.push([NaN, NaN]);
    }
    const areaUm2 = mt.area.map(a => (hasScale ? F(a * F(U * U)) : a));
    const speed = new Array(n).fill(NaN);
    for (let t = 0; t < n - 1; t++) speed[t] = F(Math.hypot(cents[t + 1][0] - cents[t][0], cents[t + 1][1] - cents[t][1]) * (U / D));
    out.states[cid] = states.map((s, t) => (edge[t] ? STATE_CODE_EDGE : STATE_CODE[s] ?? 0));
    out.per_frame[cid] = { area: areaUm2, circularity: mt.circularity, solidity: mt.solidity, speed };
    out.metrics[cid] = { ...mt, edge };
    const valid = cents.filter(c => !Number.isNaN(c[0])), k = hasScale ? U : 1;
    let total = 0, net = 0, pers = 0;
    if (valid.length >= 2) {
      for (let i = 1; i < valid.length; i++) total += Math.hypot(valid[i][0] - valid[i - 1][0], valid[i][1] - valid[i - 1][1]) * k;
      const a = valid[0], b = valid[valid.length - 1];
      net = Math.hypot(b[0] - a[0], b[1] - a[1]) * k;
      pers = total > 0 ? net / total : 0;
    }
    const known = states.filter(s => s !== 'unknown');
    out.per_track[cid] = {
      mean_speed: mean(speed), total_distance: total, net_displacement: net, persistence: pers,
      mean_area: mean(areaUm2), mean_circularity: mean(mt.circularity), mean_solidity: mean(mt.solidity),
      mean_aspect_ratio: mean(mt.aspect_ratio), mean_eccentricity: mean(mt.eccentricity),
      frames_tracked: pxs.filter(Boolean).length,
      frac_rounded: known.length ? known.filter(s => s === 'rounded').length / known.length : 0,
    };
    if (onCell) onCell(ci + 1, ids.length);
  });
  return out;
}
