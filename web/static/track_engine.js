// JS port of core/multi_cell.py::track_all_cells (the Hungarian tracker) and _label_cells_per_frame, plus a
// Hungarian solver standing in for scipy.optimize.linear_sum_assignment. Input: a label stack {labels: Int32Array
// (T*H*W), T, H, W} whose IDs need not agree between frames (raw per-frame detections). Output: tracks with, per
// frame, the detection label they took (0 = absent). Checked against the Python code by track_engine.test.mjs.
// Coordinates: pixel indices (row, col), centroids = mean pixel position (ndimage.center_of_mass).

export const LARGE = 1e9;

// Square cost matrix (array of Float64Array rows) -> assignment col[row]; minimum total cost (e-maxx Hungarian).
export function hungarian(cost) {
  const n = cost.length, INF = Infinity;
  const u = new Float64Array(n + 1), v = new Float64Array(n + 1), p = new Int32Array(n + 1), way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(INF), used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0], row = cost[i0 - 1];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = row[j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const col = new Int32Array(n);
  for (let j = 1; j <= n; j++) if (p[j]) col[p[j] - 1] = j - 1;
  return col;
}

// _label_cells_per_frame for one frame: cells in ascending label order with area >= minArea.
// Each cell: {label, cy, cx, area, px: Int32Array of pixel indices}.
export function frameCells(f, W, minArea) {
  const acc = new Map();
  for (let i = 0; i < f.length; i++) {
    const l = f[i];
    if (l <= 0) continue;
    let a = acc.get(l);
    if (!a) { a = { n: 0, sy: 0, sx: 0, idx: [] }; acc.set(l, a); }
    a.n++; a.sy += (i / W) | 0; a.sx += i % W; a.idx.push(i);
  }
  const out = [];
  for (const l of [...acc.keys()].sort((p, q) => p - q)) {
    const a = acc.get(l);
    if (a.n < minArea) continue;
    out.push({ label: l, cy: a.sy / a.n, cx: a.sx / a.n, area: a.n, px: Int32Array.from(a.idx) });
  }
  return out;
}

export const frameOf = (m, t) => m.labels.subarray(t * m.H * m.W, (t + 1) * m.H * m.W);

// track_all_cells(masks, min_area_px, max_hop_px, spawn_new_tracks, min_track_length, w_dist, w_area, w_iou,
// max_gap_frames). Returns [{labels: Int32Array(T) detection label per frame (0 absent), centroids: [[cy, cx]...]
// (centroid_history), first: first_frame, parent: index or null, n: frames present}].
export function trackAllCells(m, p) {
  const { minArea, maxHop, spawn = false, minLen = 1, wDist, wIou, wArea, maxGap } = p;
  const { T, W } = m;
  const per = [];
  for (let t = 0; t < T; t++) per.push(frameCells(frameOf(m, t), W, minArea));
  let seed = 0;
  while (seed < T && !per[seed].length) seed++;
  if (seed >= T) return [];
  let tracks = [];
  const newTrack = (t, c) => {
    const tr = { labels: new Int32Array(T), cells: new Array(T).fill(null), centroids: [[c.cy, c.cx]], first: t,
      last: t, alive: true, parent: null };
    tr.labels[t] = c.label; tr.cells[t] = c;
    return tr;
  };
  for (const c of per[seed]) tracks.push(newTrack(seed, c));
  for (let i = seed + 1; i < T; i++) {
    const cells = per[i], alive = tracks.filter(t => t.alive);
    if (!cells.length || !alive.length) {
      for (const t of alive) if (i - t.last > maxGap) t.alive = false;
      continue;
    }
    const nT = alive.length, nC = cells.length, N = Math.max(nT, nC), cur = frameOf(m, i);
    const index = new Map(cells.map((c, k) => [c.label, k]));
    const square = Array.from({ length: N }, () => new Float64Array(N).fill(LARGE));
    for (let r = 0; r < nT; r++) {
      const t = alive[r], [py, pxx] = t.centroids[t.centroids.length - 1];
      const allowed = maxHop * Math.min(Math.max(1, i - t.last), 2);
      const prev = t.cells[t.last], prevArea = prev.area;
      let inter = null;
      if (wIou > 0) {                             // overlap of the previous mask with every current cell
        inter = new Float64Array(nC);
        for (const q of prev.px) { const k = index.get(cur[q]); if (k !== undefined) inter[k]++; }
      }
      for (let k = 0; k < nC; k++) {
        const c = cells[k], d = Math.sqrt((py - c.cy) ** 2 + (pxx - c.cx) ** 2);
        let cost = wDist * d;
        if (wIou > 0 || wArea > 0) {
          if (wIou > 0) {
            let iou = 0;
            if (inter[k] > 0) { const uni = prevArea + c.area - inter[k]; if (uni > 0) iou = inter[k] / uni; }
            cost = cost + wIou * (1 - iou) * maxHop;
          }
          if (wArea > 0) cost = cost + wArea * (Math.abs(prevArea - c.area) / Math.max(Math.max(prevArea, c.area), 1)) * maxHop;
        }
        square[r][k] = d <= allowed ? cost : LARGE;
      }
    }
    const col = hungarian(square);
    const assignedT = new Set(), assignedC = new Set();
    for (let r = 0; r < N; r++) {
      const c = col[r];
      if (r >= nT || c >= nC || square[r][c] >= LARGE) continue;
      const t = alive[r], cell = cells[c];
      t.labels[i] = cell.label; t.cells[i] = cell; t.centroids.push([cell.cy, cell.cx]); t.last = i;
      assignedT.add(r); assignedC.add(c);
    }
    alive.forEach((t, r) => { if (!assignedT.has(r) && i - t.last > maxGap) t.alive = false; });
    if (spawn) cells.forEach((c, k) => { if (!assignedC.has(k)) tracks.push(newTrack(i, c)); });
  }
  for (const t of tracks) t.n = t.labels.reduce((a, l) => a + (l > 0 ? 1 : 0), 0);
  if (minLen > 1) tracks = tracks.filter(t => t.n >= minLen);
  // division heuristic: a track starting next to a track that just ended, at 20-90 % of its area
  tracks.forEach((t, a) => {
    if (t.parent !== null || t.first === 0) return;
    const [cy0, cx0] = t.centroids[0], newArea = t.cells[t.first].area;
    for (let b = 0; b < tracks.length; b++) {
      if (a === b) continue;
      const q = tracks[b];
      let pLast = -1;
      for (let s = T - 1; s >= 0; s--) if (q.labels[s]) { pLast = s; break; }
      if (pLast < 0 || (pLast !== t.first - 1 && pLast !== t.first)) continue;
      const [py, pxx] = q.centroids[q.centroids.length - 1];
      if (Math.sqrt((cy0 - py) ** 2 + (cx0 - pxx) ** 2) > maxHop) continue;
      const parentArea = q.cells[pLast].area;
      if (!parentArea) continue;
      const ratio = newArea / parentArea;
      if (ratio >= 0.2 && ratio <= 0.9) { t.parent = b; break; }
    }
  });
  return tracks.map(t => ({ labels: t.labels, centroids: t.centroids, first: t.first, parent: t.parent, n: t.n }));
}

// Relabel the detections as tracked cells: track k -> label k + 1 (0 elsewhere). Returns a new stack.
export function trackedStack(m, tracks) {
  const out = new Int32Array(m.labels.length), HW = m.H * m.W;
  for (let t = 0; t < m.T; t++) {
    const map = new Map();
    tracks.forEach((tr, k) => { if (tr.labels[t]) map.set(tr.labels[t], k + 1); });
    if (!map.size) continue;
    const f = frameOf(m, t), o = out.subarray(t * HW, (t + 1) * HW);
    for (let i = 0; i < HW; i++) { const v = map.get(f[i]); if (v) o[i] = v; }
  }
  return { labels: out, T: m.T, H: m.H, W: m.W };
}

// Per-recording px thresholds the way the Tracking GUI's _on_track gets them (DEFAULTS.pixel_thresholds) plus the
// tracker defaults: d = DEFAULTS as JSON from the API, plan = planner_engine.pixelThresholds.
export function guiTrackParams(d, px) {
  return { minArea: px.min_area_px, maxHop: px.max_hop_px, spawn: true, minLen: d.min_track_length,
    wDist: d.track_w_dist, wIou: d.track_w_iou, wArea: d.track_w_area, maxGap: d.max_gap_frames };
}

// Per-track frame -> [cy, cx] from the centroid history (one entry per assigned frame, in order).
export function trackPaths(tracks, T) {
  return tracks.map(tr => {
    const out = new Array(T).fill(null);
    let j = 0;
    for (let t = 0; t < T; t++) if (tr.labels[t]) out[t] = tr.centroids[j++];
    return out;
  });
}
