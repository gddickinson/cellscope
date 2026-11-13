// Node test: the JS engines against values computed by CellScope's Python code on the same synthetic masks
// (tests/fixtures/web_engine, written by scripts/make_web_fixture.py). Run: node web/static/engine.test.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readNpz } from './npz.js';
import { hungarian, trackAllCells, trackedStack, guiTrackParams } from './track_engine.js';
import { shapeMetrics, classifyState, computeLabelMetrics, SHAPE_KEYS } from './shape_engine.js';
import { resolveDownsample, shouldUseMirrorPad, decideBackbone, pixelThresholds, estimateFullRuntime, formatDuration, probeFrames } from './planner_engine.js';
import { npzBlob } from './npz_write.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = join(here, '..', '..', 'tests', 'fixtures', 'web_engine');
const E = JSON.parse(readFileSync(join(fx, 'expected.json'), 'utf8'));
const buf = readFileSync(join(fx, 'masks.npz'));
const { arrays } = await readNpz(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const stack = a => ({ labels: Int32Array.from(a.data), T: a.shape[0], H: a.shape[1], W: a.shape[2] });
const labels = stack(arrays.labels), shapes = stack(arrays.shapes);

let checks = 0, fails = 0;
function close(a, b, what, rtol = 1e-6, atol = 1e-9) {
  checks++;
  const na = a === null || a === undefined || Number.isNaN(a), nb = b === null || b === undefined || Number.isNaN(b);
  if (na && nb) return;
  if (typeof b === 'boolean' || typeof b === 'string' || typeof a === 'string') { if (a === b) return; }
  else if (!na && !nb && Math.abs(a - b) <= atol + rtol * Math.abs(b)) return;
  fails++;
  if (fails < 30) console.log(`FAIL ${what}: js=${a} py=${b}`);
}
const C = { DOWNSAMPLE_SMALL_PX: 1100, DOWNSAMPLE_LARGE_PX: 1500, MIRROR_PAD_MIN_DIM_PX: 1000,
  MULTICELL_THRESHOLD: 1.5, PROBE_AGGREGATOR_PERCENTILE: 75 };
const MULTS = [[5, 1.5, 'sparse'], [10, 2.0, 'medium'], [null, 2.5, 'dense']];

// Hungarian: same minimum total cost as scipy's linear_sum_assignment
for (const { cost, total } of E.lsa) {
  const col = hungarian(cost.map(r => Float64Array.from(r)));
  close(col.reduce((s, c, r) => s + cost[r][c], 0), total, `lsa n=${cost.length}`, 1e-12);
}

// Tracker
const P = p => ({ minArea: p.min_area_px ?? 200, maxHop: p.max_hop_px ?? 150, spawn: !!p.spawn_new_tracks,
  minLen: p.min_track_length ?? 1, wDist: p.w_dist ?? E.defaults.track_w_dist, wIou: p.w_iou ?? E.defaults.track_w_iou,
  wArea: p.w_area ?? E.defaults.track_w_area, maxGap: p.max_gap_frames ?? E.defaults.max_gap_frames });
for (const [ci, tc] of E.tracking.entries()) {
  const js = trackAllCells(labels, P(tc.params));
  close(js.length, tc.tracks.length, `case ${ci} n tracks`);
  tc.tracks.forEach((pt, k) => {
    const jt = js[k];
    if (!jt) return;
    close(jt.labels.join(','), pt.labels.join(','), `case ${ci} track ${k} labels`);
    close(jt.first, pt.first, `case ${ci} track ${k} first`);
    close(jt.parent ?? null, pt.parent, `case ${ci} track ${k} parent`);
    pt.centroids.forEach((c, i) => { close(jt.centroids[i][0], c[0], `c${ci} t${k} cy${i}`); close(jt.centroids[i][1], c[1], `c${ci} t${k} cx${i}`); });
  });
}
const px = pixelThresholds(E.defaults, E.um_per_px, E.dt_min);
close(JSON.stringify(P(E.tracking[0].params)), JSON.stringify(guiTrackParams(E.defaults, px)), 'GUI tracking params');

// Shape metrics + state, per detection
for (const r of E.shapes) {
  const m = r.stack === 'labels' ? labels : shapes, HW = m.H * m.W, px2 = [];
  for (let i = 0; i < HW; i++) if (m.labels[r.frame * HW + i] === r.label) px2.push(i);
  const d = shapeMetrics(Int32Array.from(px2), m.H, m.W, { minArea: E.thresholds.min_area_px });
  const tag = `${r.stack} f${r.frame} l${r.label}`;
  for (const k of SHAPE_KEYS) close(d[k], r.metrics[k], `${tag} ${k}`, 1e-9);
  close(d.edge_touch, r.metrics.edge_touch, `${tag} edge`);
  close(classifyState(d, E.thresholds, E.um_per_px), r.state_um, `${tag} state (scaled)`);
  close(classifyState(d, E.thresholds, null), r.state_noscale, `${tag} state (no scale)`);
}

// compute_label_metrics on the tracked stack, with and without a scale
const tracked = trackedStack(labels, trackAllCells(labels, P(E.tracking[0].params)));
for (const [key, um, dt] of [['metrics_scaled', E.um_per_px, E.dt_min], ['metrics_noscale', null, null]]) {
  const py = E[key], js = computeLabelMetrics(tracked, um, dt, E.thresholds);
  close(js.cell_ids.join(','), py.cell_ids.join(','), `${key} ids`);
  for (const id of py.cell_ids) {
    for (const [k, v] of Object.entries(py.per_track[id])) close(js.per_track[id][k], v, `${key} ${id} ${k}`, 1e-5);
    for (const [k, arr] of Object.entries(py.per_frame[id])) arr.forEach((v, t) => close(js.per_frame[id][k][t], v, `${key} ${id} ${k}[${t}]`, 1e-6));
    close(js.states[id].join(','), py.states[id].join(','), `${key} ${id} states`);
  }
}

// Planner
for (const c of E.planner.downsample) { const o = resolveDownsample(typeof c.spec === 'number' ? String(c.spec) : c.spec, c.shape, C); close(o[0], c.out[0], `ds ${c.spec} ${c.shape}`); close(o[1], c.out[1], `ds reason ${c.spec} ${c.shape}`); }
for (const c of E.planner.mirror) close(shouldUseMirrorPad(c.spec, c.shape, C), c.out, `mirror ${c.spec} ${c.shape}`);
for (const c of E.planner.backbone) {
  const o = decideBackbone(c.counts, C);
  for (const k of ['choice', 'p75_count', 'median_count', 'reason']) close(o[k], c.out[k], `backbone ${c.counts} ${k}`);
}
for (const c of E.planner.pixel) {
  const o = pixelThresholds(E.defaults, c.um, c.dt);
  for (const k of Object.keys(c.out)) close(o[k], c.out[k], `pixel ${c.um} ${c.dt} ${k}`, 0);
}
for (const c of E.planner.runtime) {
  const o = estimateFullRuntime(...c.args, MULTS);
  for (const k of Object.keys(c.out)) close(o[k], c.out[k], `runtime ${c.args} ${k}`);
}
for (const c of E.planner.probe) close(probeFrames(c.n, c.k).join(','), c.out.join(','), `probe frames n=${c.n}`);
close(formatDuration(42), '42s', 'fmt s'); close(formatDuration(210), '3.5 min', 'fmt min'); close(formatDuration(9720), '2.7 h', 'fmt h');

// npz writer round trip (the tracked-masks download)
const tr = trackedStack(labels, trackAllCells(labels, P(E.tracking[0].params)));
const blob = await npzBlob({ labels: { data: tr.labels, shape: [tr.T, tr.H, tr.W] } });
const back = (await readNpz(await blob.arrayBuffer())).arrays.labels;
close(back.shape.join(','), [tr.T, tr.H, tr.W].join(','), 'npz shape');
close(back.dtype, '<i4', 'npz dtype');
close(back.data.every((v, i) => v === tr.labels[i]), true, 'npz data');

console.log(`${checks - fails}/${checks} checks ok`);
if (fails) process.exit(1);
