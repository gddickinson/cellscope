// The dataset shared by the pages: the synthetic demo (default) or the user's own masks.npz (+ optional .ome.tif
// and sidecar .json), read locally in the browser. Tracking + metrics run in a Web Worker (worker.js) and are cached
// per dataset and tracker settings. Nothing is uploaded anywhere.
import { api } from './core.js';
import { readNpz } from './npz.js';
import { openTiff } from './tiff.js';
import { components } from './shape_engine.js';
import { pixelThresholds } from './planner_engine.js';
import { guiTrackParams } from './track_engine.js';

let DS = null, DEF = null;
const listeners = new Set();
export const onDataset = fn => { listeners.add(fn); return () => listeners.delete(fn); };

export async function defaults() {
  if (!DEF) DEF = await api('/defaults');
  return DEF;
}
export const thresholdsObj = d => Object.fromEntries(d.thresholds.map(r => [r.name, r.value]));

// {labels, T, H, W, key} from an .npz: key "labels" (CellScope masks.npz), else the first array. Boolean masks are
// split into 4-connected cells per frame, as core.multi_cell does for a bool stack.
export async function masksFromNpz(buf) {
  const { arrays } = await readNpz(buf);
  const key = 'labels' in arrays ? 'labels' : Object.keys(arrays)[0];
  if (!key) throw new Error('no numeric arrays in this .npz');
  const a = arrays[key];
  let shape = a.shape.length === 2 ? [1, ...a.shape] : a.shape;
  if (shape.length !== 3) throw new Error(`array "${key}" has shape (${a.shape}); expected (T, H, W)`);
  const [T, H, W] = shape, n = T * H * W, labels = new Int32Array(n);
  const big = typeof a.data[0] === 'bigint';
  for (let i = 0; i < n; i++) labels[i] = big ? Number(a.data[i]) : a.data[i];
  if (a.dtype === '|b1') {
    for (let t = 0; t < T; t++) {
      const f = labels.subarray(t * H * W, (t + 1) * H * W), { lab } = components(Uint8Array.from(f), H, W);
      f.set(lab);
    }
  }
  return { labels, T, H, W, key };
}

export async function loadDemo() {
  const d = await api('/demo');
  const [npz, tif] = await Promise.all([d.files.detections, d.files.image].map(u => fetch(u).then(r => {
    if (!r.ok) throw new Error(`${r.status} ${u}`);
    return r.arrayBuffer();
  })));
  return { name: d.name, synthetic: true, note: d.note, m: await masksFromNpz(npz), img: await openTiff(tif),
    umPerPx: d.um_per_px, dtMin: d.dt_min, truth: d.truth, files: d.files };
}

// Own files: {masks: File, image?: File, sidecar?: File, umPerPx?, dtMin?}
export async function loadOwn({ masks, image, sidecar, umPerPx, dtMin }) {
  const m = await masksFromNpz(await masks.arrayBuffer());
  let img = null, meta = {};
  if (image) {
    img = await openTiff(image);
    if (img.T !== m.T || img.H !== m.H || img.W !== m.W) throw new Error(`image is ${img.T}×${img.H}×${img.W}, masks are ${m.T}×${m.H}×${m.W}`);
  }
  if (sidecar) meta = JSON.parse(await sidecar.text());
  const um = umPerPx || meta.um_per_px || img?.meta?.um_per_px || null, dt = dtMin || meta.time_interval_min || null;
  return { name: masks.name.replace(/\.npz$/, ''), synthetic: false, m, img, umPerPx: um, dtMin: dt, truth: null,
    scaleSource: umPerPx ? 'entered' : meta.um_per_px ? 'sidecar' : img?.meta?.um_per_px ? 'OME metadata' : 'none' };
}

export async function dataset() {
  if (!DS) setDataset(await loadDemo());
  return DS;
}
export function setDataset(ds) {
  DS = ds; DS.results = new Map(); DS.params = null;
  listeners.forEach(fn => fn(DS));
}

// The tracker settings the desktop Tracking GUI would use for this dataset (DEFAULTS.pixel_thresholds).
export async function guiParams(ds) {
  const d = (await defaults()).defaults;
  return guiTrackParams(d, pixelThresholds(d, ds.umPerPx, ds.dtMin));
}
export async function currentParams(ds) {
  if (!ds.params) ds.params = await guiParams(ds);
  return ds.params;
}

let worker = null, seq = 0;
const pending = new Map();
function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = e => {
      const p = pending.get(e.data.id);
      if (!p) return;
      if (e.data.progress != null) { p.onProgress?.(e.data.progress, e.data.stage); return; }
      pending.delete(e.data.id);
      if (e.data.error) p.reject(new Error(e.data.error)); else p.resolve(e.data.result);
    };
  }
  return worker;
}

// Track + measure the dataset with `params` (default: the dataset's current settings). Cached.
export async function analyse(ds, params = null, onProgress = null) {
  params = params || await currentParams(ds);
  const key = JSON.stringify(params);
  if (ds.results.has(key)) return ds.results.get(key);
  const th = thresholdsObj(await defaults());
  const job = new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject, onProgress });
    getWorker().postMessage({ id, labels: ds.m.labels.slice(), T: ds.m.T, H: ds.m.H, W: ds.m.W, params,
      umPerPx: ds.umPerPx, dtMin: ds.dtMin, thresholds: th });
  }).then(r => ({ ...r, params, trackedStack: { labels: r.tracked, T: ds.m.T, H: ds.m.H, W: ds.m.W } }));
  ds.results.set(key, job);
  job.catch(() => ds.results.delete(key));
  return job;
}

// The true (simulated) cell behind detection `label` in frame t, or 0 (debris) / null (no truth).
export function trueCell(ds, t, label) {
  if (!ds.truth) return null;
  const hit = ds.truth.frames[t].find(d => d.label === label);
  return hit ? hit.cell : 0;
}
