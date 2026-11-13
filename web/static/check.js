// JS vs Python check: re-run the browser engines on the demo with the Tracking-GUI settings and compare with the
// Python results from the API (core.multi_cell.track_all_cells + core.mask_metrics.compute_label_metrics).
import { api, app, esc, route } from './core.js';
import { analyse, dataset, loadDemo, setDataset } from './data.js';
import { f } from './fmt.js';

route('check', async () => {
  const root = app();
  root.innerHTML = `<p class="kicker">Check</p><h1>JS vs Python</h1>
    <p class="lead">The pages compute in your browser with JS ports of CellScope's Python. Here the ports run on the
    synthetic demo with the desktop Tracking GUI's settings and are compared with what the Python code computed for
    the same masks (served by the API; exported to JSON for the static site). A node test checks the same ports on
    more cases (<code>web/static/engine.test.mjs</code>).</p><div class="progress"><div></div></div><div id="out"></div>`;
  let ds = await dataset();
  if (!ds.synthetic) { ds = await loadDemo(); setDataset(ds); }
  const ref = await api('/demo/reference');
  const params = { minArea: ref.params.min_area_px, maxHop: ref.params.max_hop_px, spawn: ref.params.spawn_new_tracks,
    minLen: ref.params.min_track_length, wDist: null, wIou: null, wArea: null, maxGap: null };
  const D = (await api('/defaults')).defaults;
  Object.assign(params, { wDist: D.track_w_dist, wIou: D.track_w_iou, wArea: D.track_w_area, maxGap: D.max_gap_frames });
  const bar = root.querySelector('.progress > div');
  const res = await analyse(ds, params, p => { bar.style.width = `${(p * 100).toFixed(0)}%`; });
  bar.style.width = '100%';
  let same = 0, n = 0, maxRel = 0, worst = '', stEq = 0, stN = 0;
  ref.tracks.forEach((t, k) => { n++; const j = res.tracks[k]; if (j && j.first === t.first && j.labels.join() === t.labels.join() && (j.parent ?? null) === t.parent) same++; });
  let vals = 0;
  for (const id of ref.cell_ids) {
    for (const [key, v] of Object.entries(ref.per_track[id])) {
      const js = res.metrics.per_track[id]?.[key];
      vals++;
      if (v === null && !Number.isFinite(js)) continue;
      const rel = Math.abs(js - v) / Math.max(Math.abs(v), 1e-12);
      if (rel > maxRel) { maxRel = rel; worst = `cell ${id} ${key}`; }
    }
    ref.states[id].forEach((s, t) => { stN++; if (res.metrics.states[id]?.[t] === s) stEq++; });
  }
  const ok = same === n && res.tracks.length === n && stEq === stN && maxRel < 1e-5;
  $out(`<div class="tiles"><div class="tile"><b class="${same === n ? 'good' : 'bad'}">${same}/${n}</b><span>tracks identical (every frame's detection, first frame, parent)</span></div>
    <div class="tile"><b class="${stEq === stN ? 'good' : 'bad'}">${stEq}/${stN}</b><span>cell-frame state codes identical</span></div>
    <div class="tile"><b>${vals}</b><span>per-cell values compared</span></div>
    <div class="tile"><b class="${maxRel < 1e-5 ? 'good' : 'bad'}">${maxRel ? maxRel.toExponential(1) : '0'}</b><span>largest relative difference${worst ? ` (${esc(worst)})` : ''}</span></div></div>
    <p class="${ok ? 'good' : 'bad'}">${ok ? 'The browser reproduces the Python results.' : 'Differences found: see the tiles above.'}</p>
    <p class="small mute">Settings: min area ${ref.params.min_area_px} px, max hop ${ref.params.max_hop_px} px, min track
    length ${ref.params.min_track_length}, new tracks on, weights ${f(D.track_w_dist)} / ${f(D.track_w_iou)} / ${f(D.track_w_area)},
    max gap ${D.max_gap_frames}. Python: <code>${esc(ref.source)}</code>. Small differences (~1e-7) are float32
    rounding: the Python metrics store per-frame values as float32.</p>`);
  function $out(h) { root.querySelector('#out').innerHTML = h; }
});
