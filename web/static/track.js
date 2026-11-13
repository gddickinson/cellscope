// Track page: the Hungarian tracker (JS port of core/multi_cell.py::track_all_cells) on the synthetic demo or your
// own masks.npz, with the tracker's knobs, the raw-vs-tracked view, a score against the simulation's ground truth
// and downloads (tracks CSV, tracked masks .npz).
import { $, app, download, esc, route, slider } from './core.js';
import { analyse, currentParams, dataset, defaults, guiParams } from './data.js';
import { cellRgb, f } from './fmt.js';
import { loaderPanel } from './loader.js';
import { npzBlob } from './npz_write.js';
import { datasetBadge, viewer } from './view.js';
import { frameOf, trackPaths } from './track_engine.js';

// Score against the simulation: per true cell, how its detections were split over tracks.
export function scoreTracks(ds, tracks) {
  const T = ds.m.T, byCell = new Map(), trackCells = tracks.map(() => new Set());
  let debris = 0;
  for (let t = 0; t < T; t++) {
    const owner = new Map();
    tracks.forEach((tr, k) => { if (tr.labels[t]) owner.set(tr.labels[t], k); });
    for (const d of ds.truth.frames[t]) {
      const k = owner.has(d.label) ? owner.get(d.label) : -1;
      if (!d.cell) { if (k >= 0) debris++; continue; }
      if (!byCell.has(d.cell)) byCell.set(d.cell, []);
      byCell.get(d.cell).push(k);
      if (k >= 0) trackCells[k].add(d.cell);
    }
  }
  const cells = [...byCell.entries()].sort((a, b) => a[0] - b[0]).map(([cell, ks]) => {
    const counts = new Map();
    ks.forEach(k => { if (k >= 0) counts.set(k, (counts.get(k) || 0) + 1); });
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
    let switches = 0, prev = null;
    for (const k of ks) { if (k < 0) continue; if (prev !== null && k !== prev) switches++; prev = k; }
    return { cell, frames: ks.length, untracked: ks.filter(k => k < 0).length, tracks: [...counts.keys()].map(k => k + 1),
      purity: best ? best[1] / ks.length : 0, switches };
  });
  return { cells, debris, shared: trackCells.filter(s => s.size > 1).length };
}

const SLIDERS = [
  ['maxHop', 'max hop (px per frame)', 5, 400, 1], ['maxGap', 'max gap (frames)', 0, 40, 1],
  ['minLen', 'min track length (frames)', 1, 30, 1], ['minArea', 'min area (px)', 1, 2000, 1],
  ['wDist', 'weight: distance', 0, 3, 0.1], ['wIou', 'weight: 1 − IoU', 0, 3, 0.1], ['wArea', 'weight: area change', 0, 3, 0.1],
];

route('track', async () => {
  const ds = await dataset(), D = (await defaults()).defaults;
  const root = app();
  root.innerHTML = `<p class="kicker">Tracking</p><h1>Follow cells across frames</h1>
    <p class="lead">A detector labels each frame on its own, so the same cell gets a different ID in every frame.
    CellScope links them with a Hungarian assignment per frame: the cost of matching a track to a detection is
    <span class="eq">w<sub>dist</sub>·distance + w<sub>IoU</sub>·(1 − IoU)·max_hop + w<sub>area</sub>·|Δarea|/max(area)·max_hop</span>,
    pairs further apart than the max hop (twice it after a missed frame) are forbidden, a track survives up to
    <i>max gap</i> missed frames, and unmatched detections start new tracks. This page runs a JS port of
    <code>core/multi_cell.py::track_all_cells</code> in your browser, starting from the settings the desktop Tracking
    GUI uses for this recording.</p>
    <div id="loader"></div>
    <p>${datasetBadge(ds)}</p>
    <div class="work"><div id="view"></div><div class="panel" id="ctl"></div></div>
    <div class="progress"><div></div></div><div id="res"></div>`;
  loaderPanel($('#loader'), ds, () => window.dispatchEvent(new HashChangeEvent('hashchange')));
  const params = { ...(await currentParams(ds)) }, ctl = $('#ctl'), gp = await guiParams(ds);
  let res = null, raw = false, timer = null;
  ctl.innerHTML = `<h3>Tracker settings</h3><div data-sl></div>
    <label class="check"><input type="checkbox" data-spawn ${params.spawn ? 'checked' : ''}> start new tracks for unmatched detections</label>
    <p><button class="ghost" data-reset>Tracking-GUI settings</button></p>
    <div class="seg" role="group" aria-label="Overlay"><button data-v="t" class="on">tracked cells</button><button data-v="r">raw detections</button></div>
    <p class="small mute">GUI settings for this recording: <code>DEFAULTS.pixel_thresholds(${ds.umPerPx ?? 'None'}, ${ds.dtMin ?? 'None'})</code>
    gives min area ${gp.minArea} px and max hop ${gp.maxHop} px; weights
    ${D.track_w_dist} / ${D.track_w_iou} / ${D.track_w_area}, max gap ${D.max_gap_frames}, min length ${D.min_track_length} (all <code>DEFAULTS</code>).</p>`;
  const inputs = {};
  for (const [key, lab, min, max, step] of SLIDERS) {
    inputs[key] = slider(ctl.querySelector('[data-sl]'), { id: key, label: lab, min, max, step, value: params[key] }, v => { params[key] = v; rerun(); });
  }
  ctl.querySelector('[data-spawn]').addEventListener('change', e => { params.spawn = e.target.checked; rerun(); });
  ctl.querySelector('[data-reset]').addEventListener('click', async () => {
    Object.assign(params, await guiParams(ds));
    for (const [key] of SLIDERS) { inputs[key].value = params[key]; inputs[key].dispatchEvent(new Event('input')); }
    ctl.querySelector('[data-spawn]').checked = params.spawn;
  });
  ctl.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
    raw = b.dataset.v === 'r';
    ctl.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b));
    view.redraw();
  }));

  const overlay = t => {
    if (raw || !res) {
      return { m: ds.m, colourOf: l => cellRgb(l * 7 + t * 13), ids: rawIds(t) };
    }
    const ids = {}, trails = {};
    res.paths.forEach((p, k) => { if (p[t]) ids[k + 1] = p[t]; trails[k + 1] = p; });
    return { m: res.trackedStack, colourOf: cellRgb, ids, trails, trailColour: id => `rgb(${cellRgb(id)})` };
  };
  const rawIds = t => {
    const f0 = frameOf(ds.m, t), acc = new Map();
    for (let i = 0; i < f0.length; i++) { const l = f0[i]; if (!l) continue; const a = acc.get(l) || [0, 0, 0]; a[0] += (i / ds.m.W) | 0; a[1] += i % ds.m.W; a[2]++; acc.set(l, a); }
    return Object.fromEntries([...acc].map(([l, a]) => [l, [a[0] / a[2], a[1] / a[2]]]));
  };
  const view = viewer($('#view'), ds, overlay);
  const bar = root.querySelector('.progress > div');

  async function run() {
    ds.params = { ...params };
    const r = await analyse(ds, ds.params, p => { bar.style.width = `${(p * 100).toFixed(0)}%`; });
    bar.style.width = '100%';
    res = { ...r, paths: trackPaths(r.tracks, ds.m.T) };
    view.redraw();
    results(ds, res);
  }
  function rerun() { clearTimeout(timer); timer = setTimeout(() => run().catch(showErr), 250); }
  const showErr = e => { $('#res').innerHTML = `<p class="err">${esc(e.message)}</p>`; };
  run().catch(showErr);
});

function results(ds, res) {
  const { tracks } = res, T = ds.m.T, host = $('#res');
  const nDet = Array.from({ length: T }, (_, t) => new Set(frameOf(ds.m, t)).size - (frameOf(ds.m, t).includes(0) ? 1 : 0)).reduce((a, b) => a + b, 0);
  const linked = tracks.reduce((a, t) => a + t.n, 0), divs = tracks.filter(t => t.parent !== null).length;
  let truth = '';
  if (ds.truth) {
    const s = scoreTracks(ds, tracks), whole = s.cells.filter(c => c.tracks.length === 1 && !c.untracked).length;
    truth = `<h2>Against the simulation's ground truth</h2>
      <p class="small mute">The demo is synthetic, so we know which true cell every detection belongs to. Purity = share
      of a cell's detected frames that sit in its main track; a switch = consecutive detected frames of one cell in
      different tracks. This score is this page's own, not CellScope's GT evaluation (<code>scripts/evaluate_against_gt.py</code>).</p>
      <div class="tiles"><div class="tile"><b>${whole}/${s.cells.length}</b><span>true cells followed by one track, every frame</span></div>
      <div class="tile"><b>${s.cells.reduce((a, c) => a + c.switches, 0)}</b><span>identity switches</span></div>
      <div class="tile"><b>${s.shared}</b><span>tracks holding more than one true cell</span></div>
      <div class="tile"><b>${s.debris}</b><span>debris detections kept in tracks</span></div></div>
      <div class="scroll"><table class="tbl"><thead><tr><th>true cell</th><th>what the simulation did</th><th>frames detected</th><th>tracks</th><th>purity</th><th>switches</th><th>untracked</th></tr></thead><tbody>
      ${s.cells.map(c => `<tr><td>${c.cell}</td><td class="doc">${esc(ds.truth.cells.find(x => x.id === c.cell)?.note || '')}</td><td>${c.frames}</td><td>${c.tracks.join(', ') || '–'}</td><td class="${c.purity === 1 ? 'good' : ''}">${f(c.purity)}</td><td>${c.switches}</td><td>${c.untracked}</td></tr>`).join('')}
      </tbody></table></div>`;
  }
  host.innerHTML = `<div class="tiles"><div class="tile"><b>${tracks.length}</b><span>tracks</span></div>
    <div class="tile"><b>${linked}/${nDet}</b><span>detections in tracks</span></div>
    <div class="tile"><b>${divs}</b><span>division links (parent set)</span></div>
    <div class="tile"><b>${T}</b><span>frames</span></div></div>
    ${truth}
    <h2>Tracks</h2>
    <p class="small mute">Parent: set by the tracker's division heuristic when a track starts within the max hop of a
    track that ended on the previous or same frame, at 20-90 % of its area.</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>track</th><th>first frame</th><th>last frame</th><th>frames</th><th>parent</th></tr></thead><tbody>
    ${tracks.map((t, k) => { const last = T - 1 - [...t.labels].reverse().findIndex(l => l > 0); return `<tr><td><span class="cls" style="--c:rgb(${cellRgb(k + 1)})">${k + 1}</span></td><td>${t.first}</td><td>${last}</td><td>${t.n}</td><td>${t.parent === null ? '–' : t.parent + 1}</td></tr>`; }).join('')}
    </tbody></table></div>
    <p><button data-csv>Download tracks (CSV)</button> <button class="ghost" data-npz>Download tracked masks (.npz)</button></p>
    <p class="small mute">The .npz has a <code>labels</code> array (track k → label k), the layout of CellScope's
    <code>masks.npz</code>, so it opens in CellScope and in <a href="https://gddickinson.github.io/cellscope_analysis/" target="_blank" rel="noopener">CellScope Analysis</a>.</p>`;
  host.querySelector('[data-csv]').addEventListener('click', () => {
    const rows = ['track,frame,detection_label,centroid_row_px,centroid_col_px,parent_track'];
    tracks.forEach((t, k) => res.paths[k].forEach((p, fr) => { if (p) rows.push([k + 1, fr, t.labels[fr], p[0], p[1], t.parent === null ? '' : t.parent + 1].join(',')); }));
    download(`${ds.name}_tracks.csv`, rows.join('\n') + '\n', 'text/csv');
  });
  host.querySelector('[data-npz]').addEventListener('click', async () => {
    download(`${ds.name}_tracked_masks.npz`, await npzBlob({ labels: { data: res.tracked, shape: [T, ds.m.H, ds.m.W] } }));
  });
}
