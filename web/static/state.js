// Cell state page: core/cell_state.py's rounded / spread / unknown rule on every tracked cell-frame, with the
// thresholds adjustable, the two rule spaces as scatter plots, a frame view coloured by state and (demo only) a
// comparison with the simulation's scripted round-up frames.
import { $, app, esc, route, slider } from './core.js';
import { analyse, dataset, defaults, thresholdsObj, trueCell } from './data.js';
import { legend, scatter } from './charts.js';
import { f, STATE_COLOR, STATE_RGB } from './fmt.js';
import { classifyState } from './shape_engine.js';
import { datasetBadge, viewer } from './view.js';

route('state', async () => {
  const ds = await dataset(), d = await defaults(), th0 = thresholdsObj(d), th = { ...th0 };
  const doc = Object.fromEntries(d.thresholds.map(r => [r.name, r.doc]));
  const root = app();
  root.innerHTML = `<p class="kicker">Cell state</p><h1>Rounded or spread?</h1>
    <p class="lead">Cells that round up (mitosis, detachment) move and look different from spread, adherent cells, so
    CellScope classifies every cell-frame and stratifies its motility by state. With the pixel size known, a
    cell-frame is <b>rounded</b> when <span class="eq">area ≤ ${th0.rounded_area_um2} µm² and eccentricity ≤ ${th0.rounded_eccentricity}</span>;
    without it, the fallback is <span class="eq">circularity ≥ ${th0.rounded_circ} and solidity ≥ ${th0.rounded_solid}</span>.
    Frames whose mask touches the image border, or is under ${th0.min_area_px} px, are <b>unknown</b>.
    Masks are cleaned first (largest piece, holes filled), as in <code>core/cell_state.py::shape_metrics_for_mask</code>.</p>
    <p class="small mute">${esc(doc.rounded_area_um2)}</p>
    <p>${datasetBadge(ds)} <span class="small mute">tracks from the <a href="#/track">Track</a> page's current settings</span></p>
    <div class="progress"><div></div></div>
    <div class="work"><div id="view"></div><div class="panel" id="ctl"></div></div>
    <div id="tiles"></div><div class="charts" id="charts"></div><div id="truth"></div>`;
  const bar = root.querySelector('.progress > div');
  const res = await analyse(ds, null, p => { bar.style.width = `${(p * 100).toFixed(0)}%`; });
  bar.style.width = '100%';
  const ids = res.metrics.cell_ids, M = res.metrics.metrics;
  let useScale = !!ds.umPerPx;
  const stateOf = (id, t) => {
    const m = M[id];
    if (m.edge[t]) return 'edge';
    return classifyState({ area: m.area[t], eccentricity: m.eccentricity[t], circularity: m.circularity[t],
      solidity: m.solidity[t], edge_touch: m.edge[t] }, th, useScale ? ds.umPerPx : null);
  };
  const ctl = $('#ctl');
  ctl.innerHTML = `<h3>Thresholds</h3>
    <label class="check"><input type="checkbox" data-scale ${useScale ? 'checked' : ''} ${ds.umPerPx ? '' : 'disabled'}>
      use the pixel size (${ds.umPerPx ? `${ds.umPerPx} µm/px` : 'not known: fallback rule'})</label><div data-sl></div>
    <p><button class="ghost" data-reset>DEFAULT_THRESHOLDS</button></p>
    ${legend([['rounded', STATE_COLOR.rounded], ['spread', STATE_COLOR.spread], ['unknown', STATE_COLOR.unknown], ['edge (excluded)', STATE_COLOR.edge]])}`;
  const sl = {};
  for (const [k, lab, min, max, step] of [['rounded_area_um2', 'rounded: max area (µm²)', 100, 3000, 10],
    ['rounded_eccentricity', 'rounded: max eccentricity', 0, 1, 0.01], ['rounded_circ', 'fallback: min circularity', 0, 1, 0.01],
    ['rounded_solid', 'fallback: min solidity', 0, 1, 0.01]]) {
    sl[k] = slider(ctl.querySelector('[data-sl]'), { id: k, label: lab, min, max, step, value: th[k] }, v => { th[k] = v; update(); });
  }
  ctl.querySelector('[data-scale]').addEventListener('change', e => { useScale = e.target.checked; update(); });
  ctl.querySelector('[data-reset]').addEventListener('click', () => {
    Object.assign(th, th0);
    for (const k of Object.keys(sl)) { sl[k].value = th[k]; sl[k].dispatchEvent(new Event('input')); }
  });
  const view = viewer($('#view'), ds, t => ({ m: res.trackedStack, colourOf: id => (M[id] ? STATE_RGB[stateOf(id, t)] : null) }));

  function update() {
    const counts = { rounded: 0, spread: 0, unknown: 0, edge: 0 }, pa = [], pc = [];
    const um2 = useScale && ds.umPerPx ? ds.umPerPx ** 2 : null;
    for (const id of ids) for (let t = 0; t < ds.m.T; t++) {
      if (!presentAt(id, t)) continue;
      const m = M[id], s = stateOf(id, t);
      counts[s]++;
      if (!Number.isFinite(m.area[t])) continue;
      const tip = `cell ${id}, frame ${t}: ${s}`;
      if (um2) pa.push({ x: m.area[t] * um2, y: m.eccentricity[t], c: STATE_COLOR[s], t: tip });
      pc.push({ x: m.circularity[t], y: m.solidity[t], c: STATE_COLOR[s], t: tip });
    }
    $('#tiles').innerHTML = `<div class="tiles">${Object.entries(counts).map(([k, v]) => `<div class="tile"><b>${v}</b><span>${k === 'edge' ? 'edge-truncated (unknown)' : k} cell-frames</span></div>`).join('')}</div>`;
    $('#charts').innerHTML = `<div class="panel"><h3>Primary rule: area × eccentricity</h3>${um2 ? scatter(pa, { xlabel: 'area (µm²)', ylabel: 'eccentricity',
      guides: [{ x: th.rounded_area_um2 }, { y: th.rounded_eccentricity }] }) : '<p class="mute small">Needs the pixel size.</p>'}
      <p class="small mute">Rounded = bottom-left box.</p></div>
      <div class="panel"><h3>Fallback rule: circularity × solidity</h3>${scatter(pc, { xlabel: 'circularity', ylabel: 'solidity',
      guides: [{ x: th.rounded_circ }, { y: th.rounded_solid }] })}<p class="small mute">Rounded = top-right box (used only without a pixel size).</p></div>`;
    if (ds.truth) truth();
    view.redraw();
  }
  const presentAt = (id, t) => res.tracks[id - 1]?.labels[t] > 0;

  function truth() {
    const rounded = new Map(ds.truth.cells.map(c => [c.id, new Set(c.rounded_frames)]));
    const tab = { tr: 0, tf: 0, fr: 0, ff: 0 }, newborn = { r: 0, n: 0 };
    for (const id of ids) for (let t = 0; t < ds.m.T; t++) {
      const tr = res.tracks[id - 1], lab = tr?.labels[t];
      if (!lab) continue;
      const cell = trueCell(ds, t, lab), s = stateOf(id, t);
      if (!cell || s === 'unknown' || s === 'edge') continue;
      const truly = rounded.get(cell).has(t), said = s === 'rounded';
      tab[(truly ? 't' : 'f') + (said ? 'r' : 'f')]++;
      const c = ds.truth.cells.find(x => x.id === cell);
      if (c.parent && t < c.start + 8) { newborn.n++; if (said) newborn.r++; }
    }
    const n = tab.tr + tab.tf + tab.fr + tab.ff;
    $('#truth').innerHTML = `<h2>Against the simulation</h2>
      <p class="small mute">The simulation shrinks a cell and removes its elongation on its scripted round-up frames;
      those are "truly rounded" here. Daughters after the division start small and round and grow over 8 frames, which
      the size rule reads as rounded although the script does not mark them: ${newborn.r} of ${newborn.n} newborn
      daughter frames are called rounded.</p>
      <div class="scroll"><table class="tbl"><thead><tr><th></th><th>classified rounded</th><th>classified spread</th></tr></thead><tbody>
      <tr><td>scripted round-up frames</td><td class="good">${tab.tr}</td><td>${tab.tf}</td></tr>
      <tr><td>other frames</td><td>${tab.fr}</td><td class="good">${tab.ff}</td></tr></tbody></table></div>
      <p class="small">Agreement ${f(n ? (tab.tr + tab.ff) / n : NaN)} over ${n} classified cell-frames of true cells.</p>`;
  }
  update();
});
