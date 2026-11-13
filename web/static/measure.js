// Measure page: core/mask_metrics.py::compute_label_metrics (the desktop "Colour masks by result" metrics) on the
// tracked cells: a per-cell table, the frame view coloured by any metric, per-frame plots and a flower plot.
import { $, app, download, esc, route } from './core.js';
import { analyse, dataset } from './data.js';
import { flower, legend, lineChart } from './charts.js';
import { cellRgb, f, label, ramp, rampCss } from './fmt.js';
import { trackPaths } from './track_engine.js';
import { datasetBadge, viewer } from './view.js';

const KEYS = ['mean_speed', 'total_distance', 'net_displacement', 'persistence', 'mean_area', 'mean_circularity',
  'mean_solidity', 'mean_aspect_ratio', 'mean_eccentricity', 'frames_tracked', 'frac_rounded'];

route('measure', async () => {
  const ds = await dataset(), root = app(), sc = !!(ds.umPerPx && ds.dtMin);
  const unit = { mean_speed: sc ? 'µm/min' : 'px/frame', total_distance: sc ? 'µm' : 'px', net_displacement: sc ? 'µm' : 'px',
    mean_area: sc ? 'µm²' : 'px' };
  root.innerHTML = `<p class="kicker">Measure</p><h1>Per-cell measurements</h1>
    <p class="lead">For each tracked cell: motion from its centroid path and shape from its masks, the quick
    whole-track metrics behind CellScope's "Colour masks by result" overlay (<code>core/mask_metrics.py</code>).
    ${sc ? '' : 'This recording has no pixel size and frame interval, so values stay in pixels and frames. '}
    These whole-track averages mix rounded and spread frames; CellScope's comparison-grade, state-stratified metrics
    (<code>core/state_analysis.py</code>, <code>per_cell.csv</code>) are desktop-only, and for group statistics use
    <a href="https://gddickinson.github.io/cellscope_analysis/" target="_blank" rel="noopener">CellScope Analysis</a>.</p>
    <p>${datasetBadge(ds)} <span class="small mute">tracks from the <a href="#/track">Track</a> page's current settings</span></p>
    <div class="progress"><div></div></div>
    <div class="work"><div id="view"></div><div class="panel" id="ctl"></div></div>
    <div id="table"></div><div class="charts" id="charts"></div>`;
  const bar = root.querySelector('.progress > div');
  const res = await analyse(ds, null, p => { bar.style.width = `${(p * 100).toFixed(0)}%`; });
  bar.style.width = '100%';
  const mt = res.metrics, ids = mt.cell_ids, paths = trackPaths(res.tracks, ds.m.T);
  let colourBy = 'cell', sel = ids[0] ?? null;
  const ctl = $('#ctl');
  ctl.innerHTML = `<h3>Colour cells by</h3><label for="cb" class="small">metric</label>
    <select id="cb"><option value="cell">cell ID</option>${KEYS.map(k => `<option value="${k}">${esc(label(k))}</option>`).join('')}</select>
    <div data-bar></div><p class="small mute">Click a cell in the image or a row in the table to plot it.</p>`;
  const range = k => { const v = ids.map(i => mt.per_track[i][k]).filter(Number.isFinite); return [Math.min(...v), Math.max(...v)]; };
  const colourOf = id => {
    if (!mt.per_track[id]) return null;
    if (colourBy === 'cell') return cellRgb(id);
    const [lo, hi] = range(colourBy), v = mt.per_track[id][colourBy];
    return ramp(hi > lo ? (v - lo) / (hi - lo) : 0.5);
  };
  const view = viewer($('#view'), ds, t => {
    const idsAt = {};
    paths.forEach((p, k) => { if (p[t]) idsAt[k + 1] = p[t]; });
    return { m: res.trackedStack, colourOf, ids: idsAt, selected: sel, trails: Object.fromEntries(paths.map((p, k) => [k + 1, p])),
      trailColour: id => `rgb(${cellRgb(id)})` };
  }, { onPick: l => { if (l) select(l); } });
  ctl.querySelector('#cb').addEventListener('change', e => {
    colourBy = e.target.value;
    const [lo, hi] = colourBy === 'cell' ? [0, 0] : range(colourBy);
    ctl.querySelector('[data-bar]').innerHTML = colourBy === 'cell' ? '' : `<div class="cbar" style="height:10px;border-radius:2px;background:${rampCss(8)}"></div>
      <div class="cbar-l small mono" style="display:flex;justify-content:space-between"><span>${f(lo)}</span><span>${f(hi)}</span></div>`;
    view.redraw();
  });

  const cols = ['cell', ...KEYS];
  $('#table').innerHTML = `<h2>Cells</h2><div class="scroll"><table class="tbl"><thead><tr>${cols.map(c => `<th>${esc(label(c))}${unit[c] ? ` (${unit[c]})` : ''}</th>`).join('')}</tr></thead>
    <tbody>${ids.map(id => `<tr class="click" tabindex="0" data-id="${id}"><td><span class="cls" style="--c:rgb(${cellRgb(id)})">${id}</span></td>${KEYS.map(k => `<td>${f(mt.per_track[id][k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    <p><button data-csv>Download per-cell table (CSV)</button></p>`;
  $('#table').querySelectorAll('tr.click').forEach(tr => {
    tr.addEventListener('click', () => select(+tr.dataset.id));
    tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(+tr.dataset.id); } });
  });
  $('#table').querySelector('[data-csv]').addEventListener('click', () => {
    const rows = [cols.join(',')].concat(ids.map(id => [id, ...KEYS.map(k => mt.per_track[id][k] ?? '')].join(',')));
    download(`${ds.name}_per_cell.csv`, rows.join('\n') + '\n', 'text/csv');
  });

  function select(id) {
    sel = id;
    $('#table').querySelectorAll('tr.click').forEach(tr => tr.classList.toggle('sel', +tr.dataset.id === id));
    view.redraw();
    const pf = mt.per_frame[id], x = Array.from({ length: ds.m.T }, (_, t) => (ds.dtMin ? t * ds.dtMin : t));
    const xl = ds.dtMin ? 'time (min)' : 'frame', c = `rgb(${cellRgb(id)})`;
    const one = (k, yl) => `<div class="panel"><h3>Cell ${id}: ${esc(yl)}</h3>${lineChart([{ name: `cell ${id}`, color: c, x, y: pf[k] }], { xlabel: xl, ylabel: yl, markers: true })}</div>`;
    const s = sc ? ds.umPerPx : 1;
    $('#charts').innerHTML = one('area', `area (${sc ? 'µm²' : 'px'})`) + one('speed', `speed (${unit.mean_speed})`)
      + one('circularity', 'circularity') + one('solidity', 'solidity')
      + `<div class="panel"><h3>All tracks from their start (${sc ? 'µm' : 'px'})</h3>${flower(paths.map((p, k) => {
        const v = p.filter(Boolean), o = v[0] || [0, 0];
        return { name: `cell ${k + 1}`, color: `rgb(${cellRgb(k + 1)})`, pts: v.map(q => [(q[1] - o[1]) * s, (q[0] - o[0]) * s]) };
      }), { unit: sc ? 'µm' : 'px' })}${legend([[`selected: cell ${id}`, c]])}</div>`;
  }
  if (sel != null) select(sel);
});
