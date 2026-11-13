// Defaults page: core/pipeline_defaults.py::DEFAULTS, core/cell_state.py::DEFAULT_THRESHOLDS, the planning constants
// and core/detection_presets.py, each with the comment that explains it in the source (read by web/defaults_doc.py).
import { $, app, esc, route } from './core.js';
import { defaults } from './data.js';

const val = v => (v === null ? 'None' : typeof v === 'boolean' ? (v ? 'True' : 'False') : typeof v === 'string' ? `"${v}"` : String(v));
const rows = list => list.map(r => `<tr data-q="${esc((r.name + ' ' + r.doc).toLowerCase())}"><td><code>${esc(r.name)}</code></td><td class="mono">${esc(val(r.value))}</td><td class="doc">${esc(r.doc) || '<span class="mute">–</span>'}</td></tr>`).join('');
const table = list => `<div class="scroll"><table class="tbl"><thead><tr><th>name</th><th>value</th><th>from the source comment</th></tr></thead><tbody>${rows(list)}</tbody></table></div>`;

route('params', async () => {
  const d = await defaults();
  const groups = [];
  for (const r of d.fields) { const g = groups.find(x => x.name === r.group); if (g) g.rows.push(r); else groups.push({ name: r.group, rows: [r] }); }
  const P = d.presets;
  app().innerHTML = `<p class="kicker">Defaults</p><h1>One source of truth</h1>
    <p class="lead">Every CellScope GUI, worker and pipeline script reads its defaults from
    <code>core/pipeline_defaults.py::DEFAULTS</code> (${d.fields.length} values) and the state thresholds from
    <code>core/cell_state.py::DEFAULT_THRESHOLDS</code>, so the same recording gives the same result whichever app runs
    it. The values below are read from those modules and the explanations are the comments above each value in the
    source.</p>
    <label for="q">filter</label><input id="q" type="search" placeholder="e.g. gap, cy5, area">
    ${groups.map(g => `<section><h2>${esc(g.name || 'Other')}</h2>${table(g.rows)}</section>`).join('')}
    <section><h2>State thresholds (DEFAULT_THRESHOLDS)</h2>${table(d.thresholds)}</section>
    <section><h2>Planning constants</h2>${table(Object.entries(d.constants).map(([name, value]) => ({ name, value: Array.isArray(value) ? JSON.stringify(value) : value, doc: '' })))}</section>
    <section><h2>Detection presets</h2><p class="small mute">A preset records only its changes from DEFAULTS
    (<code>core/detection_presets.py</code>); choosing one resets all ${P.params.length} knobs.</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>knob</th>${P.order.map(n => `<th>${esc(n)}</th>`).join('')}</tr></thead><tbody>
    ${P.params.map(k => `<tr><td><code>${esc(k)}</code></td>${P.presets.map(p => `<td class="${k in p.deltas ? 'good' : ''}">${esc(val(p.values[k]))}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    <dl class="kv">${P.presets.map(p => `<dt>${esc(p.name)}</dt><dd style="font-family:var(--sans)">${esc(p.doc)}</dd>`).join('')}</dl></section>`;
  $('#q').addEventListener('input', e => {
    const q = e.target.value.trim().toLowerCase();
    app().querySelectorAll('tr[data-q]').forEach(tr => { tr.hidden = q && !tr.dataset.q.includes(q); });
  });
});
