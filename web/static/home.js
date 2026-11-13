// Home: what CellScope is, what this site runs, counts from the code and the demo, and the page cards.
import { api, app, esc, route } from './core.js';

const CARDS = [
  ['track', 'Track', 'Link per-frame detections into cells with the Hungarian tracker; tune its cost weights, max hop and max gap; score it against the synthetic ground truth; download the tracked masks.'],
  ['state', 'Cell state', 'The rounded / spread rule (area ≤ 960 µm² and eccentricity ≤ 0.85, or the circularity / solidity fallback) on every cell-frame, with adjustable thresholds.'],
  ['measure', 'Measure', 'The per-cell colour-by metrics: speed, distance, persistence, area, circularity, solidity, fraction rounded, with per-frame plots and CSV.'],
  ['planner', 'Run planner', 'What the detection pipeline will decide for a recording: downsample factor, mirror padding, backbone (cpsam_dic or raw cpsam), pixel thresholds, runtime estimate.'],
  ['params', 'Defaults', 'Every value in DEFAULTS and DEFAULT_THRESHOLDS with the comment that explains it in the source, and the four detection presets.'],
  ['check', 'JS vs Python check', 'The browser ports re-run on the demo and compared, value by value, with CellScope\'s Python results.'],
];

route('home', async () => {
  const i = await api('/info');
  app().innerHTML = `<section class="hero"><p class="kicker">Cell detection · tracking · quantification</p>
    <h1 class="glow">CellScope</h1>
    <p class="lead">CellScope detects cell boundaries in DIC, phase-contrast and DIC + fluorescence time-lapse
    recordings, tracks the cells across frames and quantifies their migration, morphology and state. Detection runs
    deep-learning models (cellpose / cpsam, SAM2) in the desktop apps; this site runs what comes after it in your
    browser: the tracker, the state rule and the per-cell measurements, on a synthetic demo or on your own
    <code>masks.npz</code>.</p>
    <p><a class="btn" href="#/track">Track the demo</a><a class="btn alt" href="#/planner">Plan a run</a>
    <a class="btn alt" href="https://gddickinson.github.io/cellscope_analysis/" target="_blank" rel="noopener">CellScope Analysis ↗</a></p></section>
    <div class="tiles">
      <div class="tile"><b>${i.n_apps}</b><span>desktop apps (<code>main_*.py</code>)</span></div>
      <div class="tile"><b>${i.n_core_modules}</b><span>modules in <code>core/</code></span></div>
      <div class="tile"><b>${i.n_defaults}</b><span>pipeline defaults in one place</span></div>
      <div class="tile"><b>${i.n_presets}</b><span>detection presets</span></div>
      <div class="tile"><b>${i.demo.n_cells}</b><span>simulated cells in the demo, ${i.demo.n_frames} frames</span></div>
    </div>
    <div class="flow" aria-label="Pipeline">
      <div class="off"><b>1 · detect</b>cpsam_dic or raw cpsam per frame, DeepSea, Cy5 fusion. Desktop only (models, GPU).</div>
      <div><b>2 · track</b>Hungarian assignment frame to frame. <a href="#/track">Track</a></div>
      <div class="off"><b>3 · gap fill</b>cpsam, CP3, SAM2, translation. Desktop only.</div>
      <div><b>4 · classify</b>rounded / spread per cell-frame. <a href="#/state">Cell state</a></div>
      <div><b>5 · measure</b>per-cell motion and shape. <a href="#/measure">Measure</a></div>
      <div><b>6 · compare</b>groups and statistics. <a href="https://gddickinson.github.io/cellscope_analysis/" target="_blank" rel="noopener">CellScope Analysis ↗</a></div>
    </div>
    <p class="callout small">${esc(i.demo.note)} No lab recordings are on this site.</p>
    <div class="grid">${CARDS.map(([h, t, d]) => `<a class="card" href="#/${h}"><b>${t}</b><span>${d}</span></a>`).join('')}</div>`;
});
