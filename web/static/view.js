// A frame viewer used by the Track, State and Measure pages: the recording's image (if any) with a label overlay,
// trails, IDs, scale bar and time, plus a timeline (slider, play, arrow keys). The page supplies the overlay.
import { esc, onLeave } from './core.js';
import { autoLevels, draw, labelAt } from './render.js';

// overlay(t) -> {m, colourOf(label) -> [r,g,b]|null, trails?, ids?, trailColour?, selected?}
export function viewer(host, ds, overlay, { onPick = null, mode = 'fill' } = {}) {
  host.innerHTML = `<div class="stage"><canvas aria-label="Recording frame with cell masks" role="img"></canvas></div>
    <div class="timeline"><button class="ghost" data-play aria-label="Play">▶</button>
      <input type="range" min="0" max="${ds.m.T - 1}" value="0" aria-label="Frame">
      <span class="mono small" data-t></span></div>
    <div class="row small-ctl">
      <label class="check"><input type="checkbox" data-img ${ds.img ? 'checked' : 'disabled'}> image</label>
      <label class="check"><input type="checkbox" data-masks checked> masks</label>
      <label class="check"><input type="checkbox" data-outline ${mode === 'outline' ? 'checked' : ''}> outlines only</label>
      <label class="check"><input type="checkbox" data-trails checked> trails</label>
      <label class="check"><input type="checkbox" data-ids checked> IDs</label></div>`;
  const canvas = host.querySelector('canvas'), slider = host.querySelector('input[type=range]');
  const tl = host.querySelector('[data-t]'), play = host.querySelector('[data-play]');
  const opt = name => host.querySelector(`[data-${name}]`).checked;
  let t = 0, levels = null, timer = null, state = {};
  const k = Math.max(1, Math.min(4, Math.floor(720 / Math.max(ds.m.W, ds.m.H))));

  async function redraw() {
    const ov = overlay(t), layers = [];
    if (ds.img && opt('img')) {
      const plane = await ds.img.plane(t, 0);
      if (!levels) levels = autoLevels(plane);
      layers.push({ plane, lo: levels[0], hi: levels[1], rgb: [255, 255, 255] });
    }
    draw(canvas, Object.assign(state, {
      m: ov.m, t, k, layers, colourOf: ov.colourOf, mode: opt('outline') ? 'outline' : 'fill', opacity: 0.45,
      selected: ov.selected ?? null, showMasks: opt('masks'), trails: opt('trails') ? ov.trails : null, trailLen: 12,
      trailColour: ov.trailColour, ids: opt('ids') ? ov.ids : null, umPerPx: ds.umPerPx, dtMin: ds.dtMin,
      showScale: true, showTime: true,
    }));
    tl.textContent = `frame ${t + 1}/${ds.m.T}${ds.dtMin ? ` · ${(t * ds.dtMin).toFixed(0)} min` : ''}`;
  }
  const setFrame = v => { t = Math.max(0, Math.min(ds.m.T - 1, v)); slider.value = t; redraw(); };
  const stop = () => { clearInterval(timer); timer = null; play.textContent = '▶'; play.setAttribute('aria-label', 'Play'); };
  slider.addEventListener('input', () => setFrame(+slider.value));
  play.addEventListener('click', () => {
    if (timer) return stop();
    play.textContent = '❚❚'; play.setAttribute('aria-label', 'Pause');
    timer = setInterval(() => setFrame(t + 1 >= ds.m.T ? 0 : t + 1), 160);
  });
  host.querySelectorAll('.small-ctl input').forEach(i => i.addEventListener('change', redraw));
  slider.addEventListener('keydown', e => { if (e.key === ' ') { e.preventDefault(); play.click(); } });
  if (onPick) canvas.addEventListener('click', ev => { const l = labelAt(canvas, ev, overlay(t).m, t, k); onPick(l, t); });
  onLeave(stop);
  redraw();
  return { redraw, setFrame, get frame() { return t; } };
}

// "Synthetic demo" / file name badge for the current dataset.
export const datasetBadge = ds => (ds.synthetic
  ? `<span class="badge syn" title="${esc(ds.note)}">SYNTHETIC demo · simulated cells</span>`
  : `<span class="badge">your data · ${esc(ds.name)} (read in this browser, not uploaded)</span>`);
