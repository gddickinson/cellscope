// Canvas rendering for the viewer: one image channel (or a DIC-grey + fluorescence-magenta composite) with
// percentile auto-levels, the label overlay (fill or outline) coloured per cell, track trails, cell IDs, the
// selected cell, a scale bar and the frame time. Coordinates: pixel centres (x + 0.5 on the canvas grid).
import { frameOf } from './track_engine.js';

// 0.5 / 99.5 percentile levels of a plane (sampled) -> [lo, hi]
export function autoLevels(plane) {
  const step = Math.max(1, Math.floor(plane.length / 40000)), s = [];
  for (let i = 0; i < plane.length; i += step) s.push(plane[i]);
  s.sort((a, b) => a - b);
  const lo = s[Math.floor(s.length * 0.005)], hi = s[Math.min(s.length - 1, Math.floor(s.length * 0.995))];
  return [lo, hi > lo ? hi : lo + 1];
}

// Fill `img` (ImageData, W*H) from image planes. layers: [{plane, lo, hi, rgb:[r,g,b]}] blended additively.
function paintImage(img, layers, gamma = 1) {
  const d = img.data, n = d.length / 4;
  d.fill(0);
  for (const { plane, lo, hi, rgb } of layers) {
    const k = 1 / (hi - lo);
    for (let i = 0; i < n; i++) {
      let v = (plane[i] - lo) * k;
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      if (gamma !== 1) v = v ** gamma;
      d[4 * i] += v * rgb[0]; d[4 * i + 1] += v * rgb[1]; d[4 * i + 2] += v * rgb[2];
    }
  }
  for (let i = 0; i < n; i++) d[4 * i + 3] = 255;
}

// Overlay labels of frame t: colourOf(id) -> [r,g,b] | null. mode 'fill' | 'outline'.
function paintMasks(img, m, t, colourOf, { mode = 'fill', opacity = 0.45, selected = null } = {}) {
  const f = frameOf(m, t), { W, H } = m, d = img.data, cache = new Map();
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < W; x++, i++) {
    const l = f[i];
    if (l <= 0) continue;
    const edge = x === 0 || y === 0 || x === W - 1 || y === H - 1 || f[i - 1] !== l || f[i + 1] !== l || f[i - W] !== l || f[i + W] !== l;
    if (mode === 'outline' && !edge && l !== selected) continue;
    if (!cache.has(l)) cache.set(l, colourOf(l));
    const c = cache.get(l);
    if (!c) continue;
    const a = l === selected ? (edge ? 1 : Math.min(1, opacity + 0.2)) : mode === 'outline' ? 1 : edge ? Math.min(1, opacity + 0.35) : opacity;
    const sel = l === selected && edge;
    d[4 * i] = sel ? 255 : d[4 * i] * (1 - a) + c[0] * a;
    d[4 * i + 1] = sel ? 255 : d[4 * i + 1] * (1 - a) + c[1] * a;
    d[4 * i + 2] = sel ? 255 : d[4 * i + 2] * (1 - a) + c[2] * a;
  }
}

// Draw everything. o: {m, t, layers, colourOf, mode, opacity, selected, trails:{id:[[y,x]..]}, trailLen, ids:{id:[y,x]},
// umPerPx, dtMin, k (canvas px per image px), showScale, showTime, trailColour(id)}
export function draw(canvas, o) {
  const { m, t, k } = o;
  canvas.width = m.W * k; canvas.height = m.H * k;
  const g = canvas.getContext('2d');
  const off = o.off || (o.off = document.createElement('canvas'));
  off.width = m.W; off.height = m.H;
  const og = off.getContext('2d'), img = og.createImageData(m.W, m.H);
  if (o.layers.length) paintImage(img, o.layers, o.gamma || 1);
  else { for (let i = 3; i < img.data.length; i += 4) { img.data[i] = 255; img.data[i - 3] = img.data[i - 2] = img.data[i - 1] = 12; } }
  if (o.showMasks) paintMasks(img, m, t, o.colourOf, { mode: o.mode, opacity: o.opacity, selected: o.selected });
  og.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = false;
  g.drawImage(off, 0, 0, m.W * k, m.H * k);
  const P = ([y, x]) => [(x + 0.5) * k, (y + 0.5) * k];
  if (o.trails) {
    g.lineWidth = Math.max(1, k * 0.35);
    for (const [id, tr] of Object.entries(o.trails)) {
      g.strokeStyle = o.trailColour(+id); g.beginPath();
      let pen = false;
      for (let s = Math.max(0, t - o.trailLen); s <= t; s++) {
        const p = tr[s];
        if (!p || !Number.isFinite(p[0])) { pen = false; continue; }
        const [X, Y] = P(p);
        if (pen) g.lineTo(X, Y); else g.moveTo(X, Y);
        pen = true;
      }
      g.stroke();
    }
  }
  const fs = Math.max(10, Math.round(k * 4.2));
  g.font = `600 ${fs}px ui-monospace, Menlo, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
  if (o.ids) for (const [id, p] of Object.entries(o.ids)) {
    const [X, Y] = P(p);
    g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.8)'; g.strokeText(id, X, Y); g.fillStyle = '#fff'; g.fillText(id, X, Y);
  }
  g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  if (o.showTime) {
    const txt = `t = ${o.dtMin ? `${(t * o.dtMin).toFixed(0)} min` : `frame ${t}`}  (${t + 1}/${m.T})`;
    g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.8)'; g.strokeText(txt, 8, fs + 4); g.fillStyle = '#fff'; g.fillText(txt, 8, fs + 4);
  }
  if (o.showScale && o.umPerPx) {
    const target = m.W * o.umPerPx / 5, nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000].reduce((a, b) => (b <= target ? b : a), 1);
    const len = nice / o.umPerPx * k, x1 = canvas.width - 10, y = canvas.height - 12;
    g.fillStyle = '#fff'; g.fillRect(x1 - len, y, len, Math.max(3, k));
    g.textAlign = 'right'; g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,.8)';
    g.strokeText(`${nice} µm`, x1, y - 6); g.fillText(`${nice} µm`, x1, y - 6);
  }
}

// Label at canvas event position (or 0).
export function labelAt(canvas, ev, m, t, k) {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor((ev.clientX - r.left) / r.width * canvas.width / k), y = Math.floor((ev.clientY - r.top) / r.height * canvas.height / k);
  if (x < 0 || y < 0 || x >= m.W || y >= m.H) return 0;
  return frameOf(m, t)[y * m.W + x];
}
