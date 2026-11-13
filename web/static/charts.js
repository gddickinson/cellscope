// Small SVG charts (no library): line chart with optional bands, scatter with guide lines, flower plot.
// (Shared with the CellScope Analysis site, gddickinson/cellscope_analysis.)
// All return an SVG string sized by viewBox; text uses the theme's mono font via .chart CSS.
import { esc } from './core.js';
import { f } from './fmt.js';

const W = 380, H = 240, M = { l: 58, r: 12, t: 12, b: 40 };
const finite = a => a.filter(Number.isFinite);

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

function scale(d0, d1, r0, r1, log = false) {
  const t = log ? Math.log10 : v => v, a = t(d0), b = t(d1);
  const s = v => r0 + (t(v) - a) / ((b - a) || 1) * (r1 - r0);
  s.domain = [d0, d1];
  s.ticks = log ? ticks(Math.ceil(a), Math.floor(b), 6).filter(Number.isInteger).map(e => 10 ** e) : ticks(d0, d1);
  if (log && s.ticks.length < 2) s.ticks = [d0, d1];
  return s;
}

function pad(lo, hi, log) {
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi <= lo) return log ? [lo / 2, hi * 2] : [lo - 0.5, hi + 0.5];
  if (log) return [lo, hi];
  const p = (hi - lo) * 0.05;
  return [lo - p, hi + p];
}

function axes(xs, ys, xl, yl, xt = null) {
  let s = `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  if (xt) xt.forEach(([v, lab]) => { s += `<text x="${xs(v)}" y="${H - M.b + 14}" text-anchor="middle">${esc(lab)}</text>`; });
  else xs.ticks.forEach(v => { s += `<text x="${xs(v)}" y="${H - M.b + 14}" text-anchor="middle">${f(v)}</text>`; });
  ys.ticks.forEach(v => { s += `<text x="${M.l - 5}" y="${ys(v) + 4}" text-anchor="end">${f(v)}</text>`; });
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 5}" text-anchor="middle">${esc(xl)}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yl)}</text>`;
  return s;
}

const svg = (body, label) => `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${body}</svg>`;
export const legend = items => `<div class="legend">${items.map(([n, c]) => `<span class="cls" style="--c:${c}">${esc(n)}</span>`).join('')}</div>`;

// series: [{name, color, x, y, lo?, hi?, dash?, width?, opacity?}]
export function lineChart(series, { xlabel = '', ylabel = '', logx = false, logy = false, zero = false, markers = false } = {}) {
  const okx = v => Number.isFinite(v) && (!logx || v > 0), oky = v => Number.isFinite(v) && (!logy || v > 0);
  const xsAll = [], ysAll = [];
  for (const s of series) s.x.forEach((x, i) => { if (okx(x) && oky(s.y[i])) { xsAll.push(x); ysAll.push(s.y[i]); if (s.lo && oky(s.lo[i])) ysAll.push(s.lo[i]); if (s.hi && oky(s.hi[i])) ysAll.push(s.hi[i]); } });
  if (!xsAll.length) return '<p class="mute small">Nothing to plot.</p>';
  let [y0, y1] = pad(Math.min(...ysAll), Math.max(...ysAll), logy);
  if (zero && !logy) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
  const xs = scale(...pad(Math.min(...xsAll), Math.max(...xsAll), logx), M.l, W - M.r, logx);
  const ys = scale(y0, y1, H - M.b, M.t, logy);
  let b = axes(xs, ys, xlabel, ylabel);
  if (zero && !logy && y0 < 0 && y1 > 0) b += `<line class="axis" stroke-dasharray="3 3" x1="${M.l}" x2="${W - M.r}" y1="${ys(0)}" y2="${ys(0)}"/>`;
  for (const s of series) {
    if (s.lo && s.hi) {
      const idx = s.x.map((x, i) => i).filter(i => okx(s.x[i]) && oky(s.lo[i]) && oky(s.hi[i]));
      if (idx.length > 1) {
        const top = idx.map(i => `${xs(s.x[i]).toFixed(1)},${ys(s.hi[i]).toFixed(1)}`), bot = idx.slice().reverse().map(i => `${xs(s.x[i]).toFixed(1)},${ys(s.lo[i]).toFixed(1)}`);
        b += `<polygon points="${top.concat(bot).join(' ')}" fill="${s.color}" fill-opacity=".18"/>`;
      }
    }
    let d = '', pen = false;
    s.x.forEach((x, i) => { if (okx(x) && oky(s.y[i])) { d += `${pen ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(s.y[i]).toFixed(1)}`; pen = true; } else pen = false; });
    b += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.6}" stroke-opacity="${s.opacity ?? 1}"${s.dash ? ' stroke-dasharray="4 3"' : ''}><title>${esc(s.name)}</title></path>`;
    if (markers) s.x.forEach((x, i) => { if (okx(x) && oky(s.y[i])) b += `<circle cx="${xs(x).toFixed(1)}" cy="${ys(s.y[i]).toFixed(1)}" r="2" fill="${s.color}"/>`; });
  }
  return svg(b, `${ylabel} against ${xlabel}`);
}

// points: [{x, y, c, t?}]
// guides: [{x} | {y}, with optional color] drawn as dashed lines (e.g. classification thresholds).
export function scatter(points, { xlabel = '', ylabel = '', onClickAttr = null, guides = [] } = {}) {
  const pts = points.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!pts.length) return '<p class="mute small">No points to plot.</p>';
  const gx = guides.filter(g => g.x != null).map(g => g.x), gy = guides.filter(g => g.y != null).map(g => g.y);
  const xv = pts.map(p => p.x).concat(gx), yv = pts.map(p => p.y).concat(gy);
  const xs = scale(...pad(Math.min(...xv), Math.max(...xv)), M.l, W - M.r);
  const ys = scale(...pad(Math.min(...yv), Math.max(...yv)), H - M.b, M.t);
  let b = axes(xs, ys, xlabel, ylabel);
  for (const g of guides) {
    const c = g.color || 'var(--accent3)';
    if (g.x != null) b += `<line x1="${xs(g.x)}" x2="${xs(g.x)}" y1="${M.t}" y2="${H - M.b}" stroke="${c}" stroke-dasharray="4 3"/>`;
    else b += `<line y1="${ys(g.y)}" y2="${ys(g.y)}" x1="${M.l}" x2="${W - M.r}" stroke="${c}" stroke-dasharray="4 3"/>`;
  }
  for (const p of pts) b += `<circle cx="${xs(p.x).toFixed(1)}" cy="${ys(p.y).toFixed(1)}" r="3.2" fill="${p.c}" fill-opacity=".8"${onClickAttr && p.id != null ? ` ${onClickAttr}="${esc(p.id)}" class="pt"` : ''}>${p.t ? `<title>${esc(p.t)}</title>` : ''}</circle>`;
  return svg(b, `${ylabel} against ${xlabel}`);
}

// Flower plot: tracks [{color, pts:[[x,y]...]}] already origin-centred.
export function flower(tracks, { unit = 'µm' } = {}) {
  const all = tracks.flatMap(t => t.pts.flat()).filter(Number.isFinite);
  if (!all.length) return '<p class="mute small">No tracks.</p>';
  const r = Math.max(...all.map(Math.abs)) * 1.05 || 1, S = 240, c = S / 2, k = (S / 2 - 16) / r;
  let b = `<line class="axis" x1="${c}" x2="${c}" y1="8" y2="${S - 8}"/><line class="axis" y1="${c}" y2="${c}" x1="8" x2="${S - 8}"/>`;
  b += `<text x="${S - 10}" y="${c - 4}" text-anchor="end">${f(r)} ${esc(unit)}</text>`;
  for (const t of tracks) {
    const d = t.pts.filter(p => Number.isFinite(p[0])).map((p, i) => `${i ? 'L' : 'M'}${(c + p[0] * k).toFixed(1)} ${(c + p[1] * k).toFixed(1)}`).join('');
    b += `<path d="${d}" fill="none" stroke="${t.color}" stroke-width="1.3" stroke-opacity=".85"><title>${esc(t.name || '')}</title></path>`;
  }
  return `<svg class="chart flower" viewBox="0 0 ${S} ${S}" role="img" aria-label="Flower plot of origin-centred tracks">${b}</svg>`;
}
