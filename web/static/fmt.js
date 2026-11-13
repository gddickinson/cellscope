// Number formatting, colours (state colours from gui/metric_coloring.py STATE_COLORS) and metric labels.

export function f(v, d = 3) {
  if (v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '–';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return v.toLocaleString('en');
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (+v.toPrecision(d + (a >= 1 ? Math.floor(Math.log10(a)) : 0))).toString();
}
export const pval = p => (Number.isFinite(p) ? (p < 1e-4 ? p.toExponential(1) : p.toFixed(4)) : '–');

// gui/metric_coloring.py STATE_COLORS, by state code (0 unknown, 1 spread, 2 rounded, 3 edge-truncated)
export const STATE_RGB = { unknown: [130, 130, 130], spread: [60, 180, 75], rounded: [230, 50, 50], edge: [240, 160, 30] };
export const STATE_COLOR = Object.fromEntries(Object.entries(STATE_RGB).map(([k, v]) => [k, `rgb(${v})`]));
export const CODE_STATE = ['unknown', 'spread', 'rounded', 'edge'];

// Categorical palette (cells, conditions); index wraps.
export const PALETTE = ['#2ee6ff', '#ff4fd8', '#ffb02e', '#7dff6a', '#a78bff', '#ff6b6b', '#4fd1c5', '#f6e05e', '#f687b3', '#90cdf4'];
export const colour = i => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
export function cellRgb(id) {                         // stable bright colour per cell ID (golden-angle hues)
  const h = (id * 137.508) % 360, s = 0.75, l = 0.58;
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const c = n => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
  return [c(0), c(8), c(4)];
}

// Sequential ramp for colour-by-metric (dark blue -> cyan -> yellow), t in [0, 1].
const STOPS = [[40, 30, 120], [30, 120, 200], [46, 230, 255], [180, 255, 120], [255, 210, 60]];
export function ramp(t) {
  if (!Number.isFinite(t)) return [120, 120, 120];
  const x = Math.max(0, Math.min(1, t)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), u = x - i;
  return STOPS[i].map((v, k) => Math.round(v + (STOPS[i + 1][k] - v) * u));
}
export const rampCss = n => `linear-gradient(90deg, ${Array.from({ length: n }, (_, i) => `rgb(${ramp(i / (n - 1)).join(',')})`).join(',')})`;

// Readable names for the colour-by metrics (core/mask_metrics.py keys).
const LABELS = { mean_speed: 'mean speed', total_distance: 'total distance', net_displacement: 'net displacement',
  persistence: 'persistence', mean_area: 'mean area', mean_circularity: 'mean circularity', mean_solidity: 'mean solidity',
  mean_aspect_ratio: 'mean aspect ratio', mean_eccentricity: 'mean eccentricity', frames_tracked: 'frames tracked',
  frac_rounded: 'fraction rounded', area: 'area', circularity: 'circularity', solidity: 'solidity', speed: 'speed' };
export const label = c => LABELS[c] || c.replace(/_/g, ' ');
