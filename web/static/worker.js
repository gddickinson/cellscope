// Web Worker: runs the tracker (track_engine.js) and the colour-by metrics (shape_engine.js) off the page's main
// thread. Message in: {id, labels (Int32Array), T, H, W, params, umPerPx, dtMin, thresholds}.
import { trackAllCells, trackedStack } from './track_engine.js';
import { computeLabelMetrics } from './shape_engine.js';

self.onmessage = e => {
  const { id, labels, T, H, W, params, umPerPx, dtMin, thresholds } = e.data;
  try {
    const m = { labels, T, H, W };
    self.postMessage({ id, progress: 0.05, stage: 'tracking' });
    const tracks = trackAllCells(m, params);
    const tracked = trackedStack(m, tracks);
    const metrics = computeLabelMetrics(tracked, umPerPx, dtMin, thresholds,
      { onCell: (d, n) => self.postMessage({ id, progress: 0.2 + 0.8 * d / n, stage: 'measuring' }) });
    self.postMessage({ id, result: { tracks, tracked: tracked.labels, metrics } }, [tracked.labels.buffer]);
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};
