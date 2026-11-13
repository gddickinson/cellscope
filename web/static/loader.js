// "Which recording" panel for the Track page: the synthetic demo, or the user's own masks.npz (+ optional .ome.tif
// image and .json sidecar), read in the browser. Calls onLoaded(ds) after setDataset.
import { esc } from './core.js';
import { loadDemo, loadOwn, setDataset } from './data.js';

export function loaderPanel(host, ds, onLoaded) {
  host.innerHTML = `<details class="panel loader" ${ds.synthetic ? '' : 'open'}>
    <summary>Recording: <b>${esc(ds.synthetic ? 'synthetic demo' : ds.name)}</b> · ${ds.m.T} frames ·
      ${ds.m.W}×${ds.m.H} px · ${ds.umPerPx ? `${ds.umPerPx} µm/px` : 'no pixel size'}${ds.dtMin ? ` · ${ds.dtMin} min/frame` : ''}
      <span class="mute small">(change)</span></summary>
    <p class="small mute">Use your own CellScope output: a <code>masks.npz</code> (key <code>labels</code>, a (T, H, W)
      label stack; raw per-frame detections or already-tracked masks, both work), optionally the recording's
      <code>.ome.tif</code> and its <code>.json</code> sidecar for the pixel size and frame interval. Files are read in
      this browser and never uploaded.</p>
    <div class="grid2">
      <div><label for="ld-masks">masks .npz (required)</label><input id="ld-masks" type="file" accept=".npz"></div>
      <div><label for="ld-img">image .tif / .ome.tif (optional)</label><input id="ld-img" type="file" accept=".tif,.tiff"></div>
      <div><label for="ld-side">sidecar .json (optional)</label><input id="ld-side" type="file" accept=".json"></div>
      <div class="grid2">
        <div><label for="ld-um">µm per px</label><input id="ld-um" type="number" step="any" min="0" placeholder="from files"></div>
        <div><label for="ld-dt">min per frame</label><input id="ld-dt" type="number" step="any" min="0" placeholder="from files"></div>
      </div>
    </div>
    <p><button data-load>Load these files</button> <button class="ghost" data-demo>Back to the synthetic demo</button>
      <span class="small" data-msg role="status"></span></p>
  </details>`;
  const $ = s => host.querySelector(s), msg = $('[data-msg]');
  const done = d => { setDataset(d); onLoaded(d); };
  $('[data-demo]').addEventListener('click', async () => { msg.textContent = 'Loading the demo...'; done(await loadDemo()); });
  $('[data-load]').addEventListener('click', async () => {
    const masks = $('#ld-masks').files[0];
    if (!masks) { msg.innerHTML = '<span class="err">Choose a masks .npz first.</span>'; return; }
    msg.textContent = 'Reading...';
    try {
      done(await loadOwn({ masks, image: $('#ld-img').files[0], sidecar: $('#ld-side').files[0],
        umPerPx: +$('#ld-um').value || null, dtMin: +$('#ld-dt').value || null }));
    } catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  });
}
