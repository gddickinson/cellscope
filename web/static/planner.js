// Run planner: what core/unified_detection.detect_recording will decide for a recording before it runs (downsample,
// mirror padding, pixel thresholds, backbone choice from probe counts) and the Test-on-frame runtime estimate.
// JS ports of the pure functions in core/pipeline_defaults.py and core/runtime_estimate.py; constants from Python.
import { $, app, esc, route } from './core.js';
import { defaults } from './data.js';
import { f } from './fmt.js';
import { decideBackbone, estimateFullRuntime, formatDuration, pixelThresholds, probeFrames, resolveDownsample, shouldUseMirrorPad } from './planner_engine.js';

const FIELDS = [['H', 'height (px)', 2048], ['W', 'width (px)', 2048], ['T', 'frames', 97], ['um', 'µm per px', 0.6523],
  ['dt', 'minutes per frame', 10], ['sec', 'Test-on-frame time (s)', 40], ['cells', 'cells on the test frame', 4]];

route('planner', async () => {
  const d = await defaults(), C = d.constants, D = d.defaults, P = d.presets;
  app().innerHTML = `<p class="kicker">Run planner</p><h1>What will the pipeline do with my recording?</h1>
    <p class="lead">Before detecting, <code>detect_recording</code> resolves a downsample factor from the frame size,
    converts the physical-unit defaults to pixels at the detection resolution, decides on mirror padding and probes
    ${C.N_SAMPLE_FRAMES_FOR_AUTO} frames with raw cpsam to choose the backbone. The probe and the detection need the
    models (desktop); the decisions are plain rules, reproduced here with the constants read from
    <code>core/pipeline_defaults.py</code>. The example values are the 0.6523 µm/px, 2048² × 97-frame recordings quoted in
    CellScope's docs; enter your own.</p>
    <div class="side"><div class="panel" id="in"></div><div id="out"></div></div>`;
  const inp = $('#in');
  inp.innerHTML = `<h3>Recording</h3>${FIELDS.slice(0, 5).map(([k, l, v]) => `<label for="p-${k}">${l}</label><input id="p-${k}" type="number" step="any" min="0" value="${v}">`).join('')}
    <h3>Settings</h3><label for="p-preset">detection preset</label><select id="p-preset">${P.order.map(n => `<option ${n === 'Default (Balanced)' ? 'selected' : ''}>${esc(n)}</option>`).join('')}</select>
    <h3>Probe</h3><label for="p-counts">cells found on the ${C.N_SAMPLE_FRAMES_FOR_AUTO} probe frames</label>
    <input id="p-counts" value="1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0">
    <h3>Test on frame</h3>${FIELDS.slice(5).map(([k, l, v]) => `<label for="p-${k}">${l}</label><input id="p-${k}" type="number" step="any" min="0" value="${v}">`).join('')}`;
  inp.querySelectorAll('input, select').forEach(e => e.addEventListener('input', update));

  function update() {
    const v = k => +$(`#p-${k}`).value, preset = P.presets.find(p => p.name === $('#p-preset').value);
    const H = Math.max(1, Math.round(v('H'))), W = Math.max(1, Math.round(v('W'))), T = Math.max(1, Math.round(v('T')));
    const um = v('um') || null, dt = v('dt') || null;
    let ds;
    try { ds = resolveDownsample(preset.values.downsample, [H, W], C); } catch (e) { ds = [1, e.message]; }
    const [fac, why] = ds, dH = Math.floor(H / fac), dW = Math.floor(W / fac);
    const pad = shouldUseMirrorPad(preset.values.use_mirror_pad, [dH, dW], C);
    const px = pixelThresholds(D, (um || 1) * fac, dt);
    const counts = $('#p-counts').value.split(/[\s,;]+/).filter(Boolean).map(Number).filter(Number.isFinite);
    const bb = counts.length ? decideBackbone(counts, C) : null;
    const est = estimateFullRuntime(v('sec'), Math.round(v('cells')), T, C.DENSITY_MULTIPLIERS);
    const knobs = P.params.map(k => `<tr><td>${esc(k)}</td><td>${esc(String(preset.values[k]))}</td><td>${k in preset.deltas ? '<span class="badge js">changed</span>' : ''}</td></tr>`).join('');
    $('#out').innerHTML = `<div class="panel out"><h3>1 · Downsample</h3>
      <p>factor <b>${fac}</b>: ${esc(why)}. Detection runs at <b>${dW} × ${dH} px</b> (labels are scaled back up afterwards).</p>
      <p class="small mute">Auto: max(H, W) &lt; ${C.DOWNSAMPLE_SMALL_PX} → 1; below ${C.DOWNSAMPLE_LARGE_PX} → 2 (~3× faster); otherwise 2 (~5× faster).</p></div>
    <div class="panel out"><h3>2 · Mirror padding</h3><p><b>${pad ? 'on' : 'off'}</b> (setting "${esc(String(preset.values.use_mirror_pad))}"):
      ${pad ? `each frame is padded with a ${C.MIRROR_PAD_PX} px reflection so cells at the edge look whole` : `auto pads only when the detection frame's shorter side is ≥ ${C.MIRROR_PAD_MIN_DIM_PX} px`}.</p></div>
    <div class="panel out"><h3>3 · Pixel thresholds at detection resolution</h3>
      <p class="small mute"><code>DEFAULTS.pixel_thresholds(${f((um || 1) * fac)} µm/px, ${dt ?? 'None'} min)</code>${um ? '' : ' (no pixel size: 1 µm/px assumed by detect_recording)'}</p>
      <dl class="kv"><dt>min_area_px</dt><dd>${px.min_area_px} (from ${D.min_area_um2} µm²; drops smaller detections)</dd>
      <dt>max_hop_px</dt><dd>${px.max_hop_px} (from ${D.max_hop_um_per_min} µm/min; the tracker's max hop when <code>tracker_hop_mode</code> is "scale_aware" (default), else ${D.search_radius_px} px)</dd>
      <dt>search_radius_px</dt><dd>${px.search_radius_px} (from ${D.search_radius_um} µm)</dd>
      <dt>cell_diameter_px</dt><dd>${px.cell_diameter_px ?? '–'} (from ${D.expected_cell_diameter_um} µm)</dd></dl></div>
    <div class="panel out"><h3>4 · Backbone (auto-select)</h3>
      <p class="small mute">Probe frames: ${probeFrames(T, C.N_SAMPLE_FRAMES_FOR_AUTO).join(', ')}</p>
      ${bb ? `<p><b>${bb.choice}</b>: ${esc(bb.reason)} (median ${f(bb.median_count)}).</p>` : '<p class="mute">Enter the probe counts.</p>'}
      <p class="small mute">cpsam_dic (the DIC fine-tune) gives tighter outlines on isolated cells but merges touching
      ones; raw cpsam handles crowded fields. The ${C.PROBE_AGGREGATOR_PERCENTILE}th percentile keeps a few empty frames
      from pulling a crowded recording below ${C.MULTICELL_THRESHOLD} cells/frame.</p></div>
    <div class="panel out"><h3>5 · Runtime estimate (Test on frame)</h3>
      <p>detection ≈ <b>${formatDuration(est.detect_only_s)}</b>; full run ≈ <b>${formatDuration(est.full_s)}</b>
      (× ${est.multiplier.toFixed(1)}, ${est.density}: tracking + gap fill + Cy5 stages grow with cell density).</p></div>
    <div class="panel"><h3>Preset "${esc(preset.name)}"</h3><p class="small mute">${esc(preset.doc)}</p>
      <div class="scroll"><table class="tbl"><thead><tr><th>knob</th><th>value</th><th></th></tr></thead><tbody>${knobs}</tbody></table></div></div>`;
  }
  update();
});
