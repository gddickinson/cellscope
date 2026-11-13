// Minimal pure-JS TIFF reader for CellScope recordings (*.ome.tif, (T, C, H, W) uint16 or uint8), classic TIFF or
// BigTIFF, uncompressed or zlib/deflate strips, one sample per pixel. Pages are decoded lazily (one page = one
// (t, c) plane), so a large recording opened from disk is never read into memory as a whole. The T x C layout comes
// from the OME-XML (SizeT/SizeC/SizeZ, DimensionOrder), the tifffile JSON description ("shape"), or ImageJ
// "channels=" metadata, else one channel. Tiled and multi-sample TIFFs are refused with a clear message.

// A byte source: an ArrayBuffer, or a Blob/File (read in slices, nothing loaded up front).
export function source(x) {
  if (x instanceof ArrayBuffer) return { size: x.byteLength, read: async (o, n) => x.slice(o, o + n) };
  return { size: x.size, read: async (o, n) => x.slice(o, o + n).arrayBuffer() };
}

async function inflate(bytes) {
  const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8, 18: 8 };

async function readIfd(src, off, le, big) {
  const head = new DataView(await src.read(off, big ? 8 : 2));
  const n = big ? Number(head.getBigUint64(0, le)) : head.getUint16(0, le);
  const esz = big ? 20 : 12, base = off + (big ? 8 : 2);
  const dv = new DataView(await src.read(base, n * esz + (big ? 8 : 4)));
  const tags = {};
  for (let i = 0; i < n; i++) {
    const e = i * esz, tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le);
    const count = big ? Number(dv.getBigUint64(e + 4, le)) : dv.getUint32(e + 4, le);
    const bytes = (TYPE_SIZE[type] || 1) * count, inline = bytes <= (big ? 8 : 4);
    const vOff = e + (big ? 12 : 8);
    let vdv = dv, at = vOff;
    if (!inline) {
      const p = big ? Number(dv.getBigUint64(vOff, le)) : dv.getUint32(vOff, le);
      vdv = new DataView(await src.read(p, bytes)); at = 0;
    }
    if (type === 2) { tags[tag] = new TextDecoder().decode(new Uint8Array(vdv.buffer, vdv.byteOffset + at, Math.max(0, count - 1))); continue; }
    const vals = [];
    for (let k = 0; k < count; k++) {
      const o = at + k * (TYPE_SIZE[type] || 1);
      vals.push(type === 3 ? vdv.getUint16(o, le) : type === 4 ? vdv.getUint32(o, le) : type === 16 ? Number(vdv.getBigUint64(o, le))
        : type === 1 || type === 7 ? vdv.getUint8(o) : type === 8 ? vdv.getInt16(o, le) : type === 9 ? vdv.getInt32(o, le) : 0);
    }
    tags[tag] = vals;
  }
  const nx = n * esz;
  const next = big ? Number(dv.getBigUint64(nx, le)) : dv.getUint32(nx, le);
  return { tags, next };
}

function layout(desc, nPages) {
  const d = desc || '';
  let T = nPages, C = 1, Z = 1, order = 'XYCZT', meta = {};
  const om = /<Pixels\b[^>]*>/.exec(d);
  if (om) {
    const at = k => (new RegExp(`\\b${k}="([^"]*)"`).exec(om[0]) || [])[1];
    C = +at('SizeC') || 1; Z = +at('SizeZ') || 1; T = +at('SizeT') || 1; order = at('DimensionOrder') || order;
    if (at('PhysicalSizeX')) meta.um_per_px = +at('PhysicalSizeX');
    const names = [...d.matchAll(/<Channel\b[^>]*\bName="([^"]*)"/g)].map(m => m[1]);
    if (names.length === C) meta.channel_names = names;
  } else if (d.startsWith('{')) {
    try {
      const j = JSON.parse(d);
      if (Array.isArray(j.shape)) {
        const s = j.shape, ax = (j.axes || '').toUpperCase();
        if (ax.length === s.length && ax.includes('C')) { C = s[ax.indexOf('C')]; T = nPages / C; order = ax.indexOf('C') > ax.indexOf('T') ? 'XYCZT' : 'XYTZC'; }
        else if (s.length === 4) { T = s[0]; C = s[1]; }
        else if (s.length === 3) { T = s[0]; C = 1; }
      }
    } catch { /* not JSON */ }
  } else if (/channels=(\d+)/.test(d)) {
    C = +/channels=(\d+)/.exec(d)[1]; T = Math.floor(nPages / C);
  }
  if (T * C * Z !== nPages) {                    // e.g. one position of a multi-position acquisition
    if (C > 1 && nPages % C === 0) { T = nPages / C; Z = 1; } else { C = 1; T = nPages; Z = 1; }
  }
  // page index of (t, c) at z=0
  const idx = order.indexOf('C') < order.indexOf('T') ? (t, c) => (t * Z) * C + c : (t, c) => c * T * Z + t;
  return { T, C, idx, meta };
}

// Open a TIFF: returns {T, C, H, W, bits, meta, plane(t, c) -> Promise<Uint16Array|Uint8Array>}.
export async function openTiff(input, { maxPages = 100000 } = {}) {
  const src = source(input);
  const h = new DataView(await src.read(0, 16));
  const bo = h.getUint16(0, false);
  if (bo !== 0x4949 && bo !== 0x4d4d) throw new Error('not a TIFF file');
  const le = bo === 0x4949, magic = h.getUint16(2, le), big = magic === 43;
  if (magic !== 42 && !big) throw new Error('not a TIFF file');
  let off = big ? Number(h.getBigUint64(8, le)) : h.getUint32(4, le);
  const pages = [];
  while (off && pages.length < maxPages) { const r = await readIfd(src, off, le, big); pages.push(r.tags); off = r.next; }
  if (!pages.length) throw new Error('TIFF has no images');
  const p0 = pages[0], W = p0[256][0], H = p0[257][0], bits = (p0[258] || [1])[0];
  if (p0[322]) throw new Error('tiled TIFFs are not supported in the browser');
  if ((p0[277] || [1])[0] !== 1) throw new Error('RGB / multi-sample TIFFs are not supported (expected one sample per pixel)');
  if (bits !== 8 && bits !== 16) throw new Error(`${bits}-bit TIFF not supported (8 or 16 bit)`);
  const comp = (p0[259] || [1])[0];
  if (![1, 8, 32946].includes(comp)) throw new Error(`TIFF compression ${comp} not supported (uncompressed or zlib only)`);
  const { T, C, idx, meta } = layout(p0[270], pages.length);
  const cache = new Map();
  async function plane(t, c) {
    const k = idx(t, c);
    if (cache.has(k)) return cache.get(k);
    const tg = pages[k], offs = tg[273], cnts = tg[279], pc = (tg[259] || [1])[0];
    const out = new Uint8Array(W * H * bits / 8);
    let pos = 0;
    for (let s = 0; s < offs.length; s++) {
      let b = new Uint8Array(await src.read(offs[s], cnts[s]));
      if (pc !== 1) b = await inflate(b);
      out.set(b.subarray(0, Math.min(b.length, out.length - pos)), pos); pos += b.length;
    }
    let arr;
    if (bits === 8) arr = out;
    else {
      arr = new Uint16Array(W * H);
      const dv = new DataView(out.buffer);
      for (let i = 0; i < arr.length; i++) arr[i] = dv.getUint16(2 * i, le);
    }
    if (cache.size > 64) cache.delete(cache.keys().next().value);
    cache.set(k, arr);
    return arr;
  }
  return { T, C, H, W, bits, meta, nPages: pages.length, plane };
}
