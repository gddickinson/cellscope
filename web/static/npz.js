// Pure-JS reader for NumPy .npz / .npy files (CellScope masks.npz): a small ZIP reader (stored + deflate entries,
// ZIP64 sizes/offsets) whose deflate entries are inflated by the platform's DecompressionStream('deflate-raw')
// (browsers and Node >= 18), and an .npy parser for the numeric dtypes NumPy writes. No dependencies.

const SIG_EOCD = 0x06054b50, SIG_EOCD64 = 0x06064b50, SIG_LOC64 = 0x07064b50, SIG_CEN = 0x02014b50, SIG_LOCAL = 0x04034b50;

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const u64 = (dv, o) => Number(dv.getBigUint64(o, true));

// List the entries of a ZIP archive held in an ArrayBuffer: [{name, method, compSize, size, offset}].
export function zipEntries(buf) {
  const dv = new DataView(buf), n = buf.byteLength;
  let e = -1;
  for (let i = n - 22; i >= Math.max(0, n - 22 - 65535); i--) if (dv.getUint32(i, true) === SIG_EOCD) { e = i; break; }
  if (e < 0) throw new Error('not a ZIP/.npz file (no end-of-central-directory record)');
  let count = dv.getUint16(e + 10, true), cdOff = dv.getUint32(e + 16, true);
  if ((count === 0xffff || cdOff === 0xffffffff) && e >= 20 && dv.getUint32(e - 20, true) === SIG_LOC64) {
    const z = u64(dv, e - 12);
    if (dv.getUint32(z, true) !== SIG_EOCD64) throw new Error('corrupt ZIP64 directory');
    count = u64(dv, z + 32); cdOff = u64(dv, z + 48);
  }
  const out = [];
  let p = cdOff;
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== SIG_CEN) throw new Error('corrupt ZIP central directory');
    const method = dv.getUint16(p + 10, true), nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true), commentLen = dv.getUint16(p + 32, true);
    let compSize = dv.getUint32(p + 20, true), size = dv.getUint32(p + 24, true), offset = dv.getUint32(p + 42, true);
    const name = new TextDecoder().decode(new Uint8Array(buf, p + 46, nameLen));
    let x = p + 46 + nameLen;                                   // ZIP64 extra field: only the saturated values appear
    const xEnd = x + extraLen;
    while (x + 4 <= xEnd) {
      const id = dv.getUint16(x, true), len = dv.getUint16(x + 2, true);
      if (id === 1) {
        let q = x + 4;
        if (size === 0xffffffff) { size = u64(dv, q); q += 8; }
        if (compSize === 0xffffffff) { compSize = u64(dv, q); q += 8; }
        if (offset === 0xffffffff) { offset = u64(dv, q); }
      }
      x += 4 + len;
    }
    out.push({ name, method, compSize, size, offset });
    p = xEnd + commentLen;
  }
  return out;
}

// Raw bytes of one entry (inflated if needed).
export async function zipRead(buf, entry) {
  const dv = new DataView(buf);
  if (dv.getUint32(entry.offset, true) !== SIG_LOCAL) throw new Error(`corrupt ZIP entry ${entry.name}`);
  const start = entry.offset + 30 + dv.getUint16(entry.offset + 26, true) + dv.getUint16(entry.offset + 28, true);
  const raw = new Uint8Array(buf, start, entry.compSize);
  if (entry.method === 0) return raw.slice();
  if (entry.method === 8) return inflateRaw(raw);
  throw new Error(`unsupported ZIP compression method ${entry.method} in ${entry.name}`);
}

const DTYPES = {
  b1: [Uint8Array, 1], u1: [Uint8Array, 1], i1: [Int8Array, 1], u2: [Uint16Array, 2], i2: [Int16Array, 2],
  u4: [Uint32Array, 4], i4: [Int32Array, 4], f4: [Float32Array, 4], f8: [Float64Array, 8],
  i8: [BigInt64Array, 8], u8: [BigUint64Array, 8],
};

// Parse an .npy byte array -> {dtype, shape, data}. data is a typed array in C order (Fortran order is transposed).
export function parseNpy(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') throw new Error('not an .npy array');
  const major = u8[6], dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder('latin1').decode(u8.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] || '').split(',').map(s => s.trim()).filter(Boolean).map(Number);
  if (!descr) throw new Error('unreadable .npy header');
  if (descr.startsWith('|O')) throw new Error('object array (pickled) - not a numeric array');
  const order = descr[0], code = descr.slice(1);
  const spec = DTYPES[code];
  if (!spec) throw new Error(`unsupported dtype ${descr}`);
  const [Typed, size] = spec, n = shape.reduce((a, b) => a * b, 1), off = hstart + hlen;
  const body = u8.slice(off, off + n * size);                   // copy: aligned and owned
  if (order === '>' && size > 1) {                               // big-endian: swap bytes in place
    for (let i = 0; i < body.length; i += size) body.subarray(i, i + size).reverse();
  }
  let data = new Typed(body.buffer, 0, n);
  if (fortran && shape.length > 1) data = toCOrder(data, shape);
  return { dtype: descr, shape, data };
}

function toCOrder(data, shape) {
  const out = new data.constructor(data.length), nd = shape.length, idx = new Array(nd).fill(0);
  const fstride = shape.map((_, i) => shape.slice(0, i).reduce((a, b) => a * b, 1));
  for (let c = 0; c < data.length; c++) {
    let f = 0;
    for (let d = 0; d < nd; d++) f += idx[d] * fstride[d];
    out[c] = data[f];
    for (let d = nd - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
  }
  return out;
}

// Read every numeric array of an .npz (non-numeric entries are listed under `skipped`).
export async function readNpz(buf) {
  const arrays = {}, skipped = [];
  for (const e of zipEntries(buf)) {
    if (!e.name.endsWith('.npy')) continue;
    try { arrays[e.name.slice(0, -4)] = parseNpy(await zipRead(buf, e)); } catch (err) { skipped.push(`${e.name}: ${err.message}`); }
  }
  return { arrays, skipped };
}

// A CellScope masks.npz -> {labels: Int32Array (T*H*W), T, H, W, key, keys}. Like io/masks.py::load_masks: key
// `labels`, else the first stored array; a 2-D array becomes one frame.
export async function readMasks(buf) {
  const { arrays, skipped } = await readNpz(buf);
  const keys = Object.keys(arrays);
  const key = 'labels' in arrays ? 'labels' : keys[0];
  if (!key) throw new Error('no numeric arrays in this .npz' + (skipped.length ? ` (${skipped.join('; ')})` : ''));
  const a = arrays[key];
  let shape = a.shape;
  if (shape.length === 2) shape = [1, ...shape];
  if (shape.length !== 3) throw new Error(`masks array "${key}" has shape (${a.shape}); expected (T, H, W)`);
  let labels;
  if (a.data instanceof Int32Array) labels = a.data;
  else {
    labels = new Int32Array(a.data.length);
    const big = typeof a.data[0] === 'bigint';
    for (let i = 0; i < labels.length; i++) labels[i] = big ? Number(a.data[i]) : a.data[i];
  }
  return { labels, T: shape[0], H: shape[1], W: shape[2], key, keys };
}
